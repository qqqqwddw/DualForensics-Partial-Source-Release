import math
from typing import Dict, Iterable, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def parse_stage_scales(raw: Union[Iterable[int], str]) -> List[int]:
    if isinstance(raw, str):
        tokens = [t.strip() for t in raw.split(",")]
    else:
        tokens = [str(v).strip() for v in raw]

    scales = []
    for tok in tokens:
        if tok == "":
            continue
        val = int(tok)
        if val <= 0:
            continue
        scales.append(val)

    if not scales:
        return [1, 2, 4, 8]
    return sorted(set(scales))


class NoiseResidualExtractor(nn.Module):
    def __init__(self, use_grayscale: bool = True):
        super().__init__()
        self.use_grayscale = use_grayscale
        laplace = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplace_kernel", laplace, persistent=False)
        srm = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 2.0, -1.0, 0.0],
             [0.0, 2.0, -4.0, 2.0, 0.0],
             [0.0, -1.0, 2.0, -1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 5, 5)
        self.register_buffer("srm_kernel", srm, persistent=False)

    def _to_gray(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_grayscale:
            return x.mean(dim=1, keepdim=True)
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    @staticmethod
    def _normalise(noise: torch.Tensor) -> torch.Tensor:
        b = noise.shape[0]
        flat = noise.flatten(1)
        n_min = flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        n_max = flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        return ((noise - n_min) / (n_max - n_min + 1e-6)).clamp(0.0, 1.0)

    def forward(self, x_unit: torch.Tensor) -> torch.Tensor:
        x = x_unit.clamp(0.0, 1.0)
        x_gray = self._to_gray(x)

        kernel = self.laplace_kernel.to(device=x_gray.device, dtype=x_gray.dtype)
        lap = F.conv2d(x_gray, kernel, padding=1).abs()

        srm_kernel = self.srm_kernel.to(device=x_gray.device, dtype=x_gray.dtype)
        srm = F.conv2d(x_gray, srm_kernel, padding=2).abs()

        blur3 = F.avg_pool2d(x_gray, kernel_size=3, stride=1, padding=1)
        blur7 = F.avg_pool2d(x_gray, kernel_size=7, stride=1, padding=3)
        dog = (blur3 - blur7).abs()
        high = (x_gray - blur3).abs()

        noise = 0.35 * self._normalise(lap) + 0.30 * self._normalise(srm)
        noise = noise + 0.20 * self._normalise(dog) + 0.15 * self._normalise(high)
        return self._normalise(noise)


class NoiseGuidedAttentionStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        topk_ratio: float = 0.25,
        detach_mask: bool = True,
        max_tokens: int = 4096,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.topk_ratio = topk_ratio
        self.detach_mask = detach_mask
        self.max_tokens = max(64, int(max_tokens))

        self.q_noise = nn.Conv2d(1, embed_dim, kernel_size=1)
        self.k_noise = nn.Conv2d(1, embed_dim, kernel_size=1)

        self.q_img = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.k_img = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.v_img = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False),
            nn.GELU(),
        )

    def _build_noise_mask(self, noise_map: torch.Tensor):
        q = self.q_noise(noise_map).flatten(2).transpose(1, 2)
        k = self.k_noise(noise_map).flatten(2).transpose(1, 2)

        sim = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(self.embed_dim))
        sim = torch.softmax(sim, dim=-1)

        topk = max(1, int(sim.shape[-1] * self.topk_ratio))
        _, indices = torch.topk(-sim, k=topk, dim=-1)

        mask = torch.zeros_like(sim)
        mask.scatter_(2, indices, 1.0)
        if self.detach_mask:
            mask = mask.detach()

        density = mask.mean(dim=(1, 2))
        return mask, density

    def forward(self, feat_map: torch.Tensor, noise_map: torch.Tensor):
        b, _, h, w = feat_map.shape
        base_feat = feat_map
        noise_map = F.interpolate(noise_map, size=(h, w), mode="bilinear", align_corners=False)

        work_h, work_w = h, w
        if h * w > self.max_tokens:
            scale = math.sqrt(float(self.max_tokens) / float(h * w))
            work_h = max(1, int(h * scale))
            work_w = max(1, int(w * scale))
            feat_map = F.interpolate(feat_map, size=(work_h, work_w), mode="bilinear", align_corners=False)
            noise_map = F.interpolate(noise_map, size=(work_h, work_w), mode="bilinear", align_corners=False)

        mask, density = self._build_noise_mask(noise_map)

        q = self.q_img(feat_map).flatten(2).transpose(1, 2)
        k = self.k_img(feat_map).flatten(2).transpose(1, 2)
        v = self.v_img(feat_map).flatten(2).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(self.embed_dim))
        logits = logits.masked_fill(mask < 0.5, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, self.embed_dim, work_h, work_w)
        out = self.out_proj(out)
        guided = feat_map + out
        if work_h != h or work_w != w:
            guided = F.interpolate(guided, size=(h, w), mode="bilinear", align_corners=False)
        return base_feat + (guided - base_feat), density.mean()


class MultiScaleNoiseGuidance(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 128,
        stage_scales=(1, 2, 4, 8),
        topk_ratio: float = 0.25,
        detach_mask: bool = True,
        use_grayscale: bool = True,
        max_tokens: int = 4096,
    ):
        super().__init__()
        self.stage_scales = parse_stage_scales(stage_scales)
        self.noise_extractor = NoiseResidualExtractor(use_grayscale=use_grayscale)
        self.stages = nn.ModuleList(
            [
                NoiseGuidedAttentionStage(
                    in_channels=in_channels,
                    embed_dim=embed_dim,
                    topk_ratio=topk_ratio,
                    detach_mask=detach_mask,
                    max_tokens=max_tokens,
                )
                for _ in self.stage_scales
            ]
        )
        self.stage_logits = nn.Parameter(torch.zeros(len(self.stage_scales), dtype=torch.float32))
        self.fuse = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.local_evidence = nn.Sequential(
            nn.Conv2d(in_channels + 1, embed_dim, kernel_size=3, padding=1),
            _group_norm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, in_channels, kernel_size=1),
            nn.GELU(),
        )

    def _stage_hw(self, base_h: int, base_w: int, scale: int):
        return max(1, base_h // scale), max(1, base_w // scale)

    def forward(self, feat_map: torch.Tensor, x_unit: torch.Tensor):
        if feat_map is None:
            raise ValueError("feat_map must not be None when using MultiScaleNoiseGuidance.")

        base_h, base_w = feat_map.shape[-2:]
        noise_map = self.noise_extractor(x_unit)

        stage_outputs = []
        densities = []
        for stage_module, scale in zip(self.stages, self.stage_scales):
            h, w = self._stage_hw(base_h, base_w, scale)
            if h == base_h and w == base_w:
                stage_feat = feat_map
            else:
                stage_feat = F.interpolate(feat_map, size=(h, w), mode="bilinear", align_corners=False)

            guided_stage, density = stage_module(stage_feat, noise_map)
            if h != base_h or w != base_w:
                guided_stage = F.interpolate(
                    guided_stage,
                    size=(base_h, base_w),
                    mode="bilinear",
                    align_corners=False,
                )
            stage_outputs.append(guided_stage)
            densities.append(density)

        stage_weights = torch.softmax(self.stage_logits, dim=0)
        fused = torch.zeros_like(feat_map)
        for idx, guided_stage in enumerate(stage_outputs):
            fused = fused + stage_weights[idx] * guided_stage

        noise_at_feat = F.interpolate(noise_map, size=(base_h, base_w), mode="bilinear", align_corners=False)
        local = self.local_evidence(torch.cat([feat_map, noise_at_feat], dim=1))
        fused = feat_map + self.fuse(fused - feat_map) + local
        stats: Dict[str, torch.Tensor] = {
            "mask_density": torch.stack(densities).mean(),
            "stage_weights": stage_weights.detach(),
        }
        return fused, stats
