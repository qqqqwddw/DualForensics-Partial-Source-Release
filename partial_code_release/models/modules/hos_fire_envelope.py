import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mid_frequency import build_ring_band_mask


AuxForward = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
AnchorAuxForward = Callable[[torch.Tensor], torch.Tensor]


def _as_flat_label(label: torch.Tensor) -> torch.Tensor:
    return label.float().view(-1)


class HOSFireEnvelope(nn.Module):
    """HOS-guided FIRE/REM evidence module.

    The module keeps the caller interface small: given the original image in
    [0, 1], the original DINO spatial features, and a callback that forwards a
    transformed unit image through the backbone, it produces:
    - a top-k residual-evidence logit;
    - constrained HOS/FIRE frequency-mask losses;
    - optional real-centric spectral-boundary loss;
    - optional degradation/anchor consistency losses.
    """

    def __init__(
        self,
        feature_dim: int,
        mask_hidden_dim: int = 64,
        score_hidden_dim: int = 64,
        topk_ratio: float = 0.10,
        mid_low_ratio: float = 0.15,
        mid_high_ratio: float = 0.45,
        num_phase_shifts: int = 6,
        boundary_phase_eps: float = 0.12,
        boundary_margin: float = 0.25,
        rank_margin: float = 0.25,
        evidence_fake_weight: float = 1.0,
        boundary_cls_weight: float = 0.0,
        tangent_rank: int = 8,
        detach_aux_maps: bool = False,
        cdc_degrade_prob: float = 1.0,
        cdc_noise_std: float = 0.015,
        cdc_min_scale: float = 0.50,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.topk_ratio = float(topk_ratio)
        self.mid_low_ratio = float(mid_low_ratio)
        self.mid_high_ratio = float(mid_high_ratio)
        self.num_phase_shifts = max(1, int(num_phase_shifts))
        self.boundary_phase_eps = float(boundary_phase_eps)
        self.boundary_margin = float(boundary_margin)
        self.rank_margin = float(rank_margin)
        self.evidence_fake_weight = float(evidence_fake_weight)
        self.boundary_cls_weight = float(boundary_cls_weight)
        self.tangent_rank = max(1, int(tangent_rank))
        self.detach_aux_maps = bool(detach_aux_maps)
        self.cdc_degrade_prob = float(cdc_degrade_prob)
        self.cdc_noise_std = float(cdc_noise_std)
        self.cdc_min_scale = float(cdc_min_scale)

        self.mask_net = nn.Sequential(
            nn.Conv2d(3, mask_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mask_hidden_dim, mask_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mask_hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.score_net = nn.Sequential(
            nn.Conv2d(1, score_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(score_hidden_dim, score_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(score_hidden_dim, 1, kernel_size=1),
        )

    @staticmethod
    def _zero(label: torch.Tensor) -> torch.Tensor:
        return label.float().sum() * 0.0

    @staticmethod
    def _to_gray(x_unit: torch.Tensor) -> torch.Tensor:
        r, g, b = x_unit[:, 0:1], x_unit[:, 1:2], x_unit[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    @staticmethod
    def _normalise_map(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.flatten(1)
        x_min = flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        x_max = flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        return ((x - x_min) / (x_max - x_min + 1e-6)).clamp(0.0, 1.0)

    @staticmethod
    def _total_variation(x: torch.Tensor) -> torch.Tensor:
        tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return tv_h + tv_w

    def _local_residual_gray(self, x_unit: torch.Tensor) -> torch.Tensor:
        gray = self._to_gray(x_unit).float().clamp(0.0, 1.0)
        low = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        return gray - low

    def _radial_inputs(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ring = build_ring_band_mask(
            height=height,
            width=width,
            low_ratio=self.mid_low_ratio,
            high_ratio=self.mid_high_ratio,
            device=device,
            dtype=torch.float32,
        ).expand(batch_size, -1, -1, -1)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        radius = radius / (radius.max() + 1e-6)
        radius = radius.view(1, 1, height, width).expand(batch_size, -1, -1, -1)
        return ring, radius

    def _spectrum_pack(self, x_unit: torch.Tensor) -> Dict[str, torch.Tensor]:
        residual = self._local_residual_gray(x_unit)
        freq = torch.fft.fftshift(torch.fft.fft2(residual, dim=(-2, -1)), dim=(-2, -1))
        log_mag = torch.log1p(torch.abs(freq))
        log_mag = self._normalise_map(log_mag)

        b, _, h, w = residual.shape
        ring, radius = self._radial_inputs(b, h, w, residual.device)

        z = freq / (torch.abs(freq) + 1e-6)
        coupling = torch.zeros_like(log_mag)
        for idx in range(self.num_phase_shifts):
            dy = 1 + idx
            dx = 1 + (idx * 2) % max(2, w // 4)
            z1 = torch.roll(z, shifts=(dy, dx), dims=(-2, -1))
            z2 = torch.roll(z, shifts=(2 * dy, 2 * dx), dims=(-2, -1))
            closure = (z * z1 * torch.conj(z2)).real.abs()
            coupling = coupling + closure
        coupling = coupling / float(self.num_phase_shifts)
        hos_target = self._normalise_map(coupling * log_mag * ring).detach()

        mask_input = torch.cat([log_mag, ring, radius], dim=1)
        return {
            "log_mag": log_mag,
            "ring": ring,
            "radius": radius,
            "hos_target": hos_target,
            "mask_input": mask_input,
        }

    def _filter_rgb(self, x_unit: torch.Tensor, band_mask: torch.Tensor) -> torch.Tensor:
        x = x_unit.float().clamp(0.0, 1.0)
        mask = band_mask.float().clamp(0.0, 1.0)
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)

        freq = torch.fft.fftshift(torch.fft.fft2(x * 255.0, dim=(-2, -1)), dim=(-2, -1))
        filtered = freq * mask
        filtered = torch.fft.ifftshift(filtered, dim=(-2, -1))
        x_filtered = torch.fft.ifft2(filtered, dim=(-2, -1)).abs()

        b, c = x_filtered.shape[:2]
        flat = x_filtered.flatten(2)
        x_min = flat.min(dim=2, keepdim=True).values.view(b, c, 1, 1)
        x_max = flat.max(dim=2, keepdim=True).values.view(b, c, 1, 1)
        x_filtered = (x_filtered - x_min) / (x_max - x_min + 1e-6)
        return x_filtered.clamp(0.0, 1.0).to(dtype=x_unit.dtype)

    def _score_delta(self, delta_map: torch.Tensor) -> torch.Tensor:
        score_map = self.score_net(delta_map.float()).flatten(1)
        num_points = score_map.shape[1]
        k = max(1, int(round(num_points * self.topk_ratio)))
        k = min(k, num_points)
        return torch.topk(score_map, k=k, dim=1).values.mean(dim=1, keepdim=True)

    def _call_aux_forward(self, aux_map_forward: AuxForward, x_unit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.detach_aux_maps:
            with torch.no_grad():
                logit, aux_map = aux_map_forward(x_unit)
            logit = logit.detach() if logit is not None else logit
            aux_map = aux_map.detach() if aux_map is not None else aux_map
            return logit, aux_map
        return aux_map_forward(x_unit)

    @staticmethod
    def _align_aux(aux_a: torch.Tensor, aux_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if aux_a.shape[-2:] != aux_b.shape[-2:]:
            aux_b = F.interpolate(aux_b, size=aux_a.shape[-2:], mode="bilinear", align_corners=False)
        return aux_a, aux_b

    def evidence_logit(
        self,
        x_unit: torch.Tensor,
        aux_map_o: torch.Tensor,
        aux_map_forward: AuxForward,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pack = self._spectrum_pack(x_unit)
        raw_mask = self.mask_net(pack["mask_input"].to(dtype=next(self.parameters()).dtype))
        ring = pack["ring"].to(device=raw_mask.device, dtype=raw_mask.dtype)
        filter_mask = raw_mask * ring

        x_filtered = self._filter_rgb(x_unit, filter_mask)
        _, aux_map_filtered = self._call_aux_forward(aux_map_forward, x_filtered)
        if aux_map_filtered is None:
            raise RuntimeError("HOSFireEnvelope requires aux_map from filtered branch.")

        if self.detach_aux_maps:
            aux_map_o = aux_map_o.detach()
        aux_a, aux_b = self._align_aux(aux_map_o, aux_map_filtered)
        delta_map = torch.mean(torch.abs(aux_a - aux_b), dim=1, keepdim=True)
        logit = self._score_delta(delta_map)
        pack.update(
            {
                "raw_mask": raw_mask,
                "filter_mask": filter_mask,
                "x_filtered": x_filtered,
                "aux_map_filtered": aux_map_filtered,
                "delta_map": delta_map,
            }
        )
        return logit, pack

    def _mask_losses(self, pack: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_mask = pack["raw_mask"]
        ring = pack["ring"].to(device=raw_mask.device, dtype=raw_mask.dtype)
        target = pack["hos_target"].to(device=raw_mask.device, dtype=raw_mask.dtype)
        filter_mask = pack["filter_mask"]

        loss_target = F.smooth_l1_loss(filter_mask, target)
        outside = torch.mean(raw_mask * (1.0 - ring))
        tv = self._total_variation(raw_mask)
        density = torch.abs(filter_mask.mean() - target.mean().detach())
        loss_mask = outside + 0.10 * tv + density
        return loss_mask, loss_target

    def _evidence_loss(self, logit_hos: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label_flat = _as_flat_label(label)
        kwargs = {}
        if abs(self.evidence_fake_weight - 1.0) > 1e-6:
            kwargs["pos_weight"] = torch.tensor(
                self.evidence_fake_weight,
                device=logit_hos.device,
                dtype=logit_hos.dtype,
            )
        return F.binary_cross_entropy_with_logits(logit_hos.view(-1), label_flat, **kwargs)

    def _spectral_boundary_perturb(self, x_unit: torch.Tensor) -> torch.Tensor:
        x = x_unit.float().clamp(0.0, 1.0)
        b, _, h, w = x.shape
        ring = build_ring_band_mask(
            height=h,
            width=w,
            low_ratio=self.mid_low_ratio,
            high_ratio=self.mid_high_ratio,
            device=x.device,
            dtype=torch.float32,
        )
        noise = torch.randn(b, 1, h, w, device=x.device, dtype=torch.float32) * self.boundary_phase_eps
        noise = noise * ring

        freq = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        mag = torch.abs(freq)
        phase = torch.angle(freq) + noise
        perturbed = torch.polar(mag, phase)
        perturbed = torch.fft.ifftshift(perturbed, dim=(-2, -1))
        x_rec = torch.fft.ifft2(perturbed, dim=(-2, -1)).real
        x_boundary = (0.55 * x + 0.45 * x_rec).clamp(0.0, 1.0)
        return x_boundary.to(dtype=x_unit.dtype)

    def _boundary_loss(
        self,
        x_unit: torch.Tensor,
        label: torch.Tensor,
        logit_hos: torch.Tensor,
        aux_map_forward: AuxForward,
        cls_logit_o: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        label_flat = _as_flat_label(label)
        real_mask = label_flat < 0.5
        if not real_mask.any():
            return self._zero(label), None, None

        x_boundary = self._spectral_boundary_perturb(x_unit)
        cls_logit_boundary, aux_map_boundary = self._call_aux_forward(aux_map_forward, x_boundary)
        if aux_map_boundary is None:
            raise RuntimeError("HOS boundary loss requires aux_map for boundary samples.")

        logit_boundary_hos, _ = self.evidence_logit(x_boundary, aux_map_boundary, aux_map_forward)
        logit_boundary_flat = logit_boundary_hos.view(-1)
        fake_target = torch.ones_like(logit_boundary_flat[real_mask])
        loss_bce = F.binary_cross_entropy_with_logits(logit_boundary_flat[real_mask], fake_target)

        logit_real = logit_hos.view(-1).detach()
        loss_margin = F.relu(self.boundary_margin - (logit_boundary_flat[real_mask] - logit_real[real_mask])).mean()
        loss = loss_bce + loss_margin

        if cls_logit_o is not None and self.boundary_cls_weight > 0.0:
            cls_logit_boundary = cls_logit_boundary.view(-1)
            loss_cls = F.binary_cross_entropy_with_logits(cls_logit_boundary[real_mask], fake_target)
            cls_logit = cls_logit_o.view(-1).detach()
            loss_cls_margin = F.relu(self.boundary_margin - (cls_logit_boundary[real_mask] - cls_logit[real_mask])).mean()
            loss = loss + self.boundary_cls_weight * (loss_cls + loss_cls_margin)

        return loss, logit_boundary_hos, aux_map_boundary

    def _rank_loss(
        self,
        label: torch.Tensor,
        logit_hos: torch.Tensor,
        logit_boundary_hos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        label_flat = _as_flat_label(label)
        logit_flat = logit_hos.view(-1)
        real = logit_flat[label_flat < 0.5]
        fake = logit_flat[label_flat >= 0.5]
        losses = []
        if real.numel() > 0 and fake.numel() > 0:
            losses.append(F.relu(self.rank_margin + real[:, None] - fake[None, :]).mean())
        if real.numel() > 0 and logit_boundary_hos is not None:
            boundary = logit_boundary_hos.view(-1)[label_flat < 0.5]
            if boundary.numel() > 0:
                losses.append(F.relu(self.rank_margin + real - boundary).mean())
        if not losses:
            return self._zero(label)
        return torch.stack(losses).mean()

    def _tangent_loss(
        self,
        label: torch.Tensor,
        aux_map_o: torch.Tensor,
        aux_map_boundary: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if aux_map_boundary is None:
            return self._zero(label)
        label_flat = _as_flat_label(label)
        real_mask = label_flat < 0.5
        if int(real_mask.sum().item()) < 2:
            return self._zero(label)

        aux_o, aux_b = self._align_aux(aux_map_o, aux_map_boundary)
        h_real = F.adaptive_avg_pool2d(aux_o[real_mask], output_size=1).flatten(1).float()
        h_boundary = F.adaptive_avg_pool2d(aux_b[real_mask], output_size=1).flatten(1).float()
        h_real = F.normalize(h_real, dim=1)
        h_boundary = F.normalize(h_boundary, dim=1)

        centered = h_real - h_real.mean(dim=0, keepdim=True)
        n_real, dim = centered.shape
        rank = min(self.tangent_rank, n_real - 1, dim)
        if rank <= 0:
            return self._zero(label)

        with torch.no_grad():
            try:
                _, _, vh = torch.linalg.svd(centered.detach(), full_matrices=False)
                basis = vh[:rank].transpose(0, 1).contiguous()
            except RuntimeError:
                return self._zero(label)

        delta = h_boundary - h_real.detach()
        projected = torch.matmul(torch.matmul(delta, basis), basis.transpose(0, 1))
        off_manifold = delta - projected
        denom = delta.detach().pow(2).mean().clamp_min(1e-6)
        return off_manifold.pow(2).mean() / denom

    def _degrade(self, x_unit: torch.Tensor) -> torch.Tensor:
        x = x_unit.float().clamp(0.0, 1.0)
        if self.cdc_degrade_prob < 1.0:
            keep = torch.rand(1, device=x.device).item() > self.cdc_degrade_prob
            if keep:
                return x.to(dtype=x_unit.dtype)

        h, w = x.shape[-2:]
        num_ops = int(torch.randint(2, 5, (1,), device=x.device).item())
        ops = torch.randperm(6, device=x.device)[:num_ops].tolist()
        for op in ops:
            if op == 0:
                min_scale = min(max(self.cdc_min_scale, 0.30), 1.0)
                scale = float(torch.empty(1, device=x.device).uniform_(min_scale, 0.95).item())
                dh = max(16, int(round(h * scale)))
                dw = max(16, int(round(w * scale)))
                x = F.interpolate(x, size=(dh, dw), mode="bilinear", align_corners=False)
                x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            elif op == 1:
                if self.cdc_noise_std > 0:
                    x = x + torch.randn_like(x) * self.cdc_noise_std
            elif op == 2:
                # Differentiable JPEG-like quantisation. Random levels cover
                # strong social compression without calling PIL inside training.
                levels = float(torch.empty(1, device=x.device).uniform_(15.0, 63.0).item())
                x_quant = torch.round(x.clamp(0.0, 1.0) * levels) / levels
                x = x + (x_quant - x).detach()
            elif op == 3:
                factor = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.82, 1.18)
                bias = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(-0.06, 0.06)
                x = x * factor + bias
            elif op == 4:
                gamma = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.85, 1.25)
                x = x.clamp(0.0, 1.0).pow(gamma)
            else:
                crop_scale = float(torch.empty(1, device=x.device).uniform_(0.82, 1.0).item())
                ch = max(16, int(round(h * crop_scale)))
                cw = max(16, int(round(w * crop_scale)))
                if ch < h or cw < w:
                    top = int(torch.randint(0, h - ch + 1, (1,), device=x.device).item())
                    left = int(torch.randint(0, w - cw + 1, (1,), device=x.device).item())
                    x = x[:, :, top : top + ch, left : left + cw]
                    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        if torch.rand(1, device=x.device).item() < 0.50:
            x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x.clamp(0.0, 1.0).to(dtype=x_unit.dtype)

    def _cdc_loss(
        self,
        x_unit: torch.Tensor,
        label: torch.Tensor,
        cls_logit_o: Optional[torch.Tensor],
        logit_hos: torch.Tensor,
        delta_map: torch.Tensor,
        aux_map_o: torch.Tensor,
        aux_map_forward: AuxForward,
        anchor_aux_forward: Optional[AnchorAuxForward],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_deg = self._degrade(x_unit)
        logit_deg, aux_map_deg = self._call_aux_forward(aux_map_forward, x_deg)
        logit_hos_deg, pack_deg = self.evidence_logit(x_deg, aux_map_deg, aux_map_forward)

        loss_score = F.mse_loss(torch.sigmoid(logit_hos_deg), torch.sigmoid(logit_hos.detach()))
        if cls_logit_o is not None:
            loss_score = loss_score + F.mse_loss(torch.sigmoid(logit_deg), torch.sigmoid(cls_logit_o.detach()))

        delta_a, delta_b = self._align_aux(delta_map, pack_deg["delta_map"])
        loss_res = F.smooth_l1_loss(delta_b, delta_a.detach())
        loss_cdc = loss_score + loss_res

        loss_anchor_res = self._zero(label)
        if anchor_aux_forward is not None:
            anchor_o = anchor_aux_forward(x_unit)
            anchor_deg = anchor_aux_forward(x_deg)
            if anchor_o is not None and anchor_deg is not None:
                aux_o_aligned, anchor_o = self._align_aux(aux_map_o, anchor_o)
                aux_deg_aligned, anchor_deg = self._align_aux(aux_map_deg, anchor_deg)
                residual_o = aux_o_aligned - anchor_o.detach()
                residual_deg = aux_deg_aligned - anchor_deg.detach()
                loss_anchor_res = F.smooth_l1_loss(residual_deg, residual_o.detach())

        return loss_cdc, loss_anchor_res

    def forward(
        self,
        x_unit: torch.Tensor,
        aux_map_o: torch.Tensor,
        label: torch.Tensor,
        aux_map_forward: AuxForward,
        cls_logit_o: Optional[torch.Tensor] = None,
        anchor_aux_forward: Optional[AnchorAuxForward] = None,
        compute_boundary: bool = False,
        compute_cdc: bool = False,
        compute_anchor: bool = False,
    ) -> Dict[str, object]:
        label_flat = _as_flat_label(label)
        logit_hos, pack = self.evidence_logit(x_unit, aux_map_o, aux_map_forward)
        loss_evidence = self._evidence_loss(logit_hos, label_flat)
        loss_mask, loss_target = self._mask_losses(pack)

        loss_boundary = self._zero(label)
        logit_boundary_hos = None
        aux_map_boundary = None
        if compute_boundary:
            loss_boundary, logit_boundary_hos, aux_map_boundary = self._boundary_loss(
                x_unit=x_unit,
                label=label,
                logit_hos=logit_hos,
                aux_map_forward=aux_map_forward,
                cls_logit_o=cls_logit_o,
            )

        loss_rank = self._rank_loss(label, logit_hos, logit_boundary_hos)
        loss_tangent = self._tangent_loss(label, aux_map_o, aux_map_boundary)

        loss_anchor = self._zero(label)
        if compute_anchor and anchor_aux_forward is not None:
            anchor_aux = anchor_aux_forward(x_unit)
            if anchor_aux is not None:
                aux_o, anchor_aux = self._align_aux(aux_map_o, anchor_aux)
                loss_anchor = F.smooth_l1_loss(aux_o, anchor_aux.detach())

        loss_cdc = self._zero(label)
        loss_anchor_res = self._zero(label)
        if compute_cdc:
            loss_cdc, loss_anchor_res = self._cdc_loss(
                x_unit=x_unit,
                label=label,
                cls_logit_o=cls_logit_o,
                logit_hos=logit_hos,
                delta_map=pack["delta_map"],
                aux_map_o=aux_map_o,
                aux_map_forward=aux_map_forward,
                anchor_aux_forward=anchor_aux_forward,
            )

        stats = {
            "mask_density": pack["filter_mask"].detach().mean(),
            "target_density": pack["hos_target"].detach().mean(),
            "delta_mean": pack["delta_map"].detach().mean(),
        }
        return {
            "logit": logit_hos,
            "loss_evidence": loss_evidence,
            "loss_mask": loss_mask,
            "loss_target": loss_target,
            "loss_boundary": loss_boundary,
            "loss_rank": loss_rank,
            "loss_tangent": loss_tangent,
            "loss_cdc": loss_cdc,
            "loss_anchor": loss_anchor,
            "loss_anchor_res": loss_anchor_res,
            "stats": stats,
        }
