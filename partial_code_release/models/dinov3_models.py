import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_TO_ENTRYPOINT = {
    "DINOv3:ViT-S/16_svd": "dinov3_vits16",
    "DINOv3:ViT-B/16_svd": "dinov3_vitb16",
    "DINOv3:ViT-L/16_svd": "dinov3_vitl16",
    "DINOv3:ViT-L/16plus_svd": "dinov3_vitl16plus",
    "DINOv3:ViT-H/16plus_svd": "dinov3_vith16plus",
    "DINOv3:ViT-7B/16_svd": "dinov3_vit7b16",
}


def _resolve_repo_dir(repo_dir_opt):
    candidates = []
    if repo_dir_opt:
        candidates.append(Path(repo_dir_opt).expanduser())
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "dinov3-main" / "dinov3-main",
            cwd / "dinov3-main",
        ]
    )

    for candidate in candidates:
        repo_dir = candidate.resolve()
        if (repo_dir / "hubconf.py").exists() and (repo_dir / "dinov3").is_dir():
            return str(repo_dir)
    raise FileNotFoundError(
        "Cannot locate DINOv3 repo directory. "
        "Please set --dinov3_repo_dir to a folder containing hubconf.py and dinov3/."
    )


def _ensure_repo_on_sys_path(repo_dir):
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def _resolve_weights_path(weights_opt):
    if not weights_opt:
        return None
    p = Path(weights_opt).expanduser()
    if p.exists():
        return str(p.resolve())
    return weights_opt


class DinoV3Model(nn.Module):
    def __init__(self, name, opt, num_classes=1):
        super(DinoV3Model, self).__init__()
        if name not in ARCH_TO_ENTRYPOINT:
            raise ValueError(f"Unsupported DINOv3 arch: {name}")

        self.use_svd = opt.use_svd
        repo_dir = _resolve_repo_dir(getattr(opt, "dinov3_repo_dir", ""))
        _ensure_repo_on_sys_path(repo_dir)

        from dinov3.hub import backbones as dinov3_backbones

        constructor_name = ARCH_TO_ENTRYPOINT[name]
        constructor = getattr(dinov3_backbones, constructor_name)
        weights = _resolve_weights_path(getattr(opt, "dinov3_weights", ""))

        if weights is None:
            self.model = constructor(pretrained=True)
        else:
            self.model = constructor(pretrained=True, weights=weights)

        if self.use_svd:
            residual_rank = int(getattr(opt, "svd_residual_rank", 1))
            if residual_rank > 0:
                svd_rank = max(1, self.model.embed_dim - residual_rank)
            else:
                keep_ratio = float(getattr(opt, "svd_keep_rank_ratio", 0.999))
                keep_ratio = min(max(keep_ratio, 0.0), 0.999)
                svd_rank = max(1, int(round(self.model.embed_dim * keep_ratio)))
            self.model = apply_svd_residual_to_dinov3_self_attn(
                self.model,
                r=svd_rank,
                low_rank_forward=getattr(opt, "svd_low_rank_forward", False),
            )

        for param_name, param in self.model.named_parameters():
            print(f"{param_name}: {param.requires_grad}")
        num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.model.parameters())
        print(f"Number of total parameters: {num_total_param}, tunable parameters: {num_param}")

        hidden_size = self.model.embed_dim
        self.fc = nn.Linear(hidden_size, num_classes)
        self.feature_dim = self.fc.in_features

    def forward(self, x, return_feature=False, return_aux_map=False, return_patch_tokens=False):
        extracted = self.extract_features(
            x,
            return_aux_map=return_aux_map,
            return_patch_tokens=return_patch_tokens,
        )
        if return_patch_tokens:
            features, logits, spatial_feat, patch_tokens = extracted
        else:
            features, logits, spatial_feat = extracted
        if return_feature:
            if return_patch_tokens:
                return features, logits, spatial_feat, patch_tokens
            return features, logits, spatial_feat
        return logits

    def extract_features(self, x, return_aux_map=False, return_patch_tokens=False):
        outputs = self.model.forward_features(x)
        features = outputs["x_norm_clstoken"]
        logits = self.fc(features)
        patch_tokens = outputs.get("x_norm_patchtokens", None)

        spatial_feat = None
        if return_aux_map:
            tokens = patch_tokens  # B x N x C
            if tokens.ndim == 3:
                bsz, num_tokens, channels = tokens.shape
                patch_size = getattr(self.model, "patch_size", 16)
                if isinstance(patch_size, tuple):
                    patch_size = patch_size[0]

                feat_h = x.shape[-2] // patch_size
                feat_w = x.shape[-1] // patch_size
                if feat_h * feat_w == num_tokens:
                    spatial_feat = tokens.transpose(1, 2).reshape(bsz, channels, feat_h, feat_w)
                else:
                    side = int(round(math.sqrt(num_tokens)))
                    if side * side == num_tokens:
                        spatial_feat = tokens.transpose(1, 2).reshape(bsz, channels, side, side)
                    else:
                        spatial_feat = tokens.transpose(1, 2).unsqueeze(-2)
        if return_patch_tokens:
            return features, logits, spatial_feat, patch_tokens
        return features, logits, spatial_feat

    @contextmanager
    def svd_residual_enabled(self, enabled: bool):
        modules = [module for module in self.model.modules() if isinstance(module, SVDResidualLinear)]
        previous = [module.residual_enabled for module in modules]
        for module in modules:
            module.residual_enabled = bool(enabled)
        try:
            yield
        finally:
            for module, value in zip(modules, previous):
                module.residual_enabled = value

    def extract_anchor_aux_map(self, x):
        """Return frozen-main DINO spatial features by disabling SVD residual paths."""
        was_training = self.model.training
        with torch.no_grad():
            try:
                self.model.eval()
                with self.svd_residual_enabled(False):
                    _, _, spatial_feat = self.extract_features(x, return_aux_map=True)
            finally:
                self.model.train(was_training)
        return None if spatial_feat is None else spatial_feat.detach()

    def svd_regularization_losses(self):
        zero = self.fc.weight.sum() * 0.0
        orth_loss = zero
        ksv_loss = zero
        num_orth = 0
        num_ksv = 0

        for module in self.modules():
            if not isinstance(module, SVDResidualLinear):
                continue

            if module.U_residual is not None and module.V_residual is not None:
                eye_u = torch.eye(
                    module.U_residual.shape[1],
                    device=module.U_residual.device,
                    dtype=module.U_residual.dtype,
                )
                eye_v = torch.eye(
                    module.V_residual.shape[0],
                    device=module.V_residual.device,
                    dtype=module.V_residual.dtype,
                )
                orth_u = torch.norm(module.U_residual.t() @ module.U_residual - eye_u, p="fro") ** 2
                orth_v = torch.norm(module.V_residual @ module.V_residual.t() - eye_v, p="fro") ** 2
                orth_loss = orth_loss + orth_u + orth_v
                num_orth += 1

            if module.S_residual is not None and hasattr(module, "S_residual_init"):
                denom = torch.sum(module.S_residual_init**2) + 1e-8
                numer = torch.sum(module.S_residual**2)
                ksv_loss = ksv_loss + torch.abs(numer / denom - 1.0)
                num_ksv += 1

        if num_orth > 0:
            orth_loss = orth_loss / num_orth
        if num_ksv > 0:
            ksv_loss = ksv_loss / num_ksv
        return orth_loss, ksv_loss


class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None, low_rank_forward=False):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.low_rank_forward = low_rank_forward

        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.S_r = None
        self.U_r = None
        self.V_r = None
        self.S_residual = None
        self.U_residual = None
        self.V_residual = None
        self.residual_enabled = True

    def forward(self, x):
        if self.residual_enabled and self.S_residual is not None:
            if self.low_rank_forward:
                main_out = F.linear(x, self.weight_main, self.bias)
                low_rank = x.matmul(self.V_residual.t())
                low_rank = low_rank * self.S_residual
                delta_out = low_rank.matmul(self.U_residual.t())
                return main_out + delta_out
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)


def apply_svd_residual_to_dinov3_self_attn(model, r, low_rank_forward=False):
    for block in getattr(model, "blocks", []):
        if not hasattr(block, "attn"):
            continue
        attn = block.attn
        if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
            attn.qkv = replace_with_svd_residual(attn.qkv, r, low_rank_forward=low_rank_forward)
        if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
            attn.proj = replace_with_svd_residual(attn.proj, r, low_rank_forward=low_rank_forward)

    for param_name, param in model.named_parameters():
        if any(k in param_name for k in ["S_residual", "U_residual", "V_residual"]):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def replace_with_svd_residual(module, r, low_rank_forward=False):
    if not isinstance(module, nn.Linear):
        return module

    in_features = module.in_features
    out_features = module.out_features
    bias = module.bias is not None

    new_module = SVDResidualLinear(
        in_features,
        out_features,
        r,
        bias=bias,
        init_weight=module.weight.data.clone(),
        low_rank_forward=low_rank_forward,
    )

    if bias and module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)

    new_module.weight_original_fnorm = torch.norm(module.weight.data, p="fro")
    U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
    r = min(r, len(S))

    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    weight_main = U_r @ torch.diag(S_r) @ Vh_r

    new_module.weight_main_fnorm = torch.norm(weight_main.data, p="fro")
    new_module.weight_main.data.copy_(weight_main)

    U_residual = U[:, r:]
    S_residual = S[r:]
    Vh_residual = Vh[r:, :]

    if len(S_residual) > 0:
        new_module.S_residual = nn.Parameter(S_residual.clone())
        new_module.U_residual = nn.Parameter(U_residual.clone())
        new_module.V_residual = nn.Parameter(Vh_residual.clone())
        new_module.register_buffer("S_residual_init", S_residual.clone(), persistent=False)
    else:
        new_module.S_residual = None
        new_module.U_residual = None
        new_module.V_residual = None
    return new_module
