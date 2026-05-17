import numpy as np
import torch
import torch.nn as nn


class RealManifoldDistanceScorer(nn.Module):
    """Real-centric distance evidence fitted from real-image features.

    The scorer is intentionally non-parametric: statistics are fitted offline
    from real images, then inference maps distance-to-real-manifold into a
    standardized fake-evidence logit.
    """

    def __init__(self, stats_path: str):
        super().__init__()
        if not stats_path:
            raise ValueError("stats_path is required for RealManifoldDistanceScorer.")

        stats = np.load(stats_path, allow_pickle=False)
        self.stats_path = stats_path
        files = set(stats.files)
        self.global_weight = float(stats["global_weight"]) if "global_weight" in files else 1.0
        self.patch_weight = float(stats["patch_weight"]) if "patch_weight" in files else 0.0
        self.patch_topk_ratio = float(stats["patch_topk_ratio"]) if "patch_topk_ratio" in files else 0.10
        self.score_mean = float(stats["score_mean"])
        self.score_std = max(float(stats["score_std"]), 1e-6)

        self.register_buffer("global_mean", torch.from_numpy(stats["global_mean"]).float(), persistent=False)
        self.register_buffer("global_inv_var", torch.from_numpy(stats["global_inv_var"]).float(), persistent=False)

        if "patch_mean" in files and "patch_inv_var" in files:
            self.register_buffer("patch_mean", torch.from_numpy(stats["patch_mean"]).float(), persistent=False)
            self.register_buffer("patch_inv_var", torch.from_numpy(stats["patch_inv_var"]).float(), persistent=False)
        else:
            self.patch_mean = None
            self.patch_inv_var = None

    @staticmethod
    def _as_float_vector(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("global features should be shaped BxC.")
        return x.float()

    def _global_distance(self, feat: torch.Tensor) -> torch.Tensor:
        feat = self._as_float_vector(feat)
        mean = self.global_mean.to(device=feat.device, dtype=feat.dtype)
        inv_var = self.global_inv_var.to(device=feat.device, dtype=feat.dtype)
        return ((feat - mean).pow(2) * inv_var).mean(dim=1)

    def _patch_distance(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if self.patch_mean is None or self.patch_inv_var is None:
            return None
        if patch_tokens is None:
            return None
        if patch_tokens.ndim != 3:
            raise ValueError("patch tokens should be shaped BxNxC.")
        tokens = patch_tokens.float()
        mean = self.patch_mean.to(device=tokens.device, dtype=tokens.dtype)
        inv_var = self.patch_inv_var.to(device=tokens.device, dtype=tokens.dtype)
        patch_dist = ((tokens - mean.view(1, 1, -1)).pow(2) * inv_var.view(1, 1, -1)).mean(dim=2)
        num_tokens = patch_dist.shape[1]
        k = max(1, int(round(num_tokens * self.patch_topk_ratio)))
        k = min(k, num_tokens)
        return torch.topk(patch_dist, k=k, dim=1).values.mean(dim=1)

    def raw_score(self, feat: torch.Tensor, patch_tokens: torch.Tensor = None) -> torch.Tensor:
        parts = []
        weights = []
        if self.global_weight != 0.0:
            parts.append(self._global_distance(feat))
            weights.append(abs(self.global_weight))

        patch_score = self._patch_distance(patch_tokens)
        if patch_score is not None and self.patch_weight != 0.0:
            parts.append(patch_score)
            weights.append(abs(self.patch_weight))

        if not parts:
            raise RuntimeError("No active real-manifold distance component.")
        weighted = torch.zeros_like(parts[0])
        norm = 0.0
        for score, weight in zip(parts, weights):
            weighted = weighted + float(weight) * score
            norm += float(weight)
        return weighted / max(norm, 1e-6)

    def forward(self, feat: torch.Tensor, patch_tokens: torch.Tensor = None) -> torch.Tensor:
        raw = self.raw_score(feat, patch_tokens)
        return ((raw - self.score_mean) / self.score_std).unsqueeze(1)
