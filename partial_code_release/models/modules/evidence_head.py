import torch
import torch.nn as nn
import torch.nn.functional as F

from .mid_frequency import MidFrequencyPrior
from .noise_guidance import NoiseResidualExtractor


class ForensicEvidenceHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.noise_extractor = NoiseResidualExtractor(use_grayscale=True)
        self.mid_extractor = MidFrequencyPrior(use_grayscale=True)
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 3 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _map_stats(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        return torch.stack(
            [
                flat.mean(dim=1),
                flat.std(dim=1, unbiased=False),
                flat.amax(dim=1),
            ],
            dim=1,
        )

    def forward(
        self,
        cls_feat: torch.Tensor,
        patch_tokens: torch.Tensor,
        x_unit: torch.Tensor,
    ) -> torch.Tensor:
        patch_mean = patch_tokens.mean(dim=1)
        patch_max = patch_tokens.amax(dim=1)

        noise_map = self.noise_extractor(x_unit)
        mid_map = self.mid_extractor(x_unit)
        forensic_stats = torch.cat(
            [
                self._map_stats(noise_map),
                self._map_stats(mid_map),
            ],
            dim=1,
        ).to(dtype=cls_feat.dtype)

        feat = torch.cat([cls_feat, patch_mean, patch_max, forensic_stats], dim=1)
        return self.net(feat)


class MultiViewFusionHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, num_views: int = 3):
        super().__init__()
        self.num_views = int(num_views)
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(self.num_views * (self.feature_dim + 1), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features, logits) -> torch.Tensor:
        if len(features) != self.num_views or len(logits) != self.num_views:
            raise ValueError(f"MultiViewFusionHead expects {self.num_views} views.")

        parts = []
        for feat, logit in zip(features, logits):
            if logit.ndim == 1:
                logit = logit.unsqueeze(1)
            parts.extend([feat, logit])
        return self.net(torch.cat(parts, dim=1))


class DeltaMultiViewFusionHead(nn.Module):
    """Fuse original/fixed/random views through explicit transform-instability evidence."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim * 6 + 10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _as_column(logit: torch.Tensor) -> torch.Tensor:
        return logit.unsqueeze(1) if logit.ndim == 1 else logit

    def forward(self, features, logits) -> torch.Tensor:
        if len(features) != 3 or len(logits) != 3:
            raise ValueError("DeltaMultiViewFusionHead expects original/fixed/random views.")

        feat_o, feat_f, feat_r = features
        logit_o, logit_f, logit_r = [self._as_column(logit) for logit in logits]

        logit_stack = torch.cat([logit_o, logit_f, logit_r], dim=1)
        logit_delta = torch.cat(
            [
                torch.abs(logit_o - logit_f),
                torch.abs(logit_o - logit_r),
                torch.abs(logit_f - logit_r),
            ],
            dim=1,
        )
        logit_stats = torch.cat(
            [
                logit_stack.mean(dim=1, keepdim=True),
                logit_stack.std(dim=1, keepdim=True, unbiased=False),
                logit_stack.amax(dim=1, keepdim=True),
                logit_stack.amin(dim=1, keepdim=True),
            ],
            dim=1,
        )

        feat = torch.cat(
            [
                feat_o,
                feat_f,
                feat_r,
                torch.abs(feat_o - feat_f),
                torch.abs(feat_o - feat_r),
                torch.abs(feat_f - feat_r),
                logit_stack,
                logit_delta,
                logit_stats,
            ],
            dim=1,
        )
        return self.net(feat)


class PatchMILHead(nn.Module):
    """LOGER-style top-k patch aggregation over DINO patch tokens."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, topk_ratio: float = 0.10):
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _topk_mean(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 2:
            raise ValueError("PatchMILHead scores should be shaped BxN.")
        num_tokens = scores.shape[1]
        k = max(1, int(round(num_tokens * self.topk_ratio)))
        k = min(k, num_tokens)
        return torch.topk(scores, k=k, dim=1).values.mean(dim=1, keepdim=True)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("PatchMILHead expects patch tokens shaped BxNxC.")
        patch_logits = self.scorer(patch_tokens).squeeze(-1)
        return self._topk_mean(patch_logits)


class ForensicQueryMILHead(nn.Module):
    """Learnable forensic queries that attend to patch tokens before MIL pooling.

    This is a lightweight SIDA-inspired local detector: the queries play the
    role of task tokens, but stay inside the DINO feature space and do not
    require VLM text, masks, or segmentation labels.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_queries: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        topk_ratio: float = 0.50,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_queries = int(num_queries)
        self.topk_ratio = float(topk_ratio)
        if self.num_queries <= 0:
            raise ValueError("ForensicQueryMILHead requires num_queries > 0.")
        if self.feature_dim % int(num_heads) != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")

        self.query_tokens = nn.Parameter(torch.empty(self.num_queries, self.feature_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.token_norm = nn.LayerNorm(self.feature_dim)
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def _topk_mean(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 2:
            raise ValueError("ForensicQueryMILHead scores should be shaped BxK.")
        num_queries = scores.shape[1]
        k = max(1, int(round(num_queries * self.topk_ratio)))
        k = min(k, num_queries)
        return torch.topk(scores, k=k, dim=1).values.mean(dim=1, keepdim=True)

    def diversity_loss(self) -> torch.Tensor:
        if self.num_queries <= 1:
            return self.query_tokens.sum() * 0.0
        q = F.normalize(self.query_tokens, dim=1)
        gram = q @ q.t()
        eye = torch.eye(self.num_queries, device=gram.device, dtype=gram.dtype)
        return ((gram - eye) ** 2).sum() / float(self.num_queries * (self.num_queries - 1))

    def forward(self, patch_tokens: torch.Tensor, return_attention: bool = False):
        if patch_tokens.ndim != 3:
            raise ValueError("ForensicQueryMILHead expects patch tokens shaped BxNxC.")
        bsz = patch_tokens.shape[0]
        tokens = self.token_norm(patch_tokens)
        queries = self.query_tokens.unsqueeze(0).expand(bsz, -1, -1)
        queries = self.query_norm(queries)
        attended, attn = self.cross_attn(
            query=queries,
            key=tokens,
            value=tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        query_logits = self.scorer(attended).squeeze(-1)
        image_logit = self._topk_mean(query_logits)
        if return_attention:
            return image_logit, attn, query_logits
        return image_logit


class CrossViewPatchDisagreementHead(nn.Module):
    """Top-k local disagreement between original/fixed/random patch tokens."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, topk_ratio: float = 0.10):
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _align_tokens(tokens):
        min_tokens = min(t.shape[1] for t in tokens)
        return [t[:, :min_tokens, :] for t in tokens]

    def _topk_mean(self, scores: torch.Tensor) -> torch.Tensor:
        num_tokens = scores.shape[1]
        k = max(1, int(round(num_tokens * self.topk_ratio)))
        k = min(k, num_tokens)
        return torch.topk(scores, k=k, dim=1).values.mean(dim=1, keepdim=True)

    def forward(
        self,
        patch_tokens_o: torch.Tensor,
        patch_tokens_f: torch.Tensor,
        patch_tokens_r: torch.Tensor,
    ) -> torch.Tensor:
        if patch_tokens_o.ndim != 3 or patch_tokens_f.ndim != 3 or patch_tokens_r.ndim != 3:
            raise ValueError("CrossViewPatchDisagreementHead expects BxNxC token tensors.")
        patch_tokens_o, patch_tokens_f, patch_tokens_r = self._align_tokens(
            [patch_tokens_o, patch_tokens_f, patch_tokens_r]
        )
        diff = torch.cat(
            [
                torch.abs(patch_tokens_o - patch_tokens_f),
                torch.abs(patch_tokens_o - patch_tokens_r),
                torch.abs(patch_tokens_f - patch_tokens_r),
            ],
            dim=-1,
        )
        patch_logits = self.scorer(diff).squeeze(-1)
        return self._topk_mean(patch_logits)


class FireErrorEvidenceHead(nn.Module):
    """Top-k spatial scorer for FIRE-style feature reconstruction error maps."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, topk_ratio: float = 0.10):
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, delta_map: torch.Tensor) -> torch.Tensor:
        if delta_map.ndim != 4:
            raise ValueError("FireErrorEvidenceHead expects BxCxHxW error maps.")
        score_map = self.conv(delta_map).flatten(1)
        num_points = score_map.shape[1]
        k = max(1, int(round(num_points * self.topk_ratio)))
        k = min(k, num_points)
        return torch.topk(score_map, k=k, dim=1).values.mean(dim=1, keepdim=True)
