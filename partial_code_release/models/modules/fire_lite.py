import torch
import torch.nn as nn
import torch.nn.functional as F

from .mid_frequency import build_ring_band_mask


class MidBandMaskHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReconScoreHead(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
        return self.fc(feat)


def build_mid_target_mask(
    batch_size: int,
    height: int,
    width: int,
    low_ratio: float,
    high_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = build_ring_band_mask(
        height=height,
        width=width,
        low_ratio=low_ratio,
        high_ratio=high_ratio,
        device=device,
        dtype=dtype,
    )
    return mask.expand(batch_size, -1, -1, -1)


def fft_band_filter_rgb(x_unit: torch.Tensor, band_mask: torch.Tensor) -> torch.Tensor:
    if x_unit.ndim != 4 or x_unit.shape[1] != 3:
        raise ValueError("fft_band_filter_rgb expects input shaped as Bx3xHxW.")
    if band_mask.ndim != 4 or band_mask.shape[1] != 1:
        raise ValueError("band_mask must be shaped as Bx1xHxW.")
    if x_unit.shape[0] != band_mask.shape[0] or x_unit.shape[2:] != band_mask.shape[2:]:
        raise ValueError("x_unit and band_mask must have matching B/H/W.")

    x = x_unit.clamp(0.0, 1.0)
    mask = band_mask.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

    # Follow FIRE-style frequency filtering: FFT -> mask -> IFFT magnitude.
    # Scaling by 255 keeps amplitude range close to image-domain intensity.
    freq = torch.fft.fft2(x * 255.0, dim=(-2, -1))
    freq = torch.fft.fftshift(freq, dim=(-2, -1))
    filtered = freq * mask
    filtered = torch.fft.ifftshift(filtered, dim=(-2, -1))
    x_pse = torch.fft.ifft2(filtered, dim=(-2, -1)).abs()

    # Normalize each sample/channel to [0, 1] for stable downstream forwarding.
    flat = x_pse.flatten(2)
    x_min = flat.min(dim=2, keepdim=True).values.view(x_pse.shape[0], x_pse.shape[1], 1, 1)
    x_max = flat.max(dim=2, keepdim=True).values.view(x_pse.shape[0], x_pse.shape[1], 1, 1)
    x_pse = (x_pse - x_min) / (x_max - x_min + 1e-6)
    return x_pse.clamp(0.0, 1.0)


def feature_l2_distance(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    if feat_a.shape != feat_b.shape:
        raise ValueError("feature_l2_distance expects tensors with identical shapes.")
    return torch.norm(feat_a - feat_b, p=2, dim=1)
