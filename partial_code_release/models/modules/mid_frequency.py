import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def build_ring_band_mask(
    height: int,
    width: int,
    low_ratio: float,
    high_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = math.sqrt(cy * cy + cx * cx)
    low = low_ratio * max_radius
    high = high_ratio * max_radius
    mask = ((dist >= low) & (dist <= high)).to(dtype)
    return mask.unsqueeze(0).unsqueeze(0)


def tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError("tokens_to_map expects tensor shaped as BxNxC.")
    b, n, c = tokens.shape
    h = int(round(math.sqrt(n)))
    if h * h != n:
        raise ValueError(f"Token count {n} cannot be reshaped to a square map.")
    return tokens.transpose(1, 2).reshape(b, c, h, h)


class MidFrequencyPrior(nn.Module):
    """Fixed FFT forensic prior extractor.

    The default output is backward compatible with the old single-channel
    mid-band response. Enabling ``return_multiband`` exposes multiple radial
    bands and residual cues for stronger training-time supervision.
    """

    def __init__(
        self,
        use_grayscale: bool = True,
        low_ratio: float = 0.15,
        high_ratio: float = 0.45,
        use_absolute_response: bool = True,
        band_ratios=None,
        include_residual: bool = True,
    ):
        super().__init__()
        self.use_grayscale = use_grayscale
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio
        self.use_absolute_response = use_absolute_response
        self.band_ratios = band_ratios or (
            (0.05, 0.15),
            (0.15, 0.30),
            (0.30, 0.45),
            (0.45, 0.75),
        )
        self.include_residual = bool(include_residual)

    def _to_gray(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_grayscale:
            return x.mean(dim=1, keepdim=True)
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    @staticmethod
    def _normalise_per_sample(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.flatten(1)
        x_min = flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        x_max = flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        return ((x - x_min) / (x_max - x_min + 1e-6)).clamp(0.0, 1.0)

    def _band_response(self, freq: torch.Tensor, low_ratio: float, high_ratio: float) -> torch.Tensor:
        _, _, h, w = freq.shape
        mask = build_ring_band_mask(
            height=h,
            width=w,
            low_ratio=float(low_ratio),
            high_ratio=float(high_ratio),
            device=freq.device,
            dtype=freq.real.dtype,
        )
        masked = freq * mask
        masked = torch.fft.ifftshift(masked, dim=(-2, -1))
        response = torch.fft.ifft2(masked, dim=(-2, -1))
        if self.use_absolute_response:
            response = response.abs()
        else:
            response = response.real
        return self._normalise_per_sample(response)

    def forward(self, x_unit: torch.Tensor, return_multiband: bool = False) -> torch.Tensor:
        x = x_unit.clamp(0.0, 1.0)
        x_gray = self._to_gray(x)

        freq = torch.fft.fft2(x_gray, dim=(-2, -1))
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        response = self._band_response(freq, self.low_ratio, self.high_ratio)
        if not return_multiband:
            return response

        bands = [self._band_response(freq, low, high) for low, high in self.band_ratios]
        if self.include_residual:
            lowpass = F.avg_pool2d(x_gray, kernel_size=5, stride=1, padding=2)
            residual = self._normalise_per_sample((x_gray - lowpass).abs())
            lap = F.pad(x_gray, (1, 1, 1, 1), mode="reflect")
            lap = (
                -lap[:, :, :-2, :-2]
                - lap[:, :, :-2, 1:-1]
                - lap[:, :, :-2, 2:]
                - lap[:, :, 1:-1, :-2]
                + 8.0 * lap[:, :, 1:-1, 1:-1]
                - lap[:, :, 1:-1, 2:]
                - lap[:, :, 2:, :-2]
                - lap[:, :, 2:, 1:-1]
                - lap[:, :, 2:, 2:]
            )
            bands.extend([residual, self._normalise_per_sample(lap.abs())])
        return torch.cat(bands, dim=1).clamp(0.0, 1.0)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _group_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _group_norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class MidPriorPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128, out_channels: int = 1, depth: int = 2):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            _group_norm(hidden_dim),
            nn.GELU(),
        ]
        for _ in range(max(0, int(depth))):
            layers.append(ResidualConvBlock(hidden_dim))
        layers.extend(
            [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, int(out_channels), kernel_size=1),
                nn.Sigmoid(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, spatial_feat: torch.Tensor) -> torch.Tensor:
        return self.net(spatial_feat)
