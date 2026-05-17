import torch
import torch.nn as nn


def _to_uint8_indices(x_unit: torch.Tensor) -> torch.Tensor:
    x_unit = x_unit.clamp(0.0, 1.0)
    return (x_unit * 255.0).round().long().clamp(0, 255)


def _remap_lut_to_unit(lut: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    lut_min = lut.min(dim=-1, keepdim=True).values
    lut_max = lut.max(dim=-1, keepdim=True).values
    return (lut - lut_min) / (lut_max - lut_min + eps)


class FixedPixelMapping(nn.Module):
    """Deterministic 256-entry lookup mapping shared by all channels."""

    def __init__(self, remap_to_unit: bool = True, decimals: int = 2):
        super().__init__()
        values = torch.arange(256, dtype=torch.float32)
        rounded = torch.round((values / 256.0) * (10 ** decimals)) / (10 ** decimals)
        lut = values - rounded * 256.0
        if remap_to_unit:
            lut = _remap_lut_to_unit(lut.unsqueeze(0)).squeeze(0)
        self.register_buffer("lut", lut, persistent=False)
        self.remap_to_unit = remap_to_unit

    def forward(self, x_unit: torch.Tensor) -> torch.Tensor:
        idx = _to_uint8_indices(x_unit)
        mapped = self.lut[idx]
        if self.remap_to_unit:
            mapped = mapped.clamp(0.0, 1.0)
        return mapped


class RandomPixelMapping(nn.Module):
    """Per-sample, per-channel random lookup mapping."""

    def __init__(
        self,
        remap_to_unit: bool = True,
        random_range: float = 1.0,
        deterministic: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.remap_to_unit = remap_to_unit
        self.random_range = random_range
        self.deterministic = deterministic
        self.seed = seed
        self._generator = None
        if deterministic:
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(seed)

    def _sample_tables(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._generator is None:
            tables = torch.rand(batch_size, 3, 256, dtype=torch.float32)
        else:
            tables = torch.rand(batch_size, 3, 256, dtype=torch.float32, generator=self._generator)
        tables = (tables * 2.0 - 1.0) * self.random_range
        if self.remap_to_unit:
            tables = _remap_lut_to_unit(tables)
        return tables.to(device=device, non_blocking=True)

    def forward(self, x_unit: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_unit.shape
        if c != 3:
            raise ValueError("RandomPixelMapping expects 3-channel input.")

        idx = _to_uint8_indices(x_unit)
        tables = self._sample_tables(b, x_unit.device)

        b_idx = torch.arange(b, device=x_unit.device).view(b, 1, 1, 1).expand_as(idx)
        c_idx = torch.arange(c, device=x_unit.device).view(1, c, 1, 1).expand_as(idx)
        mapped = tables[b_idx, c_idx, idx]
        if self.remap_to_unit:
            mapped = mapped.clamp(0.0, 1.0)
        return mapped
