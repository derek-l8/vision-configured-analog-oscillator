from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int = 16
    input_width: int = 576
    input_height: int = 768
    # Missing field in legacy checkpoints means BatchNorm, preserving loadability.
    normalization: str = "batch"
    switch_pool_grid: int = 1

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, normalization: str = "batch") -> None:
        if normalization not in {"batch", "group"}:
            raise ValueError("normalization must be batch or group")
        def norm() -> nn.Module:
            return (nn.GroupNorm(math.gcd(8, out_channels), out_channels)
                    if normalization == "group" else nn.BatchNorm2d(out_channels))
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            norm(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            norm(),
            nn.ReLU(inplace=True),
        )


class MultiTaskCNN(nn.Module):
    """Small from-scratch encoder/decoder with three task heads."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.base_channels
        norm = self.config.normalization
        self.stem = ConvBlock(3, c, stride=2, normalization=norm)       # 1/2 resolution
        self.encoder2 = ConvBlock(c, c * 2, stride=2, normalization=norm)  # 1/4
        self.encoder3 = ConvBlock(c * 2, c * 4, stride=2, normalization=norm)  # 1/8
        self.encoder4 = ConvBlock(c * 4, c * 6, stride=2, normalization=norm)  # 1/16
        self.orientation_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c * 6, 2))
        grid = self.config.switch_pool_grid
        if grid < 1:
            raise ValueError("switch_pool_grid must be positive")
        # Preserve coarse position so left/right actuator evidence is not blended
        # into one whole-image average. No manually supplied control crop is used.
        self.switch_head = nn.Sequential(nn.AdaptiveAvgPool2d((grid, grid)), nn.Flatten(), nn.Linear(c * 6 * grid * grid, 2))
        self.decode3 = ConvBlock(c * 10, c * 4, normalization=norm)
        self.decode2 = ConvBlock(c * 6, c * 2, normalization=norm)
        self.decode1 = ConvBlock(c * 3, c, normalization=norm)
        self.heatmap_head = nn.Conv2d(c, 2, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(image)
        enc2 = self.encoder2(stem)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        raw_orientation = self.orientation_head(enc4)
        orientation = F.normalize(raw_orientation, p=2, dim=1, eps=1e-6)
        switch_logits = self.switch_head(enc4)

        dec3 = F.interpolate(enc4, size=enc3.shape[-2:], mode="bilinear", align_corners=False)
        dec3 = self.decode3(torch.cat((dec3, enc3), dim=1))
        dec2 = F.interpolate(dec3, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.decode2(torch.cat((dec2, enc2), dim=1))
        dec1 = F.interpolate(dec2, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.decode1(torch.cat((dec1, stem), dim=1))
        heatmap_logits = self.heatmap_head(dec1)
        return {
            "orientation_sin_cos": orientation,
            "switch_logits": switch_logits,
            "heatmap_logits": heatmap_logits,
        }


def model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> MultiTaskCNN:
    config = ModelConfig(**checkpoint["model_config"])
    model = MultiTaskCNN(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model
