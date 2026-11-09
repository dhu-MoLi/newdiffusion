from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class DraemSegmenter(nn.Module):
    """Compact two-class U-Net used only for the P2-C0 downstream pilot."""

    def __init__(self, *, base_channels: int = 16) -> None:
        super().__init__()
        if type(base_channels) is not int or base_channels < 4:
            raise ValueError("base_channels must be an integer of at least four")
        self.encoder_1 = _ConvBlock(3, base_channels)
        self.encoder_2 = _ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = _ConvBlock(base_channels * 2, base_channels * 4)
        self.up_2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.decoder_2 = _ConvBlock(base_channels * 4, base_channels * 2)
        self.up_1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.decoder_1 = _ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape N x 3 x H x W")
        if image.shape[-2] % 4 or image.shape[-1] % 4:
            raise ValueError("image spatial dimensions must be divisible by four")
        first = self.encoder_1(image)
        second = self.encoder_2(functional.max_pool2d(first, 2))
        middle = self.bottleneck(functional.max_pool2d(second, 2))
        decoded_2 = self.decoder_2(torch.cat((self.up_2(middle), second), dim=1))
        decoded_1 = self.decoder_1(torch.cat((self.up_1(decoded_2), first), dim=1))
        return self.head(decoded_1)


def focal_cross_entropy(logits: Tensor, target: Tensor, *, gamma: float) -> Tensor:
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("logits must have shape N x 2 x H x W")
    if target.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise ValueError("target must align with logits")
    if target.dtype != torch.long or not 0.0 <= gamma <= 8.0:
        raise ValueError("target must be long and gamma must be in [0, 8]")
    cross_entropy = functional.cross_entropy(logits, target, reduction="none")
    probability = torch.exp(-cross_entropy)
    return ((1.0 - probability).pow(gamma) * cross_entropy).mean()
