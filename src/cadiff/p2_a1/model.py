from __future__ import annotations

import torch
from torch import Tensor, nn

from cadiff.p2_a0.model import (
    SD15_DOWN_RESIDUAL_SLOTS,
    AdapterResiduals,
    MaskedInfillAdapter,
    _time_embedding,
)
from cadiff.p2_a1.config import P2A1Config


class FactorizedMaskedInfillAdapter(MaskedInfillAdapter):
    def __init__(self, config: P2A1Config) -> None:
        if not isinstance(config, P2A1Config):
            raise ValueError("config must be a P2A1Config")
        super().__init__(config)
        type_count = len(config.defect_types)
        sources = (128,) * 3 + (256,) * 3 + (512,) * 6
        self.output_heads = nn.ModuleList(nn.ModuleList() for _ in range(type_count))
        self.mid_heads = nn.ModuleList(nn.Identity() for _ in range(type_count))
        self.shared_output_heads = nn.ModuleList(
            nn.Conv2d(source, slot.channels, 1)
            for source, slot in zip(sources, SD15_DOWN_RESIDUAL_SLOTS, strict=True)
        )
        self.delta_down_heads = nn.ModuleList(
            nn.ModuleList(nn.Conv2d(source, config.type_rank, 1) for source in sources)
            for _ in range(type_count)
        )
        self.delta_up_heads = nn.ModuleList(
            nn.ModuleList(
                nn.Conv2d(config.type_rank, slot.channels, 1)
                for slot in SD15_DOWN_RESIDUAL_SLOTS
            )
            for _ in range(type_count)
        )
        self.shared_mid_head = nn.Conv2d(512, 1280, 1)
        self.delta_mid_down = nn.ModuleList(
            nn.Conv2d(512, config.type_rank, 1) for _ in range(type_count)
        )
        self.delta_mid_up = nn.ModuleList(
            nn.Conv2d(config.type_rank, 1280, 1) for _ in range(type_count)
        )
        for head in (*self.shared_output_heads, self.shared_mid_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for route in self.delta_up_heads:
            for head in route:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        for head in self.delta_mid_up:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _type_delta_modules(self, route_index: int) -> tuple[nn.Module, ...]:
        return (
            *self.delta_down_heads[route_index],
            *self.delta_up_heads[route_index],
            self.delta_mid_down[route_index],
            self.delta_mid_up[route_index],
        )

    def routed_parameters(self, route_index: object) -> tuple[nn.Parameter, ...]:
        if type(route_index) is not int or not 0 <= route_index < len(self.output_heads):
            raise ValueError("route_index must identify a configured defect type")
        inactive = {
            id(parameter)
            for index in range(len(self.output_heads))
            if index != route_index
            for module in self._type_delta_modules(index)
            for parameter in module.parameters()
        }
        return tuple(
            parameter for parameter in self.parameters() if id(parameter) not in inactive
        )

    def type_specific_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for index in range(len(self.output_heads))
            for module in self._type_delta_modules(index)
            for parameter in module.parameters()
        )

    def forward(
        self,
        noisy: Tensor,
        context: Tensor,
        mask: Tensor,
        timesteps: Tensor,
        type_embedding: Tensor,
        route_index: int,
    ) -> AdapterResiduals:
        self._validate(noisy, context, mask, timesteps, type_embedding, route_index)
        condition = self.conditioning(
            torch.cat((_time_embedding(timesteps, noisy.dtype), type_embedding), dim=1)
        )
        feature64 = self.block64(
            self.stem(torch.cat((noisy, context, mask), dim=1)), condition
        )
        feature32 = self.block32(self.down32(feature64), condition)
        feature16 = self.block16(self.down16(feature32), condition)
        feature8 = self.block8(self.down8(feature16), condition)
        features = (
            (feature64,) * 3
            + (feature32,) * 3
            + (feature16,) * 3
            + (feature8,) * 3
        )
        down = tuple(
            shared(feature)
            + up(torch.nn.functional.silu(reduce(feature)))
            for shared, reduce, up, feature in zip(
                self.shared_output_heads,
                self.delta_down_heads[route_index],
                self.delta_up_heads[route_index],
                features,
                strict=True,
            )
        )
        mid = self.shared_mid_head(feature8) + self.delta_mid_up[route_index](
            torch.nn.functional.silu(self.delta_mid_down[route_index](feature8))
        )
        return AdapterResiduals(down=down, mid=mid)
