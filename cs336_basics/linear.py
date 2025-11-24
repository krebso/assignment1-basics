from typing import final, override
import torch

from torch import Tensor
from torch.nn import Module, Parameter

from einops import einsum

@final
class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        w = torch.empty([self.out_features, self.in_features], dtype=self.dtype, device=self.device)
        sigma = 2 / (self.in_features + self.out_features)
        _ = torch.nn.init.trunc_normal_(w, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

        self.w = Parameter(w)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.w, "... in_features, out_features in_features -> ... out_features").to(self.device)
