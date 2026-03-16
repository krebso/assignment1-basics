from typing import final
import torch
from torch.nn import Module, Parameter
from einops import einsum, reduce


@final
class RMSNorm(Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.g = Parameter(torch.ones(self.d_model, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        upcast_x = x.to(torch.float32)

        sum = reduce(torch.square(upcast_x), "... in_features -> ... 1", "sum")
        rms = torch.sqrt((sum / self.d_model) + self.eps)
        mul = einsum(upcast_x, self.g, "... in_features, in_features -> ... in_features")

        return (mul / rms).to(in_dtype)
