from typing import final, override

import torch

from torch import Tensor
from torch.nn import Module, Parameter

from einops import einsum


@final
class SwiGLU(Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()

        self.d_model = d_model

        self.d_ff = 64 * int(8 / 3 * self.d_model / 64)

        # assert self.d_ff % 64 == 0, self.d_ff

        w1 = torch.empty([self.d_ff, self.d_model])
        w2 = torch.empty([self.d_model, self.d_ff])
        w3 = torch.empty([self.d_ff, self.d_model])

        self.w1 = Parameter(w1)
        self.w2 = Parameter(w2)
        self.w3 = Parameter(w3)

    def silu(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)

    @override
    def forward(self, x: Tensor) -> Tensor:
        silu = self.silu(einsum(x, self.w1, "... d_model, ... d_ff d_model-> ... d_ff"))
        gate = einsum(x, self.w3, "... d_model, ... d_ff d_model -> ... d_ff")
        position_wise = einsum(silu, gate, "... d_ff, ... d_ff -> ... d_ff")
        return einsum(position_wise, self.w2, "... d_ff, ... d_model d_ff -> ... d_model")
