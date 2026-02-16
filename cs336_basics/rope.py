import torch

from torch import Tensor
from torch.nn import Module

from einops import einsum, rearrange


class RotaryPositionalEmbedding(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=torch.device | None) -> None:
        super().__init__()
        i = torch.arange(max_seq_len)
        k = torch.arange(d_k // 2)
        exponents = (2 * k) / d_k
        freq = 1 / (theta**exponents)
        angles = torch.outer(i, freq)

        self.register_buffer("sin", torch.sin(angles), persistent=False)
        self.register_buffer("cos", torch.cos(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        sin = self.sin[token_positions]  # type: ignore
        cos = self.cos[token_positions]  # type: ignore

        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos

        y = torch.empty_like(x)
        y[..., 0::2] = y_even
        y[..., 1::2] = y_odd

        return y
