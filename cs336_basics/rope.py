import torch

from torch import Tensor
from torch.nn import Module

from einops import einsum, rearrange


class RotaryPositionalEmbedding(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=torch.device | None) -> None:
        i = torch.arange(max_seq_len)  # TODO may start at 1?
        k = torch.arange(1, d_k//2 + 1)
        exponents = (2 * k - 2) / d_k
        freq = 1 / (theta ** exponents)
        angles = torch.outer(i, freq)

        self.sin = torch.sin(angles)
        self.cos = torch.cos(angles)


    
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos

        y = torch.empty_like(x)
        y[..., 0::2] = y_even
        y[..., 1::2] = y_odd

        return y
