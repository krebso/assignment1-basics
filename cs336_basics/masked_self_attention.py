import torch

from torch import Tensor
from torch.nn import Module, Parameter

from einops import rearrange, einsum

from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding

class MultiheadSelfAttention(Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None) -> None:
        super().__init__()
        Q = torch.empty([d_model, d_model])
        K = torch.empty([d_model, d_model])
        V = torch.empty([d_model, d_model])
        O = torch.empty([d_model, d_model])

        self.num_heads = num_heads

        self.Q = Parameter(Q)
        self.K = Parameter(K)
        self.V = Parameter(V)
        self.O = Parameter(O)

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        else:
            self.rope = None

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sl = x.size()[-2]

        q = einsum(self.Q, x, "dm dm2, ... sl dm2 -> ... sl dm")
        k = einsum(self.K, x, "dm dm2, ... sl dm2 -> ... sl dm")
        v = einsum(self.V, x, "dm dm2, ... sl dm2 -> ... sl dm")

        q = rearrange(q, "... sl (h dk) -> ... h sl dk", h=self.num_heads)
        k = rearrange(k, "... sl (h dk) -> ... h sl dk", h=self.num_heads)
        v = rearrange(v, "... sl (h dk) -> ... h sl dk", h=self.num_heads)

        mask = torch.ones([sl, sl]).tril().to(torch.bool)

        if self.rope is not None and token_positions is not None:
            q = self.rope.forward(q, token_positions)
            k = self.rope.forward(k, token_positions)

        o = rearrange(
            scaled_dot_product_attention(q, k, v, mask),
            "... h sl dk -> ... sl (h dk)"
        )

        return einsum(self.O, o, "... dm dm2, ... sl dm2 -> ... sl dm")
