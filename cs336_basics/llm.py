import torch

from torch.nn import Module
from torch import Tensor

from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear


class Transformer(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.emb = Embedding(vocab_size, d_model, device=device)
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, device=device)
            for _ in range(num_layers)
        ]
        self.norm = RMSNorm(d_model, device=device)
        self.linear = Linear(d_model, vocab_size, device=device)

    def forward(self, x: Tensor) -> Tensor:
        emb = self.emb(x)

        y = emb

        for block in self.blocks:
            y = block.forward(y)

        y_norm = self.norm(y)

        head = self.linear(y_norm.data)

        return head
