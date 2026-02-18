from torch.nn import Module
from torch import Tensor

from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.softmax import softmax


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
    ) -> None:
        super().__init__()

        self.emb = Embedding(vocab_size, d_model)
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length) for _ in range(num_layers)
        ]
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        emb = self.emb(x)

        y = emb

        for block in self.blocks:
            y = block.forward(y)

        y = self.norm(y)

        return self.linear(y)
