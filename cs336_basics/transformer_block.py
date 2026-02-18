from torch.nn import Module

from torch import Tensor

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.masked_self_attention import MultiheadSelfAttention
from cs336_basics.positionwise_feedforward import SwiGLU


class TransformerBlock(Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, theta: float | None = None, max_seq_len: int | None = None
    ) -> None:
        super().__init__()
        self.rmsnorm_mha = RMSNorm(d_model)
        self.rmsnorm_ff = RMSNorm(d_model)
        self.mha = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        mha = self.mha.forward(self.rmsnorm_mha.forward(x))
        y = x + mha
        ff = self.ff.forward(self.rmsnorm_ff.forward(y))
        return y + ff
