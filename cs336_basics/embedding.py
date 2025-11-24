from typing import final, override
import torch

from torch.nn import Module, Parameter
from einops import einsum

LAST_DIM = -1


@final
class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        e = torch.empty([self.num_embeddings, self.embedding_dim], dtype=self.dtype)
        _ = torch.nn.init.trunc_normal_(e, mean=0, std=1, a=-3, b=3)
        self.e = Parameter(e)

    def _one_hot(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return (
            torch.zeros(*token_ids.shape, self.num_embeddings, device=self.device, dtype=self.dtype)
            .scatter_(LAST_DIM, token_ids.unsqueeze(LAST_DIM), 1.0)
            .to(self.device)
        )

    @override
    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return einsum(
            self._one_hot(token_ids),
            self.e,
            "... seq_length num_embeddings, num_embeddings embedding_dim -> ... seq_length embedding_dim",
        ).to(self.device)
