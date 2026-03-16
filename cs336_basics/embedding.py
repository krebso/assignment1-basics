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
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        e = torch.empty([num_embeddings, embedding_dim], dtype=dtype, device=device)
        torch.nn.init.trunc_normal_(e, mean=0, std=1, a=-3, b=3)
        self.e = Parameter(e)

    @property
    def device(self):
        return self.e.device  # always tracks wherever the parameter lives

    def _one_hot(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return torch.zeros(*token_ids.shape, self.num_embeddings, device=self.device, dtype=self.dtype).scatter_(
            LAST_DIM, token_ids.unsqueeze(LAST_DIM).to(self.device), 1.0
        )

    @override
    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return einsum(
            self._one_hot(token_ids),
            self.e,
            "... seq_length num_embeddings, num_embeddings embedding_dim -> ... seq_length embedding_dim",
        )
