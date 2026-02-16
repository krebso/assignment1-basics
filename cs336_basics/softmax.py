import torch

from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    x -= torch.max(x, keepdim=True, dim=dim).values
    x = torch.exp(x)
    return torch.div(x, torch.sum(x, dim=dim, keepdim=True))
