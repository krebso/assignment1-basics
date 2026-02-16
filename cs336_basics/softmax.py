import torch

from torch import Tensor
from torch.nn import Module


class Softmax(Module):
    def __init__(self) -> None:
        pass

    def forward(self, x: Tensor, dim: int) -> Tensor:
        x -= torch.max(x, keepdim=True, dim=dim).values
        x = torch.exp(x)
        return torch.div(x, torch.sum(x, dim=dim, keepdim=True))
