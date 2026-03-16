import torch
import math

from torch import Tensor

from einops import reduce


def cross_entropy(preds: Tensor, targets: Tensor) -> Tensor:
    targets = targets.to(preds.device)

    # softmax
    norm_preds = preds - reduce(preds, "... vs -> ... 1", "max")

    num = torch.gather(norm_preds, -1, targets.unsqueeze(-1))
    denom = reduce(torch.exp(norm_preds), "... vs -> ... 1", "sum")

    logs = -1 * (num - torch.log(denom))

    return reduce(logs, "... -> 1", "sum") / math.prod(preds.size()[:-1])
