import torch

from torch import Tensor

from cs336_basics.softmax import softmax

from einops import reduce


def cross_entropy(preds: Tensor, targets: Tensor) -> ...:
    bs = preds.size()[0]

    # softmax
    preds -= torch.max(preds, keepdim=True, dim=-1).values

    num = torch.gather(preds, -1, targets.unsqueeze(1))
    denom = reduce(torch.exp(preds), "bs ... vc -> bs ... 1", "sum")

    logs = -1 * (num - torch.log(denom))

    return reduce(logs, "bs ... 1 -> ...", "sum") / bs
