import torch
from collections.abc import Iterable
from math import sqrt


def gradient_clipping(ps: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    square_sum = 0

    for p in ps:
        if p.grad is None:
            continue

        square_sum += torch.sum(torch.flatten(torch.square(p.grad.data))).item()

    l2_norm = sqrt(square_sum)

    if max_l2_norm < l2_norm:
        scale = max_l2_norm / (l2_norm + eps)

        for p in ps:
            if p.grad is None:
                continue

            p.grad.data *= scale
