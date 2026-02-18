import torch

from math import sqrt

from torch import Tensor

from einops import einsum

from cs336_basics.softmax import softmax


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    qTk = einsum(q, k, "... sl1 dk, ... sl2 dk -> ... sl1 sl2") / sqrt(q.size()[-1])
    if mask is not None:
        qTk = qTk.masked_fill(~mask, float("-inf"))

    sqTk = softmax(qTk, dim=-1)

    return einsum(sqTk, v, "... sl1 sl2, ... sl2 dv -> ... sl1 dv")
