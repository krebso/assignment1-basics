# Write a function that takes a numpy array x (integer array with token IDs), a
# batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), and returns
# a pair of tensors: the sampled input sequences and the corresponding next-token targets. Both ten-
# sors should have shape (batch_size, context_length) containing token IDs, and both should be
# placed on the requested device.

import numpy as np
import torch
import random


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_index = len(x) - context_length
    batch_size = min(batch_size, max_index)
    input_sequences, targets = (
        torch.zeros([batch_size, context_length], device=device),
        torch.zeros([batch_size, context_length], device=device),
    )

    for b, i in enumerate(random.sample(range(max_index), batch_size)):
        input_sequences[b] = torch.from_numpy(x[i : i + context_length])
        targets[b] = torch.from_numpy(x[i + 1 : i + 1 + context_length])

    return input_sequences, targets
