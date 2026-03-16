from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.softmax import softmax
from cs336_basics.data_loading import get_batch
from cs336_basics.optim import AdamW
from cs336_basics.llm import Transformer


import numpy as np
import torch

TRAINING_DATASET = ""
VALIDATION_DATASET = ""
DEVICE = "mps"

# Model params
VOCAB_SIZE = 10
CONTEXT_LENGTH = 10
D_MODEL = 128
NUM_LAYERS = 2
NUM_HEADS = 4
D_FF = 512
ROPE_THETA = 10000

# Training params
EPOCHS = 10000
N_BATCH = 4
BATCH_SIZE = 4

# training_dataset = np.memmap(TRAINING_DATASET)
# validation_dataset = np.memmap(VALIDATION_DATASET)

model = Transformer(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    rope_theta=ROPE_THETA,
)

optimizer = AdamW(model.parameters(), lr=1e-3)


def train():
    for e in range(1, EPOCHS + 1):
        # x, y = get_batch(training_dataset, BATCH_SIZE, CONTEXT_LENGTH, "mps")
        x = torch.randint(VOCAB_SIZE, [BATCH_SIZE, CONTEXT_LENGTH], device=DEVICE)
        y = x

        logits = model.forward(x)
        # print("Logits: ", logits.data)
        loss = cross_entropy(logits, y)
        if e % 10 == 0:
            print(f"Training loss for epoch {e}: {loss.item()}")

        optimizer.zero_grad()
        optimizer.step(loss.backward)

        # x, y = get_batch(validation_dataset, BATCH_SIZE, CONTEXT_LENGTH, "mps")
        # x = torch.randint(10, size=[CONTEXT_LENGTH], device=DEVICE)
        # y = x
        # logits = model.forward(x)
        # loss = cross_entropy(logits, y)
        # print(f"Training loss: {loss.item()}")


if __name__ == "__main__":
    # Uniform logits should give ln(vocab_size)
    # logits = torch.zeros(4, 10, 10)  # (batch, seq, vocab)
    # targets = torch.zeros(4, 10, dtype=torch.long)
    # print(cross_entropy(logits, targets))  # should be ~2.3

    train()
