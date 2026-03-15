from collections.abc import Callable
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1,
        eps: float = 1e-8,
    ) -> None:
        defaults = {"alpha": lr, "beta_1": betas[0], "beta_2": betas[1], "weight_decay_rate": weight_decay, "eps": eps}
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["m"] = torch.zeros_like(p)
                state["v"] = torch.zeros_like(p)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            weight_decay_rate = group["weight_decay_rate"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

            state = self.state[p]
            m = state["m"]
            v = state["v"]
            t = state.get("t", 1)
            grad = p.grad.data

            m *= beta_1
            m += (1 - beta_1) * grad
            v *= beta_2
            v += (1 - beta_2) * torch.square(grad)

            alpha_t = alpha * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)

            p.data -= alpha_t * m / (torch.sqrt(v) + eps)
            p.data -= alpha * weight_decay_rate * p.data

            state["t"] = t + 1

        return loss
