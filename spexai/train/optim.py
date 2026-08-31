"""Optimizers for the training-recipe program.

Muon (Jordan et al. 2024, https://kellerjordan.github.io/posts/muon/):
orthogonalized momentum updates for 2-D weight matrices via a
Newton-Schulz iteration; non-matrix parameters (biases, embeddings,
norm constants) are handled by a companion AdamW.
"""

import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate UV^T for G = USV^T (quintic Newton-Schulz)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon for 2-D parameters only; pair with AdamW for the rest."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                buf = st["buf"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"])  # nesterov
                u = zeropower_via_newtonschulz5(g, group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(u, alpha=-group["lr"] * scale)


def split_matrix_params(model):
    """(matrix params for Muon, everything else) - embeddings excluded."""
    mats, rest = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in name:
            mats.append(p)
        else:
            rest.append(p)
    return mats, rest


def mup_param_groups(model, lr, base_width):
    """Approximate muP for Adam: hidden-matrix learning rate scaled by
    base_width / fan_in (Yang & Hu 2021); embeddings, biases and vectors
    keep the base rate."""
    by_fanin, rest = {}, []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in name and p.shape[1] > base_width:
            by_fanin.setdefault(p.shape[1], []).append(p)
        else:
            rest.append(p)
    groups = [{"params": rest, "lr": lr}]
    for fanin, ps in by_fanin.items():
        groups.append({"params": ps, "lr": lr * base_width / fanin})
    return groups
