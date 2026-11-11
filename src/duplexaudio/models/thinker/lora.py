"""LoRA on the frozen Qwen3-Omni thinker's attention projections (q/k/v/o of the text decoder). B=0 at init, so a
fresh module is the identity; the checkpoint loader fills A/B from the trained deltas."""
from __future__ import annotations
import math
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)                       # B=0 -> starts as identity (base unchanged)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.drop(self.A(x)))


def inject_lora(root, targets, r, alpha, dropout, dtype, dev):
    n = 0
    for _, mod in list(root.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cn in targets:
                setattr(mod, cn, LoRALinear(child, r, alpha, dropout).to(dev).to(dtype))
                n += 1
    return n
