"""User-audio front end of the thinker: 8 Mimi RVQ codebook tables (summed) + a TBL_DIM -> hidden bridge, one
embedding per 80 ms Mimi frame. Frozen at inference; its weights come from the thinker checkpoint."""
from __future__ import annotations
import torch
import torch.nn as nn

N_CB, CARD, TBL_DIM = 8, 2049, 4096


class FrontEnd(nn.Module):
    """8 Mimi RVQ codebook tables (summed) + a 4096 -> hidden bridge."""
    def __init__(self, hidden: int):
        super().__init__()
        self.user_codebook_emb = nn.Embedding(N_CB * CARD, TBL_DIM)
        self.user_in_proj = nn.Linear(TBL_DIM, hidden)
        self.register_buffer("_cb_off", torch.arange(N_CB) * CARD, persistent=False)

    def forward(self, user_codes: torch.Tensor) -> torch.Tensor:      # (..., 8) -> (..., hidden)
        e = self.user_codebook_emb(user_codes.long() + self._cb_off).sum(dim=-2)
        return self.user_in_proj(e)
