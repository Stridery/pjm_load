# src/models/_moe_head.py
"""
Shared Mixture-of-Experts output head + hard routing.

This is the part of the MoE that is INDEPENDENT of the encoder: given a per-day
representation z (feat_dim) and a per-sample season index, it runs the 12 regime
expert heads (from REGIME_MAP / SEASON_ORDER) and hard-selects each sample's own
season. Both MoETransformer (attention encoder) and MoEMSTNN (conv encoder) plug
their z into it, so the routing lives in exactly one place.

Routing is deterministic (no gating network):
  season -> per sample, picks which season's experts run
  hour   -> per output index, each expert owns a fixed set of hours (REGIME_MAP)
Only the selected season's experts appear in the loss, so gradient flows to those
experts + the shared encoder and nowhere else.
"""

import torch
import torch.nn as nn

from src.config import REGIME_MAP, SEASON_ORDER


def validate_regime_map(out_dim):
    """Each season's hour lists must be disjoint and tile 0..out_dim-1."""
    for season, experts in REGIME_MAP.items():
        hours = [h for hrs in experts.values() for h in hrs]
        if sorted(hours) != list(range(out_dim)):
            raise ValueError(
                f"REGIME_MAP['{season}'] hours {sorted(hours)} do not tile "
                f"0..{out_dim - 1} exactly (check for gaps/overlaps)."
            )


def _expert_head(in_dim, fc_hidden, n_out, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, fc_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_hidden, n_out),
    )


class RegimeHead(nn.Module):
    """12 regime expert heads over a shared representation, with hard season routing.

    feat_dim : width of the representation z the encoder hands in (encoder dim + any
               static-skip features).
    """
    def __init__(self, feat_dim, out_dim, fc_hidden, dropout):
        super().__init__()
        self.out_dim = out_dim
        validate_regime_map(out_dim)

        self.experts = nn.ModuleDict()
        self._season_expert_names = {}          # season -> [expert names] (build order)
        for season in SEASON_ORDER:
            names = list(REGIME_MAP[season].keys())
            self._season_expert_names[season] = names
            concat_hours = []
            for name in names:
                hrs = REGIME_MAP[season][name]
                self.experts[f'{season}__{name}'] = _expert_head(feat_dim, fc_hidden, len(hrs), dropout)
                concat_hours.extend(hrs)
            # inv_perm[h] = column of the season's concatenated expert output holding hour h
            inv_perm = [concat_hours.index(h) for h in range(out_dim)]
            self.register_buffer(f'inv_perm_{season}', torch.tensor(inv_perm, dtype=torch.long))

    def _season_forward(self, z, season):
        """Full out_dim-vector prediction as if the whole batch were this season."""
        cols = [self.experts[f'{season}__{n}'](z) for n in self._season_expert_names[season]]
        cat = torch.cat(cols, dim=1)             # (batch, out_dim) in expert order
        return cat[:, getattr(self, f'inv_perm_{season}')]   # reorder to hour 0..out_dim-1

    def forward(self, z, season_idx):
        # Compute every season's head, then hard-select each sample's own season.
        # Unselected season outputs receive no gradient (sliced away from the loss).
        stacked = torch.stack([self._season_forward(z, s) for s in SEASON_ORDER], dim=0)  # (S,B,out)
        batch_ar = torch.arange(z.size(0), device=z.device)
        return stacked[season_idx, batch_ar]     # (batch, out_dim)
