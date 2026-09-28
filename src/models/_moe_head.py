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


# ---------------------------------------------------------------------------
# Soft-gated MoE — the mainstream formulation
# ---------------------------------------------------------------------------
# RegimeHead above partitions BOTH the data (by calendar month) and the output vector (by
# hour band). On 1461 training days that is the wrong trade: the twelve experts see 1,428 to
# 7,018 target scalars each, against 35,064 for a single head, and every MoE variant in this
# repo scores worse than its plain counterpart -- by 0.09-0.12 pp in residual mode and by
# 0.65-1.79 pp in direct mode. A mixture that divides the data cannot win when the data is
# already the binding constraint.
#
# The standard formulation (Jacobs et al. 1991; Shazeer et al. 2017) does not divide anything:
#
#     y = sum_i  g_i(z) * E_i(z),        g = softmax(W z)
#
# Every expert outputs the full 24-hour vector and every expert receives gradient from every
# sample, weighted by its gate. Specialisation is something the gate is free to discover, not
# something the calendar imposes -- and it can decline to specialise, in which case the mixture
# degenerates to an ensemble of heads over a shared encoder, which is a safe floor rather than
# a penalty.
#
# The gate reads z, which in every encoder here is (encoder output + the static block). That
# block already carries tmrw_month_sin/cos, day-of-week, the holiday flag and the 31
# forecast-weather departures, so the gate sees strictly more than the month index hard routing
# used -- and it needs no per-sample timestamp, which is what forced the MoE models to carry
# their own trainer. These can use the shared _seq_trainer unchanged.

class SoftMoEHead(nn.Module):
    """N full-width expert heads over a shared representation, combined by a learned gate.

    feat_dim   width of z (encoder dim + static skip)
    n_experts  how many. Small: the experts are not fed disjoint data, but each still needs
               enough distinct gradient to differentiate, and 12 on 1461 days is optimistic.
    gate_noise standard deviation of Gaussian noise added to the gate logits during training
               only (Shazeer's noisy gating). Exploration: without it the gate can commit to
               one expert in the first epochs and never revisit.
    """

    def __init__(self, feat_dim, out_dim, fc_hidden, dropout, n_experts=4, gate_noise=0.0,
                 gate_slice=None):
        super().__init__()
        self.out_dim = out_dim
        self.n_experts = n_experts
        self.gate_noise = gate_noise
        # gate_slice restricts what the ROUTER sees; the experts always see all of z. None means
        # the whole representation. Narrowing it is how the calendar-only variant is built: the
        # mixture is unchanged, only the information the route may depend on.
        self.gate_slice = gate_slice
        gate_in = feat_dim if gate_slice is None else (gate_slice[1] - gate_slice[0])
        self.experts = nn.ModuleList(
            [_expert_head(feat_dim, fc_hidden, out_dim, dropout) for _ in range(n_experts)]
        )
        self.gate = nn.Linear(gate_in, n_experts)
        # Start the mixture as a near-uniform average of its experts. A gate that begins
        # opinionated picks its favourite before any expert has learned anything, and the
        # others then never catch up.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self._last_gate = None          # (batch, n_experts), kept for the balance loss

    def forward(self, z):
        gz = z if self.gate_slice is None else z[:, self.gate_slice[0]:self.gate_slice[1]]
        logits = self.gate(gz)
        if self.training and self.gate_noise > 0:
            logits = logits + torch.randn_like(logits) * self.gate_noise
        g = torch.softmax(logits, dim=1)                       # (B, N)
        self._last_gate = g
        stacked = torch.stack([e(z) for e in self.experts], dim=1)   # (B, N, out)
        return (stacked * g.unsqueeze(-1)).sum(dim=1)          # (B, out)

    def balance_loss(self):
        """Shazeer's importance loss: squared coefficient of variation of per-expert gate mass.

        Zero when the batch spreads evenly over the experts, large when one expert takes
        everything. Without it a softmax gate has no reason not to collapse -- the cheapest way
        to lower training loss early is to back the expert that is currently best and let the
        rest atrophy, which turns an N-expert mixture into one head with N times the parameters.
        """
        if self._last_gate is None:
            return torch.zeros((), device=self.gate.weight.device)
        importance = self._last_gate.sum(dim=0)                # (N,)
        return (importance.std(unbiased=False) / (importance.mean() + 1e-8)) ** 2

    def gate_summary(self):
        """Mean gate weight per expert over the last batch — for checking collapse."""
        if self._last_gate is None:
            return None
        return self._last_gate.mean(dim=0).detach().cpu().numpy()
