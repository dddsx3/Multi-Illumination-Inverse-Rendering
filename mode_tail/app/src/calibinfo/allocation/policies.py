"""Unified sequential selection policies (preregistered frame).

Every policy produces one full ordering of the light universe:

    l_t = argmax over unselected lights of J_l(Lambda_t)

with an immediate precision update of the selected light
(Lambda_l -> regime * Lambda_l, i.e. Sigma_phi,l divided by the regime factor)
before the next step. Budgets are prefixes of the ordering.

Policies:
  mode_aware : J = G_l — the **adaptive normalized weak-Fisher-mode sensitivity
               heuristic** (the frozen name for the current implementation, per
               the math freeze v1.0 §33): bottom-5 Euclidean eigenvectors of the
               plain DeltaF, recomputed at every step, per-mode sensitivities
               normalized across lights, then averaged into G_l. This is a
               heuristic aggregate, NOT an exact retention-gradient allocation
               (that would require F_inf-normalized generalized eigenvectors).
  e_opt      : J = lambda_min gain of the candidate-updated DeltaF
  a_opt      : J = trace(pinv(DeltaF)) decrease of the candidate update
               (positive-subspace pseudo-A as implemented)
  d_opt      : J = positive-subspace logdet increase of the candidate update
  random     : uniform permutation (paired raw-draw expectation estimator)

Leakage surface (V-B4): SelectionState carries the whitened linearization assets
and precision blocks ONLY -- u, M0, Lam0, Finf and the active mask, plus DeltaF/R
derived internally. The nominal calibratables consumed by the reconstruction, the
data images and any reconstruction error are not representable in this
state and unreachable from select_ordering (the reconstruction-side identifiers
are absent from this module by construction; see test V-B4).

Tie-break: ascending light index among equal J values (strict-improvement scan).
Inactive lights (u_l = 0) have an identity candidate update, so their objective
gain is exactly 0; the engine evaluates this exactly instead of re-running an
eigen-solve that provably returns the same value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibinfo.allocation.blocks import (
    LightBlocks,
    EPS,
    mode_gains,
    objective_of,
    sym_inv,
)

N_MODES = 5


@dataclass(frozen=True)
class SelectionState:
    """Prediction-side selection inputs (the complete leakage surface)."""

    u: np.ndarray        # (L, P, 3) per-light whitened design columns
    M0: np.ndarray       # (L, 3, 3) per-light nuisance Gram B_l^T B_l
    lam0: np.ndarray     # (L, 3, 3) base precision blocks (update direction)
    finf: np.ndarray     # (P,) F-infinity diagonal
    active: np.ndarray   # (L,) bool: lights with Fisher presence


POLICIES = ("mode_aware", "e_opt", "a_opt", "d_opt", "random")


def budget_scales(ordering, k, regime):
    """Per-light corruption std scale for a budget prefix.

    Exactly k lights carry scale 1/sqrt(regime); all others 1. The identical count
    across policies at a given budget is the mechanical budget-fairness guarantee.
    """
    scales = np.ones(len(ordering))
    for pos in ordering[:k]:
        scales[pos] = 1.0 / np.sqrt(regime)
    return scales


def _apply_update(state, lam, DF, idx, regime):
    K_old = sym_inv(state.M0[idx] + lam[idx])
    lam[idx] = regime * lam[idx]
    K_new = sym_inv(state.M0[idx] + lam[idx])
    return DF + (state.u[idx] @ (K_old - K_new)) @ state.u[idx].T


def select_ordering(state, policy, regime=10.0, rng=None):
    """Full light ordering under the frozen sequential frame.

    Returns (ordering, steps): ordering is a permutation of range(L); steps records
    per-step {"index", "J"} for the deterministic policies (empty for random).

    Exact-evaluation skip: inactive candidates (u_l = 0) have an identity candidate
    update, so their gain is exactly 0.0 and never beats a strictly positive active
    gain. When no active candidate strictly improves, the full-scan argmax with
    ascending-index tie-break resolves over the zero-gain pool (inactive lights +
    exactly-zero active gains) -- the branch below reproduces that resolution
    exactly (V-B6's full-scan reference guards the equivalence).
    """
    L = state.u.shape[0]
    if policy == "random":
        if rng is None:
            raise ValueError("random policy requires a permutation rng")
        perm = rng.permutation(L)
        return [int(i) for i in perm], []

    blocks = LightBlocks(state.u, state.M0, state.lam0, state.finf, state.active)
    lam = state.lam0.copy()
    DF = blocks.assemble(lam)
    kind = {"e_opt": "e", "a_opt": "a", "d_opt": "d"}[policy]
    cur = objective_of(DF, kind)
    ordering, steps = [], []
    unselected = set(range(L))

    for _t in range(L):
        active_unselected = sorted(i for i in unselected if state.active[i])
        gains = {}
        for idx in active_unselected:                      # ascending-index scan
            K_old = sym_inv(state.M0[idx] + lam[idx])
            K_new = sym_inv(state.M0[idx] + regime * lam[idx])
            upd = (state.u[idx] @ (K_old - K_new)) @ state.u[idx].T
            cand = objective_of(DF + upd, kind)
            gain = (cur - cand) if kind == "a" else (cand - cur)
            gains[idx] = gain
        max_active = max(gains.values(), default=0.0)
        if max_active > 0.0:
            best_idx = min(i for i, gv in gains.items() if gv == max_active)
            best_j = max_active
        else:
            zero_pool = [i for i in unselected if not state.active[i]]
            zero_pool += [i for i, gv in gains.items() if gv == 0.0]
            if zero_pool:
                best_idx, best_j = min(zero_pool), 0.0
            else:                                          # all gains negative
                best_idx = min(gains, key=lambda i: (-gains[i], i))
                best_j = gains[best_idx]
        ordering.append(best_idx)
        unselected.remove(best_idx)
        steps.append({"index": best_idx, "J": float(best_j)})
        if state.active[best_idx]:
            DF = _apply_update(state, lam, DF, best_idx, regime)
            cur = (cur - best_j) if kind == "a" else (cur + best_j)
    return ordering, steps


def select_ordering_mode_aware(state, regime=10.0):
    """mode_aware policy (separate loop: its J is the aggregated G_l, not a
    Fisher objective gain; the bottom-5 mode basis is recomputed at every step)."""
    L = state.u.shape[0]
    blocks = LightBlocks(state.u, state.M0, state.lam0, state.finf, state.active)
    lam = state.lam0.copy()
    DF = blocks.assemble(lam)
    ordering, steps = [], []
    unselected = set(range(L))
    for _t in range(L):
        _w, V = np.linalg.eigh(DF)
        modes = V[:, :N_MODES]                              # bottom-5, ascending
        g = mode_gains(state.u, state.M0, lam, state.lam0, state.active, modes)
        uns = sorted(unselected)
        denom = g[uns].sum(0) + EPS                         # (m,)
        Gs = (g / denom).mean(1)                            # (L,)
        best_idx, best_j = None, None
        for idx in uns:                                     # ascending tie-break
            if best_j is None or Gs[idx] > best_j:
                best_idx, best_j = idx, float(Gs[idx])
        ordering.append(best_idx)
        unselected.remove(best_idx)
        steps.append({"index": best_idx, "J": best_j})
        if state.active[best_idx]:
            DF = _apply_update(state, lam, DF, best_idx, regime)
    return ordering, steps
