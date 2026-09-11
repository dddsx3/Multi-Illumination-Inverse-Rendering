"""MF-0.6 · allocation rank invariance check（数学冻结 v1.0 §57 MF-0.6）。

问题：allocation 的 pseudo-A/D 指标（objective_of 的 positive-subspace 口径）
只有在 **common identifiable subspace / 数值秩跨 candidates 与 budgets 保持**
时才能称为标准 A/D-optimal criteria；若秩会变，则必须写
"implemented pseudo-A / positive-subspace D criteria"（秩变化会让
tr(F⁺)/logdet 出现反直觉跳变——v1.0 §36 的 F(ε) 反例：F=diag(1,0) →
diag(1,ε) 使 tr(F⁺) 从 1 跳到 1+ε⁻¹）。

本模块提供纯预测侧检查器：给定 SelectionState 与（冻结或新选的）ordering，
沿 budget 前缀推进 precision 更新，在每个状态记录 ΔF 的数值秩与正子空间维
（objective_of 同族的相对容差），返回逐状态报告与不变性判定。
"""
from __future__ import annotations

import numpy as np

from calibinfo.allocation.blocks import LightBlocks, sym_inv

RANK_TOL_REL = 1e-10          # 数值秩的相对容差（objective_of 同族口径）


def _pos_dim(ev, tol_rel):
    """正子空间维（objective_of 的 a/d 口径：ev > tol_rel·max(ev, 1)）。"""
    tol = tol_rel * max(float(ev[-1]), 1.0)
    return int((ev > tol).sum())


def numerical_rank(M, tol_rel=RANK_TOL_REL):
    """对称 PSD 矩阵的数值秩（相对容差，eigh 路径）；返回 (rank, 特征值升序)。"""
    ev = np.linalg.eigvalsh(np.asarray(M, float))
    if ev.size == 0:
        return 0, ev
    tol = tol_rel * max(float(ev[-1]), 1.0)
    return int((ev > tol).sum()), ev


def rank_invariance_report(state, ordering, regime=10.0, budget_counts=(),
                           rank_tol_rel=RANK_TOL_REL):
    """沿一条 ordering 记录每个 budget 前缀状态的 ΔF 数值秩。

    state      : SelectionState（预测侧泄露面）
    ordering   : list[int]，完整 light 排序（冻结 selection_orders.json 的行）
    regime     : precision 提升因子（λ_l → regime·λ_l）
    budget_counts: 前缀长度列表（如 [14, 28, 57, 85, 114]）；空列表 = 每步都记

    返回 dict(
      ranks          : {k: rank(ΔF after k selections)}（含 k=0 基态）
      pos_dims       : {k: 正子空间维}（a/d 指标的工作子空间维）
      lam_min        : {k: λmin(ΔF)}
      rank_invariant : bool——所有记录状态秩相同
      finf_rank      : rank(diag(F∞))（基准可识别维）
    )
    """
    blocks = LightBlocks(state.u, state.M0, state.lam0, state.finf, state.active)
    lam = state.lam0.copy()
    DF = blocks.assemble(lam)
    finf_rank, _ = numerical_rank(np.diag(state.finf), rank_tol_rel)

    marks = sorted(set(int(k) for k in budget_counts)) if budget_counts \
        else list(range(1, len(ordering) + 1))
    ranks, pos_dims, lam_min = {}, {}, {}
    r0, ev0 = numerical_rank(DF, rank_tol_rel)
    ranks[0], lam_min[0] = r0, float(ev0[0])
    pos_dims[0] = _pos_dim(ev0, rank_tol_rel)
    seen = 0
    for idx in ordering:
        idx = int(idx)
        if state.active[idx]:
            K_old = sym_inv(state.M0[idx] + lam[idx])
            lam[idx] = regime * lam[idx]
            K_new = sym_inv(state.M0[idx] + lam[idx])
            DF = DF + (state.u[idx] @ (K_old - K_new)) @ state.u[idx].T
        seen += 1
        if seen in marks:
            r, ev = numerical_rank(DF, rank_tol_rel)
            ranks[seen] = r
            lam_min[seen] = float(ev[0])
            pos_dims[seen] = _pos_dim(ev, rank_tol_rel)
    vals = [v for v in ranks.values()]
    return dict(ranks=ranks, pos_dims=pos_dims, lam_min=lam_min,
                rank_invariant=bool(len(set(vals)) == 1),
                finf_rank=finf_rank)
