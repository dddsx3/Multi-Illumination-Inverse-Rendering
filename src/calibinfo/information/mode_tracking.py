"""B1 · mode_tracking：跨 λ 的连续性模式追踪（R3 修订）。

红线（宪法 §0 禁令 + 红队报告三 攻击三/R3）：跨 λ 模式识别**必须**用连续性追踪
（相邻 λ 特征向量重叠度 assignment），**禁止**"第 j 小特征值"索引排序——
N=1 时特征基 λ 不变（V6 观测 0.00°），N>1 时可旋转 88°+。
绑定测试：test_information_modules.py + test_v6_mode_tracking.py（红队 V 系列）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def track_modes(prev_vecs, curr_vecs, k=None, degenerate_tol=1e-3):
    """基于 |V_prevᵀ V_curr| 重叠度 assignment 的连续性追踪。

    参数
    ----
    prev_vecs, curr_vecs : (n, k) 相邻两个 λ（或 τ）处的特征基（列 = 模式）
    k                    : 追踪的模式数（默认全列）
    degenerate_tol       : 重叠矩阵奇异值相对间隔 < tol 视为简并（assignment 不可靠）

    返回
    ----
    dict(
      assignment   : (k,) 数组——curr 模式 j 对应的 prev 模式号（最大化总重叠）
      overlap      : (k, k) |V_prevᵀ V_curr|
      degenerate   : bool——重叠矩阵存在近简并奇异值（该步 assignment 需人工复核）
      min_gap      : 相邻奇异值最小相对间隔
    )
    """
    prev_vecs = np.asarray(prev_vecs, float)
    curr_vecs = np.asarray(curr_vecs, float)
    k = k or prev_vecs.shape[1]
    assert curr_vecs.shape == prev_vecs.shape
    O = np.abs(prev_vecs[:, :k].T @ curr_vecs[:, :k])   # (k, k) 重叠度
    row, col = linear_sum_assignment(-O)                 # 最大化总重叠
    assignment = np.empty(k, dtype=int)
    assignment[row] = col
    sv = np.linalg.svd(O, compute_uv=False)
    sv_sorted = np.sort(sv)[::-1]
    gaps = (sv_sorted[:-1] - sv_sorted[1:]) / np.maximum(sv_sorted[:-1], 1e-300)
    min_gap = float(gaps.min()) if gaps.size else float("inf")
    return dict(assignment=assignment, overlap=O,
                degenerate=bool(min_gap < degenerate_tol), min_gap=min_gap)
