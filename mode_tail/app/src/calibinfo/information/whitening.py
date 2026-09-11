"""B1 · whitening：W = Σ_y^{-1/2} 白化。

所有理论与代码从加权/白化形式开始：对原始观测 y = Ax + Bδc + ε, ε~N(0,Σ_y)，
先以 W = Σ_y^{-1/2} 白化 A、B、残差，下游（schur.delta_f 等）只接受白化后系统。
绑定测试：test_information_modules.py::test_whiten_system（含异方差 + 稠密协方差）。
"""

from __future__ import annotations

import numpy as np


def whiten_system(A, B, Sigma_y):
    """返回 (Aw, Bw, meta)：Aw = W A，Bw = W B，W = Σ_y^{-1/2}。

    Sigma_y 支持：
      - (m,) 对角方差向量（快路径）；
      - (m, m) 对称正定协方差（eigh 构造正定平方根，微负特征值按数值零截断）。
    meta: method、cond（条件数）、n_truncated（截断的特征值数，应通常为 0）。
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    Sigma_y = np.asarray(Sigma_y, float)
    m = A.shape[0]
    assert B.shape[0] == m
    if Sigma_y.ndim == 1:
        assert Sigma_y.shape[0] == m
        assert np.all(Sigma_y > 0), "对角方差必须严格为正"
        w = 1.0 / np.sqrt(Sigma_y)
        meta = dict(method="diag", cond=float(Sigma_y.max() / Sigma_y.min()),
                    n_truncated=0)
        return w[:, None] * A, w[:, None] * B, meta
    assert Sigma_y.shape == (m, m)
    Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)              # 对称化
    ev, V = np.linalg.eigh(Sigma_y)
    tol = max(ev.max(), 1.0) * 1e-12
    n_trunc = int((ev <= tol).sum())
    assert n_trunc == 0, f"Σ_y 有 {n_trunc} 个非正特征值——先修噪声模型再白化"
    w = 1.0 / np.sqrt(ev)
    W = V @ np.diag(w) @ V.T
    meta = dict(method="eigh_dense",
                cond=float(ev.max() / ev.min()), n_truncated=0)
    return W @ A, W @ B, meta
