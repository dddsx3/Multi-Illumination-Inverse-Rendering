"""低秩恒等式与 Woodbury 路线测试（M2；src/calibinfo/information/lowrank.py）。

已知答案（小规模合成，无原始数据依赖）：
  1. 谱恒等式：低秩路线 vs 稠密 eigh(R) ≤1e-10；ρ≡1 计数 = P−3L；
  2. tr ΔF⁻¹：低秩 Woodbury vs 稠密 inv ≤1e-10；
  3. 低秩梯度 vs 中心差分 ≤1e-6；
  4. 非平凡谱的 regime 移动（精度↑ ⇒ 弱模式 ρ↑——probe e11 的 rho_none/rho_all
     结构的机器可查形式）。
"""

import numpy as np

from calibinfo.allocation.blocks import sym_inv
from calibinfo.information.lowrank import (
    retention_spectrum_lowrank,
    woodbury_trace_inv,
    woodbury_trace_inv_grad,
)

SEED = 20260917


def _blocks(seed=SEED, L=4, P=60):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=(L, P))
    s = rng.uniform(0.2, 1.5, size=(L, P))
    B = rng.normal(0.0, 0.5, size=(L, P, 3))
    u = (w * s)[:, :, None] * B
    M0 = np.einsum("kpi,kpj->kij", B, w[:, :, None] * B)
    lam0 = np.stack([np.diag([30.0, 800.0, 800.0])] * L)
    finf = (w * s ** 2).sum(0)
    active = np.ones(L, bool)
    return u, M0, lam0, finf, active


def _assemble(u, M0, lam0, finf, active, t):
    DF = np.diag(finf)
    for k in np.flatnonzero(active):
        K = sym_inv(M0[k] + t[k] * lam0[k])
        DF = DF - (u[k] @ K) @ u[k].T
    return DF


def test_spectrum_lowrank_matches_dense():
    """谱恒等式：低秩 vs 稠密 ≤1e-10；ρ≡1 计数 = P − 3L。"""
    u, M0, lam0, finf, active = _blocks()
    L, P = u.shape[0], u.shape[1]
    t = np.full(L, 3.0)
    low = retention_spectrum_lowrank(finf, u, M0, lam0, active, t)
    Fh_inv = np.diag(1.0 / np.sqrt(finf))
    DF = _assemble(u, M0, lam0, finf, active, t)
    R = Fh_inv @ DF @ Fh_inv
    rho_dense = np.sort(np.linalg.eigvalsh(0.5 * (R + R.T)))
    assert low["n_unit"] == P - 3 * L
    assert np.max(np.abs(rho_dense - low["rho"])) <= 1e-10


def test_woodbury_trace_inv_matches_dense():
    u, M0, lam0, finf, active = _blocks(seed=SEED + 1)
    t = np.full(4, 2.0)
    DF = _assemble(u, M0, lam0, finf, active, t)
    f_dense = float(np.trace(np.linalg.inv(DF)))
    f_low = woodbury_trace_inv(finf, u, M0, lam0, active, t)
    assert abs(f_dense - f_low) <= 1e-10 * max(1.0, abs(f_dense))


def test_woodbury_gradient_matches_fd():
    u, M0, lam0, finf, active = _blocks(seed=SEED + 2)
    rng = np.random.default_rng(SEED + 3)
    t = rng.uniform(1.0, 4.0, 4)
    f0, grad = woodbury_trace_inv_grad(finf, u, M0, lam0, active, t)
    h = 1e-6
    for k in range(4):
        tp, tm = t.copy(), t.copy()
        tp[k] += h
        tm[k] -= h
        fd = (woodbury_trace_inv(finf, u, M0, lam0, active, tp)
              - woodbury_trace_inv(finf, u, M0, lam0, active, tm)) / (2 * h)
        assert abs(grad[k] - fd) <= 1e-6 * max(1.0, abs(fd))
    assert np.all(grad <= 1e-12)                       # tr ΔF⁻¹ 关于 t 递减


def test_nontrivial_tail_moves_with_regime():
    """精度↑ ⇒ 弱模式 ρ 上升（e11 rho_none/rho_all 结构的回归形式）。"""
    u, M0, lam0, finf, active = _blocks(seed=SEED + 4)
    low_lo = retention_spectrum_lowrank(finf, u, M0, lam0, active, np.ones(4))
    low_hi = retention_spectrum_lowrank(finf, u, M0, lam0, active, np.full(4, 25.0))
    assert low_lo["rho"][0] < low_hi["rho"][0]         # 最弱模式被抬升
    assert low_lo["n_unit"] == low_hi["n_unit"] == 60 - 12
