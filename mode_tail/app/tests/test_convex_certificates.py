"""Certificate machinery tests（plan v2 L4；src/calibinfo/allocation/convex.py）。

已知答案（小规模合成，无原始数据依赖）：
  1. LMO 正确性：预算约束线性目标的最优解与随机可行点暴力对拍；
  2. FW 间隙全程 ≥ 0 且（通常）单调下降；返回点的间隙是可行下界证书；
  3. FW 终点 J ≤ 任意随机可行点的 J − gap 余量内（证书不被打穿）；
  4. 预算可行性不变式：Σ(t−1) ≤ B、t ∈ [1,κ]。
"""

import numpy as np
import pytest

from calibinfo.allocation.blocks import LightBlocks
from calibinfo.allocation.convex import CertificateProblem, budget_for_k


def _problem(seed=0, L=6, P=30):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=(L, P))
    s = rng.uniform(0.2, 1.5, size=(L, P))
    B = rng.normal(0.0, 0.5, size=(L, P, 3))
    u = (w * s)[:, :, None] * B
    M0 = np.einsum("kpi,kpj->kij", B, w[:, :, None] * B)
    lam0 = np.stack([np.diag([30.0, 800.0, 800.0])] * L)
    finf = (w * s ** 2).sum(0)
    blocks = LightBlocks(u, M0, lam0, finf, np.ones(L, bool))
    return CertificateProblem(blocks=blocks, kappa=5.0)


# ------------------------------------------------------- 1 LMO
def test_lmo_matches_brute_force():
    """LMO 解析解与随机可行点暴力对拍：⟨g, s*⟩ ≤ ⟨g, s⟩ − tol。"""
    prob = _problem(seed=1)
    rng = np.random.default_rng(20260914)
    for _ in range(50):
        g = rng.normal(size=prob.blocks.L)
        B = float(rng.uniform(0, prob.blocks.L * (prob.kappa - 1)))
        s_star = prob.lmo(g, B)
        # 可行性
        assert np.all(s_star >= 1.0 - 1e-12)
        assert np.all(s_star <= prob.kappa + 1e-12)
        assert np.sum(s_star - 1.0) <= B + 1e-9
        # 随机可行点不能更优（目标 = ⟨g, s⟩，最小化）
        b = rng.uniform(0, prob.kappa - 1.0, size=prob.blocks.L)
        b *= min(1.0, B / max(np.sum(b), 1e-12))
        s_rand = 1.0 + b
        assert g @ s_star <= g @ s_rand + 1e-9 * max(1.0, np.abs(g @ s_rand))


# ------------------------------------------------------- 2 FW 证书
def test_frank_wolfe_gap_nonnegative_and_feasible():
    prob = _problem(seed=2)
    B = budget_for_k(2, prob.kappa)
    out = prob.frank_wolfe(B, iters=40)
    t = out["t"]
    assert np.all(t >= 1.0 - 1e-12) and np.all(t <= prob.kappa + 1e-12)
    assert np.sum(t - 1.0) <= B + 1e-9
    assert out["gap"] >= 0.0
    for h in out["history"]:
        assert h["gap"] >= 0.0


def test_frank_wolfe_certificate_not_beaten_by_random_points():
    """证书不被打穿：J* ≥ J_FW − gap 对随机可行点成立（核心验收）。"""
    prob = _problem(seed=3)
    B = budget_for_k(3, prob.kappa)
    out = prob.frank_wolfe(B, iters=60)
    lower = out["J_A"] - out["gap"]
    rng = np.random.default_rng(20260915)
    for _ in range(30):
        b = rng.uniform(0, prob.kappa - 1.0, size=prob.blocks.L)
        b *= min(1.0, B / max(np.sum(b), 1e-12))
        t_rand = 1.0 + b
        assert prob.J_A(t_rand) >= lower - 1e-9 * max(1.0, abs(lower))


def test_frank_wolfe_gap_decreases_and_small():
    """间隙随迭代下降（凸程序收敛），且终值相对 J 的量级很小。"""
    prob = _problem(seed=4)
    B = budget_for_k(2, prob.kappa)
    out = prob.frank_wolfe(B, iters=60)
    gaps = [h["gap"] for h in out["history"]]
    assert gaps[-1] <= gaps[0]                       # 总体下降
    assert gaps[-1] <= 0.05 * max(1.0, abs(out["J_A"]))


# ------------------------------------------------------- woodbury 路线等价
def test_woodbury_route_equals_dense_route():
    """woodbury 低秩路线与 dense 路线的 J_A/梯度/FW 结果在 PD 域上等价。"""
    from calibinfo.information.lowrank import woodbury_trace_inv_grad
    prob_d = _problem(seed=11)
    prob_w = _problem(seed=11)
    prob_w.route = "woodbury"
    rng = np.random.default_rng(20260916)
    for _ in range(20):
        t = rng.uniform(1.0, 5.0, size=prob_d.blocks.L)
        f_d = prob_d.J_A(t)
        f_w, g_w = woodbury_trace_inv_grad(
            prob_w.blocks.finf, prob_w.blocks.u, prob_w.blocks.M0,
            prob_w.blocks.lam0, prob_w.blocks.active, t)
        assert abs(f_d - f_w) <= 1e-9 * max(1.0, abs(f_d))
        g_d = prob_d.grad_J_A(t)
        assert np.max(np.abs(g_d - g_w)) <= 1e-8 * max(1.0, float(np.max(np.abs(g_d))))
    # FW 终点一致（同一证书程序）
    B = budget_for_k(2, prob_d.kappa)
    out_d = prob_d.frank_wolfe(B, iters=30)
    out_w = prob_w.frank_wolfe(B, iters=30)
    assert abs(out_d["J_A"] - out_w["J_A"]) <= 1e-8 * max(1.0, abs(out_d["J_A"]))
