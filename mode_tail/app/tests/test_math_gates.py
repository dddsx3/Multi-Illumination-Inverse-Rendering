"""数学冻结 v1.0 §57 correctness gates：MF-0.1 / MF-0.4 / MF-0.5 / MF-0.6。

全部为已知答案测试（无原始数据依赖，CI 可跑）：
  MF-0.1  noise_fit / calibrate 的 polyfit 系数顺序（slope/intercept 错位修复）
  MF-0.4  协方差定理坐标关系 F∞^{1/2} Cov(x̂) F∞^{1/2}/σ² = R⁻¹
          （matched GLS 系综；禁止的 F∞^{-1/2} 形式作对照）
  MF-0.5  奇异 proper covariance：B=[1,1]、Σ_c=diag(1,0)、σ=1、A=[1]
          → 协方差因子/边缘化路线 = 0.5；无约束伪逆精度捷径 = 0（禁止等价宣称）
  MF-0.6  allocation 秩不变：ΔF 数值秩跨 candidates/budgets 保持
          （保持 ⇒ E/A/D-optimal 命名合法；破坏 ⇒ 必须写 pseudo-A/D 口径）
"""

import numpy as np
import pytest

from calibinfo.allocation.policies import (
    SelectionState,
    select_ordering,
    select_ordering_mode_aware,
)
from calibinfo.allocation.rank_invariance import (
    numerical_rank,
    rank_invariance_report,
)
from calibinfo.information.retention import retention_spectrum
from calibinfo.information.schur import (delta_f, delta_f_marginal,
                                         delta_f_marginal_factor, nuisance_factor)

SEED = 20260910


# =================================================================== MF-0.1
def _noise_fit(I, s_hat, convention):
    from experiments.openillumination_validation import noise_fit
    return noise_fit(I, s_hat, convention=convention)


def test_mf01_noise_fit_known_answer():
    """known-answer（精确）：resid² = a₀ + b₀·I 点态成立的合成数据必须逐位
    拟合回 (a₀, b₀)。legacy convention 必然返回错位对（历史 bug 的记录性
    断言）；corrected convention 返回 (a₀, b₀)。"""
    pytest.importorskip("PIL")
    rng = np.random.default_rng(SEED)
    a0, b0 = 0.01, 0.05
    s_hat = rng.uniform(0.2, 1.0, size=50_000)
    r = np.sqrt(a0 + b0 * s_hat)                 # 确定性残差：resid² 点态 = 模型
    I = s_hat + r
    a_c, b_c = _noise_fit(I, s_hat, "corrected")
    assert a_c == pytest.approx(a0, rel=1e-10)
    assert b_c == pytest.approx(b0, rel=1e-10)
    # legacy 的记录性断言：错位顺序（a 位拿到 slope、b 位拿到 intercept）
    a_l, b_l = _noise_fit(I, s_hat, "legacy")
    assert a_l == pytest.approx(b0, rel=1e-10)
    assert b_l == pytest.approx(a0, rel=1e-10)


def test_mf01_calibrate_known_answer():
    """joint_map.calibrate corrected convention 的 known-answer（统计口径）。

    64 灯/像素的合成异方差场景（Var = a₀ + b₀·I）必须拟合回 (a₀, b₀)；
    legacy convention 必然返回错位对（a↔b 互换，偏差 ~150%——与统计偏差
    ~15% 清晰分离）。容差 0.35 覆盖 per-pixel ρ 重估带来的回归失真。"""
    from calibinfo.estimators.joint_map import calibrate
    rng = np.random.default_rng(SEED + 1)
    N, P = 64, 30_000
    n_gt = rng.normal(size=(P, 3))
    n_gt /= np.linalg.norm(n_gt, axis=1, keepdims=True)
    dirs = rng.normal(size=(N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    nl = np.clip(n_gt @ dirs.T, 0, None)                    # (P, N)
    I_hat = nl.T                                            # ρ = 1 ⇒ I_hat = nlᵀ
    a0, b0 = 0.02, 0.05
    resid = rng.normal(0.0, 1.0, size=I_hat.shape) * np.sqrt(a0 + b0 * I_hat)
    _rho, a_l, b_l, _r = calibrate(n_gt, dirs, I_hat + resid, convention="legacy")
    _rho, a_c, b_c, _r = calibrate(n_gt, dirs, I_hat + resid, convention="corrected")
    assert a_c == pytest.approx(a0, rel=0.35)
    assert b_c == pytest.approx(b0, rel=0.35)
    assert a_l == pytest.approx(b0, rel=0.35)               # legacy 错位记录
    assert b_l == pytest.approx(a0, rel=0.35)


def test_mf01_noise_fit_rejects_negative_slope_both_conventions():
    """负斜率（方差随强度下降，非物理）在两种 convention 下都回退平坦噪声——
    守卫判定对象（slope）跨 convention 一致。"""
    pytest.importorskip("PIL")
    rng = np.random.default_rng(SEED + 2)
    x = np.linspace(0.2, 1.0, 5000)
    sd = np.sqrt(0.05 - 0.04 * x)                           # 斜率 −0.04 < 0
    r = rng.normal(0.0, 1.0, size=x.size) * sd
    assert _noise_fit(x + r, x, "legacy") == (1e-4, 0.0)
    assert _noise_fit(x + r, x, "corrected") == (1e-4, 0.0)


# =================================================================== MF-0.4
def test_mf04_retention_inverse_algebraic_identity():
    """代数恒等式（全秩）：F∞^{1/2} ΔF⁻¹ F∞^{1/2} = R⁻¹，逐元素 <1e-10。

    两条独立计算路线：R⁻¹ 走 R 的直接特征分解；lhs 走 ΔF⁻¹ 与 F∞^{1/2}。"""
    rng = np.random.default_rng(SEED + 3)
    m, n, q = 60, 8, 4
    Q, _ = np.linalg.qr(rng.normal(size=(m, n)))
    s = np.geomspace(1.0, 100.0, n)
    A = Q * s[None, :]                                      # F∞ = diag(s²)
    B = rng.normal(size=(m, q))
    Sig_c = np.diag(rng.uniform(0.5, 1.5, size=q))
    sigma = 0.1
    Lam = sigma ** 2 * np.linalg.inv(Sig_c)
    DF, _M, diag = delta_f(A, B, Lam)
    assert not diag["singular"]
    Finf = A.T @ A
    w, V = np.linalg.eigh(Finf)
    Fh = V @ np.diag(1.0 / np.sqrt(w)) @ V.T                # F∞^{-1/2}
    Finvh = V @ np.diag(np.sqrt(w)) @ V.T                   # F∞^{1/2}
    # R⁻¹ 路线一：R = F∞^{-1/2}ΔF F∞^{-1/2} 的直接特征分解
    R = 0.5 * (Fh @ DF @ Fh + (Fh @ DF @ Fh).T)
    wR, UR = np.linalg.eigh(R)
    Rinv = UR @ np.diag(1.0 / wR) @ UR.T
    # 路线二交叉：retention_spectrum 的 (rho, modes) 与上完全一致
    spec = retention_spectrum(DF, Finf)
    assert spec["n_identifiable"] == n
    assert np.allclose(spec["rho"], wR, atol=1e-12)
    assert np.allclose(np.abs(spec["modes"].T @ UR), np.eye(n), atol=1e-8)
    # 已知答案：F∞^{1/2}·(σ²ΔF⁻¹)·F∞^{1/2}/σ² == R⁻¹
    lhs = Finvh @ (sigma ** 2 * np.linalg.inv(DF)) @ Finvh / sigma ** 2
    rel = np.linalg.norm(lhs - Rinv) / np.linalg.norm(Rinv)
    assert rel < 1e-10
    # 禁止形式的对照（v1.0 §47）：F∞^{-1/2}·Cov·F∞^{-1/2} ≠ R⁻¹
    wrong = Fh @ (sigma ** 2 * np.linalg.inv(DF)) @ Fh / sigma ** 2
    assert np.linalg.norm(wrong - Rinv) / np.linalg.norm(Rinv) > 1e-3


def test_mf04_matched_gls_dual_coordinate_variance():
    """MC 已知答案：matched GLS 系综下 Var(z_j) = σ²/ρ_j，
    z_j = u_jᵀ F∞^{1/2}(x̂ − x)（严格坐标）；普通坐标 u_jᵀ(x̂ − x) 偏离 1/ρ_j。"""
    rng = np.random.default_rng(SEED + 4)
    m, n, q = 80, 10, 4
    Q, _ = np.linalg.qr(rng.normal(size=(m, n)))
    s = np.geomspace(1.0, 1000.0, n)                        # F∞ 强尺度不均匀
    A = Q * s[None, :]
    B = rng.normal(size=(m, q))
    Sig_c = np.diag(rng.uniform(0.5, 1.5, size=q))
    sigma = 0.05
    Lam = sigma ** 2 * np.linalg.inv(Sig_c)
    DF, _M, diag = delta_f(A, B, Lam)
    assert not diag["singular"]
    DFinv = np.linalg.inv(DF)
    G = B.T @ B + Lam
    Ginv_Bt = np.linalg.lstsq(G, B.T, rcond=None)[0]
    Finf = A.T @ A
    spec = retention_spectrum(DF, Finf)
    rho = spec["rho"]
    U = spec["modes"]                                       # R 的特征向量（升序）
    w, V = np.linalg.eigh(Finf)
    Fh = V @ np.diag(np.sqrt(w)) @ V.T                      # F∞^{1/2}
    assert w[-1] / w[0] > 1e4                               # 坐标差可见的前提

    a_true = rng.normal(size=n)
    trials = 6000
    E = np.empty((trials, n))
    for t in range(trials):
        dc = rng.multivariate_normal(np.zeros(q), Sig_c)
        eps = rng.normal(0.0, sigma, size=m)
        y = A @ a_true + B @ dc + eps
        My = y - B @ (Ginv_Bt @ y)
        E[t] = DFinv @ (A.T @ My) - a_true

    Z_dual = (E @ Fh) @ U                                   # z_j = u_jᵀ F∞^{1/2} e
    Z_naive = E @ U                                         # 普通坐标（错误对象）
    var_dual = Z_dual.var(axis=0, ddof=1) / sigma ** 2
    var_naive = Z_naive.var(axis=0, ddof=1) / sigma ** 2
    target = 1.0 / rho
    # dual 坐标：中位相对偏差 ≤5%（trials=6000 的 MC 预算）
    med_rel_dual = float(np.median(np.abs(var_dual - target) / target))
    assert med_rel_dual < 0.05, f"dual 坐标偏差过大: {med_rel_dual:.4f}"
    # 普通坐标必须偏离（方向性断言：F∞ 强尺度不均匀时必然可见）
    med_rel_naive = float(np.median(np.abs(var_naive - target) / target))
    assert med_rel_naive > 0.05, "普通坐标意外贴合 1/ρ_j——对照失效"


# =================================================================== MF-0.5
def test_mf05_singular_covariance_known_answer():
    """v1.0 §57 MF-0.5 反例：B=[1,1]、Σ_c=diag(1,0)、σ=1、A=[1]。

    协方差因子 / 边缘化路线 → 0.5（唯一正确值）；
    无约束 Σ_c⁺ 精度捷径（把支撑方向当 flat）→ 0，禁止等价宣称。"""
    A = np.array([[1.0]])
    B = np.array([[1.0, 1.0]])
    Sigma_c = np.diag([1.0, 0.0])
    sigma = 1.0
    F_marg, _ = delta_f_marginal(A, B, Sigma_c, sigma)
    assert float(F_marg[0, 0]) == pytest.approx(0.5, abs=1e-12)
    F_fact, meta_f = delta_f_marginal_factor(A, B, Sigma_c, sigma)
    assert float(F_fact[0, 0]) == pytest.approx(0.5, abs=1e-12)
    assert meta_f["rank_c"] == 1
    L, r_c, _meta = nuisance_factor(Sigma_c)
    assert r_c == 1 and L.shape == (2, 1)
    assert np.allclose(L @ L.T, Sigma_c)
    # 支撑约束 profiling（c ∈ range(Σ_c) ⇒ c₂ ≡ 0；v1.0 §6 形式）：
    # x=1：min_{c₁} (1 − c₁)² + c₁² = 1/2 = xᵀΔFx ✓（解析：c₁* = 1/2）
    c1 = np.linspace(-3, 3, 200_001)
    obj = (1.0 - c1) ** 2 + c1 ** 2
    assert obj.min() == pytest.approx(0.5, rel=1e-6)
    # 禁止捷径：Λ = σ²Σ_c⁺ 塞进无约束 profiling → ΔF = 0 ≠ 0.5
    DF_short, _M, _diag = delta_f(A, B, sigma ** 2 * np.diag([1.0, 0.0]))
    assert float(DF_short[0, 0]) < 1e-12
    assert abs(float(DF_short[0, 0]) - 0.5) > 0.1           # 明确不等价


def test_mf05_full_rank_three_route_agreement():
    """满秩对照：Schur / 边缘化 / 因子三路线逐元素一致 <1e-10（因子路线回归）。"""
    rng = np.random.default_rng(SEED + 5)
    m, n, q = 40, 6, 5
    A = rng.normal(size=(m, n))
    B = rng.normal(size=(m, q))
    Sig_c = np.diag(rng.uniform(0.2, 2.0, size=q))
    sigma = 0.3
    Lam = sigma ** 2 * np.linalg.inv(Sig_c)
    DF, _M, _diag = delta_f(A, B, Lam)
    F1, _ = delta_f_marginal(A, B, Sig_c, sigma)
    F2, _ = delta_f_marginal_factor(A, B, Sig_c, sigma)
    scale = np.linalg.norm(DF)
    assert np.linalg.norm(DF - F1) / scale < 1e-10
    assert np.linalg.norm(DF - F2) / scale < 1e-10


# =================================================================== MF-0.6
def _blocks_state(seed=0, L=12, P=60, n_inactive=2):
    """CI04 whitened per-light 形态的随机状态（同 allocation known-answer 家族）。"""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=(L, P))
    s = rng.uniform(0.2, 1.5, size=(L, P))
    B = rng.normal(0.0, 0.5, size=(L, P, 3))
    if n_inactive:
        w[-n_inactive:] = 0.0
        s[-n_inactive:] = 0.0
        B[-n_inactive:] = 0.0
    u = (w * s)[:, :, None] * B
    M0 = np.einsum("kpi,kpj->kij", B, w[:, :, None] * B)
    lam0 = np.stack([np.diag([30.0, 800.0, 800.0])] * L)
    finf = (w * s ** 2).sum(0)
    active = np.ones(L, bool)
    if n_inactive:
        active[-n_inactive:] = False
    return SelectionState(u=u, M0=M0, lam0=lam0, finf=finf, active=active)


@pytest.mark.parametrize("policy", ["mode_aware", "e_opt", "a_opt", "d_opt"])
def test_mf06_rank_invariant_across_budgets(policy):
    """ΔF 数值秩跨每一步 selection/precision 更新保持（MF-0.6 已知答案）。

    budget_counts=[] ⇒ 每步一记，等价于逐 candidate 检查。"""
    state = _blocks_state(seed=1)
    if policy == "mode_aware":
        ordering, _ = select_ordering_mode_aware(state)
    else:
        ordering, _ = select_ordering(state, policy)
    rep = rank_invariance_report(state, ordering, regime=10.0)
    assert rep["rank_invariant"], f"{policy}: 秩发生变化 {rep['ranks']}"
    assert rep["ranks"][0] == rep["finf_rank"]
    # 正子空间维同样不变（a/d 指标的工作子空间）
    assert len(set(rep["pos_dims"].values())) == 1


def test_mf06_rank_invariant_random_orderings():
    """random ordering 下同样秩不变（抽查 5 条排列，逐步记录）。"""
    for p_i in range(5):
        state = _blocks_state(seed=10 + p_i)
        ordering, _ = select_ordering(state, "random",
                                      rng=np.random.default_rng(100 + p_i))
        rep = rank_invariance_report(state, ordering, regime=100.0)
        assert rep["rank_invariant"], f"perm {p_i}: {rep['ranks']}"


def test_mf06_numerical_rank_detector_contrast():
    """记录性对照：构造会改变数值秩的 PSD 更新，检测器必须能发现
    （rank_invariant 判据不是恒真式）。"""
    M = np.diag([1.0, 1e-14, 1e-14])                        # 数值秩 1（相对容差）
    r0, _ = numerical_rank(M)
    assert r0 == 1
    M2 = M + np.diag([0.0, 1.0, 0.0])                       # PSD 更新抬升零方向
    r1, _ = numerical_rank(M2)
    assert r1 == 2                                          # 秩确实变了
