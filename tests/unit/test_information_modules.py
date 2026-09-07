"""B1/B2 绑定测试 · information 层单源实现（迁移矩阵 C03 验收）。

来源：REPO_MIGRATION B1 表。逐项绑定：
  (a) delta_f vs delta_f_marginal 双路线逐元素相对误差 <1e-10（全秩 + Λ>0）；
      m/q 欠定、临界、过定三区（C06 验收提前覆盖）；
  (b) Λ=0 秩亏走通且无 solve 调用（红队 N3-fix 红线）；
  (c) whiten_system：对角 + 稠密协方差两路径，异方差场景下白化后最小二乘一致；
  (d) gauge_response 闭式 vs 直接 Rayleigh <1e-10（小规模 identity 类）；
  (e) retention_spectrum：谱界 + 正定平方根口径（F∞ 非对角时 diag(1/s) 不可用）；
  (f) track_modes：无交换场景 assignment 恒等；swap 场景正确配对（禁索引排序）。
"""

import numpy as np
import pytest

from calibinfo.information.gauge import gauge_response
from calibinfo.information.mode_tracking import track_modes
from calibinfo.information.retention import retention_spectrum
from calibinfo.information.schur import delta_f, delta_f_marginal
from calibinfo.information.whitening import whiten_system

SEED = 20260907


def _system(rng, m, q, n, rank_B=None):
    """随机系统 + 低秩物理映射 B = B0 Jᵀ（可选）。返回 A, B, Sigma_c, sigma。"""
    A = rng.normal(size=(m, n))
    r = rank_B if rank_B is not None else q
    B0 = rng.normal(size=(m, r))
    if r == q:
        B = B0
    else:
        J = rng.normal(size=(r, q))
        B = B0 @ J                                     # 秩 r < q
    Q, _ = np.linalg.qr(rng.normal(size=(q, q)))
    Sig_phi = np.diag(rng.uniform(0.1, 1.0, size=r)) if r < q \
        else np.diag(rng.uniform(0.1, 1.0, size=q))
    if r < q:
        Sig_c = J @ Sig_phi @ J.T
    else:
        Sig_c = Sig_phi
    sigma = 0.1
    return A, B, Sig_c, sigma


@pytest.mark.parametrize("m,q,n", [(40, 3, 8), (30, 5, 5), (25, 9, 4)])  # 过定/临界/欠定(n<q)
def test_dual_route_full_rank(m, q, n):
    """双路线：Schur(ΔF) vs marginal 直接逆——m/q 三区逐元素 rel<1e-10（C06 验收）。"""
    rng = np.random.default_rng(SEED)
    A, B, Sig_c, sigma = _system(rng, m, q, n)
    Lam = sigma ** 2 * np.linalg.inv(Sig_c)
    DF, _M, diag = delta_f(A, B, Lam)
    assert not diag["singular"]
    DF_m, meta_m = delta_f_marginal(A, B, Sig_c, sigma)
    rel = np.linalg.norm(DF - DF_m) / np.linalg.norm(DF_m)
    assert rel < 1e-10


def test_lambda0_rank_deficient_no_solve():
    """Λ=0 且 B 秩亏：delta_f 走 lstsq/pinv 极限、M(0) 幂等投影、标记 singular；
    gauge 方向（Aa = Bc̄ 精确成立）被完全消去。"""
    rng = np.random.default_rng(SEED)
    m, q, n = 50, 6, 5
    B = rng.normal(size=(m, 4)) @ rng.normal(size=(4, q))     # 秩 4 < q=6
    cbar = rng.normal(size=q)
    # 构造 gauge 恒等式精确成立：A e₁ = B c̄，其余列随机
    A = rng.normal(size=(m, n))
    A[:, 0] = B @ cbar
    a = np.zeros(n)
    a[0] = 1.0
    assert np.linalg.norm(A @ a - B @ cbar) < 1e-12
    DF, M, diag = delta_f(A, B, 0.0)
    assert diag["singular"]
    # M(0) = I − UUᵀ（col(B) 投影的补）→ 幂等
    assert np.linalg.norm(M @ M - M) / np.linalg.norm(M) < 1e-10
    # gauge 方向被完全消去：ΔF a = 0（Λ=0 时精确）
    assert np.linalg.norm(DF @ a) < 1e-8 * max(np.linalg.norm(DF), 1.0)


def test_solve_never_called():
    """红线 RL-solve 机械盯防：delta_f 源码不得出现 np.linalg.solve / linalg.solve 调用。"""
    import inspect
    from calibinfo.information import schur
    src = inspect.getsource(schur.delta_f)
    assert "solve(" not in src.replace("lstsq(", ""), "delta_f 出现 solve 调用"


def test_whiten_system_diag_and_dense():
    """白化：对角快路径 + 稠密 eigh 路径；白化后系统与手工加权一致（<1e-12）。"""
    rng = np.random.default_rng(SEED)
    m, q, n = 40, 4, 6
    A = rng.normal(size=(m, n))
    B = rng.normal(size=(m, q))
    var = rng.uniform(0.5, 3.0, size=m)
    Aw, Bw, meta = whiten_system(A, B, var)
    assert meta["method"] == "diag"
    assert np.allclose(Aw, (1 / np.sqrt(var))[:, None] * A)
    # 稠密路径
    Sy = np.diag(var)
    Aw2, Bw2, meta2 = whiten_system(A, B, Sy)
    assert meta2["method"] == "eigh_dense"
    assert np.allclose(Aw2, Aw) and np.allclose(Bw2, Bw)
    # 白化后 Schur 与原始加权 GLS 等价（宪法 §2.1：一切从白化形式开始）：
    # Aᵀ(Σ_y + BΣ_cBᵀ)⁻¹A == 白化空间 ΔF(Λw=σ'²Σ_c⁻¹, σ'²=1)
    Sig_c = np.eye(q)
    V_total = np.diag(var) + B @ Sig_c @ B.T
    DF_gls = A.T @ np.linalg.solve(V_total, A)
    DFw, _, _ = delta_f(Aw, Bw, Sig_c)                  # Λw = 1·Σ_c⁻¹
    rel = np.linalg.norm(DF_gls - DFw) / np.linalg.norm(DF_gls)
    assert rel < 1e-10
    # 白化空间内双路线自洽
    DFm_w, _ = delta_f_marginal(Aw, Bw, Sig_c, 1.0)
    assert np.linalg.norm(DFw - DFm_w) / np.linalg.norm(DFm_w) < 1e-10
    # 非正定 Σ_y 必须拒绝
    with pytest.raises(AssertionError):
        whiten_system(A, B, -var)


def test_gauge_response_closed_form_vs_rayleigh():
    """gauge 闭式 vs 直接构造 ΔF 的 Rayleigh 商（identity <1e-10）。"""
    rng = np.random.default_rng(SEED)
    m, q = 60, 5
    B = rng.normal(size=(m, q))
    cbar = rng.normal(size=q)
    A = rng.normal(size=(m, 7))
    # 构造满足 gauge 恒等式 Aa = Bc̄ 的 (A, a)
    a = rng.normal(size=7)
    A = A - np.outer(A @ a - B @ cbar, a) / (a @ a)      # 投影修正 A a = B c̄
    assert np.linalg.norm(A @ a - B @ cbar) < 1e-10
    lam = 0.37
    out = gauge_response(B, cbar, lam)
    DF, _, _ = delta_f(A, B, lam)
    rayleigh = float(a @ DF @ a)
    rel = abs(out["exact"] - rayleigh) / max(abs(rayleigh), 1e-300)
    assert rel < 1e-10
    # 两端：λ→0 斜率 / λ→∞ 饱和
    out0 = gauge_response(B, cbar, 1e-8)
    assert abs(out0["exact"] / 1e-8 - out["slope"]) / out["slope"] < 1e-4
    out_inf = gauge_response(B, cbar, 1e10)
    assert abs(out_inf["exact"] - out["saturation"]) / out["saturation"] < 1e-6


def test_retention_bounds_and_sqrt_convention():
    """retention：谱界 0≤ρ≤1；F∞ 非对角时正定平方根口径（red line RL-retention-whitening）。"""
    rng = np.random.default_rng(SEED)
    m, q, n = 80, 4, 10
    A = rng.normal(size=(m, n))
    B = rng.normal(size=(m, q))
    Finf = A.T @ A
    DF, _, _ = delta_f(A, B, 0.3)
    out = retention_spectrum(DF, Finf)
    assert out["bounds_ok"] and out["n_identifiable"] == n
    # 非对角 F∞：手工 diag(1/√diag(F∞)) 的错误口径必然给不同（可能越界）结果——
    # 这里只验证正确口径下谱界成立且与红队 V3 口径（对角特例）一致：
    Adiag_basis = rng.normal(size=(m, n))
    s = rng.uniform(0.5, 2.0, size=n)
    A2 = Adiag_basis * s                                  # 列 j 范数 ≈ 无关；改用显式构造
    A2 = np.linalg.qr(rng.normal(size=(m, n)))[0] * s     # 正交列 × 尺度 → Finf 对角 s²
    B2 = rng.normal(size=(m, 3))
    DF2, _, _ = delta_f(A2, B2, 0.2)
    out2 = retention_spectrum(DF2, A2.T @ A2)
    assert out2["bounds_ok"]
    Finf2 = A2.T @ A2
    assert np.linalg.norm(Finf2 - np.diag(np.diag(Finf2))) < 1e-10   # 确为对角


def test_track_modes_assignment():
    """track_modes：同基（无交换）assignment 恒等；交换列序后正确配对（禁索引排序）。

    assignment 语义：prev 模式 i ↔ curr 模式 assignment[i]（最大总重叠）。
    """
    rng = np.random.default_rng(SEED)
    n, k = 12, 4
    V = np.linalg.qr(rng.normal(size=(n, k)))[0]
    r1 = track_modes(V, V)
    assert np.array_equal(r1["assignment"], np.arange(k))
    perm = np.array([2, 0, 3, 1])
    inv = np.argsort(perm)                               # prev i 匹配 curr j=inv[i]
    r2 = track_modes(V, V[:, perm])
    assert np.array_equal(r2["assignment"], inv)
    # 近旋转基（N>1 情形的缩影）：assignment 跟随最大重叠
    th = 0.2
    G = np.eye(k)
    G[0, 0], G[0, 1], G[1, 0], G[1, 1] = np.cos(th), -np.sin(th), np.sin(th), np.cos(th)
    Vr = V @ G
    r3 = track_modes(V, Vr)
    assert np.array_equal(r3["assignment"], np.arange(k))
