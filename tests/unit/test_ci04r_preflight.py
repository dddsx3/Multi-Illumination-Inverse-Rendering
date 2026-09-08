"""CI04-R Preflight P1–P5（已知答案测试，T3.B；任一失败 → STOP）。

P1 λmin 读取：禁 F.min()，必须 eigvalsh(F).min()；diag(1,…,1,1e-7) → 1e-7。
P2 trace 稀释：d=100, diag(1×99, ε) 的 trace ratio / Δlogdet / λmin 已知答案。
P3 并联和秩亏：S:Λ = S(S+Λ)⁺Λ 一般式（Anderson–Duffin）；禁裸 inv/solve。
P4 gauge 恒等式回归：aᵀΔFa = c̄ᵀ[(BᵀB):Λ]c̄（随机秩亏 B、各向异性 Λ）。
P5 identifiable-subspace overlap：秩亏 F∞ 下 R2-C overlap 无 NaN/Inf、0≤q≤1+1e-10。
"""
import numpy as np
import pytest

from calibinfo.metrics.ci04r_predictors import (
    lambda_min,
    trace_ratio,
    logdet_deficit,
    parallel_sum,
    identifiable_overlap,
)


# --------------------------------------------------------------- P1
def test_p1_lambda_min_reads_diagonal_last_entry():
    d = 100
    F = np.diag(np.concatenate([np.ones(d - 1), [1e-7]]))
    assert lambda_min(F) == pytest.approx(1e-7, rel=1e-12)
    # 稠密对称矩阵情形：随机正交旋转对角阵，min 特征值仍为 1e-7，
    # 而 F.min()（逐元素）取到非对角元 ≠ 1e-7 —— 这正是必须 eigvalsh 的原因
    rng = np.random.default_rng(20260908)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Fd = (Q * np.concatenate([np.ones(d - 1), [1e-7]])) @ Q.T
    Fd = (Fd + Fd.T) / 2
    assert lambda_min(Fd) == pytest.approx(1e-7, rel=1e-9)
    assert abs(Fd.min() - 1e-7) > 1e-4                 # F.min() 陷阱的文档性断言


# --------------------------------------------------------------- P2
@pytest.mark.parametrize("eps,tr_ratio,lmin,logdef",
                         [(1e-1, 0.991, 1e-1, -2.30), (1e-7, 0.990, 1e-7, -16.12)])
def test_p2_trace_dilution_known_answers(eps, tr_ratio, lmin, logdef):
    d = 100
    F = np.diag(np.concatenate([np.ones(d - 1), [eps]]))
    assert lambda_min(F) == pytest.approx(eps, rel=1e-12)
    assert trace_ratio(F) == pytest.approx(tr_ratio, abs=5e-4)
    assert logdet_deficit(F, log_eps=1e-12) == pytest.approx(logdef, abs=5e-3)


# --------------------------------------------------------------- P3
def test_p3_parallel_sum_rank_deficient_known_answer():
    S = np.diag([1.0, 0.0])
    Lam = np.diag([2.0, 3.0])
    out = parallel_sum(S, Lam)
    assert np.allclose(out, np.diag([2.0 / 3.0, 0.0]), atol=1e-14)


def test_p3_parallel_sum_random_rank_deficient_300():
    rng = np.random.default_rng(20260908)
    for _ in range(300):
        m = rng.integers(2, 8)
        A = rng.normal(size=(m, m))
        # PSD 构造（并联和仅对 S,Λ ≽ 0 有定义；行置零+对称化会产生负特征值——非法输入）
        k = m if rng.random() >= 0.5 else int(rng.integers(1, m))   # 一半案例秩亏
        U, sv, Vt = np.linalg.svd(A)
        S = U[:, :k] @ np.diag(sv[:k]) @ U[:, :k].T
        Lam = np.diag(rng.uniform(0.1, 3.0, size=m))
        Lam[rng.integers(0, m), rng.integers(0, m)] = 0.0
        Lam = (Lam + Lam.T) / 2
        # 独立参考路线：G = S+Λ 正定时用无正则 lstsq（与实现的 eigh 截断路径不同源）；
        # G 奇异（S 与 Λ 零空间重合）时跳过——此时参考路线自身无定义
        G = S + Lam
        if np.linalg.eigvalsh((G + G.T) / 2).min() <= 1e-10 * max(np.linalg.eigvalsh((G + G.T) / 2).max(), 1e-300):
            continue
        ref = S @ np.linalg.lstsq(G, Lam, rcond=None)[0]
        out = parallel_sum(S, Lam)
        scale = max(np.linalg.norm(ref), 1e-300)
        assert np.linalg.norm(out - ref) <= 1e-12 * scale + 1e-13   # rel 1e-12×scale + abs 浮点地板


def test_p3_banned_formula_gives_wrong_answer():
    """文档性断言：任务书禁止的 [S⁺+Λ⁻¹]⁻¹ 一般式在 S 秩亏时给出错误第二分量 3（正确 0）。"""
    S = np.diag([1.0, 0.0])
    Lam = np.diag([2.0, 3.0])
    banned = np.linalg.pinv(np.linalg.pinv(S) + np.linalg.pinv(Lam))
    assert abs(banned[1, 1] - 3.0) < 1e-12              # 错误值
    correct = parallel_sum(S, Lam)
    assert abs(correct[1, 1]) < 1e-14                   # 正确值 0


# --------------------------------------------------------------- P4
def test_p4_gauge_identity_random_rank_deficient_200():
    rng = np.random.default_rng(20260908)
    from calibinfo.information.schur import delta_f
    from calibinfo.metrics.ci04r_predictors import gauge_parallel_form
    checked_rel = 0
    for _ in range(200):
        m, q = int(rng.integers(30, 80)), 9
        B = rng.normal(size=(m, 4)) @ rng.normal(size=(4, q))     # 秩亏 B
        A = rng.normal(size=(m, q))
        cbar = rng.normal(size=q)
        # 构造 gauge 恒等式：A a = B c̄（a 取 A 的一个方向并投影修正）
        a = rng.normal(size=q)
        A = A - np.outer(A @ a - B @ cbar, a) / (a @ a)
        assert np.linalg.norm(A @ a - B @ cbar) < 1e-10 * max(np.linalg.norm(B @ cbar), 1e-12)
        lam = float(rng.uniform(0.05, 5.0))
        Lam = lam * np.eye(q)
        DF, _, _ = delta_f(A, B, Lam)
        lhs = float(a @ DF @ a)
        rhs = gauge_parallel_form(B, cbar, Lam)          # c̄ᵀ[(BᵀB):Λ]c̄
        scale = max(abs(rhs), 1e-12)
        if abs(rhs) > 1e-12 * scale:                     # 真值非近零 → 相对误差
            assert abs(lhs - rhs) < 1e-10 * scale
            checked_rel += 1
        else:                                            # 真值近零 → 绝对误差
            assert abs(lhs - rhs) < 1e-12 * scale
    assert checked_rel > 100                             # 大多数随机案例非退化


# --------------------------------------------------------------- P5
def test_p5_identifiable_overlap_rank_deficient():
    rng = np.random.default_rng(20260908)
    n = 12
    # 秩亏 F∞：只有 7 个正方向
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    w = np.concatenate([rng.uniform(0.5, 2.0, size=7), np.zeros(5)])
    F_inf = (Q * w) @ Q.T
    # R 也在子空间内构造
    R = (Q * np.concatenate([rng.uniform(0, 1, size=7), np.zeros(5)])) @ Q.T
    R = (R + R.T) / 2
    ajs = rng.normal(size=(5, n))
    qj = identifiable_overlap(R, F_inf, ajs)
    assert qj.shape == (5,)
    assert np.all(np.isfinite(qj))
    assert np.all(qj >= -1e-10) and np.all(qj <= 1 + 1e-10)
