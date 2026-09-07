"""V1 · Prop 1 有限样本协方差恒等式（红队已知答案测试）。

来源：legacy_redteam/redteam3_exp.py §V1/V1b/V1-fix（移植，场景参数与 rng 消耗顺序逐位一致，
种子 20260907 → 与红队原始 run 同场景）。判定：红队报告三 §1 V1/V1-fix。
阈值分层（宪法 §6.2）：V1b identity 类 <1e-10（观测 2.05e-16，余量 5+ 个数量级）；
MC 类锁 median/IQR 区间 + Frobenius 预算（观测 median 1.0045 / IQR [0.984, 1.025] /
Frobenius 6.1% @ cond(ΔF)=1.8e8）；fixed-δc 诊断锁闭式复现比（观测 bias 1.04 / 条件方差 1.0015）。
预注册带取红队观测值 ± MC 误差预算量级，不处于阈值敏感区。
"""

import numpy as np
import pytest

from _redteam_reference import SEED, delta_f_reference, make_scene

P, N, SIGMA, SIG_C = 1200, 3, 0.02, 0.05
TRIALS = 2000


@pytest.fixture(scope="module")
def v1_world():
    rng = np.random.default_rng(SEED)
    _, a, ss, Bs, cs, _ = make_scene(rng, P, N)

    act = np.zeros(P, bool)
    for s in ss:
        act |= s > 0
    idx = np.where(act)[0]
    O = len(idx)
    s_l = [s[idx] for s in ss]
    B_l = [B[idx] for B in Bs]
    a_O = a[idx]

    A_st = np.vstack([np.diag(s) for s in s_l])          # (N·O, O)
    B_blk = np.zeros((N * O, N * 9))
    for k in range(N):
        B_blk[k * O:(k + 1) * O, k * 9:(k + 1) * 9] = B_l[k]
    Lam = (SIGMA / SIG_C) ** 2 * np.eye(9)
    Lam_blk = np.kron(np.eye(N), Lam)

    G = B_blk.T @ B_blk + Lam_blk
    Ginv_Bt = np.linalg.lstsq(G, B_blk.T, rcond=None)[0]
    M = np.eye(N * O) - B_blk @ Ginv_Bt
    DF = A_st.T @ M @ A_st
    DFinv = np.linalg.inv(DF)

    # 两个系综按红队脚本的 rng 顺序一次跑完（marginal 先、fixed 用其 dc_all[0]），
    # 使每个断言与 legacy 原始 run 的样本逐位一致。
    dc_all = rng.normal(0, SIG_C, size=(TRIALS, N, 9))
    E = np.empty((TRIALS, O))
    for t in range(TRIALS):
        dcst = dc_all[t].reshape(-1)
        y = A_st @ a_O + B_blk @ dcst + rng.normal(0, SIGMA, size=N * O)
        My = y - B_blk @ (Ginv_Bt @ y)
        E[t] = DFinv @ (A_st.T @ My) - a_O
    dc_fix = dc_all[0].reshape(-1)
    bias = DFinv @ A_st.T @ M @ (B_blk @ dc_fix)
    Ef = np.empty((TRIALS, O))
    for t in range(TRIALS):
        y = A_st @ a_O + B_blk @ dc_fix + rng.normal(0, SIGMA, size=N * O)
        My = y - B_blk @ (Ginv_Bt @ y)
        Ef[t] = DFinv @ (A_st.T @ My) - a_O

    return dict(a_O=a_O, s_l=s_l, B_l=B_l, O=O, A_st=A_st, B_blk=B_blk,
                Lam=Lam, Ginv_Bt=Ginv_Bt, M=M, DF=DF, DFinv=DFinv,
                E=E, Ef=Ef, bias=bias)


def test_v1b_joint_block_equals_sum_of_per_image(v1_world):
    """联合块对角 nuisance ΔF == Σ_k ΔF_k（红队 V1b，观测 2.05e-16）。"""
    w = v1_world
    DF_sum = np.zeros((w["O"], w["O"]))
    for s, B in zip(w["s_l"], w["B_l"]):
        DF_sum += delta_f_reference(s, B, w["Lam"])
    rel = np.linalg.norm(w["DF"] - DF_sum) / np.linalg.norm(w["DF"])
    assert rel < 1e-10


def test_v1_marginal_covariance_identity(v1_world):
    """marginal Cov(x̂) 对角 = σ²ΔF⁻¹ 对角（红队观测 median 1.0045, IQR [0.984,1.025]，
    Frobenius 6.1% @ cond 1.8e8）。"""
    w = v1_world
    Cov_mc = np.cov(w["E"].T)
    crb_diag = SIGMA ** 2 * np.diag(w["DFinv"])
    ratio = np.diag(Cov_mc) / crb_diag
    fro = np.linalg.norm(Cov_mc - SIGMA ** 2 * w["DFinv"]) \
        / np.linalg.norm(SIGMA ** 2 * w["DFinv"])
    assert 0.97 < np.median(ratio) < 1.04
    assert 0.93 < np.percentile(ratio, 25) and np.percentile(ratio, 75) < 1.08
    assert fro < 0.12          # 2× 红队观测 0.061（cond 驱动的 MC 预算）


def test_v1_fixed_dc_bias_and_conditional_variance(v1_world):
    """fixed-δc：条件偏差被 bias=ΔF⁻¹AᵀMBδc 解释（观测 1.04）；条件方差
    = σ²ΔF⁻¹AᵀM²AΔF⁻¹（观测 1.0015）且在弱模式端系统性低于 marginal（观测约 0.8；
    M 是收缩映射，非幂等投影）。"""
    w = v1_world
    A_st, B_blk, DFinv, M = w["A_st"], w["B_blk"], w["DFinv"], w["M"]
    Ef, E_marg, bias = w["Ef"], w["E"], w["bias"]

    # (1) 条件均值 = 闭式 bias（红队观测 1.04）
    assert 0.9 < np.linalg.norm(Ef.mean(0)) / np.linalg.norm(bias) < 1.2
    # (2) marginal 均值 ≈ 0（红队观测残差 5.2%）——"必须 joint sampling"的一阶答案
    assert np.linalg.norm(E_marg.mean(0)) / np.linalg.norm(bias) < 0.2
    # (3) 条件方差 = σ²ΔF⁻¹AᵀM²AΔF⁻¹（红队观测 median 1.0015）
    cond_pred = SIGMA ** 2 * np.diag(DFinv @ (A_st.T @ (M @ M) @ A_st) @ DFinv)
    ratio_f = np.diag(np.cov(Ef.T)) / cond_pred
    assert 0.9 < np.median(ratio_f) < 1.1
    # (4) 方向性（红队口径=弱模式端，约 0.8）：最弱 5% 模式处条件方差 < marginal 方差
    crb_diag = SIGMA ** 2 * np.diag(DFinv)
    k = max(1, int(0.05 * w["O"]))
    weak = np.argsort(crb_diag)[-k:]            # 预测方差最大 = 信息最弱的模式
    ratio_marg = np.diag(np.cov(E_marg.T)) / crb_diag
    assert np.median(ratio_f[weak]) < np.median(ratio_marg[weak])
