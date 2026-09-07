"""V2 · Prop 2 gauge-lifting 精确谱闭式（红队已知答案测试）。

来源：legacy_redteam/redteam3_exp.py §V2（场景 = 种子 20260907 的首个 make_scene(1200,3)，
与红队原始 run 同场景同数字）。判定：红队报告三 §1 V2。
观测余量：25 个数量级 λ 网格 max rel err 3.9e-8；线性抬升系数与 Σαᵢ² 逐位一致（0.7162）；
饱和比 0.999948。
阈值分层（宪法 §6.2）：闭式 vs 直接计算锁 1e-6（= 红队观测 ×25，且容许本机 BLAS 差异；
CI02 实验本身的 <1e-8 gate 适用于 CI02 场景构造精度，与本单元测试基线分层记录）。
"""

import numpy as np
import pytest

from _redteam_reference import SEED, delta_f_reference, make_scene

P, N = 1200, 3


@pytest.fixture(scope="module")
def v2_scene():
    rng = np.random.default_rng(SEED)
    _, a, ss, Bs, cs, _ = make_scene(rng, P, N)
    act = np.zeros(P, bool)
    for s in ss:
        act |= s > 0
    idx = np.where(act)[0]
    s_l = [s[idx] for s in ss]
    B_l = [B[idx] for B in Bs]
    a_O = a[idx]
    return s_l, B_l, cs, a_O


def test_v2_closed_form_over_25_decades(v2_scene):
    """aᵀΔF(λ)a == Σαᵢ²sᵢ²λ/(sᵢ²+λ)（闭式 via BᵀB 特征分解 vs 直接 Rayleigh）。"""
    s_l, B_l, cs, a_O = v2_scene
    s, B, c = s_l[0], B_l[0], cs[0]
    i1 = np.where(s > 0)[0]
    s, B, a1 = s[i1], B[i1], a_O[i1]
    cbar = -c                                       # gauge nuisance 方向
    ev9, V9 = np.linalg.eigh(B.T @ B)
    alpha = V9.T @ cbar

    lams = np.logspace(-6, 6, 25)
    lhs = np.array([a1 @ delta_f_reference(s, B, lam * np.eye(9)) @ a1
                    for lam in lams])
    rhs = np.array([np.sum(alpha ** 2 * ev9 * lam / (ev9 + lam)) for lam in lams])
    rel = np.abs(lhs - rhs) / np.abs(rhs)
    assert rel.max() < 1e-6                          # 红队观测 3.9e-8


def test_v2_saturation_endpoint(v2_scene):
    """λ→∞ 饱和到 ‖Aa‖² = ‖Bc̄‖²（红队观测 0.999948）。"""
    s_l, B_l, cs, a_O = v2_scene
    s, B, c = s_l[0], B_l[0], cs[0]
    i1 = np.where(s > 0)[0]
    s, B, a1 = s[i1], B[i1], a_O[i1]
    sat = np.linalg.norm(s * a1) ** 2
    lhs_end = a1 @ delta_f_reference(s, B, 1e6 * np.eye(9)) @ a1
    ratio = lhs_end / sat
    assert 0.999 < ratio <= 1.0 + 1e-9


def test_v2_linear_lifting_coefficient(v2_scene):
    """λ→0 线性抬升系数 == Σαᵢ²（sᵢ>0）（红队观测逐位一致 0.7162）。"""
    s_l, B_l, cs, a_O = v2_scene
    s, B, c = s_l[0], B_l[0], cs[0]
    i1 = np.where(s > 0)[0]
    s, B, a1 = s[i1], B[i1], a_O[i1]
    cbar = -c
    ev9, V9 = np.linalg.eigh(B.T @ B)
    alpha = V9.T @ cbar
    lam = 1e-6
    lhs = a1 @ delta_f_reference(s, B, lam * np.eye(9)) @ a1
    coef = lhs / lam
    sum_alpha2 = np.sum((alpha ** 2)[ev9 > 1e-10 * ev9.max()])
    assert abs(coef - sum_alpha2) / sum_alpha2 < 1e-4   # O(λ/s²) 修正量级 1e-6，余量充足
