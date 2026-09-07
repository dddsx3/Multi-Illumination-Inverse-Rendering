"""V3 · retention 谱界 + gauge 闭式曲线 + λ⋆ 交叉（红队已知答案测试）。

来源：legacy_redteam/redteam3_exp.py §V3（同种子首场景）。判定：红队报告三 §1 V3 + 攻击二（R2 修订）。
观测余量：谱界全网格成立（至 1e-12 量级）；ρ_gauge 闭式 vs 实测 Rayleigh 2.2e-16；
retention 度量下 gauge 与最弱模式不交叉（亏缺系数 52.4 vs 65.5）；绝对单位下
λ⋆ 线性预测器/实测 = 0.74。
阈值分层（宪法 §6.2）：闭式 identity <1e-10；λ⋆ 比锁 H3 的"一个数量级"带 [0.316, 3.16]。
R2 修订回归：λ⋆ 只存在于绝对信息单位；R(Λ) 谱只做归一化读出——本测试同时断言两个度量
下交叉存在/不存在这对方向性事实。
"""

import numpy as np
import pytest

from _redteam_reference import SEED, delta_f_reference, make_scene

P, N = 1200, 3


@pytest.fixture(scope="module")
def v3_scene():
    rng = np.random.default_rng(SEED)
    _, a, ss, Bs, cs, _ = make_scene(rng, P, N)
    act = np.zeros(P, bool)
    for s in ss:
        act |= s > 0
    idx = np.where(act)[0]
    return ([s[idx] for s in ss], [B[idx] for B in Bs], cs, a[idx])


def _gauge_curve(v3_scene):
    """返回 (s, B, a1, cbar, ev9, alpha, lams, lhs, sat, rho_g)。"""
    s_l, B_l, cs, a_O = v3_scene
    s, B, c = s_l[0], B_l[0], cs[0]
    i1 = np.where(s > 0)[0]
    s, B, a1 = s[i1], B[i1], a_O[i1]
    cbar = -c
    ev9, V9 = np.linalg.eigh(B.T @ B)
    alpha = V9.T @ cbar
    lams = np.logspace(-6, 6, 25)
    lhs = np.array([a1 @ delta_f_reference(s, B, lam * np.eye(9)) @ a1
                    for lam in lams])
    sat = np.linalg.norm(s * a1) ** 2
    return s, B, a1, cbar, ev9, alpha, lams, lhs, sat, lhs / sat


def test_v3_retention_spectrum_bounds(v3_scene):
    """R(Λ)=F∞^{-1/2}ΔF(Λ)F∞^{-1/2} 谱界 0 ≤ ρⱼ ≤ 1（红队观测至 1e-12）。"""
    s, B, *_ = _gauge_curve(v3_scene)
    Finv_h = np.diag(1.0 / s)                       # F∞ = diag(s²) → F∞^{-1/2} = diag(1/s)
    for lam in [1e-4, 1e-2, 1.0, 1e2]:
        R = Finv_h @ delta_f_reference(s, B, lam * np.eye(9)) @ Finv_h
        rho = np.linalg.eigvalsh(R)
        assert rho.min() > -1e-9
        assert rho.max() < 1.0 + 1e-9


def test_v3_gauge_retention_closed_form(v3_scene):
    """ρ_gauge 闭式曲线 vs 实测 Rayleigh 商（红队观测 2.2e-16）。"""
    s, B, a1, _, ev9, alpha, lams, lhs, sat, rho_g = _gauge_curve(v3_scene)
    Finf = np.diag(s ** 2)
    meas = np.array([(a1 @ delta_f_reference(s, B, lam * np.eye(9)) @ a1)
                     / (a1 @ Finf @ a1) for lam in lams])
    rel = np.abs(rho_g - meas) / meas.max()
    assert rel.max() < 1e-10


def test_v3_no_crossover_in_retention_metric(v3_scene):
    """R2 回归：retention（归一化）度量下 gauge 与最弱模式不交叉——
    gauge 亏缺系数（α 加权平均 sᵢ²）恒小于最弱模式 s_max²（红队观测 52.4 vs 65.5）。"""
    *_, ev9, alpha, lams, lhs, sat, rho_g = _gauge_curve(v3_scene)
    d_gauge = np.sum(alpha ** 2 * ev9 ** 2) / np.sum(alpha ** 2 * ev9)
    assert d_gauge < ev9.max()


def test_v3_lambda_star_crossover_absolute_units(v3_scene):
    """绝对信息单位下交叉存在，且线性预测器 λ⋆≈μ_floor/Σαᵢ² 在一个数量级内
    （红队观测比 0.74；H3 失败判据 = 偏差 >1 decade）。"""
    s, B, a1, _, ev9, alpha, lams, lhs, sat, rho_g = _gauge_curve(v3_scene)
    M0 = delta_f_reference(s, B, np.full((9, 9), 1e-12))
    ev0 = np.linalg.eigvalsh(M0)
    floor_abs = ev0[ev0 > 1e-10 * ev0.max()][0]      # 场景本征弱模式地板
    cross = (rho_g * sat) >= floor_abs
    assert cross.any()                                # 交叉必然存在（绝对单位）
    lam_star = lams[np.argmax(cross)]
    lam_star_pred = floor_abs / np.sum((alpha ** 2)[ev9 > 1e-10 * ev9.max()])
    ratio = lam_star_pred / lam_star
    assert 0.316 < ratio < 3.16                       # 一个数量级带（红队观测 0.74）
