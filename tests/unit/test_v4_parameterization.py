"""V4 · 参数化不变性 + raw 坐标陷阱（红队已知答案测试）。

来源：legacy_redteam/redteam3_exp.py §V4（独立 P=600 场景，红队 rng 顺序：V1 场景(1200,3)
→ V4 场景 make_scene(600,1)，种子 20260907 连续消耗 → 本测试按相同消耗顺序重建同场景）。
判定：红队报告三 §1 V4 + 攻击五（实现层事故样本）。
观测余量：φ 空间 profiled vs c 空间 marginal 6.4e-15；同名义 λ=0.16 raw SH vs 白化差 10.0%、
vs φ 坐标差 3.3%。
阈值分层：不变性 identity <1e-10；陷阱为方向性断言（>5%），非精确锁值。
"""

import numpy as np
import pytest

from _redteam_reference import SEED, delta_f_reference, j_phi, make_scene

P_V4 = 600
_SIGMA, _SIG_C = 0.02, 0.05
_TRIALS = 2000


def _replay_v1_rng_consumption(rng):
    """复刻红队脚本 V1→V4 之间的 rng 流消耗（场景才能逐位一致）：
    make_scene(1200,3) → dc_all (2000,3,9) → marginal 2000 次噪声 (3O,)
    → fixed 2000 次噪声 (3O,)。仅消耗流，不做计算。"""
    _, a, ss, Bs, cs, _ = make_scene(rng, 1200, 3)
    act = np.zeros(1200, bool)
    for s in ss:
        act |= s > 0
    O = int(act.sum())
    rng.normal(0, _SIG_C, size=(_TRIALS, 3, 9))
    for _ in range(2 * _TRIALS):
        rng.normal(0, _SIGMA, size=3 * O)


@pytest.fixture(scope="module")
def v4_scene():
    # 红队脚本 rng 顺序：V1 场景(1200,3)+V1 两个 2000 次系综先消耗，V4 场景后消耗
    rng = np.random.default_rng(SEED)
    _replay_v1_rng_consumption(rng)
    _, a, ss, Bs, cs, dirs = make_scene(rng, P_V4, 1)
    return ss[0], Bs[0], cs[0], dirs[0], a


def test_v4_parameterization_invariance(v4_scene):
    """φ 空间 profiled ΔF == c 空间 marginal ΔF（红队观测 6.4e-15）。"""
    s4, B4, c4, d4, _ = v4_scene
    i4 = np.where(s4 > 0)[0]
    s4, B4 = s4[i4], B4[i4]
    Jp = j_phi(c4, d4)
    Sig_phi = np.diag(np.array([0.05, 0.10, 0.10]) ** 2)
    Sig_c = Jp @ Sig_phi @ Jp.T
    s2 = 0.02 ** 2
    dF_phi = delta_f_reference(s4, B4 @ Jp, s2 * np.linalg.inv(Sig_phi))
    Marg = s2 * np.eye(len(s4)) + B4 @ Sig_c @ B4.T
    dF_c = s2 * (np.diag(s4) @ np.linalg.solve(Marg, np.diag(s4)))
    rel = np.linalg.norm(dF_phi - dF_c) / np.linalg.norm(dF_c)
    assert rel < 1e-10


def test_v4_raw_coordinate_pitfall(v4_scene):
    """方向性陷阱：同名义 λ=0.16，raw SH 坐标 λI 与白化正确形式差显著
    （红队观测 10.0%；raw 与 φ 坐标差 3.3%）→ 修订 #8 必需而非形式主义。"""
    s4, B4, c4, d4, _ = v4_scene
    i4 = np.where(s4 > 0)[0]
    s4, B4 = s4[i4], B4[i4]
    Jp = j_phi(c4, d4)
    Sig_phi = np.diag(np.array([0.05, 0.10, 0.10]) ** 2)
    Sig_c = Jp @ Sig_phi @ Jp.T
    lam_nom = 0.16
    dF_raw = delta_f_reference(s4, B4, lam_nom * np.eye(9))
    dF_phiI = delta_f_reference(s4, B4 @ Jp, lam_nom * np.eye(3))
    dF_whit = delta_f_reference(s4, B4, lam_nom * np.linalg.pinv(Sig_c))
    d1 = np.linalg.norm(dF_raw - dF_whit) / np.linalg.norm(dF_whit)
    d2 = np.linalg.norm(dF_raw - dF_phiI) / np.linalg.norm(dF_phiI)
    assert d1 > 0.05                                 # 红队观测 0.100
    assert d2 > 0.01                                 # 红队观测 0.033
