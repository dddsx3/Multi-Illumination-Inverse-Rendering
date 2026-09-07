"""V5 · Λ = σ²Σ_c⁻¹ 尺度一致性（红队已知答案测试）。

来源：legacy_redteam/redteam3_exp.py §V5（同种子首场景首灯）。判定：红队报告三 §1 V5。
观测余量：(σ,Σ_c)→(10σ,100Σ_c) retention 谱逐位不变（max diff 0.0）；
σ 单独 ×100（λ×100）谱移动 max diff 0.806。
阈值分层：不变性 identity <1e-10；移动为方向性断言（>0.1）。
"""

import numpy as np
import pytest

from _redteam_reference import SEED, delta_f_reference, make_scene

P, N, SIGMA, SIG_C = 1200, 3, 0.02, 0.05


@pytest.fixture(scope="module")
def v5_scene():
    rng = np.random.default_rng(SEED)
    _, a, ss, Bs, _, _ = make_scene(rng, P, N)
    i5 = np.where(ss[0] > 0)[0]
    return ss[0][i5], Bs[0][i5]


def test_v5_scale_invariance(v5_scene):
    """(σ,Σ_c)→(t·σ, t²·Σ_c) 时 retention 谱不变（红队观测 max diff 0.0）。"""
    s, B = v5_scene
    Finv_h = np.diag(1.0 / s)

    def retention(lam):
        return np.linalg.eigvalsh(
            Finv_h @ delta_f_reference(s, B, lam * np.eye(9)) @ Finv_h)

    t = 10.0
    lam1 = SIGMA ** 2 / SIG_C ** 2
    lam2 = (t ** 2 * SIGMA ** 2) / (t ** 2 * SIG_C ** 2)
    assert np.abs(retention(lam1) - retention(lam2)).max() < 1e-10


def test_v5_sigma_alone_shifts_continuum(v5_scene):
    """σ 单独 ×t²（固定 Σ_c）→ λ ×t²，连续谱位置移动（红队观测 max diff 0.806）。"""
    s, B = v5_scene
    Finv_h = np.diag(1.0 / s)

    def retention(lam):
        return np.linalg.eigvalsh(
            Finv_h @ delta_f_reference(s, B, lam * np.eye(9)) @ Finv_h)

    t = 10.0
    lam1 = SIGMA ** 2 / SIG_C ** 2
    lam3 = (t ** 2 * SIGMA ** 2) / SIG_C ** 2
    assert np.abs(retention(lam1) - retention(lam3)).max() > 0.1
