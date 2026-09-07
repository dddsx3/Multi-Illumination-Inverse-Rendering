"""V6 · 模式追踪：R(Λ) 特征基旋转（红队已知答案测试）。

来源：legacy_redteam/redteam3_exp.py §V6（同种子首场景）。判定：红队报告三 §1 V6 + 攻击三（R3 修订）。
观测：N=1 最弱保留 9 维子空间在 λ 跨 4 个数量级旋转 0.00°（col(B) 固定正交基 U 结构原因）；
N=3 同子空间旋转 88.4°（各灯 U_k 混合 + 近简并方向任意基贡献）。
阈值分层（宪法 §6.2）：只断言两个方向性事实——N=1 ≈ 0°，N>1 可大幅旋转；
角度判定对简并基伪影稳健（结论只依赖"索引排序不可靠"），协议本身即单元测试（R3）。
"""

import numpy as np
import pytest

from _redteam_reference import SEED, delta_f_reference, make_scene, principal_angle

P, N = 1200, 3


@pytest.fixture(scope="module")
def v6_scene():
    rng = np.random.default_rng(SEED)
    _, a, ss, Bs, _, _ = make_scene(rng, P, N)
    return ss, Bs


def _bottom9_angle(s_list, B_list):
    """λ ∈ {1e-2, 1e2} 的最弱保留 9 维子空间最大主角度（度）。"""
    idxN = np.where(np.logical_or.reduce([sv > 0 for sv in s_list]))[0]
    sl = [sv[idxN] for sv in s_list]
    bl = [bv[idxN] for bv in B_list]
    Finf = sum(np.diag(sv ** 2) for sv in sl)
    w9, Vinf = np.linalg.eigh(Finf)
    Fh = Vinf @ np.diag(1 / np.sqrt(w9)) @ Vinf.T
    Vs = {}
    for lam in [1e-2, 1e2]:
        Ftot = sum(delta_f_reference(sv, bv, lam * np.eye(9))
                   for sv, bv in zip(sl, bl))
        Rl = Fh @ Ftot @ Fh
        _, Vl = np.linalg.eigh(Rl)
        Vs[lam] = Vl[:, :9]                    # 最弱保留（最小特征值端）
    return principal_angle(Vs[1e-2], Vs[1e2])


def test_v6_single_image_basis_lambda_invariant(v6_scene):
    """N=1：bottom-9 子空间跨 4 个数量级 λ 旋转 ≈ 0°（红队观测 0.00°）。"""
    ss, Bs = v6_scene
    ang = _bottom9_angle(ss[:1], Bs[:1])
    assert ang < 0.5


def test_v6_multi_image_modes_rotate(v6_scene):
    """N=3：同子空间大幅旋转（红队观测 88.4°）→ 禁"第 j 小特征值"索引排序（R3）。"""
    ss, Bs = v6_scene
    ang = _bottom9_angle(ss, Bs)
    assert ang > 30.0
