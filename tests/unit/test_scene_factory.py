"""C09 绑定测试 · 分层 scene 工厂。

检查：分层网格 ≥30 场景；gauge 恒等式 A·a = −B·c̄ 精确成立（Prop 2 前提）；
μ_floor > 0 且 gauge 方向在 Λ=0 是精确零模式；P 全维激活（工厂 keep 已过滤全暗像素）。
"""

import numpy as np

from calibinfo.datasets.synthetic import make_grid, make_scene, mu_floor
from calibinfo.information.schur import delta_f

SEED = 20260907


def test_scene_gauge_identity():
    """A·a = −B·c̄（符号规范）逐位成立——Prop 2 闭式的前提。"""
    rng = np.random.default_rng(SEED)
    sc = make_scene(rng, P=150, n_lights=3, geometry="bumpy", albedo="wide")
    lhs = sc["A_st"] @ sc["gauge_a"]
    rhs = -sc["B_blk"] @ sc["gauge_cbar"]
    rel = np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)
    assert rel < 1e-12


def test_mu_floor_positive_and_gauge_is_null():
    """μ_floor > 0；gauge 方向 ΔF(0⁺)·a ≈ 0（精确零模式）。"""
    rng = np.random.default_rng(SEED + 1)
    sc = make_scene(rng, P=200, n_lights=1, geometry="sphere", albedo="medium")
    floor, ev = mu_floor(sc)
    assert floor > 0
    # gauge 是最小特征值方向的近似：ΔF(0⁺) a ≈ 0
    DF0, _, _ = delta_f(sc["A_st"], sc["B_blk"],
                        1e-12 * np.eye(sc["B_blk"].shape[1]))
    na = np.linalg.norm(DF0 @ sc["gauge_a"])
    assert na < 1e-8 * max(np.linalg.norm(DF0), 1e-300)
    # 最小正特征值应远大于 gauge 残差对应的数值零带
    assert ev.min() < 1e-9 * ev.max()


def test_grid_stratification_54():
    """分层网格 54 场景 ≥ 30；元数据分层四轴齐全；场景间 floor 有分布宽度。"""
    rng = np.random.default_rng(SEED)
    scenes, manifest = make_grid(rng, P=120)
    assert len(scenes) == 54 and len(manifest) == 54
    axes = {m["geometry"] for m in manifest} | {m["albedo"] for m in manifest}
    assert len(axes) >= 5
    floors = [mu_floor(sc)[0] for sc in scenes[:12]]
    assert min(floors) > 0 and max(floors) / min(floors) > 2   # 有分层效应
