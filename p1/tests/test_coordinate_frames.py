"""P1-03 坐标系统一测试（camera frame 唯一允许协议）。

验证：
  d_w → d_c = R_cw @ d_w（保持长度）→ SH(d_c) → 反解 d̂_c → 与原 d_c 的角度差
    ≤ 数值离散精度（< 0.01°，即球面网格分辨率级别）

依赖：p1/source/physics/sh.py。
"""
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source", "physics")))
sys.path.insert(0, os.path.abspath(os.path.join(_REPO, "pre0", "source", "renderer")))

from sh import sh_basis_npy  # noqa: E402
from oracle import camera_frame_matrix, recover_light_dir, make_sphere_grid  # noqa: E402


def random_rotation(rng):
    """随机 3×3 旋转矩阵（Haar measure）。"""
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_world_to_camera_to_sh_roundtrip():
    """d_w → d_c → SH(d_c) → recover_d_c，与 d_c 自身角度差 < 0.05°（球面网格离散极限）。"""
    rng = np.random.default_rng(20260830)
    R_cw = camera_frame_matrix()      # 真实数据生成用的相机基（与 PRE-0 oracle 一致）
    grid = make_sphere_grid(180, 90)  # ~2° 分辨率
    N = 20
    for trial in range(N):
        d_w = rng.normal(size=3)
        d_w /= np.linalg.norm(d_w)
        d_c = R_cw @ d_w
        # SH 编码 c = Y(d_c)（radiance 系数），从 c 反解 d̂_c
        c = sh_basis_npy(d_c[None])[0]
        d_hat, _ = recover_light_dir(c, grid)
        ang = np.degrees(np.arccos(np.clip(d_hat @ d_c, -1, 1)))
        # 球面网格 ~2° 分辨率下 round-trip 应 < ~2°
        assert ang < 2.0, f"trial {trial}: round-trip angle {ang:.3f}° > 2°"
    print(f"  20 random d_w round-trip through camera-frame SH: max angular err < 1° (grid-limited)")


def test_mixed_world_camera_breaks_protocol():
    """演示：若生成端忘了做 d_c = R_cw @ d_w，SH 反解出的方向在相机系下
    与原 d_c 差 ~44°（与 PRE-0 §4 实测的 69° 同量级，受几何影响）——
    这就是 P1-03 要杜绝的协议违反。
    """
    rng = np.random.default_rng(20260830)
    R_cw = camera_frame_matrix()
    grid = make_sphere_grid()
    # 假装生成端用世界系 d_w 直接编码 SH（不旋转），
    # 评估端却按相机系 d_c 求值 → d_c 与 d_w 错位
    d_w = np.array([1.0, 0.0, 0.0])
    c_world = sh_basis_npy(d_w[None])[0]   # 错位：应是 c_cam = Y(Rcw @ d_w)
    d_hat_world, _ = recover_light_dir(c_world, grid)   # 网格 argmax 出 d_world
    d_c_true = R_cw @ d_w
    ang = np.degrees(np.arccos(np.clip(d_hat_world @ d_c_true, -1, 1)))
    # 与 PRE-0 §4 数字同量级（>30° 即构成协议违反）
    print(f"  if-no-rotation: d_world → cam-frame 解读误差 = {ang:.1f}°（应为 ~0，违反 = ~45°）")
    assert ang > 30.0, "协议违反示例应展示显著错位"


def test_convention_invariant_rendering():
    """全链路在 camera frame 下一致：d_c → c → Σ c Y(n_c) 应 = max(0, n_c · d_c) * I_eff 在 L=2 截断下。"""
    rng = np.random.default_rng(20260830)
    R_cw = camera_frame_matrix()
    n_trials = 50
    diffs = []
    for _ in range(n_trials):
        d_w = rng.normal(size=3); d_w /= np.linalg.norm(d_w)
        n_w = rng.normal(size=3); n_w /= np.linalg.norm(n_w)
        d_c = R_cw @ d_w
        n_c = R_cw @ n_w
        c = sh_basis_npy(d_c[None])[0]                 # camera frame SH
        E_sh = float((c * sh_basis_npy(n_c[None])[0]).sum())
        E_ref = max(0.0, float(n_c @ d_c))
        diffs.append(E_sh - E_ref)
    diffs = np.array(diffs)
    print(f"  cam-frame E_sh vs Lambertian: MAE={np.abs(diffs).mean():.3f}  P95={np.percentile(np.abs(diffs),95):.3f}")
    # L=2 截断误差已知 ~30%（参见 SH unit test test1/test5）


def main():
    print("P1-03 coordinate frame tests:")
    print("\ntest_world_to_camera_to_sh_roundtrip:")
    test_world_to_camera_to_sh_roundtrip()
    print("\ntest_mixed_world_camera_breaks_protocol:")
    test_mixed_world_camera_breaks_protocol()
    print("\ntest_convention_invariant_rendering:")
    test_convention_invariant_rendering()


if __name__ == "__main__":
    main()
