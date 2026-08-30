"""P1-02/R1 SH 解析单元测试（修正版：Lambertian convolution Â=[π,2π/3,π/4]）。

历史注记：P1-R0 停线修正——旧版 K_L=[√π,√(π/3),√(π/5)] 为逐带畸变，
且旧 Test5 错误地用 reproducing kernel Σ Y(d)Y(n)（无卷积）对照
Lambertian，得出"L=2 有 75% 误差"的错误结论。本版按
Ramamoorthi & Hanrahan 2001 Eq.7-9 的正确乘子重写全部对照。

解析预期值（单方向光，I_eff=1）：
  E_L2(n,d) = Σ_l Â_l Σ_m Y_lm(d)Y_lm(n)   （加法定理）
  n=d   : 0.25 + 0.5 + 0.3125 = 1.0625      （vs ref 1.0，+6.25%）
  n⊥d   : 0.25 − 0.15625      = 0.09375     （vs ref 0，ringing +0.094）
  n=−d  : 0.25 − 0.5 + 0.3125 = 0.0625      （vs ref 0，ringing +0.0625）
  Monte Carlo（均匀 n,d）: MAE≈0.031, RMSE≈0.036, max≈0.094
"""
import math
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source", "physics")))

from sh import C0, C1, C2, A_L, sh_basis_npy, sh_directional_irradiance  # noqa: E402


def E_L2_analytic(d: np.ndarray, n: np.ndarray, I_eff: float = 1.0):
    """E_L2(n,d) = Σ_l Â_l Σ_m Y_lm(d)Y_lm(n)（irradiance，L=2 截断）。

    标量向量输入返回 float；批量输入返回 [N] 数组。
    """
    scalar = np.asarray(d).ndim == 1
    Yd = sh_basis_npy(np.atleast_2d(d))
    Yn = sh_basis_npy(np.atleast_2d(n))
    A = np.array([A_L[0], A_L[1], A_L[1], A_L[1], A_L[2], A_L[2], A_L[2], A_L[2], A_L[2]])
    out = I_eff * ((Yd * Yn * A).sum(-1))
    return float(out[0]) if scalar else out


def test1_frontal_light():
    """Test 1: n=d → E_L2 = 1.0625·I（vs Lambertian ref 1.0，+6.25%）。"""
    d = np.array([0., 0., 1.]); n = d.copy()
    E = float(E_L2_analytic(d, n))
    assert abs(E - 1.0625) < 1e-3, f"frontal E={E}（应=1.0625）"
    assert abs(E - 1.0) / 1.0 < 0.08, f"frontal 截断误差 {abs(E-1):.3f} > 8%"
    # 生成端路径（sh_directional_irradiance + 求值）必须与解析式一致
    c = sh_directional_irradiance(d, 1.0)
    E_via_c = float((c * sh_basis_npy(n[None])[0]).sum())
    assert abs(E_via_c - E) < 1e-6, "生成端系数与解析式不一致"
    print(f"  Test1: E_L2={E:.4f} ref=1.0  err={abs(E-1):.4f} (6.25% 预期)")


def test2_orthogonal_normal():
    """Test 2: n⊥d → ref=0；E_L2 = 0.09375（ringing 上界 ~9%，与文献一致）。"""
    d = np.array([0., 0., 1.]); n = np.array([1., 0., 0.])
    E = float(E_L2_analytic(d, n))
    assert abs(E - 0.09375) < 1e-3, f"orthogonal E={E}（应=0.09375）"
    print(f"  Test2: E_L2={E:.4f} ref=0  ringing=+{E:.4f}")


def test3_back_facing():
    """Test 3: n=−d → ref=0；E_L2 = 0.0625（小正 ringing；ReLU 后仍 0.0625）。"""
    d = np.array([0., 0., 1.]); n = np.array([0., 0., -1.])
    E = float(E_L2_analytic(d, n))
    assert abs(E - 0.0625) < 1e-3, f"back-facing E={E}（应=0.0625）"
    relu = max(0.0, E)
    assert 0.05 < relu < 0.08
    print(f"  Test3: E_L2={E:.4f} ref=0  ReLU 残余={relu:.4f}")


def test4_rotation_equivariance():
    """Test 4: 旋转等变性——E_L2 只依赖 n·d，旋转后不变（加法定理推论）。"""
    rng = np.random.default_rng(20260830)
    for trial in range(5):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        E0 = float(E_L2_analytic(d, n))
        A = rng.normal(size=(3, 3)); Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0: Q[:, 0] *= -1
        E_rot = float(E_L2_analytic(Q @ d, Q @ n))
        # 基常数为 6 位截断（承袭 physics_renderer），等变性精度受此限制
        assert abs(E0 - E_rot) < 1e-4, f"trial {trial}: {E0} vs {E_rot}"
    print("  Test4: E_L2(Rd, Rn) == E_L2(d, n) 对 5 个随机旋转成立（<1e-4，基常数 6 位截断极限）")


def test5_monte_carlo_reference():
    """Test 5: 20k 随机 (n,d) 对，E_L2 vs max(0, n·d)。预期 MAE≈0.031。"""
    rng = np.random.default_rng(20260830)
    N = 20000
    d = rng.normal(size=(N, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = rng.normal(size=(N, 3)); n /= np.linalg.norm(n, axis=1, keepdims=True)
    E_sh = E_L2_analytic(d, n)
    E_ref = np.maximum((n * d).sum(-1), 0.0)
    diff = E_sh - E_ref
    mae = float(np.abs(diff).mean()); rmse = float(np.sqrt((diff ** 2).mean()))
    p95 = float(np.percentile(np.abs(diff), 95)); mx = float(np.abs(diff).max())
    print(f"  Test5: MAE={mae:.4f}  RMSE={rmse:.4f}  P95={p95:.4f}  max={mx:.4f}")
    print(f"        （专家独立核算预期 MAE≈0.031 / RMSE≈0.036 / max≈0.094）")
    assert mae < 0.05, f"MAE={mae} 明显偏离修正后的预期（~0.031）"
    assert mx < 0.12, f"max={mx} 超过 ringing 上界"
    return dict(mae=mae, rmse=rmse, p95=p95, max_err=mx)


def main():
    print("P1-02/R1 SH physics unit tests（Â=[π, 2π/3, π/4]）:")
    for fn in (test1_frontal_light, test2_orthogonal_normal, test3_back_facing,
               test4_rotation_equivariance, test5_monte_carlo_reference):
        print(f"\n{fn.__name__}:")
        fn()


if __name__ == "__main__":
    main()
