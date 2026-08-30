"""P1 阶段统一 SH 工具（与 LIGHTING_MODEL.md 完全一致）。

常量与顺序（不可改）：
  basis order: [Y00, Y1-1, Y10, Y1+1, Y2-2, Y2-1, Y20, Y2+1, Y2+2]
  C0, C1, C2[0..4] 与 p1/protocol/LIGHTING_MODEL.md §2 表完全一致
  Lambertian convolution k_l = [sqrt(pi), sqrt(pi/3), sqrt(pi/5)]
"""
import math

import numpy as np

C0 = 0.282095                                # 0.5*sqrt(1/pi)
C1 = 0.488603                                # sqrt(3/(4*pi))
C2 = [1.092548, 1.092548, 0.315392, 1.092548, 0.546274]
# Lambertian clamped-cosine convolution multipliers（正交归一实 SH，P1-R1 修正；
# Ramamoorthi & Hanrahan 2001 Eq.7-9：A_hat_0=pi, A_hat_1=2pi/3, A_hat_2=pi/4）
# 旧值 K_L=[sqrt(pi),sqrt(pi/3),sqrt(pi/5)] 为逐带畸变（P1_R0_STOP_LINE 问题1），作废。
A_L = [math.pi, 2.0 * math.pi / 3.0, math.pi / 4.0]


def sh_basis_npy(n: np.ndarray) -> np.ndarray:
    """n: [...,3] 单位向量 -> [...,9] SH basis（与 p1/protocol/LIGHTING_MODEL.md §2 同序）"""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack([
        np.full_like(x, C0),
        C1 * y, C1 * z, C1 * x,
        C2[0] * x * y, C2[1] * y * z, C2[2] * (3.0 * z * z - 1.0),
        C2[3] * x * z, C2[4] * (x * x - y * y),
    ], axis=-1)




def sh_directional_irradiance(direction: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """direction [3] 单位向量, intensity I_eff（albedo 乘子域，见下）
    -> 9 维 irradiance SH coefficients（Lambertian convolution 已应用）

    c_lm = A_hat_l · I_eff · Y_lm(d)

    I_eff 语义（P1-R1）：albedo 乘子域的 irradiance —— 即
    E(n) = Σ c Y(n) 直接乘 albedo 得到像素值（Lambertian BSDF 的 1/π
    已并入 I_eff：若 Blender SUN strength=S，则 I_eff = S/π）。
    """
    Y = sh_basis_npy(direction[None])[0]                        # [9]
    k = np.array([A_L[0], A_L[1], A_L[1], A_L[1], A_L[2], A_L[2], A_L[2], A_L[2], A_L[2]])
    return intensity * k * Y


def sh_directional_radiance(direction: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """direction [3] -> 9 维 radiance SH（未卷积，仅作中间/对照）"""
    return intensity * sh_basis_npy(direction[None])[0]


def irradiance_from_sh(c: np.ndarray, n: np.ndarray) -> np.ndarray:
    """E(n) = Σ c_lm Y_lm(n)，c [9] + n[...,3] -> irradiance [...]"""
    Y = sh_basis_npy(n)
    return (Y * c).sum(axis=-1)


def lambertian_reference(direction: np.ndarray, n: np.ndarray) -> np.ndarray:
    """E_ref = max(0, n·d) * I_eff（无卷积）"""
    return np.maximum(n @ direction, 0.0)
