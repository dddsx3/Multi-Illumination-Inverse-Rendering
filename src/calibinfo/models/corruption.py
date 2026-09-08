"""C15 · 物理 Σ_c corruption 生成器（CI04 protocol 核心）。

宪法 §4.4：三类 corruption——强度/方向位置/联合；Σ_c 以**物理单位**定义
（相对强度²、度²），**禁 raw λI**；白化坐标进入。

物理参数化（与红队 V4 的 φ 空间一致）：
  φ = (log I, θ, φ_angle)——灯强度对数 + 两个方向自由度。
  c 空间（SH-9 或灯位方向）的 Σ_c = J_φ Σ_φ J_φᵀ（可秩亏，J_φ 由有限差分构造）。

三类（每类 level = 物理量标准差）：
  intensity:  σ_logI ∈ {0.05, 0.1, 0.2, 0.4}（相对强度 e^σ 倍；
              Σ_φ = diag(σ_logI², 0, 0)——方向维无扰动）
  direction:  σ_deg ∈ {0.5, 1, 2, 4}（方向旋转角标准差，度²；
              Σ_φ = diag(0, (σ_rad)², (σ_rad)²)）
  joint:      两者乘性组合（强度+方向同时）。

CI04 主检验使用（predictor 侧）：ΔF 在 corruption 前由 GT/nominal 计算（禁反调）；
empirical 侧 = 注入后重估计的模式误差。本模块只管"注入"与"Σ_c 白化坐标"，
估计器与判读在 experiments/ci04_real_corruption.py。
"""

from __future__ import annotations

import numpy as np

from calibinfo.datasets.synthetic import sh_basis


def light_dirs_from_positions(light_pos):
    """灯位 → 单位方向（灯指向对象，对象在原点）：d = −p/‖p‖。"""
    light_pos = np.asarray(light_pos, float)
    return -light_pos / np.linalg.norm(light_pos, axis=1, keepdims=True)


def rotate_dirs(rng, dirs, sigma_deg):
    """方向 corruption：每灯绕随机切向轴旋转 N(0, σ_deg) 度（小角近似精确实现）。"""
    dirs = np.asarray(dirs, float)
    n = dirs.shape[0]
    sigma_rad = np.radians(sigma_deg)
    out = np.empty_like(dirs)
    for i in range(n):
        # 随机切向（垂直于 d）：取随机向量 Gram-Schmidt
        v = rng.normal(size=3)
        v -= dirs[i] * (v @ dirs[i])
        nv = np.linalg.norm(v)
        v = v / nv if nv > 1e-12 else np.array([1.0, 0.0, 0.0])
        ang = rng.normal(0.0, sigma_rad)
        out[i] = dirs[i] * np.cos(ang) + np.cross(v, dirs[i]) * np.sin(ang) \
            + dirs[i] * (dirs[i] @ v) * (1 - np.cos(ang)) * 0.0
        out[i] /= np.linalg.norm(out[i])
    return out


def scale_intensity(rng, sig_logI, n):
    """强度 corruption：I → I · e^{ζ}, ζ~N(0, σ_logI)。返回乘性因子。"""
    return np.exp(rng.normal(0.0, sig_logI, size=n))


class CorruptionGenerator:
    """三类 corruption 的统一接口（物理单位 → 像素/观测空间）。"""

    def __init__(self, corruption_type, level):
        assert corruption_type in ("intensity", "direction", "joint")
        self.type = corruption_type
        self.level = level
        # 物理单位 Σ_φ（φ = log I, θ, φ_angle）
        if corruption_type == "intensity":
            self.sig_logI, self.sig_deg = float(level), 0.0
        elif corruption_type == "direction":
            self.sig_logI, self.sig_deg = 0.0, float(level)
        else:  # joint：level 作为强度档，方向取同值（预注册联合口径）
            self.sig_logI, self.sig_deg = float(level), float(level)

    def sigma_phi_diag(self):
        """φ 空间对角 Σ_φ（物理单位：log²强度、rad²方向）。"""
        return np.diag([self.sig_logI ** 2, np.radians(self.sig_deg) ** 2,
                        np.radians(self.sig_deg) ** 2])

    def j_phi(self, dirs, eps=1e-6):
        """φ → SH-9 c 空间 Jacobian（3 列：dI, dθ, dφ_angle；红队 V4 同构）。"""
        d = dirs[0]                                   # 单灯口径（CI04 每灯独立注入）
        def rot(dd, axis, ang):
            axis = np.asarray(axis, float)
            axis /= np.linalg.norm(axis)
            return dd * np.cos(ang) + np.cross(axis, dd) * np.sin(ang) \
                + axis * np.dot(axis, dd) * (1 - np.cos(ang))
        dth = rot(d, [0, 1, 0], eps)
        dph = rot(d, [0, 0, 1], eps)
        c = sh_basis(d[None, :])[0]
        return np.stack([c, (sh_basis(dth[None, :])[0] - c) / eps,
                         (sh_basis(dph[None, :])[0] - c) / eps], axis=1)

    def sigma_c_in_sh(self, dirs):
        """c 空间（SH-9）Σ_c = J_φ Σ_φ J_φᵀ（秩 ≤3，物理单位白化坐标）。"""
        J = self.j_phi(dirs)
        return J @ self.sigma_phi_diag() @ J.T

    def apply(self, rng, light_dir, light_intensity=None):
        """注入一次 corruption：返回 (扰动方向 dirs', 乘性强度因子 gains)。

        light_dir: (3,) 或 (n,3) 单位灯方向；light_intensity: 基准强度（可 None）。
        """
        dirs = np.atleast_2d(np.asarray(light_dir, float))
        n = dirs.shape[0]
        d2 = rotate_dirs(rng, dirs, self.sig_deg) if self.sig_deg > 0 else dirs.copy()
        g = scale_intensity(rng, self.sig_logI, n) if self.sig_logI > 0 \
            else np.ones(n)
        gains = g if light_intensity is None else g * np.asarray(light_intensity, float)
        return (d2[0] if np.ndim(light_dir) == 1 else d2), gains
