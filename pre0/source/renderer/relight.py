"""PRE-02 解析补光：为 N>5 的子集生成协议模型重打光。

仅用于 PRE-02（GT 信息量实验），不用于 probe 训练。
模型：I = A^GT ⊙ ReLU(Σ c_i Y_i(n^GT))，c = I_eff·Y(d)，
方向取 15 个固定序列方向（仰角带 [20°,70°] + 黄金角方位，seed=20260829），
强度同协议 I_eff = 100/(4π·2.99²)。

语义边界（写入协议）：补光图像由解析协议模型生成，
不含 Cycles 特有效应（近场衰减、间接光、路径追踪噪声）。
PRE-02 中真实 5 光子集来自磁盘 PNG（含上述效应），
解析补光子集不与之混用绝对值，只用于 N>5 的趋势分析；
新光方向在"世界系"定义（与 sh_coeffs 生成语义一致）。
"""
import math

import numpy as np

from oracle import camera_frame_matrix

RELIGHT_SEED = 20260829
N_RELIGHT = 15
I_EFF = 100.0 / (4 * math.pi * 2.99 ** 2)
C0, C1 = 0.282095, 0.488603
C2 = [1.092548, 1.092548, 0.315392, 1.092548, 0.546274]


def sh_basis_np(n: np.ndarray) -> np.ndarray:
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack([
        np.full_like(x, C0),
        C1 * y, C1 * z, C1 * x,
        C2[0] * x * y, C2[1] * y * z, C2[2] * (3 * z * z - 1),
        C2[3] * x * z, C2[4] * (x * x - y * y),
    ], axis=-1)


def fib_dirs(n: int = N_RELIGHT, seed: int = RELIGHT_SEED,
             el_lo: float = 20.0, el_hi: float = 70.0) -> np.ndarray:
    """n 个固定序列方向（世界系）：仰角带内等步进 + 黄金角方位 + 微扰"""
    rng = np.random.default_rng(seed)
    golden = math.pi * (3 - math.sqrt(5))
    dirs = []
    for i in range(n):
        t = (i + 0.5) / n
        el = math.radians(el_lo + (el_hi - el_lo) * t)
        az = golden * i + rng.uniform(0, 0.05)
        dirs.append([math.cos(el) * math.cos(az),
                     math.cos(el) * math.sin(az),
                     math.sin(el)])
    d = np.stack(dirs)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def analytic_relight(scene: dict) -> np.ndarray:
    """scene: scene_loader.load_scene 输出 -> [15,H,W] 线性域解析重打光图像"""
    n_cam = scene["normal"].transpose(1, 2, 0)       # [H,W,3] 法线系
    M = camera_frame_matrix()                        # d_cam = M @ d_world
    n_world = n_cam @ M.T                            # 转回世界系
    A = scene["albedo"][0]
    Y = sh_basis_np(n_world)                         # [H,W,9]
    imgs = []
    for d in fib_dirs():
        c = I_EFF * sh_basis_np(d[None])[0]
        imgs.append(A * np.maximum(Y @ c, 0.0))
    return np.stack(imgs)
