"""红队参考实现（只读基线）— 从 legacy_redteam/redteam3_exp.py 原样移植的数值 helper。

用途：tests/unit 下 V1–V6 已知答案单元测试的共享基线（卡 C02，宪法 §6.2 红线 #8 素材）。
生产单源实现位于 src/calibinfo/information/（卡 C06 落地 delta_f/whiten_system 等）；
届时生产实现与本参考的互检是 C06 验收的一部分。本模块仅服务测试，不进实验主路径。

种子纪律：三轮红队统一种子 20260907，与 legacy_redteam 脚本一致；改动种子 = 判定失效。
"""

import numpy as np

SEED = 20260907


def sh_basis(n):
    """SH-9 基（与红队脚本逐位一致；系数取前 9 项常用近似）。"""
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    A0, A1, A2, A3, A4 = 0.282095, 0.488603, 1.092548, 0.315392, 0.546274
    return np.stack([A0 * np.ones_like(x), A1 * y, A1 * z, A1 * x,
                     A2 * x * y, A2 * y * z, A3 * (3 * z**2 - 1), A2 * x * z,
                     A4 * (x**2 - y**2)], axis=1)


def rand_dirs(rng, n, zmin=0.15):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 2] = np.abs(d[:, 2]) + zmin
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def make_scene(rng, P, N):
    """合成场景：a 对数均匀，N 灯方向随机；返回 Y, a, ss, Bs, cs, dirs（与红队脚本同序）。"""
    n = rand_dirs(rng, P)
    Y = sh_basis(n)
    a = np.exp(rng.uniform(np.log(0.3), np.log(1.0), size=P))
    dirs = [rand_dirs(rng, 1)[0] for _ in range(N)]
    ss, Bs, cs = [], [], []
    for d in dirs:
        c = sh_basis(d[None, :])[0]
        s_full = Y @ c
        H = (s_full > 0).astype(float)
        ss.append(np.maximum(s_full, 0.0))
        Bs.append((a * H)[:, None] * Y)   # B_k = D(a) H Y
        cs.append(c)
    return Y, a, ss, Bs, cs, dirs


def delta_f_reference(s, B, Lam):
    """ΔF(Λ) 红队 lstsq 路径（红线 RL-solve：Λ=0/秩亏禁 solve，统一 lstsq）。

    A = D(s)：ΔF = Aᵀ[I − B(BᵀB+Λ)⁻¹Bᵀ]A。
    """
    G = B.T @ B + Lam
    w = np.linalg.lstsq(G, B.T, rcond=None)[0]
    M = np.eye(len(s)) - B @ w
    return s[:, None] * M * s[None, :]


def principal_angle(V1, V2, k=9):
    """两个 k 维列空间的最大主角度（度）。"""
    _, sv, _ = np.linalg.svd(V1[:, :k].T @ V2[:, :k])
    return np.degrees(np.arccos(np.clip(sv[-1], -1, 1)))


def j_phi(c, d, eps=1e-6):
    """φ 空间 → c 空间 Jacobian：列 [c (log-intensity), dc/dθ, dc/dφ]。"""
    def rot(dd, axis, ang):
        axis = np.asarray(axis, float)
        axis /= np.linalg.norm(axis)
        return dd * np.cos(ang) + np.cross(axis, dd) * np.sin(ang) \
            + axis * np.dot(axis, dd) * (1 - np.cos(ang))

    dth = rot(d, [0, 1, 0], eps)
    dph = rot(d, [0, 0, 1], eps)
    return np.stack([c, (sh_basis(dth[None, :])[0] - sh_basis(d[None, :])[0]) / eps,
                     (sh_basis(dph[None, :])[0] - sh_basis(d[None, :])[0]) / eps], axis=1)
