"""C09 · 分层 scene 工厂（CI02/CI03 的合成场景供给）。

宪法 §4.2：≥30 合成场景分层采样：几何条件数 × albedo spread × 灯数 × B 奇异谱；
每场景保存真实 gauge (a, c̄) 与 μ_floor 所需基线。

场景模型（红队 V 系列/CI01 photometric 同构）：
  法线 n (P,3)（几何分层：球面/起伏/重终止子），SH-9 基 Y = SH(n)，
  反照率 a 对数均匀（spread 分层），N 灯方向 d_k → c_k = SH9(d_k)，
  s_full_k = Y c_k，H_k = [s_full_k>0]，s_k = relu；
  A = D(s_k) 堆叠（N·P, P），B = 块对角 (a∘H_k)Y_k（N·P, N·9）；
  gauge 对 (a, c̄)，c̄_k = −c_k（A a = −B c̄ 的符号规范：aᵀΔF a 对 c̄ 符号不变）。

manifest：分层参数 + 种子全部入 results/raw（run_manifest 落盘），列表见 make_grid。
"""

from __future__ import annotations

import numpy as np

from calibinfo.information.schur import delta_f

SH_COEF = (0.282095, 0.488603, 1.092548, 0.315392, 0.546274)


def sh_basis(n):
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    A0, A1, A2, A3, A4 = SH_COEF
    return np.stack([A0 * np.ones_like(x), A1 * y, A1 * z, A1 * x,
                     A2 * x * y, A2 * y * z, A3 * (3 * z**2 - 1), A2 * x * z,
                     A4 * (x**2 - y**2)], axis=1)


def rand_dirs(rng, n, zmin=0.15):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 2] = np.abs(d[:, 2]) + zmin
    return d / np.linalg.norm(d, axis=1, keepdims=True)


GEOMETRIES = {          # 几何分层：法线分布 → 终止子/阴影结构差异
    "sphere": dict(zmin=0.45, bump=0.00),
    "bumpy": dict(zmin=0.25, bump=0.35),
    "terminator_heavy": dict(zmin=0.05, bump=0.10),
}
ALBEDO_SPREADS = {      # albedo 分层：对数均匀支撑宽度
    "narrow": (0.6, 1.0),
    "medium": (0.3, 1.0),
    "wide": (0.1, 1.0),
}


def make_scene(rng, P, n_lights, geometry="sphere", albedo="medium",
               light_elev_deg=None):
    """单场景构造。返回 dict（全部数组 + 分层元数据 + gauge 对）。"""
    g = GEOMETRIES[geometry]
    n = rand_dirs(rng, P, zmin=g["zmin"])
    if g["bump"] > 0:                                   # 起伏法线（几何条件数↑）
        n = n + g["bump"] * rng.normal(size=n.shape)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        n[:, 2] = np.abs(n[:, 2]) + 0.02
        n /= np.linalg.norm(n, axis=1, keepdims=True)
    Y = sh_basis(n)
    lo, hi = ALBEDO_SPREADS[albedo]
    a = np.exp(rng.uniform(np.log(lo), np.log(hi), size=P))

    if light_elev_deg is None:
        light_elev_deg = float(np.degrees(np.arcsin(rng.uniform(0.2, 0.9))))
    az = rng.uniform(0, 2 * np.pi)
    el = np.radians(light_elev_deg)
    dirs = [np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])]
    for k in range(n_lights - 1):                       # 多灯：方位角分层铺开
        az_k = az + 2 * np.pi * (k + 1) / max(n_lights, 1)
        el_k = np.radians(light_elev_deg * rng.uniform(0.6, 1.0))
        dirs.append(np.array([np.cos(el_k) * np.cos(az_k),
                              np.cos(el_k) * np.sin(az_k), np.sin(el_k)]))
    dirs = [d / np.linalg.norm(d) for d in dirs]

    cs, ss, Hs = [], [], []
    for d in dirs:
        c = sh_basis(d[None, :])[0]
        s_full = Y @ c
        H = (s_full > 0).astype(float)
        cs.append(c)
        ss.append(np.maximum(s_full, 0.0))
        Hs.append(H)
    act = np.logical_or.reduce(Hs)                       # 至少一灯照亮的像素
    keep = np.where(act)[0]

    N = n_lights
    A_st = np.vstack([np.diag(s[keep]) for s in ss])    # (N·P', P')
    B_blk = np.zeros((N * len(keep), N * 9))
    for k in range(N):
        B_blk[k * len(keep):(k + 1) * len(keep), k * 9:(k + 1) * 9] = \
            (a[keep] * Hs[k][keep])[:, None] * Y[keep]
    return dict(
        P=int(len(keep)), n_gt=n[keep], Y=Y[keep], a=a[keep],
        cs=np.array(cs), ss=[s[keep] for s in ss], Hs=[H[keep] for H in Hs],
        dirs=np.array(dirs), A_st=A_st, B_blk=B_blk,
        gauge_a=a[keep], gauge_cbar=np.array([-c for c in cs]).reshape(-1),
        meta=dict(geometry=geometry, albedo=albedo, n_lights=N,
                  light_elev_deg=light_elev_deg, P_full=int(act.sum())),
    )


def make_grid(rng, P=300, geometries=("sphere", "bumpy", "terminator_heavy"),
              albedos=("narrow", "medium", "wide"), light_elevs=(25.0, 45.0, 65.0),
              n_lights_choices=(1, 3), seed_offset=0):
    """分层枚举场景网格（3×3×3×2 = 54 ≥ 30），保留 stratification manifest。"""
    scenes, manifest = [], []
    i = 0
    for geo in geometries:
        for alb in albedos:
            for elev in light_elevs:
                for n_l in n_lights_choices:
                    s_rng = np.random.default_rng(rng.integers(0, 2**63 - 1) + seed_offset + i)
                    sc = make_scene(s_rng, P, n_l, geometry=geo, albedo=alb,
                                    light_elev_deg=elev)
                    sc["scene_id"] = f"s{i:02d}_{geo}_{alb}_e{int(elev)}_n{n_l}"
                    scenes.append(sc)
                    manifest.append(dict(scene_id=sc["scene_id"], **sc["meta"]))
                    i += 1
    return scenes, manifest


def mu_floor(scene, tol_rel=1e-10, lam_eps=1e-12):
    """μ_floor 操作性定义（宪法 λ⋆ Diagnostic；红队 V3 同口径）：

    Λ→0⁺（lam_eps·I）下联合 ΔF 的最小**正**特征值（> tol_rel·λ_max）——
    即 uncalibrated 基线的场景本征弱模式地板；gauge 方向与结构零模式
    （全暗像素已由工厂 keep 过滤）不入地板。
    """
    DF0, _, _ = delta_f(scene["A_st"], scene["B_blk"],
                        lam_eps * np.eye(scene["B_blk"].shape[1]))
    ev = np.linalg.eigvalsh(DF0)
    pos = ev[ev > tol_rel * max(ev.max(), 1e-300)]
    return float(pos[0]), ev
