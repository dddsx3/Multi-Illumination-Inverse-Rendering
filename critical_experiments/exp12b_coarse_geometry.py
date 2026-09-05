#!/usr/bin/env python3
"""exp12b(卡 D②)· 粗几何鲁棒性: 深度代理三档下的诊断排序一致性

设计(卡 D 步骤 ②): 深度代理三档——高斯平滑(σ=2px)/加噪(5%)/降分辨率(32² 双线性
上采)→ 由代理深度经 Sobel 重算法线 → 重算诊断 tr(S_θ⁻¹) → 与真值几何的诊断排序
做场景内 Spearman。判据: 一致性 ≥0.8 → "训练前可用"; <0.8 → 降格"后验诊断"。
(两分支都可写, 先有数。)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))

from exp2_joint_fisher_schur import DATA, load_scene_compat, sh2, sh2_d, sobel_sparse  # noqa: E402
from exp11v3_kappa_expansion import trace_theta_inv_safe  # noqa: E402

OUT = HERE / "exp12b_coarse_geometry.json"
SCENES = ["sphere", "cube", "cylinder", "hemisphere"]
RES = 64
N_LIGHTS = 5
N_CONFIGS = 20
SEED = 20260906


def normals_from_depth(z, H, W):
    Sx, Sy = sobel_sparse(H, W)
    v = np.stack([-(Sx @ z), -(Sy @ z), np.ones_like(z)], axis=1)
    nv = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
    return v / nv[:, None]


def depth_and_normals(scene, res=RES):
    sc = load_scene_compat(str(DATA / scene))
    z, mask = sc["depth"], sc["mask"]
    H0, W0 = mask.shape
    H = W = res
    i0, j0 = (H0 - H) // 2, (W0 - H) // 2
    z = z[i0:i0+H, j0:j0+W].ravel().astype(float)
    mk = mask[i0:i0+H, j0:j0+W].ravel() > 0
    z = np.where(mk & (z < 1e8), z, np.nan)
    return z, mk.astype(bool).ravel(), H, W


def fill_nan(z):
    """NaN(天空)用最近有效值填充(行优先近似——诊断用, 不做精确测地)。"""
    z = z.copy()
    n = len(z)
    idx = np.where(np.isfinite(z))[0]
    if len(idx) == 0:
        return np.zeros_like(z)
    bad = ~np.isfinite(z)
    z[bad] = np.interp(np.where(bad)[0], idx, z[idx])
    return z


def main():
    rng = np.random.default_rng(SEED)
    out = {"scenes": {}, "meta": dict(proxies=["smooth_sigma2", "noise_5pct", "downres32"],
                                      n_configs=N_CONFIGS, res=RES)}
    for scene in SCENES:
        z_true, valid, H, W = depth_and_normals(scene)
        z_fill = fill_nan(z_true)
        n_true = normals_from_depth(z_fill, H, W)[valid]
        pool = load_scene_compat(str(DATA / scene))["sh"][:32].astype(float)
        rho = np.load(str(DATA / scene) + "/albedo.npy")[0]
        # ρ 对齐 valid 像素(用 mask=valid 的近似——与 exp12 相同口径)
        a_full = np.load(str(DATA / scene) + "/albedo.npy")[0].ravel()
        a_full = np.where(np.isfinite(a_full), a_full, np.nan)
        # 用 valid 的索引对齐: valid 是 (H,W) bool → 直接取
        a2 = np.load(str(DATA / scene) + "/albedo.npy")[0]
        i0 = (a2.shape[0] - H) // 2; j0 = (a2.shape[1] - W) // 2
        a_v = a2[i0:i0+H, j0:j0+W].ravel()[valid]
        a_v = np.where(np.isfinite(a_v), a_v, np.median(a_v[np.isfinite(a_v)]))
        cfgs = [np.sort(rng.choice(32, N_LIGHTS, replace=False)) for _ in range(N_CONFIGS)]
        # 真值几何诊断
        tr_true = []
        for sel in cfgs:
            C = pool[sel]
            tr_true.append(trace_theta_inv_safe(n_true, a_v, C)[1])
        # 三档代理
        proxies = {}
        z2 = z_fill.reshape(H, W)
        zs = gaussian_filter1d(z2, sigma=2, axis=0)
        zs = gaussian_filter1d(zs, sigma=2, axis=1)
        proxies["smooth_sigma2"] = zs.ravel()
        proxies["noise_5pct"] = z_fill + rng.normal(0, 0.05 * (np.nanmax(z_true) - np.nanmin(z_true)), H*W) * valid
        z32 = z_true.reshape(H, W)
        small = z32[::2, ::2]
        up = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
        proxies["downres32"] = fill_nan(up.ravel()) * valid + z_fill * (~valid)
        rows = {"true": tr_true}
        for pname, zp in proxies.items():
            n_p = normals_from_depth(fill_nan(zp), H, W)[valid]
            tr_p = []
            for sel in cfgs:
                C = pool[sel]
                try:
                    tr_p.append(trace_theta_inv_safe(n_p, a_v, C)[1])
                except Exception:
                    tr_p.append(np.nan)
            fin = np.isfinite(tr_p) & np.isfinite(tr_true)
            r_, p_ = spearmanr(np.array(tr_p)[fin], np.array(tr_true)[fin])
            proxies_out = dict(spearman_vs_true=float(r_), p=float(p_), n=int(fin.sum()))
            rows[pname] = proxies_out
            print(f"  {scene:10s} {pname:12s}: 排序一致性 ρ={r_:.3f} (p={p_:.3e})")
        ok = all(abs(rows[p]["spearman_vs_true"]) >= 0.8 for p in
                 ("smooth_sigma2", "noise_5pct", "downres32"))
        out["scenes"][scene] = dict(results=rows, all_ge_08=bool(ok))
        print(f"{scene:10s} 三档全 ≥0.8: {ok}")
    # 汇总
    all_ok = all(v["all_ge_08"] for v in out["scenes"].values())
    out["verdict"] = dict(
        acceptance="三档代理排序一致性 ≥0.8 → 训练前可用; <0.8 → 后验诊断",
        result=("排序一致性达标(训练前可用)" if all_ok else "未达标(降格为给定几何估计的后验诊断)"),
        per_scene={s: v["all_ge_08"] for s, v in out["scenes"].items()})
    print("\n[exp12b] 判定:", out["verdict"]["result"])
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp12b] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
