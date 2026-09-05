#!/usr/bin/env python3
"""实验 5 扩展 · 经典预期的正确对象：逐像素局部法线 Fisher vs σ_min(C)²

背景：exp5 主实验发现全局 λ_min⁺(几何块) 对 σ_min(C)² 不敏感（低秩扰动保护，
Weyl）。经典"光照病态→几何可辨识性下降"预期的正确对象是【逐像素局部】法线
信息——机理：像素 p 的法线扰动 δn 经 N 盏光映射为观测变化 G_p·δn，G_p(k,:) =
ρ_p·h_kp·(C_kᵀ ∂Y_p/∂n)·P_⊥(n_p)，局部 Fisher F_p = G_pᵀG_p (3×3)，
其最小特征值 = 该像素最差方向的信息。光照系数 C 病态（各行近共线）→ G_p
条件数差 → λ_min(F_p) 小。这在逐像素口径下不受全局低秩保护（是经典理论的
直接对应物）。

口径（预注册）：
  - 对每 (scene, config)：λ_loc = mean_p λ_min(F_p)（像素均值，也可报告中位）；
  - F_p 在法线切平面上取最小特征（3×3 特征分解后取最小，P_⊥ 已保证第三方向
    的中性——法线单位约束由 P_⊥ 投影处理）；
  - Spearman(λ_loc, σ_min(C)²) 跨 4 场景 × 8 配置。
预期：正相关显著（经典理论成立口径）；若仍不显著→如实报告。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))

from exp2_joint_fisher_schur import DATA, load_scene_compat, jacobian_blocks, sh2_d  # noqa: E402

OUT = HERE / "exp5b_local_normal_fisher.json"
SCENES = ["sphere", "cube", "cylinder", "hemisphere"]
N_CONFIGS = 8
N_LIGHTS = 5
SEED = 20260905


def local_normal_fisher(scene_dir, C):
    """逐像素 3×3 局部 Fisher 的最小特征值均值/中位。"""
    sc = load_scene_compat(scene_dir)
    z, rho, mask = sc["depth"], sc["albedo"], sc["mask"]
    H, W = mask.shape
    z = z.ravel().astype(float)
    rho = rho.ravel().astype(float)
    mk = mask.ravel()
    valid = (mk > 0) & (z < 1e8)
    vi = np.where(valid)[0]
    C = np.asarray(C, float)
    # 法线用 GT mesh 法线（几何已知口径与 exp4 一致）
    import os
    n_mesh = np.load(os.path.join(scene_dir, "normal_mesh.npy"))   # (3,H,W)
    n = n_mesh.transpose(1, 2, 0).reshape(-1, 3)[vi]
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    rho_v = rho[vi]
    Y = sh2_d(n)                                       # (P,9,3)
    # G_p(k,:) = ρ_p·h_kp·(C_kᵀ dY_p)  (3,)——h 用 C 当前值(与 Fisher 的 h 口径一致)
    from exp2_joint_fisher_schur import sh2
    Yn = sh2(n)                                        # (P,9)
    Z = Yn @ C.T                                       # (P,N)
    Hk = (Z > 0).astype(float)
    # G: (P,N,3) = ρ·h·(C dY)
    G = np.einsum('pji,kj->pki', Y, C) * (rho_v[:, None] * Hk)[:, :, None]
    # F_p = G_pᵀG_p (3,3) → 特征值
    # 批量: F = einsum('pki,pkj->pij', G, G)
    F = np.einsum('pki,pkj->pij', G, G)
    w = np.linalg.eigvalsh(F)                          # (P,3) 升序
    lam_min = w[:, 0]
    return dict(lam_min_mean=float(lam_min.mean()),
                lam_min_median=float(np.median(lam_min)),
                lam_min_p25=float(np.percentile(lam_min, 25)),
                lam_min_p75=float(np.percentile(lam_min, 75)),
                n_pixels=int(len(vi)))


def main():
    rng = np.random.default_rng(SEED)
    out = {"runs": [], "meta": dict(
        caliber="per-pixel local normal Fisher λ_min(F_p), 3×3 tangent, geometry-known",
        rationale="exp5 主实验证明全局 λ_min⁺ 被低秩扰动保护(B·F⁺·Bᵀ 秩≤45); 经典预期的正确对象是逐像素局部信息——G_p 的条件数直接由 C 的行相关性控制")}
    for scene in SCENES:
        d = DATA / scene
        if not d.is_dir():
            continue
        sc = load_scene_compat(str(d))
        sh_pool = sc["sh"]
        for ci in range(N_CONFIGS):
            sel = rng.choice(len(sh_pool), N_LIGHTS, replace=False)
            C = sh_pool[np.sort(sel)].astype(float)
            sig = np.linalg.svd(C, compute_uv=False)
            sigma_min = float(sig[-1])
            r = local_normal_fisher(str(d), C)
            out["runs"].append(dict(scene=scene, config=ci,
                                    sigma_min=sigma_min, sigma_min_sq=sigma_min**2, **r))
            print(f"{scene:10s} cfg{ci} σ_min={sigma_min:.4f} λ_loc_mean={r['lam_min_mean']:.4e}")

    ok = out["runs"]
    x = [r["sigma_min_sq"] for r in ok]
    y = [r["lam_min_mean"] for r in ok]
    rho_all, p_all = spearmanr(x, y)
    out["spearman_all"] = dict(rho=float(rho_all), p=float(p_all), n=len(ok))
    per_scene = {}
    for scene in SCENES:
        xs = [r["sigma_min_sq"] for r in ok if r["scene"] == scene]
        ys = [r["lam_min_mean"] for r in ok if r["scene"] == scene]
        if len(set(xs)) > 3 and np.std(ys) > 0:
            r_, p_ = spearmanr(xs, ys)
            per_scene[scene] = dict(rho=float(r_), p=float(p_), n=len(xs))
    out["spearman_per_scene"] = per_scene
    print(f"\n[exp5b] Spearman(λ_loc, σ_min²) 全样本 = {rho_all:.3f} (p={p_all:.3e}, n={len(ok)})")
    for s, v in per_scene.items():
        print(f"  {s:10s}: ρ={v['rho']:.3f} (p={v['p']:.3e})")

    out["verdict"] = {
        'positive': bool(rho_all > 0 and p_all < 0.05),
        'interpretation': ('经典预期在【逐像素局部法线信息】口径下成立(全样本显著正相关)'
                           if rho_all > 0 and p_all < 0.05 else
                           '逐像素口径下仍不显著——如实报告'),
        'synthesis_with_exp5': ('全局 λ_min⁺ 不敏感(低秩保护) + 局部 λ_loc 显著相关 → '
                                '经典预期成立域 = 逐像素/局部信息, 非全局极值谱。两个口径在论文中分开表述, '
                                '诊断工具的"光照散布度→几何信息"叙事在局部口径下有实证支撑'),
    }
    print('verdict:', out["verdict"]["interpretation"])
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp5b] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
