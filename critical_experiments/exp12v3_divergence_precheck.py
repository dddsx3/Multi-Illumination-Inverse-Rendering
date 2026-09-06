#!/usr/bin/env python3
"""卡 R 步骤 1 · exp12v3 残差预检: "抓现行"发散的直接观测

按 v3.2 卡 R:
  - 按原种子重跑 exp12v2 的 8 配置 × 5 实现(确定性复现 res 32 × 3 迭代);
  - 每个"90° 解"算 残差比 = ‖I_obs − Î(ẑ,ρ̂,Ĉ)‖ / 真值处残差;
  - 预期 ≫ 10 → 发散从"合理推断"升格为"直接观测事实";
  - 补存估计解(.npz);
  - 1 步无阻尼 GN 抓现行: 检查单步更新后阴影边界附近像素的 z 更新量相对场景尺度的倍数。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))

from exp2_joint_fisher_schur import (  # noqa: E402
    DATA, load_scene_compat, sobel_sparse, sh2, sh2_d, build_J_z_sparse, jacobian_blocks,
)
from exp12v2_discrimination_matched import (  # noqa: E402
    scene_data, compute_normals, joint_gn, normal_angle_mae,
)

OUT = HERE / "exp12v3_divergence_precheck.json"
NPZ = HERE / "exp12v3_estimates.npz"
RES = 32
N_TRIALS = 5
N_CONFIGS = 8
NOISE = 0.01
SEED = 20260906


def render_full(z, rho, C, H, W, Sx, Sy):
    n = compute_normals(z, H, W, Sx, Sy)
    Y = sh2(n)
    S = np.maximum(Y @ C.T, 0.0)
    return rho[:, None] * S


def main():
    rng = np.random.default_rng(SEED)
    scene = "sphere"   # 抓现行只需一个场景(卡R 步骤1: 中度病态 config)
    z_true, rho_true, valid, vi, C_true, H, W, Sx, Sy = scene_data(scene, RES)
    pool = load_scene_compat(str(DATA / scene))["sh"][:32].astype(float)
    n_true = compute_normals(z_true, H, W, Sx, Sy)[valid]
    rho_v = rho_true[valid]
    Y_v = sh2(n_true)
    I_true = rho_v[:, None] * np.maximum(Y_v @ C_true.T, 0)
    sigma = NOISE * np.abs(I_true).max()

    rows = []
    sol_store = {}
    for ci in range(N_CONFIGS):
        sel = np.sort(rng.choice(32, 5, replace=False))
        C_cfg = pool[sel]
        S_cfg = np.maximum(Y_v @ C_cfg.T, 0)
        for t in range(N_TRIALS):
            I_n = I_true + rng.normal(0, sigma, I_true.shape)
            z0 = z_true.copy()
            rho0 = rho_true.copy()
            rho0[valid] *= (1 + rng.normal(0, 0.01, rho0[valid].shape))
            C0 = C_cfg + rng.normal(0, 0.01, C_cfg.shape)
            z_e, rho_e, C_e = joint_gn(z0, rho0, C0, I_n, H, W, Sx, Sy, valid, vi)
            mae = normal_angle_mae(z_e, z_true, valid, H, W, Sx, Sy)
            # 残差: 全图渲染但在 valid 像素上比
            I_est = render_full(z_e, rho_e, C_e, H, W, Sx, Sy)[valid]
            r_est = np.linalg.norm(I_n - I_est)
            r_true = np.linalg.norm(I_n - I_true)   # ≈ 纯噪声
            ratio = r_est / max(r_true, 1e-300)
            rows.append(dict(config=ci, trial=t, mae=mae,
                             residual_est=float(r_est), residual_true=float(r_true),
                             residual_ratio=float(ratio)))
            sol_store[f"cfg{ci}_t{t}"] = dict(z=z_e, rho=rho_e, C=C_e)
            print(f"  cfg{ci:02d} t{t}: MAE={mae:6.2f}°  残差比={ratio:8.2f}  "
                  f"(噪声底 {r_true:.3f})")

    ratios = np.array([r["residual_ratio"] for r in rows])
    # 1 步 GN 抓现行(中度病态 config)
    ci_mid = 3
    sel = np.sort(rng.choice(32, 5, replace=False))
    C_cfg = pool[sel]
    I_n = I_true + rng.normal(0, sigma, I_true.shape)
    z0, rho0, C0 = z_true.copy(), rho_true.copy(), C_cfg + rng.normal(0, 0.01, C_cfg.shape)
    # 单步: 走 joint_gn n_iters=1
    z_1, rho_1, C_1 = joint_gn(z0, rho0, C0, I_n, H, W, Sx, Sy, valid, vi, n_iters=1)
    dz_1 = np.abs(z_1 - z_true)[valid]
    # 阴影边界像素(n·C_k ≈ 0)
    Z = sh2(compute_normals(z_true, H, W, Sx, Sy)) @ C_cfg.T
    boundary = valid & (np.abs(Z).min(axis=1) < 0.05 * np.abs(Z).max())
    scale = np.abs(z_true[valid]).mean()
    print(f"\n抓现行: 单步 GN 后 z 更新 中位|Δz|={np.median(dz_1):.2e} "
          f"(场景尺度 {scale:.1f}, 比值 {np.median(dz_1)/scale:.2f})")
    if boundary.any():
        dz_b = np.abs(z_1 - z_true)[boundary]
        print(f"  阴影边界({boundary.sum()}px): |Δz| 中位 {np.median(dz_b):.2e} "
              f"(×场景尺度 {np.median(dz_b)/scale:.2f})")

    out = dict(
        scene=scene, res=RES, rows=rows,
        summary=dict(n=len(rows), residual_ratio_median=float(np.median(ratios)),
                     residual_ratio_max=float(ratios.max()),
                     residual_ratio_min=float(ratios.min()),
                     n_ratio_gt_10=int((ratios > 10).sum()),
                     mae_range=[float(min(r['mae'] for r in rows)),
                                float(max(r['mae'] for r in rows))]),
        one_step_gn=dict(median_dz_over_scale=float(np.median(dz_1) / scale),
                         boundary_median_dz=float(np.median(np.abs(z_1 - z_true)[boundary]))
                         if boundary.any() else None),
        verdict=dict(
            finding=f"残差比中位 {np.median(ratios):.0f}×, 最大 {ratios.max():.0f}×, "
                    f"{int((ratios>10).sum())}/{len(ratios)} 个解残差比>10",
            conclusion=("发散从'合理推断'升格为'直接观测事实'——无阻尼 GN 3 步迭代"
                        "产生与观测完全不相关的解(残差≫噪声底), MAE 90° 恒定确认为数值发散"
                        if np.median(ratios) > 10 else
                        "残差比不高——发散假设需复核(与 v3.2 预期不符, 停手上报)")),
    )
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(NPZ, **{f"{k}_z": v["z"] for k, v in sol_store.items()})
    np.savez(NPZ, **{f"{k}_rho": v["rho"] for k, v in sol_store.items()})
    np.savez(NPZ, **{f"{k}_C": v["C"] for k, v in sol_store.items()})
    print(f"\n[exp12v3-precheck] 残差比中位 {np.median(ratios):.0f}× | 落盘 -> {OUT}")
    print(f"  判定: {out['verdict']['conclusion'][:60]}")


if __name__ == "__main__":
    main()
