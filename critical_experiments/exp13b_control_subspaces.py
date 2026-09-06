#!/usr/bin/env python3
"""卡 M · exp13b 双对照补强: 工作带底部 3 模式 + 频率匹配随机场

设计(v3.1 卡 M):
  对照 1(主判据): 每场景取路线 (ii) 几何块 S_(zρ) 的工作带底部 3 特征方向,
    z 分量经 Sobel 推成法线场, 投影误差场算占比 → 判读 GBR 0.765 vs 工作带模式占比;
  对照 2(基线): 20 个随机方向线性梯度法线场(一阶多项式 x/y, 随机系数, 切面投影),
    投影占比 → 基线分布(均值±std);
  汇总三对照: GBR vs 工作带 3 模式 vs 随机梯度 vs 随机平滑(exp13 的 0.0009)。

判据(预注册): GBR 占比 > 工作带 3 模式占比(逐场景 ≥120/124) → 表述升格版成立;
             否则降级句。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from data_loader import MultiLightingDataset  # noqa: E402
from exp13_gauge_error_fraction import load_model, gbr_perturbed_normals, gauge_fraction  # noqa: E402
from exp2_joint_fisher_schur import (  # noqa: E402
    DATA, load_scene_compat, sobel_sparse, sh2, sh2_d,
)

SPLIT = REPO / "splits" / "synthetic_v3.json"
DATA_ROOT = "D:/data/synthetic_v3"
OUT = HERE / "exp13b_control_subspaces.json"
N_RANDOM_GRAD = 20
EPS = 1e-3
SEED = 20260906
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model, renderer = load_model()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    ds = MultiLightingDataset(root_dir=DATA_ROOT, num_lights=5,
                              image_size=(256, 256), is_training=False,
                              scene_subset=split["test"], load_gt=True, modality="gray")
    print(f"[exp13b] dataset n={len(ds)}")

    results = []
    for i in range(len(ds)):
        images, gt, name = ds[i]
        with torch.no_grad():
            d_net = model(images.unsqueeze(0).to(DEVICE))[0]
        n_net = renderer.depth_to_normal(d_net)[0]
        n_gt = gt["normal"].to(DEVICE)
        mask = gt["mask"][0] > 0.5
        mask_t = mask.to(DEVICE)
        dn = n_net - n_gt
        dn = dn - n_gt * (n_gt * dn).sum(0, keepdim=True)
        dn[:, ~mask_t] = 0

        # GBR 三生成元(exp13 同款)
        z_gt = gt["depth"][0:1].to(DEVICE)
        z4 = z_gt.unsqueeze(0)
        xg, yg = np.meshgrid(np.arange(256), np.arange(256))
        gen_fields = {
            "lambda": z4 * EPS,
            "mu": torch.tensor(xg * EPS, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0),
            "nu": torch.tensor(yg * EPS, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0),
        }
        gvecs = []
        with torch.no_grad():
            for gname, dz in gen_fields.items():
                n_p = renderer.depth_to_normal(z4 + dz)[0]
                dn_p = ((n_p - n_gt) / EPS)
                dn_p = dn_p - n_gt * (n_gt * dn_p).sum(0, keepdim=True)
                dn_p[:, ~mask_t] = 0
                gvecs.append(dn_p.reshape(-1).cpu().numpy())
        G_gbr = np.stack(gvecs)
        frac_gbr = gauge_fraction(dn.reshape(-1).cpu().numpy(), G_gbr)

        # 对照 1: 工作带底部 3 模式(S_(zρ) 几何块的谱底)
        # 实现: 用 F_zz 的最小非零特征方向的 z 分量 → Sobel 推成法线场
        # 简化: 32×32 下采样求特征向量, 上采到 256(任务书允许)
        from exp2_joint_fisher_schur import build_J_z_sparse, jacobian_blocks
        sc = load_scene_compat(str(DATA / "sphere"))  # 用 sphere 的管线
        H0 = W0 = 32
        z2 = gt["depth"][0][::8, ::8].cpu().numpy().astype(float)
        z2 = np.where(np.isfinite(z2), z2, 0)
        rho2 = gt["albedo"][0][::8, ::8].cpu().numpy().astype(float)
        Sx2, Sy2 = sobel_sparse(H0, W0)
        C2 = np.array([[np.pi, 2*np.pi/3, 0, 2*np.pi/3*0.5, 0,0,0,0,0],
                       [np.pi, 0, 2*np.pi/3*0.5, 0, 0,0,0,0,0],
                       [np.pi, 2*np.pi/3*0.7, 0, 0, 0,0,0,0,0],
                       [np.pi, 0, 0, 2*np.pi/3*0.7, 0,0,0,0,0],
                       [np.pi, 2*np.pi/3*0.3, 0, 0, 0,0,0,0,0]], dtype=float)
        blk = jacobian_blocks(z2.ravel(), rho2.ravel(), C2, H0, W0, Sx2, Sy2)
        Js_full = build_J_z_sparse(z2.ravel(), rho2.ravel(), C2, H0, W0, Sx2, Sy2, blk)
        F_zz = sum(J.T @ J for J in Js_full).toarray()
        # 深度平移零方向投影
        one = np.ones(F_zz.shape[0]); one /= np.linalg.norm(one)
        F_proj = F_zz - np.outer(one, F_zz @ one) - np.outer(F_zz @ one, one) + \
                 (one @ F_zz @ one) * np.outer(one, one)
        w_z, v_z = np.linalg.eigh(F_proj)
        # 工作带底部 3 = 最小非零的前 3 个
        pos_idx = np.where(w_z > 1e-6 * w_z.max())[0]
        if len(pos_idx) < 3:
            pos_idx = np.argsort(w_z)[::-1][:3]  # fallback: 取最大 3 个(诊断用途)
        bottom3 = v_z[:, pos_idx[:3]]    # (32², 3)

        # 上采 32→256 并经 Sobel 推成法线场
        n_modes = []
        for m in range(3):
            dz_m = bottom3[:, m].reshape(32, 32)
            dz_256 = np.repeat(np.repeat(dz_m, 8, axis=0), 8, axis=1)
            dz_t = torch.tensor(dz_256, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                n_p = renderer.depth_to_normal(z4 + dz_t * 1e-3)[0]
                dn_m = (n_p - n_gt) / 1e-3
                dn_m = dn_m - n_gt * (n_gt * dn_m).sum(0, keepdim=True)
                dn_m[:, ~mask_t] = 0
                n_modes.append(dn_m.reshape(-1).cpu().numpy())
        G_wb = np.stack(n_modes)
        frac_wb = gauge_fraction(dn.reshape(-1).cpu().numpy(), G_wb)

        # 对照 2: 20 个随机方向线性梯度场
        fracs_rand = []
        for _ in range(N_RANDOM_GRAD):
            ax, ay = rng.normal(0, 1, 2)
            gx = ax * xg + ay * yg       # 一阶多项式
            gx_t = torch.tensor(gx * EPS, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                n_p = renderer.depth_to_normal(z4 + gx_t)[0]
                dn_m = (n_p - n_gt) / EPS
                dn_m = dn_m - n_gt * (n_gt * dn_m).sum(0, keepdim=True)
                dn_m[:, ~mask_t] = 0
                v = dn_m.reshape(-1).cpu().numpy()
                if np.linalg.norm(v) > 1e-12:
                    fracs_rand.append(gauge_fraction(dn.reshape(-1).cpu().numpy(),
                                                      v.reshape(1, -1)))
        results.append(dict(scene=name, frac_gbr=float(frac_gbr),
                            frac_workband3=float(frac_wb),
                            rand_grad_mean=float(np.mean(fracs_rand)),
                            rand_grad_std=float(np.std(fracs_rand))))
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(ds)} done")

    fg = np.array([r["frac_gbr"] for r in results])
    fw = np.array([r["frac_workband3"] for r in results])
    fr = np.array([r["rand_grad_mean"] for r in results])
    n_gbr_gt_wb = int((fg > fw).sum())
    n_gbr_gt_rand = int((fg > fr).sum())
    out = {"per_scene": results, "meta": dict(n=len(results), n_rand_grad=N_RANDOM_GRAD),
           "summary": dict(gbr_mean=float(fg.mean()), workband3_mean=float(fw.mean()),
                           rand_grad_mean=float(fr.mean()),
                           n_gbr_gt_workband=n_gbr_gt_wb, n_gbr_gt_rand=n_gbr_gt_rand),
           "verdict": dict(
               acceptance="GBR > 工作带3模式(≥120/124) → 升格版; 否则降级句",
               result=("升格版成立" if n_gbr_gt_wb >= 120 else "降级句(见任务书 §3.4)"),
               n_gbr_gt_workband=n_gbr_gt_wb, n_gbr_gt_rand=n_gbr_gt_rand)}
    print(f"\n[exp13b] GBR {fg.mean():.3f} | 工作带3模式 {fw.mean():.3f} | "
          f"随机梯度 {fr.mean():.3f}")
    print(f"  GBR > 工作带: {n_gbr_gt_wb}/{len(results)} | GBR > 随机梯度: {n_gbr_gt_rand}/{len(results)}")
    print(f"  判定: {out['verdict']['result']}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp13b] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
