#!/usr/bin/env python3
"""关键实验 13(卡 E)· 网络误差的 gauge 子空间能量占比

设计(任务书 v3.0 卡 E, 预注册):
  1. 每 test 场景一次前向 → 逐像素法线误差场 δn_p(网络法线 vs GT 法线的角差向量场);
  2. 对每场景构造 GBR 三生成元在法线场上的诱导变化: z 分别 +λz、+μx、+νy(小步长 ε),
     用与网络相同的 Sobel 管线重算法线, δn^(λ,μ,ν) = (变换后法线 − 原法线)/ε;
  3. 在全局内积 ⟨A,B⟩ = Σ_p ⟨A_p,B_p⟩ 下把 δn 投影到 span{三生成元}, 算能量占比;
  4. 对照: 3 个随机平滑场(同范数同带宽)做同样投影占比作为零假设分布;
  5. 汇总 124 场景分布。
验收: E_gauge 显著高于随机对照 → 升格直接测量; 不显著 → 如实报告。
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
from fusion_unet import FusionUNet           # noqa: E402
from physics_renderer import PhysicsRenderer  # noqa: E402
from residual_modules import HierarchicalResidual  # noqa: E402

CKPT = Path("D:/MIR_Archive_20260829/checkpoints/A3-0_f_n5gray_seed42/best_model.pth")
SPLIT = REPO / "splits" / "synthetic_v3.json"
DATA_ROOT = "D:/data/synthetic_v3"
OUT = HERE / "exp13_gauge_error_fraction.json"
N_RANDOM_CONTROLS = 3
EPS = 1e-3
SEED = 20260906
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    msd = ckpt["model_state_dict"]
    tcfg = ckpt.get("config", {})
    model = FusionUNet(num_images=int(tcfg.get("num_lights", 5)),
                       in_channels=int(msd["stem.net.0.weight"].shape[1]),
                       base_channels=int(tcfg.get("base_channels", 32)),
                       use_per_light_albedo="delta_head.0.weight" in msd,
                       sh_constraint=str(tcfg.get("sh_constraint", "clamp")))
    model.load_state_dict(msd)
    renderer = PhysicsRenderer()
    renderer.load_state_dict(ckpt["renderer_state_dict"])
    r_sd = ckpt.get("residual_state_dict") or {}
    residual = HierarchicalResidual(use_local_residual=True,
                                    num_images=int(tcfg.get("num_lights", 5)),
                                    feature_channels=int(tcfg.get("base_channels", 32)),
                                    hidden_channels=int(r_sd["local_net.net.0.weight"].shape[0]) if r_sd else 64)
    if r_sd:
        residual.load_state_dict(r_sd)
    model.to(DEVICE).eval(); renderer.to(DEVICE).eval(); residual.to(DEVICE).eval()
    return model, renderer


def normals_from_depth_torch(depth, renderer):
    """depth [1,1,H,W] → 法线 [3,H,W](与网络管线同一 renderer)。"""
    return renderer.depth_to_normal(depth)[0]


def gbr_perturbed_normals(z_t, renderer, H, W):
    """z_t [1,1,H,W] → 法线 [3,H,W]。"""
    return renderer.depth_to_normal(z_t)[0]


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model, renderer = load_model()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    ds = MultiLightingDataset(root_dir=DATA_ROOT, num_lights=5,
                              image_size=(256, 256), is_training=False,
                              scene_subset=split["test"], load_gt=True, modality="gray")
    print(f"[exp13] dataset n={len(ds)}")
    results = []
    for i in range(len(ds)):
        images, gt, name = ds[i]
        # 网络法线
        with torch.no_grad():
            d_net = model(images.unsqueeze(0).to(DEVICE))[0]
        n_net = renderer.depth_to_normal(d_net)[0]              # [3,H,W]
        n_gt = gt["normal"].to(DEVICE)                          # [3,H,W]
        mask = gt["mask"][0] > 0.5
        mask_t = mask.to(DEVICE)
        # 误差场 δn(取切向投影后的角差向量)
        dn = n_net - n_gt
        dn = dn - n_gt * (n_gt * dn).sum(0, keepdim=True)       # 投到 GT 法线切面
        dn[:, ~mask_t] = 0
        # GBR 三生成元: GT depth 场加小步长扰动 → renderer 重算法线
        z_gt = gt["depth"][0:1].to(DEVICE)                      # [1,H,W]
        z4 = z_gt.unsqueeze(0)                                  # [1,1,H,W]
        xg, yg = np.meshgrid(np.arange(256), np.arange(256))
        gen_fields = {
            "lambda": z4 * (1 + EPS) - z4,                       # δz = ε·z
            "mu": torch.tensor(xg * EPS, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0),
            "nu": torch.tensor(yg * EPS, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0),
        }
        gvecs = []
        with torch.no_grad():
            for gname, dz in gen_fields.items():
                n_p = normals_from_depth_torch(z4 + dz, renderer)
                dn_p = ((n_p - n_gt) / EPS)
                dn_p = dn_p - n_gt * (n_gt * dn_p).sum(0, keepdim=True)
                dn_p[:, ~mask_t] = 0
                gvecs.append(dn_p.reshape(-1))
        G = np.stack([g.cpu().numpy() for g in gvecs])           # (3, D)
        # 随机平滑对照场(同范数)
        ctrl_fracs = []
        for _ in range(N_RANDOM_CONTROLS):
            noise = rng.normal(0, 1, (3, 256, 256)).astype(np.float32)
            # 平滑: 简易 box blur ×3(避免 scipy 依赖)
            for ax in (1, 2):
                noise = (np.roll(noise, 1, ax) + noise + np.roll(noise, -1, ax)) / 3
            noise_t = torch.tensor(noise, device=DEVICE)
            noise_t = noise_t - n_gt * (n_gt * noise_t).sum(0, keepdim=True)
            noise_t[:, ~mask_t] = 0
            noise_t = noise_t / max(np.linalg.norm(noise_t.reshape(-1).cpu().numpy()), 1e-12) \
                * np.linalg.norm(dn.reshape(-1).cpu().numpy())
            frac_c = gauge_fraction(noise_t.reshape(-1).cpu().numpy(), G)
            ctrl_fracs.append(frac_c)
        frac = gauge_fraction(dn.reshape(-1).cpu().numpy(), G)
        results.append(dict(scene=name, E_gauge=float(frac),
                            control_mean=float(np.mean(ctrl_fracs)),
                            control_max=float(np.max(ctrl_fracs))))
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(ds)} 场景完成")

    fr = np.array([r["E_gauge"] for r in results])
    cm = np.array([r["control_mean"] for r in results])
    out = {"per_scene": results, "meta": dict(n=len(results), eps=EPS,
                                              n_controls=N_RANDOM_CONTROLS),
           "summary": dict(E_gauge_mean=float(fr.mean()), E_gauge_median=float(np.median(fr)),
                           control_mean=float(cm.mean()),
                           n_above_control=int((fr > cm).sum())),
           "verdict": {
               "note": "E_gauge 显著高于随机对照 → 网络误差集中在解析歧义子空间(直接测量); 否则如实报告",
               "comparison": "见 summary 数字",
           }}
    print(f"\n[exp13] E_gauge 均值 {fr.mean():.4f} 中位 {np.median(fr):.4f} | "
          f"随机对照均值 {cm.mean():.4f} | 高于对照场景数 {(fr > cm).sum()}/{len(fr)}")
    (HERE / "exp13_gauge_error_fraction.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp13] 落盘 -> {HERE / 'exp13_gauge_error_fraction.json'}")


def gauge_fraction(vec, G):
    """vec (D,) 在 span(G 行) 上的能量占比。"""
    nrm2 = np.linalg.norm(vec) ** 2
    if nrm2 < 1e-300:
        return 0.0
    Q, _ = np.linalg.qr(G.T)
    proj = Q @ (Q.T @ vec)
    return float(np.linalg.norm(proj) ** 2 / nrm2)


if __name__ == "__main__":
    main()
