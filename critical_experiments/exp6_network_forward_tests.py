#!/usr/bin/env python3
"""关键实验 6 · 网络三个前向测试（零训练，A3-0 checkpoint）

设计文档原文语义：
  1) 复制测试：某场景 N 张图全部换成同一张（light_001 复制 5 遍）
     vs 正常 5 张不同图 → 输出法线 MAE 差异。若几乎一样 → 网络没用多光照信息。
  2) 打乱测试：保留 1 张原场景图，其余 N−1 张换成其他随机场景的图
     → 输出法线与正常输入的差异。若变化很小 → 网络对输入光照不敏感。
  3) 敏感度测试：对正常 N 张图输入，算 ∂n̂/∂I_i 的梯度范数 ‖·‖ 在 N 张图上的分布。

实现口径（与 evaluate_model.py 加载完全一致，逐符号对齐）：
  - FusionUNet + PhysicsRenderer + HierarchicalResidual，ckpt = A3-0 best_model.pth；
  - gray 模态、sh_constraint 从 checkpoint 元数据 auto 解析；
  - 法线对比：模型输出的深度头 → renderer.depth_to_normal（与训练物理一致），
    GT 法线角度误差 per-scene，正常/复制/打乱 三态同口径；
  - 测试集：synthetic_v3 test split 的 124 场景（scene 级口径同 EX 系列），
    敏感度测试抽样 32 场景控制算力（torch.autograd 逐像素回传，如实注记）。

产物：critical_experiments/exp6_network_forward_tests.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_loader import MultiLightingDataset  # noqa: E402
from fusion_unet import FusionUNet           # noqa: E402
from physics_renderer import PhysicsRenderer  # noqa: E402
from residual_modules import HierarchicalResidual  # noqa: E402

CKPT = Path("D:/MIR_Archive_20260829/checkpoints/A3-0_f_n5gray_seed42/best_model.pth")
SPLIT = REPO / "splits" / "synthetic_v3.json"
DATA_ROOT = "D:/data/synthetic_v3"
OUT = REPO / "critical_experiments" / "exp6_network_forward_tests.json"
N_SENSITIVITY_SCENES = 32
SEED = 20260905
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    msd = ckpt["model_state_dict"]
    tcfg = ckpt.get("config", {})
    num_images = int(tcfg.get("num_lights", 5))
    in_channels = int(msd["stem.net.0.weight"].shape[1])
    use_pl_alb = "delta_head.0.weight" in msd
    sh_constraint = str(tcfg.get("sh_constraint", "clamp"))
    model = FusionUNet(num_images=num_images, in_channels=in_channels,
                       base_channels=int(tcfg.get("base_channels", 32)),
                       use_per_light_albedo=use_pl_alb,
                       sh_constraint=sh_constraint)
    model.load_state_dict(msd)
    renderer = PhysicsRenderer()
    renderer.load_state_dict(ckpt["renderer_state_dict"])
    r_sd = ckpt.get("residual_state_dict") or {}
    res_hidden = int(r_sd["local_net.net.0.weight"].shape[0]) if r_sd else 64
    residual = HierarchicalResidual(use_local_residual=True, num_images=num_images,
                                    feature_channels=int(tcfg.get("base_channels", 32)),
                                    hidden_channels=res_hidden)
    if r_sd:
        residual.load_state_dict(r_sd)
    model.to(DEVICE).eval(); renderer.to(DEVICE).eval(); residual.to(DEVICE).eval()
    return model, renderer, residual, tcfg


def normals_from_model(model, renderer, images):
    """images [N,H,W]（或 [B,N,H,W]）→ 输出法线 [3,H,W]。
    FusionUNet 前向返回五元组 + 逐光反照率；深度头输出经 renderer.depth_to_normal。"""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    out = model(images.to(DEVICE))
    depth = out[0] if isinstance(out, (tuple, list)) else out["depth"]
    if depth.dim() == 5:
        depth = depth[:, 0]      # [B,N,1,H,W] → 共享深度(均值池化后)
        depth = depth.mean(dim=1)
    if depth.dim() == 4:
        depth = depth
    normal = renderer.depth_to_normal(depth)
    return normal[0]             # [3,H,W]


def angular_mae(n1, n2, mask):
    d = (n1 * n2).sum(0).clamp(-1, 1)
    ang = torch.rad2deg(torch.acos(d))
    return float(ang[mask].mean())


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    test_names = split["test"]

    model, renderer, residual, tcfg = load_model()
    print(f"[exp6] model loaded, device={DEVICE}, tcfg keys={list(tcfg)[:6]}")

    ds = MultiLightingDataset(root_dir=DATA_ROOT, num_lights=5,
                              image_size=(256, 256), is_training=False,
                              scene_subset=test_names, load_gt=True, modality="gray")
    print(f"[exp6] dataset n={len(ds)}")

    # 预取全部场景张量（124 × 5×256×256 float32 ≈ 400MB，可行）
    cache = {}
    for i in range(len(ds)):
        images, gt, name = ds[i]
        cache[name] = (images, gt)
    names = list(cache.keys())

    results = {"replicate_test": {}, "shuffle_test": {}, "sensitivity_test": {}}

    # ---------- 测试 1：复制测试 ----------
    rep_maes = []
    for name in names:
        images, gt = cache[name]
        # 正常输入
        with torch.no_grad():
            n_normal = normals_from_model(model, renderer, images)
        # 复制输入：全部换成 light_001
        img_dup = images[0:1].repeat(5, 1, 1)
        with torch.no_grad():
            n_dup = normals_from_model(model, renderer, img_dup)
        mask = gt["mask"][0] > 0.5
        mae_dup = angular_mae(n_dup, gt["normal"].to(DEVICE), mask.to(DEVICE))
        mae_norm = angular_mae(n_normal, gt["normal"].to(DEVICE), mask.to(DEVICE))
        # 法线自身对复制的输出漂移（网络输入敏感性直接量）
        drift = angular_mae(n_dup, n_normal, mask.to(DEVICE))
        rep_maes.append(dict(scene=name, mae_normal=mae_norm, mae_replicated=mae_dup,
                             output_drift_deg=drift))
    arr_dn = np.array([r["mae_normal"] for r in rep_maes])
    arr_drep = np.array([r["mae_replicated"] for r in rep_maes])
    arr_drift = np.array([r["output_drift_deg"] for r in rep_maes])
    results["replicate_test"] = dict(
        per_scene=rep_maes,
        mae_normal_mean=float(arr_dn.mean()), mae_normal_median=float(np.median(arr_dn)),
        mae_replicated_mean=float(arr_drep.mean()), mae_replicated_median=float(np.median(arr_drep)),
        output_drift_mean=float(arr_drift.mean()), output_drift_median=float(np.median(arr_drift)),
        delta_mae_mean=float((arr_drep - arr_dn).mean()),
        verdict_hint="复制输入的 MAE 恶化(Δ)与输出漂移(drift)给出网络对'光照多样性缺失'的敏感性",
    )
    print(f"[exp6-1 复制] normal MAE {arr_dn.mean():.2f}° → replicated {arr_drep.mean():.2f}° "
          f"(Δ={ (arr_drep-arr_dn).mean():+.2f}°, 输出漂移 median {np.median(arr_drift):.2f}°)")

    # ---------- 测试 2：打乱测试 ----------
    shuf_maes = []
    for name in names:
        images, gt = cache[name]
        with torch.no_grad():
            n_normal = normals_from_model(model, renderer, images)
        # 保留 light_001，其余 4 张换成随机其他场景的对应图
        others = rng.choice([n for n in names if n != name], 4, replace=False)
        imgs_mix = images.clone()
        for j, other in enumerate(others):
            imgs_mix[1 + j] = cache[other][0][rng.integers(5)]
        with torch.no_grad():
            n_mix = normals_from_model(model, renderer, imgs_mix)
        mask = gt["mask"][0] > 0.5
        mae_mix = angular_mae(n_mix, gt["normal"].to(DEVICE), mask.to(DEVICE))
        drift = angular_mae(n_mix, n_normal, mask.to(DEVICE))
        shuf_maes.append(dict(scene=name, mae_mixed=mae_mix,
                              mae_normal=float(angular_mae(n_normal, gt["normal"].to(DEVICE), mask.to(DEVICE))),
                              output_drift_deg=drift))
    arr_sm = np.array([r["mae_mixed"] for r in shuf_maes])
    arr_sdrift = np.array([r["output_drift_deg"] for r in shuf_maes])
    results["shuffle_test"] = dict(
        per_scene=shuf_maes,
        mae_mixed_mean=float(arr_sm.mean()), mae_mixed_median=float(np.median(arr_sm)),
        output_drift_mean=float(arr_sdrift.mean()), output_drift_median=float(np.median(arr_sdrift)),
    )
    print(f"[exp6-2 打乱] mixed MAE {arr_sm.mean():.2f}° (漂移 median {np.median(arr_sdrift):.2f}°)")

    # ---------- 测试 3：敏感度测试 ----------
    sens = []
    sub = rng.choice(names, N_SENSITIVITY_SCENES, replace=False).tolist()
    for name in sub:
        images, gt = cache[name]
        x = images.unsqueeze(0).to(DEVICE).requires_grad_(True)   # [1,5,H,W]
        out = model(x)
        depth = out[0]            # [1,1,H,W]
        normal = renderer.depth_to_normal(depth)
        # 输出标量: 法线的 mask 内均值范数(对每张输入图分别求梯度)
        mask = gt["mask"][0] > 0.5
        loss = (normal * normal).sum()      # 法线能量的代理标量(避免 3×H×W 逐元素存梯度)
        loss.backward()
        g = x.grad[0]                        # [5,H,W]
        per_light = [float(g[k][mask.to(DEVICE)].pow(2).sum().sqrt()) for k in range(5)]
        tot = float(np.sqrt(sum(v ** 2 for v in per_light)))
        sens.append(dict(scene=name, grad_norm_per_light=per_light, grad_norm_total=tot))
    tot_arr = np.array([r["grad_norm_total"] for r in sens])
    share_max = [max(r["grad_norm_per_light"]) / max(r["grad_norm_total"], 1e-30) for r in sens]
    results["sensitivity_test"] = dict(
        per_scene=sens, n_scenes=len(sub),
        grad_total_mean=float(tot_arr.mean()), grad_total_median=float(np.median(tot_arr)),
        max_share_mean=float(np.mean(share_max)),
        share_note="max_share=单张图占梯度能量的比例; ~1/N=0.2 为均匀, ~1 为集中于单图",
    )
    print(f"[exp6-3 敏感度] ‖∂n̂/∂I‖ 总量 median {np.median(tot_arr):.4f}, "
          f"单图最大占比 {np.mean(share_max):.2f} (均匀基准 0.20)")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[exp6] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
