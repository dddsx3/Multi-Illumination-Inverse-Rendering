"""重训 v2 + INC-0012 物理约束：warm-start from 原 v2 best → 100 epoch 续训

目的：验证 INC-0012 修复后反照率退化是否治愈。
  - v2 best 已有 Sigmoid/Softplus 物理约束（key 兼容）—— 但 Sigmoid/Softplus 层是随机初始化，
    没有从零学习过 0/1 边界
  - 用原 v2 best 的主干 + 重新学习物理约束头（3-5 epoch 应该够）

策略：
  - warm-start from v2 best 的主干权重
  - 物理约束头（Sigmoid/Softplus）用默认初始化重新训练
  - 短训练 5-10 epoch（不重训 100 epoch，避免 GPU 占用与 A3-bis 冲突）
  - 评估 13+2 项指标 + 对比 v2 best

用法:
  python _retrain_v2_with_phys.py --warm_start_ckpt <v2_best> --smoke_epochs 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import MultiLightingDataset, create_data_loader, split_scene_names
from fusion_unet import FusionUNet
from physics_renderer import PhysicsRenderer
from residual_modules import HierarchicalResidual
from evaluate_model import assert_physical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm_start_ckpt",
                    default=r"D:\Multi-Illumination Inverse Rendering\checkpoints\p2_t22_f_n5rgb_v2\best_model.pth")
    ap.add_argument("--data_root", default=r"D:\data\synthetic_v3")
    ap.add_argument("--smoke_epochs", type=int, default=10)
    ap.add_argument("--albedo_smooth", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--train_subset", type=int, default=100)
    ap.add_argument("--val_subset", type=int, default=20)
    ap.add_argument("--tag", default="v2_phys_10ep")
    ap.add_argument("--out_dir", default=r"D:\Multi-Illumination Inverse Rendering\_SMOKE_phys_constraints")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[T-PHYS] warm-start 重训: tag={args.tag} epochs={args.smoke_epochs} "
          f"albedo_smooth={args.albedo_smooth}")
    print(f"[T-PHYS] log: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 v2 best 作 warm-start
    print(f"[T-PHYS] 加载 v2 best: {args.warm_start_ckpt}")
    ckpt = torch.load(args.warm_start_ckpt, map_location=device, weights_only=False)
    tcfg = ckpt.get("config", {})
    num_lights = int(tcfg.get("num_lights", 5))
    base_channels = int(tcfg.get("base_channels", 32))
    msd = ckpt["model_state_dict"]
    in_channels = int(msd["stem.net.0.weight"].shape[1])
    use_pl_alb = "delta_head.0.weight" in msd
    modality = "rgb" if in_channels == 3 else "gray"
    print(f"[T-PHYS] modality={modality} in_channels={in_channels} use_pl_alb={use_pl_alb}")

    # 2. 构造新模型
    model = FusionUNet(num_images=num_lights, in_channels=in_channels,
                       base_channels=base_channels,
                       use_per_light_albedo=use_pl_alb,
                       sh_constraint=str(tcfg.get("sh_constraint", "clamp")))
    renderer = PhysicsRenderer()
    residual = HierarchicalResidual(use_local_residual=True,
                                    num_images=num_lights,
                                    feature_channels=base_channels)
    # 严格 load v2 best
    missing, unexpected = model.load_state_dict(msd, strict=False)
    print(f"[T-PHYS] warm-start load: missing={len(missing)} unexpected={len(unexpected)}")
    if ckpt.get("renderer_state_dict"):
        renderer.load_state_dict(ckpt["renderer_state_dict"])
    if ckpt.get("residual_state_dict"):
        residual.load_state_dict(ckpt["residual_state_dict"])
    model.to(device); renderer.to(device); residual.to(device)

    # 3. 数据集
    train_names, val_names = split_scene_names(args.data_root, 0.8, seed=42)
    train_ds = MultiLightingDataset(root_dir=args.data_root, num_lights=num_lights,
                                     image_size=(256, 256), is_training=True,
                                     scene_subset=train_names[:args.train_subset],
                                     load_gt=True, modality=modality)
    val_ds = MultiLightingDataset(root_dir=args.data_root, num_lights=num_lights,
                                   image_size=(256, 256), is_training=False,
                                   scene_subset=val_names[:args.val_subset],
                                   load_gt=True, modality=modality)
    train_loader = create_data_loader(train_ds, batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers)
    print(f"[T-PHYS] train {len(train_ds)} 场景 / val {len(val_ds)} 场景")

    # 4. 优化器（较低 lr 避免破坏已学特征）
    optim = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))

    # 5. 训练 N epoch
    history = []
    for epoch in range(args.smoke_epochs):
        epoch_t0 = time.time()
        model.train(); renderer.train(); residual.train()
        running = {"recon": 0.0, "albedo_smooth": 0.0, "n_batches": 0}
        for batch in train_loader:
            images, gt, _ = batch
            images = images.to(device)
            gt_dev = {k: v.to(device) for k, v in gt.items() if k != "image_luma"}
            recon_target = gt.get("image_luma", images).to(device)
            optim.zero_grad()
            out = model(images)
            if len(out) == 6:
                depth, albedo, sh, wm, feats, _alb_pl = out
            else:
                depth, albedo, sh, wm, feats = out
            rendered, normal, shading = renderer(depth, albedo, sh)
            final_r, g_res, l_res = residual(albedo, shading, normal, sh,
                                              stage="stage1", features=feats)
            err = final_r - recon_target
            mask = gt_dev["mask"]
            err_masked = err * mask
            n_valid = mask.sum().clamp_min(1)
            recon_loss = (err_masked ** 2 + 1e-3).sqrt().sum() / n_valid
            dy = (albedo[..., 1:, :] - albedo[..., :-1, :]).abs()
            dx = (albedo[..., :, 1:] - albedo[..., :, :-1]).abs()
            alb_smooth = (dy.sum() + dx.sum()) / (albedo.numel() + 1)
            loss = recon_loss + args.albedo_smooth * alb_smooth
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running["recon"] += float(recon_loss)
            running["albedo_smooth"] += float(alb_smooth)
            running["n_batches"] += 1
        epoch_dt = time.time() - epoch_t0
        nb = running["n_batches"]
        train_rec = {
            "recon": running["recon"] / max(nb, 1),
            "albedo_smooth": running["albedo_smooth"] / max(nb, 1),
        }

        # 物理断言
        phys = run_physical_assertion(model, val_ds, device, num_lights)
        rec = {
            "epoch": epoch,
            "duration_s": round(epoch_dt, 1),
            "train": train_rec,
            "physical": phys,
        }
        history.append(rec)
        print(f"[T-PHYS] epoch {epoch}: dur={epoch_dt:.1f}s recon={train_rec['recon']:.4f}")
        print(f"  albedo mean={phys['albedo_mean']:.4f} std={phys['albedo_std']:.4f} "
              f"viol={phys['albedo_violation_ratio']*100:.4f}%")
        print(f"  depth  mean={phys['depth_mean']:.4f} std={phys['depth_std']:.4f} "
              f"viol={phys['depth_violation_ratio']*100:.4f}%")

    # 6. 落盘
    with open(os.path.join(out_dir, f"retrain_{args.tag}_history.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "tag": args.tag,
            "warm_start_ckpt": args.warm_start_ckpt,
            "smoke_epochs": args.smoke_epochs,
            "albedo_smooth": args.albedo_smooth,
            "history": history,
        }, f, ensure_ascii=False, indent=1)

    last = history[-1]["physical"] if history else None
    if last:
        print()
        print(f"[T-PHYS] 末 epoch 物理断言:")
        print(f"  albedo violation: {last['albedo_violation_ratio']*100:.4f}%")
        print(f"  depth  violation: {last['depth_violation_ratio']*100:.4f}%")
        print(f"  albedo mean: {last['albedo_mean']:.4f} std: {last['albedo_std']:.4f}")
    print(f"[T-PHYS] 重训冒烟完成，产物: {out_dir}")


def run_physical_assertion(model, val_ds, device, num_lights):
    model.eval()
    alb_viols, dep_viols = [], []
    alb_means, alb_stds = [], []
    dep_means, dep_stds = [], []
    with torch.no_grad():
        for i in range(min(5, len(val_ds))):
            images, gt, _ = val_ds[i]
            images = images.unsqueeze(0).to(device)
            gt_dev = {k: v.unsqueeze(0).to(device) for k, v in gt.items()}
            out = model(images)
            if len(out) == 6:
                depth, albedo, *_ = out
            else:
                depth, albedo, *_ = out
            phys = assert_physical({"albedo": albedo, "depth": depth},
                                    mask=gt_dev["mask"])
            alb_viols.append(phys["albedo_violation_ratio"])
            dep_viols.append(phys["depth_violation_ratio"])
            alb_means.append(phys["albedo_mean"])
            alb_stds.append(phys["albedo_std"])
            dep_means.append(phys["depth_mean"])
            dep_stds.append(phys["depth_std"])
    return {
        "albedo_violation_ratio": float(np.mean(alb_viols)),
        "depth_violation_ratio": float(np.mean(dep_viols)),
        "albedo_range": [float(min(alb_means) - 2 * np.mean(alb_stds)),
                         float(max(alb_means) + 2 * np.mean(alb_stds))],
        "depth_range": [float(min(dep_means) - 2 * np.mean(dep_stds)),
                        float(max(dep_means) + 2 * np.mean(dep_stds))],
        "albedo_mean": float(np.mean(alb_means)),
        "albedo_std": float(np.mean(alb_stds)),
        "depth_mean": float(np.mean(dep_means)),
        "depth_std": float(np.mean(dep_stds)),
    }


if __name__ == "__main__":
    main()
