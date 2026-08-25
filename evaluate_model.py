import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import MultiLightingDataset, create_data_loader, split_scene_names
from unet_model import IntrinsicUNet
from physics_renderer import PhysicsRenderer
from residual_modules import HierarchicalResidual
from evaluate import compute_all


def main():
    ap = argparse.ArgumentParser(description="Phase 1 T1.6 量化评估")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", choices=["val", "train", "test"], default="test",
                    help="test=冻结正式测试集（推荐）；需配合 --split_manifest")
    ap.add_argument("--split_manifest", default=None,
                    help="划分清单 JSON（C1 正式化）；提供后优先于 --split 的计算划分")
    ap.add_argument("--train_val_split", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="只评估前 N 个场景（0=全部）")
    ap.add_argument("--out_dir", default="eval_output")
    # T2.2/T2.3/T2.5：架构与变体参数。默认 auto 从 checkpoint 自动识别；
    # 无法从权重形状推断的项（sh_constraint、residual_off）优先读训练期
    # 存档的 config 元数据，读不到时回落默认值或显式旗标。
    ap.add_argument("--model", choices=["auto", "unet", "fusion"], default="auto")
    ap.add_argument("--modality", choices=["auto", "gray", "rgb"], default="auto")
    ap.add_argument("--sh_constraint", choices=["auto", "clamp", "softplus"],
                    default="auto")
    ap.add_argument("--res_hidden", type=int, default=0,
                    help="残差隐藏通道；0=从 checkpoint 形状自动识别")
    ap.add_argument("--residual_off", action="store_true",
                    help="F-resA：推理时残差缩放置零（与训练期行为一致）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tcfg = ckpt.get("config", {})
    num_images = int(tcfg.get("num_lights", 5))
    base_channels = int(tcfg.get("base_channels", 32))
    print("checkpoint epoch:", ckpt.get("epoch"), "| saved stage:",
          tcfg.get("current_stage"), "| val_loss@save:", ckpt.get("val_loss"))

    # 划分来源优先级：清单 JSON（C1 正式化，test 冻结）> 确定性计算
    if args.split_manifest:
        from split_manifest import load_split
        names = load_split(args.split_manifest, args.split)
    else:
        train_names, val_names = split_scene_names(
            args.data_root, args.train_val_split, args.seed)
        names = val_names if args.split == "val" else train_names
    print(f"split={args.split}: {len(names)} scenes")

    # 架构/模态/约束解析：显式旗标 > checkpoint 训练期元数据 > 权重形状 > 默认
    msd_probe = ckpt["model_state_dict"]
    if args.model != "auto":
        architecture = args.model
    elif "aggregator.proj.weight" in msd_probe:
        architecture = "fusion"
    else:
        architecture = "unet"
    if args.modality != "auto":
        modality = args.modality
    elif tcfg.get("modality"):
        modality = str(tcfg["modality"])
    elif architecture == "fusion":
        modality = "rgb" if msd_probe["stem.net.0.weight"].shape[1] == 3 else "gray"
    else:
        modality = "gray"
    if args.sh_constraint != "auto":
        sh_constraint = args.sh_constraint
    else:
        sh_constraint = str(tcfg.get("sh_constraint", "clamp"))
    print(f"[arch] model={architecture} modality={modality} "
          f"sh_constraint={sh_constraint}")

    dataset = MultiLightingDataset(
        root_dir=args.data_root,
        num_lights=num_images,
        image_size=(256, 256),
        is_training=False,
        scene_subset=names,
        load_gt=True,
        modality=modality,
    )

    msd = ckpt["model_state_dict"]
    if architecture == "fusion":
        from fusion_unet import FusionUNet
        in_channels = int(msd["stem.net.0.weight"].shape[1])
        use_pl_alb = "delta_head.0.weight" in msd
        model = FusionUNet(
            num_images=num_images, in_channels=in_channels,
            base_channels=base_channels,
            use_per_light_albedo=use_pl_alb, sh_constraint=sh_constraint)
        print(f"[auto] FusionUNet in_channels={in_channels} "
              f"per_light_albedo={use_pl_alb} sh_constraint={sh_constraint}")
    else:
        model = IntrinsicUNet(num_images=num_images, base_channels=base_channels)
    renderer = PhysicsRenderer()
    r_sd = ckpt.get("residual_state_dict") or {}
    res_hidden = args.res_hidden or int(
        r_sd.get("local_net.net.0.weight", torch.empty(64, 1)).shape[0])
    residual = HierarchicalResidual(
        use_local_residual=True, num_images=num_images, feature_channels=base_channels,
        hidden_channels=res_hidden)
    if getattr(args, "residual_off", False) or tcfg.get("residual_off"):
        # F-resA：阶段3 残差缩放恒 0，推理与训练期口径一致
        residual.residual_scales = {"stage1": 0.0, "stage2": 0.0, "stage3": 0.0}
        print("[F-resA] 推理残差缩放置零")
    model.load_state_dict(msd)
    renderer.load_state_dict(ckpt["renderer_state_dict"])
    if r_sd:
        residual.load_state_dict(r_sd)
    model.to(device).eval(); renderer.to(device).eval(); residual.to(device).eval()

    loader = create_data_loader(dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    per_scene = {}
    n = 0
    with torch.no_grad():
        for images, gt, names_b in loader:
            images = images.to(device)
            gt_dev = {k: v.to(device) for k, v in gt.items()} if gt is not None else None
            # rgb 模态：网络吃 [B,K,3,H,W]；重建/图像指标用与灰度链路
            # 逐位同源的 BT.709 luma 目标（数据加载器在编码域生成）
            recon_target = images
            if images.dim() == 5:
                recon_target = gt_dev["image_luma"] if (
                    gt_dev is not None and "image_luma" in gt_dev) else \
                    (0.2126 * images[:, :, 0] + 0.7152 * images[:, :, 1]
                     + 0.0722 * images[:, :, 2])
                if gt_dev is not None and "image_luma" in gt_dev:
                    gt_dev.pop("image_luma")
            out = model(images)
            if len(out) == 6:  # FusionUNet（S2 开启）：末位为逐光照反照率
                depth, albedo, sh_coeffs, weight_map, features, _alb_pl = out
            else:
                depth, albedo, sh_coeffs, weight_map, features = out
            rendered, normal, shading = renderer(depth, albedo, sh_coeffs)
            final_render, g_res, l_res = residual(
                albedo, shading, normal, sh_coeffs,
                stage="stage3", features=features)
            if gt_dev is None:
                continue
            m = compute_all(
                pred={"normal": normal, "depth": depth, "albedo": albedo,
                      "image": final_render},
                gt={"normal": gt_dev["normal"], "depth": gt_dev["depth"],
                    "albedo": gt_dev["albedo"], "image": recon_target},
                mask=gt_dev["mask"])

            # 深度对齐评估（Eigen 惯例）：网络预测的绝对深度存在尺度+偏移
            # 量规自由度（重建损失不约束绝对尺度），逐场景最小二乘对齐后
            # 再计算 RMSE/MAE，作为与原始未对齐指标并列的报告项
            mb = gt_dev["mask"]
            p = depth[mb > 0]
            g = gt_dev["depth"][mb > 0]
            if p.numel() > 8:
                A = torch.stack([p, torch.ones_like(p)], dim=1)
                try:
                    sol = torch.linalg.lstsq(A, g.unsqueeze(1)).solution
                    d_al = (sol[0, 0] * depth + sol[1, 0])
                    err = ((d_al - gt_dev["depth"]) * mb).abs()
                    m["depth_mae_aligned"] = float(err.sum() / mb.sum())
                    se = ((d_al - gt_dev["depth"]) ** 2 * mb)
                    m["depth_rmse_aligned"] = float(torch.sqrt(se.sum() / mb.sum()))
                except Exception:
                    pass
            for i, sname in enumerate(names_b):
                per_scene[sname] = {k: float(v[i] if isinstance(v, (list, tuple)) else v)
                                     for k, v in m.items()}
                n += 1
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break

    if not per_scene:
        print("[FAIL] 无可评估场景"); sys.exit(1)

    keys = sorted(next(iter(per_scene.values())).keys())
    agg = {}
    for k in keys:
        vals = np.array([per_scene[s][k] for s in per_scene])
        agg[k] = (float(vals.mean()), float(vals.std()))

    csv_path = os.path.join(args.out_dir, "per_scene_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene"] + keys)
        for s, m in sorted(per_scene.items()):
            w.writerow([s] + [m[k] for k in keys])

    json_path = os.path.join(args.out_dir, "eval_summary.json")
    summary = {
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "split": args.split,
        "split_manifest": args.split_manifest,
        "architecture": architecture,
        "modality": modality,
        "sh_constraint": sh_constraint,
        "res_hidden": res_hidden,
        "residual_off": bool(getattr(args, "residual_off", False) or tcfg.get("residual_off")),
        "scenes": len(per_scene),
        "metrics_mean_std": {k: {"mean": agg[k][0], "std": agg[k][1]} for k in keys},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print()
    print(f"=== 13 项指标（{len(per_scene)} 场景, mean ± std）===")
    for k in keys:
        mu, sd = agg[k]
        print(f"  {k:28s} {mu:10.4f} ± {sd:.4f}")
    print("CSV ->", csv_path)
    print("JSON ->", json_path)


if __name__ == "__main__":
    main()
