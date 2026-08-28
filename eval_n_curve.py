"""
T2.6b / T2.5：N 光子集敏感性曲线评估（单一模型，多 N 推理）

对冻结 test 的每个场景：从其 5 张灰度图中抽 N 张（M=10 个随机子集，
seed 固定且**子集索引入库**供审计重放），推理并计算指标；聚合出
N∈{1..5} × M 子集的曲线原始数据。

注意（如实声明）：v3 每场景仅渲染 5 光，故 N>5 无法评估；
任务书中 {7,10} 需要未来渲染更多光照后才能补测。

用法:
  python eval_n_curve.py --checkpoint ../checkpoints/best_model.pth \
      --data_root D:/data/synthetic_v3 --split_manifest splits/synthetic_v3.json \
      --out_dir eval_output/n_curve
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import MultiLightingDataset
from split_manifest import load_split
from unet_model import IntrinsicUNet
from physics_renderer import PhysicsRenderer
from residual_modules import HierarchicalResidual
from evaluate import compute_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--out_dir", default="eval_output/n_curve")
    ap.add_argument("--ns", default="1,2,3,4,5")
    ap.add_argument("--subsets_per_n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=1)   # 逐场景逐子集
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tcfg = ckpt.get("config", {})
    num_images = int(tcfg.get("num_lights", 5))
    base_channels = int(tcfg.get("base_channels", 32))
    msd = ckpt["model_state_dict"]
    is_fusion = "aggregator.proj.weight" in msd
    if is_fusion:
        from fusion_unet import FusionUNet
        in_channels = int(msd["stem.net.0.weight"].shape[1])
        use_pl_alb = "delta_head.0.weight" in msd
        model = FusionUNet(num_images=num_images, in_channels=in_channels,
                           base_channels=base_channels,
                           use_per_light_albedo=use_pl_alb)
    else:
        model = IntrinsicUNet(num_images=num_images, base_channels=base_channels)
    renderer = PhysicsRenderer()
    residual = HierarchicalResidual(use_local_residual=True,
                                    num_images=num_images,
                                    feature_channels=base_channels)
    model.load_state_dict(ckpt["model_state_dict"])
    renderer.load_state_dict(ckpt["renderer_state_dict"])
    if ckpt.get("residual_state_dict"):
        residual.load_state_dict(ckpt["residual_state_dict"])
    model.to(device).eval(); renderer.to(device).eval(); residual.to(device).eval()

    test_names = sorted(load_split(args.split_manifest, "test"))
    # 自动探测 modality（与 evaluate_model.py 保持一致）
    if tcfg.get("modality"):
        modality = str(tcfg["modality"])
    elif is_fusion:
        modality = "rgb" if msd["stem.net.0.weight"].shape[1] == 3 else "gray"
    else:
        modality = "gray"
    ds = MultiLightingDataset(root_dir=args.data_root, num_lights=num_images,
                              image_size=(256, 256), is_training=False,
                              scene_subset=test_names, modality=modality,
                              load_gt=True)
    rng = np.random.default_rng(args.seed)

    ns = [int(x) for x in args.ns.split(",")]
    protocol = {}
    curve = {n: [] for n in ns}

    import itertools
    for idx in range(len(ds)):
        images, gt, sname = ds[idx]
        images = images.unsqueeze(0).to(device)
        gt_dev = {k: v.unsqueeze(0).to(device) for k, v in gt.items()} \
            if gt is not None else None
        if gt_dev is None:
            continue
        # 重建目标用 BT.709 luma（与 evaluate_model.py 保持一致）
        recon_target = images
        if images.dim() == 5:
            if "image_luma" in gt_dev:
                recon_target = gt_dev["image_luma"]
            else:
                recon_target = (0.2126 * images[:, :, 0]
                                + 0.7152 * images[:, :, 1]
                                + 0.0722 * images[:, :, 2])
            gt_dev.pop("image_luma", None)

        for N in ns:
            if N > num_images:
                continue
            combos = list(itertools.combinations(range(num_images), N))
            rng_c = np.random.default_rng(args.seed + N * 1000 + idx)
            take = min(args.subsets_per_n, len(combos))
            chosen = [combos[i] for i in
                      rng_c.choice(len(combos), size=take, replace=False)]

            subset_metrics = []
            for si, combo in enumerate(chosen):
                sel = list(combo) + [combo[0]] * (num_images - N)
                x = images[:, sel]                        # [1,N,H,W]
                with torch.no_grad():
                    out = model(x)
                    if len(out) == 6:
                        depth, albedo, sh, wm, feats, _alb_pl = out
                    else:
                        depth, albedo, sh, wm, feats = out
                    rendered, normal, shading = renderer(depth, albedo, sh)
                    fr, gr, lr = residual(albedo, shading, normal, sh,
                                          stage="stage3", features=feats)
                m = compute_all(
                    pred={"normal": normal, "depth": depth, "albedo": albedo,
                          "image": fr},
                    gt={"normal": gt_dev["normal"], "depth": gt_dev["depth"],
                        "albedo": gt_dev["albedo"],
                        "image": recon_target},
                    mask=gt_dev["mask"])
                m["subset_indices"] = list(combo)
                m["scene"] = sname
                m["n"] = N
                m["subset_id"] = si
                curve[N].append(m)

            protocol.setdefault(str(N), []).append(
                {"scene": sname, "subsets": [list(c) for c in chosen]})

    agg = {}
    for N in sorted(curve):
        rows = curve[N]
        keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
        agg[N] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                      "std": float(np.std([r[k] for r in rows]))}
                  for k in keys}
        print(f"N={N}:")
        for k, s in agg[N].items():
            print(f"   {k}: {s['mean']:.4f} ± {s['std']:.4f}")

    with open(os.path.join(args.out_dir, "n_curve_raw.json"), "w",
              encoding="utf-8") as f:
        json.dump({"per_subset": [m for n in curve for m in curve[n]],
                   "protocol": protocol}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out_dir, "n_curve_agg.json"), "w",
              encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=1)
    print("JSON ->", args.out_dir)


if __name__ == "__main__":
    main()
