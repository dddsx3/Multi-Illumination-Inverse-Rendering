"""
Phase 1 T1.7：DiLiGenT 迁移验证（G7）

协议：
- 每物体从 96 光中按索引均匀抽 5 张作为模型输入
- 彩色光照归一化：I_c /= L_c（逐通道），BT.709 luma 转灰度，clip [0,1]
  （数值为线性强度，直接作为张量输入——与训练链路"解码后线性值"一致）
- 几何对齐：中心裁剪 612->512 宽度 + 缩放 256x256；GT 法线同步处理并重新单位化
- 掩码：GT 法线长度 > 0.5（背景在 GT 中存储为零向量）
- 预测法线来自 渲染器(预测深度)；与 GT 夹角统计（若整体反向自动翻转并标记）

用法:
  python evaluate_diligent.py --root D:/data/DiLiGenT/pmsData \
      --checkpoint ../checkpoints/best_model.pth [--out_dir eval_diligent]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unet_model import IntrinsicUNet
from physics_renderer import PhysicsRenderer
from residual_modules import HierarchicalResidual

OBJECTS = ["ballPNG", "bearPNG", "buddhaPNG", "catPNG", "cowPNG",
           "gobletPNG", "harvestPNG", "pot1PNG", "pot2PNG", "readingPNG"]
LUMA = (0.2126, 0.7152, 0.0722)


def angular_stats(pred, gt, mask):
    pn = pred / np.clip(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-8, None)
    gn = gt / np.clip(np.linalg.norm(gt, axis=-1, keepdims=True), 1e-8, None)
    dot = np.clip((pn * gn).sum(axis=-1), -1.0, 1.0)
    deg = np.degrees(np.arccos(dot))[mask]
    flipped = False
    if deg.size and deg.mean() > 90.0:
        deg = 180.0 - deg
        flipped = True
    return {
        "mae": float(deg.mean()),
        "median": float(np.median(deg)),
        "acc_11_25": float((deg <= 11.25).mean()),
        "acc_22_5": float((deg <= 22.5).mean()),
        "acc_30": float((deg <= 30.0).mean()),
        "flipped": flipped,
        "pixels": int(mask.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/data/DiLiGenT/pmsData")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_lights", type=int, default=5)
    ap.add_argument("--out_dir", default="eval_diligent")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tcfg = ckpt.get("config", {})
    NI = int(tcfg.get("num_lights", 5))
    BC = int(tcfg.get("base_channels", 32))

    model = IntrinsicUNet(num_images=NI, base_channels=BC)
    renderer = PhysicsRenderer()
    residual = HierarchicalResidual(use_local_residual=True,
                                    num_images=NI, feature_channels=BC)
    model.load_state_dict(ckpt["model_state_dict"])
    renderer.load_state_dict(ckpt["renderer_state_dict"])
    if ckpt.get("residual_state_dict"):
        residual.load_state_dict(ckpt["residual_state_dict"])
    model.to(device).eval(); renderer.to(device).eval(); residual.to(device).eval()

    results = {}
    for obj in OBJECTS:
        d = os.path.join(args.root, obj)
        li = np.loadtxt(os.path.join(d, "light_intensities.txt"))
        n_total = li.shape[0]

        sel = np.unique(np.linspace(0, n_total - 1, NI).astype(int))
        while len(sel) < NI:
            sel = np.append(sel, sel[-1] + 1)

        H, W = 512, 612
        crop_l = (W - H) // 2

        grays = []
        for k in sel:
            img = np.asarray(Image.open(
                os.path.join(d, f"{k + 1:03d}.png"))).astype(np.float32) / 255.0
            img_n = img / np.clip(li[k].astype(np.float32)[None, None, :],
                                  np.float32(0.05), None)
            gray = (LUMA[0] * img_n[..., 0] + LUMA[1] * img_n[..., 1]
                    + LUMA[2] * img_n[..., 2])
            gray = np.clip(gray[:, crop_l:crop_l + H], 0.0, 1.0)
            grays.append(torch.from_numpy(gray))

        import scipy.io as sio
        n_mat = sio.loadmat(os.path.join(d, "Normal_gt.mat"))["Normal_gt"]
        n_map = np.asarray(n_mat, dtype=np.float32)
        if n_map.shape[0] == 3:
            n_map = n_map.transpose(1, 2, 0)
        n_map = n_map[:, crop_l:crop_l + H, :]
        gt_t = F.interpolate(
            torch.from_numpy(n_map).permute(2, 0, 1)[None],
            size=(256, 256), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).numpy()

        images = torch.stack(grays, dim=0).unsqueeze(0).to(device)   # [1,K,H,W]
        images = F.interpolate(images, size=(256, 256), mode="bilinear",
                               align_corners=False)

        with torch.no_grad():
            depth, albedo, sh, wm, feats = model(images)
            rendered, normal, shading = renderer(depth, albedo, sh)
            final_r, g_res, l_res = residual(albedo, shading, normal, sh,
                                             stage="stage3", features=feats)
            normal_p = normal[0].permute(1, 2, 0).float().cpu().numpy()

        mask = np.linalg.norm(gt_t, axis=-1) > 0.5
        if mask.sum() < 100:
            results[obj] = {"error": "有效前景过少"}
            print(f"[{obj}] 有效前景过少")
            continue

        st = angular_stats(normal_p, gt_t, mask)
        st["lights_used"] = [int(x) + 1 for x in sel]
        results[obj] = st
        flip_tag = " (FLIPPED)" if st["flipped"] else ""
        print(f"[{obj}] MAE={st['mae']:.2f} med={st['median']:.2f} "
              f"acc11.25={st['acc_11_25']:.3f} px={st['pixels']}{flip_tag}")

    ok = {o: r for o, r in results.items() if "mae" in r}
    if ok:
        agg = {k: round(float(np.mean([r[k] for r in ok.values()])), 4)
               for k in ("mae", "median", "acc_11_25", "acc_22_5", "acc_30")}
        print("\n=== G7 aggregate over", len(ok), "objects ===")
        for k, v in agg.items():
            print(f"  {k}: {v}")

    with open(os.path.join(args.out_dir, "diligent_results.json"), "w",
              encoding="utf-8") as f:
        json.dump({"per_object": results, "aggregate": agg if ok else {}},
                  f, ensure_ascii=False, indent=1)
    print("JSON ->", os.path.join(args.out_dir, "diligent_results.json"))


if __name__ == "__main__":
    main()
