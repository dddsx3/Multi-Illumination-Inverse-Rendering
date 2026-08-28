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
    ap.add_argument("--num_lights", type=int, default=5,
                    help="每个 DiLiGenT 物体使用的光照数 N")
    ap.add_argument("--num_lights_subsets", type=int, default=3,
                    help="N 子集评估每个物体的随机采样数（中期审计 v2 P1 协议 M=3）")
    ap.add_argument("--n_curve_ns", default="",
                    help="中期审计 v2 P1 N 双轨协议："
                         "逗号分隔 N 值列表如 '1,2,3,5,7,10,15'。"
                         "非空时启用 N 曲线模式，每个 N 用 M=num_lights_subsets 个随机子集评估，"
                         "与单 N 模式互斥。")
    ap.add_argument("--out_dir", default="eval_diligent")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tcfg = ckpt.get("config", {})
    NI = int(tcfg.get("num_lights", 5))
    BC = int(tcfg.get("base_channels", 32))

    results = {}
    n_curve_mode = bool(args.n_curve_ns.strip())
    use_local = (not n_curve_mode)   # N 曲线时跳过残差（local 通道数 N-敏感）
    is_fusion = "aggregator.proj.weight" in ckpt["model_state_dict"]
    if tcfg.get("modality"):
        modality = str(tcfg["modality"])
    elif is_fusion:
        modality = "rgb" if ckpt["model_state_dict"]["stem.net.0.weight"].shape[1] == 3 else "gray"
    else:
        modality = "gray"
    print(f"[arch] modality={modality} is_fusion={is_fusion} use_local_residual={use_local}")

    model = IntrinsicUNet(num_images=NI, base_channels=BC)
    if is_fusion:
        from fusion_unet import FusionUNet
        in_channels = int(ckpt["model_state_dict"]["stem.net.0.weight"].shape[1])
        use_pl_alb = "delta_head.0.weight" in ckpt["model_state_dict"]
        model = FusionUNet(num_images=NI, in_channels=in_channels,
                           base_channels=BC,
                           use_per_light_albedo=use_pl_alb)
        print(f"[auto] FusionUNet in_channels={in_channels} use_pl_alb={use_pl_alb}")
        if n_curve_mode:
            print("[N-curve] 残差 local 通道数不兼容 N=1..15 → 跳过残差")
    renderer = PhysicsRenderer()
    residual = HierarchicalResidual(use_local_residual=use_local,
                                    num_images=NI, feature_channels=BC)
    if not use_local:
        residual.local_net = None   # 禁用 local 残差网络
    model.load_state_dict(ckpt["model_state_dict"])
    renderer.load_state_dict(ckpt["renderer_state_dict"])
    if ckpt.get("residual_state_dict") and use_local:
        residual.load_state_dict(ckpt["residual_state_dict"])
    model.to(device).eval(); renderer.to(device).eval(); residual.to(device).eval()

    results = {}
    # N 曲线模式：每个 N 独立 M 个子集；单 N 模式：等价于 M=1 等距采样
    if n_curve_mode:
        ns_to_eval = [int(x) for x in args.n_curve_ns.split(",") if x.strip()]
        print(f"[N-curve DiLiGenT] ns={ns_to_eval}, M={args.num_lights_subsets}/N")
        curve = {N: {obj: [] for obj in OBJECTS} for N in ns_to_eval}
    else:
        ns_to_eval = [args.num_lights]
        curve = None

    H, W = 512, 612
    crop_l = (W - H) // 2
    import scipy.io as sio
    rng = np.random.default_rng(42)

    for obj in OBJECTS:
        d = os.path.join(args.root, obj)
        li = np.loadtxt(os.path.join(d, "light_intensities.txt"))
        n_total = li.shape[0]
        # 预加载所有 96 张光照图（一次性 IO 优化）
        # modality 决定缓存：rgb 缓存 [96, H, W, 3]，gray 缓存 [96, H, W]
        if modality == "rgb":
            all_imgs = []
            for k in range(n_total):
                img = np.asarray(Image.open(
                    os.path.join(d, f"{k + 1:03d}.png"))).astype(np.float32) / 255.0
                img_n = img / np.clip(li[k].astype(np.float32)[None, None, :],
                                      np.float32(0.05), None)
                img_n = np.clip(img_n[:, crop_l:crop_l + H], 0.0, 1.0)
                all_imgs.append(img_n)
            all_imgs_arr = np.stack(all_imgs, axis=0)   # [96, H, W, 3]
        else:
            all_grays = []
            for k in range(n_total):
                img = np.asarray(Image.open(
                    os.path.join(d, f"{k + 1:03d}.png"))).astype(np.float32) / 255.0
                img_n = img / np.clip(li[k].astype(np.float32)[None, None, :],
                                      np.float32(0.05), None)
                gray = (LUMA[0] * img_n[..., 0] + LUMA[1] * img_n[..., 1]
                        + LUMA[2] * img_n[..., 2])
                gray = np.clip(gray[:, crop_l:crop_l + H], 0.0, 1.0)
                all_grays.append(gray)
            all_imgs_arr = np.stack(all_grays, axis=0)   # [96, H, W]

        n_mat = sio.loadmat(os.path.join(d, "Normal_gt.mat"))["Normal_gt"]
        n_map = np.asarray(n_mat, dtype=np.float32)
        if n_map.shape[0] == 3:
            n_map = n_map.transpose(1, 2, 0)
        n_map = n_map[:, crop_l:crop_l + H, :]
        gt_t = F.interpolate(
            torch.from_numpy(n_map).permute(2, 0, 1)[None],
            size=(256, 256), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).numpy()
        mask = np.linalg.norm(gt_t, axis=-1) > 0.5
        if mask.sum() < 100:
            results[obj] = {"error": "有效前景过少"}
            print(f"[{obj}] 有效前景过少")
            continue

        for N in ns_to_eval:
            if n_curve_mode:
                # 随机子集采样 M 次——不枚举所有 C(n_total, N) 组合
                # （N 较大时 C(96,15)≈4.5e16 直接爆内存）
                subsets_to_eval = []
                seen = set()
                max_tries = args.num_lights_subsets * 20
                tries = 0
                while len(subsets_to_eval) < args.num_lights_subsets and tries < max_tries:
                    sub = tuple(sorted(rng.choice(n_total, size=N, replace=False).tolist()))
                    if sub not in seen:
                        seen.add(sub)
                        subsets_to_eval.append(list(sub))
                    tries += 1
            else:
                # 单 N 等距采样（保持原行为）
                sel = np.unique(np.linspace(0, n_total - 1, N).astype(int))
                while len(sel) < N:
                    sel = np.append(sel, sel[-1] + 1)
                subsets_to_eval = [list(sel)]

            for si, sel in enumerate(subsets_to_eval):
                sel_arr = np.stack([all_imgs_arr[k] for k in sel], axis=0)  # [N, H, W, C] or [N, H, W]
                if modality == "rgb":
                    sel_t = torch.from_numpy(sel_arr).permute(0, 3, 1, 2)  # [N, 3, H, W]
                else:
                    sel_t = torch.from_numpy(sel_arr).unsqueeze(1)         # [N, 1, H, W]
                # 合并 N 与 batch：treat as [N, C, H, W] 直接 interpolate
                sel_t = sel_t.to(device)
                images = F.interpolate(sel_t, size=(256, 256), mode="bilinear",
                                       align_corners=False)
                images = images.unsqueeze(0)   # [1, N, C, 256, 256]
                with torch.no_grad():
                    out = model(images)
                    if len(out) == 6:
                        depth, albedo, sh, wm, feats, _alb_pl = out
                    else:
                        depth, albedo, sh, wm, feats = out
                    rendered, normal, shading = renderer(depth, albedo, sh)
                    final_r, g_res, l_res = residual(albedo, shading, normal, sh,
                                                     stage="stage3", features=feats)
                    normal_p = normal[0].permute(1, 2, 0).float().cpu().numpy()

                st = angular_stats(normal_p, gt_t, mask)
                st["lights_used"] = [int(x) + 1 for x in sel]
                if n_curve_mode:
                    curve[N][obj].append(st)
                    print(f"[{obj} N={N} sub{si}] MAE={st['mae']:.2f} "
                          f"acc11.25={st['acc_11_25']:.3f} lights={st['lights_used'][:3]}...")
                else:
                    results[obj] = st
                    flip_tag = " (FLIPPED)" if st["flipped"] else ""
                    print(f"[{obj}] MAE={st['mae']:.2f} med={st['median']:.2f} "
                          f"acc11.25={st['acc_11_25']:.3f} px={st['pixels']}{flip_tag}")

    if n_curve_mode:
        # 聚合 N 曲线：每个 N 聚合 10 物体 × M 子集
        out_curve = {}
        for N in sorted(curve):
            per_obj_mae = []
            per_obj_med = []
            per_obj_acc = []
            for obj, sts in curve[N].items():
                if not sts:
                    continue
                per_obj_mae.append(np.mean([s["mae"] for s in sts]))
                per_obj_med.append(np.mean([s["median"] for s in sts]))
                per_obj_acc.append(np.mean([s["acc_11_25"] for s in sts]))
            if per_obj_mae:
                out_curve[str(N)] = {
                    "mae_mean": round(float(np.mean(per_obj_mae)), 4),
                    "mae_std":  round(float(np.std(per_obj_mae)), 4),
                    "median_mean": round(float(np.mean(per_obj_med)), 4),
                    "acc_11_25_mean": round(float(np.mean(per_obj_acc)), 4),
                    "n_objects": len(per_obj_mae),
                    "n_subsets": args.num_lights_subsets,
                }
        out_payload = {"n_curve": out_curve, "ns_evaluated": ns_to_eval,
                       "m_per_n": args.num_lights_subsets}
        out_path = os.path.join(args.out_dir, "diligent_n_curve.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=1)
        print(f"\n=== N-curve DiLiGenT ({len(curve)} N values × {args.num_lights_subsets} subsets) ===")
        for N_str, s in out_curve.items():
            print(f"  N={N_str:>3s}: MAE={s['mae_mean']:.2f} ± {s['mae_std']:.2f}  "
                  f"acc11.25={s['acc_11_25_mean']:.3f}")
        print(f"N-curve JSON -> {out_path}")
        return

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
