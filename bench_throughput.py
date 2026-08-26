#!/usr/bin/env python3
"""吞吐标定：测真实 s/batch、峰值显存、GPU 利用率、数据加载占比。

用途：A10（或任意目标机）上跑训练前先标定，让 run_a10.py 能用**实测**
数字做预算规划，而不是拍脑袋估计。也可单独运行做静态体检。

关键产出（--json 写盘）：
  sec_per_batch_train  端到端训练一个 batch 的秒数（含数据）
  sec_per_batch_data   只跑数据加载的秒数/批（判断是否 I/O 受限）
  peak_mem_gb          峰值显存（决定能并行几个车道）
  gpu_util_mean        采样均值（低利用率 => 并行化收益大）
  est_sec_per_epoch    按 batches_per_epoch 折算
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def gpu_query(fields="temperature.gpu,utilization.gpu,memory.used"):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return [float(x) for x in out.split(",")]
    except Exception:
        return None


def build(args, device):
    import torch
    from data_loader import MultiLightingDataset, create_data_loader
    from split_manifest import load_split
    from fusion_unet import FusionUNet
    from unet_model import IntrinsicUNet
    from physics_renderer import PhysicsRenderer
    from residual_modules import HierarchicalResidual

    names = load_split(args.split_manifest, "train")
    ds = MultiLightingDataset(
        root_dir=args.data_root, num_lights=5, image_size=(256, 256),
        is_training=True, scene_subset=names, load_gt=True, modality=args.modality)
    loader = create_data_loader(ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True)
    in_ch = 3 if args.modality == "rgb" else 1
    if args.model == "fusion":
        model = FusionUNet(num_images=5, in_channels=in_ch, base_channels=32)
    else:
        model = IntrinsicUNet(num_images=5, base_channels=32)
    renderer = PhysicsRenderer()
    residual = HierarchicalResidual(use_local_residual=True, num_images=5,
                                    feature_channels=32)
    model.to(device); renderer.to(device); residual.to(device)
    params = list(model.parameters()) + list(residual.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    return loader, model, renderer, residual, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_manifest", default=str(HERE / "splits" / "synthetic_v3.json"))
    ap.add_argument("--model", default="fusion", choices=["fusion", "unet"])
    ap.add_argument("--modality", default="gray", choices=["gray", "rgb"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--batches", type=int, default=12, help="计时批次数（不含预热）")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batches_per_epoch", type=int, default=56)
    ap.add_argument("--temp_abort", type=int, default=0,
                    help=">0 时超过该温度立即中止（低散热平台自保）")
    ap.add_argument("--skip_data_probe", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("[FAIL] 无 CUDA 设备"); sys.exit(2)
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    cc = torch.cuda.get_device_capability(0)
    bf16_ok = torch.cuda.is_bf16_supported()
    print(f"[env] GPU={name} | 显存={total_gb:.1f}GB | 算力={cc[0]}.{cc[1]} "
          f"| BF16={bf16_ok} | torch={torch.__version__} | cpu={os.cpu_count()}")
    if not bf16_ok:
        print("[WARN] 该设备不支持 BF16：训练口径要求 bf16，请勿用 fp16 替代（数值口径不同）")

    loader, model, renderer, residual, opt = build(args, device)
    from loss_functions import CharbonnierLoss
    charb = CharbonnierLoss()

    it = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    utils, temps = [], []

    def step():
        nonlocal it
        try:
            images, gt, _ = next(it)
        except StopIteration:
            it = iter(loader)
            images, gt, _ = next(it)
        images = images.to(device, non_blocking=True)
        target = images if images.dim() == 4 else (
            0.2126 * images[:, :, 0] + 0.7152 * images[:, :, 1] + 0.0722 * images[:, :, 2])
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            out = model(images)
            depth, albedo, sh, wmap, feats = out[:5]
            rendered, normal, shading = renderer(depth, albedo, sh)
            final, _, _ = residual(albedo, shading, normal, sh,
                                   stage="stage3", features=feats)
            loss = charb(final, target)
        loss.backward()
        opt.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    t0 = time.time()
    for i in range(args.batches):
        step()
        s = gpu_query()
        if s:
            temps.append(s[0]); utils.append(s[1])
            if args.temp_abort and s[0] >= args.temp_abort:
                print(f"[ABORT] 温度 {s[0]}C >= {args.temp_abort}C，提前结束标定")
                args.batches = i + 1
                break
    torch.cuda.synchronize()
    train_spb = (time.time() - t0) / max(args.batches, 1)
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    reserved_gb = torch.cuda.max_memory_reserved() / 1024 ** 3

    data_spb = None
    if not args.skip_data_probe:
        it2 = iter(loader)
        for _ in range(2):
            next(it2)
        t1 = time.time()
        nb = min(args.batches, 8)
        for _ in range(nb):
            try:
                next(it2)
            except StopIteration:
                break
        data_spb = (time.time() - t1) / nb

    res = {
        "gpu": name, "total_mem_gb": round(total_gb, 2),
        "compute_capability": f"{cc[0]}.{cc[1]}", "bf16": bf16_ok,
        "torch": torch.__version__, "cpu_count": os.cpu_count(),
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "model": args.model, "modality": args.modality,
        "sec_per_batch_train": round(train_spb, 4),
        "sec_per_batch_data": round(data_spb, 4) if data_spb else None,
        "data_bound_ratio": round(data_spb / train_spb, 3) if data_spb else None,
        "peak_mem_gb": round(peak_gb, 2), "peak_reserved_gb": round(reserved_gb, 2),
        "gpu_util_mean": round(sum(utils) / len(utils), 1) if utils else None,
        "gpu_temp_max": max(temps) if temps else None,
        "batches_per_epoch": args.batches_per_epoch,
        "est_sec_per_epoch": round(train_spb * args.batches_per_epoch, 1),
    }
    print("\n=== 标定结果 ===")
    for k, v in res.items():
        print(f"  {k:22s} {v}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("JSON ->", args.json)
    return res


if __name__ == "__main__":
    main()
