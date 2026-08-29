"""P1-15 最小 Probe 重训（variable-N sampling）。

相对 PRE-03 关键改进（per P1-15 任务书）：
  - **训练时 variable-N 采样**：N ~ Uniform{3..15}（per batch）
  - **同时训 FixedN=5 baseline**（对照组）
  - 统一预算（同 PRE-03，~0.71M 参数 / 40 epoch / linear 域 / edge_aware=False）
  - 数据：P1 协议 dataset（P1-04 输出格式）
  - 输出：probe_{varN,fixed5}_best.pth + summary

用法：
  python p1/source/probes/train_probe_p1.py --probe A --mode varN
  python p1/source/probes/train_probe_p1.py --probe A --mode fixed5
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "physics")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pre0", "source", "probe_models")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pre0", "source", "dataset")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pre0", "source", "train")))

from pre0.source.probe_models.probes import PROBES, count_params  # noqa: E402
from pre0.source.train.train_probe import SceneBatcher, depth_to_normal, sh_shading  # noqa: E402

DATA_ROOT = "D:/data/synthetic_v4"   # 实际是 P1 协议 dataset
SEED = 20260830
BATCH = 8
EPOCHS = 40
LR = 2e-4


def si_mae_torch(pred, gt, mask):
    out = []
    for b in range(pred.shape[0]):
        p = pred[b][0][mask[b, 0] > 0.5]
        g = gt[b][0][mask[b, 0] > 0.5]
        d = (p * p).sum()
        if d < 1e-8:
            out.append(float("nan"))
            continue
        s = (p * g).sum() / d
        out.append(float((s * p - g).abs().mean()))
    return out


@torch.no_grad()
def evaluate(model, batcher, device):
    model.eval()
    single = SceneBatcher(batcher.dirs, 1, False, device)
    si, dl1, sherr, nang, psnrs = [], [], [], [], []
    for imgs, albedo, depth, mask, sh, normal in single:
        d, a, sh_p = model(imgs)
        n = depth_to_normal(d, mask)
        s = sh_shading(n, sh_p)
        rec = a * s
        imgs_k = imgs[:, :, 0]
        m = mask > 0.5
        si += si_mae_torch(a, albedo, mask)
        dl1.append(float((d - depth).abs()[m].mean()))
        sherr.append(float(((sh_p - sh) ** 2).mean() ** 0.5))
        dot = (n * normal).sum(1).clamp(-1, 1)[m[:, 0]]
        nang.append(float(torch.rad2deg(torch.acos(dot)).mean()))
        mse = float((((rec - imgs_k) * mask) ** 2).sum() / mask.sum())
        psnrs.append(10 * math.log10(1 / max(mse, 1e-12)))
    model.train()
    return dict(si_mae=float(np.nanmean(si)),
                depth_l1=float(np.mean(dl1)),
                sh_rmse=float(np.mean(sherr)),
                normal_ang=float(np.mean(nang)),
                recon_psnr=float(np.mean(psnrs)))


def make_subset_sampler(N, K, seed):
    """返回每 epoch 重生成的 3..15 随机子集（per scene 一次）"""
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(K, N, replace=False).tolist()) for _ in range(1000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=["A", "B", "C"])
    ap.add_argument("--mode", required=True, choices=["varN", "fixed5"])
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    manifest = os.path.join(_REPO, "p1", "protocol", "split_manifest.json")
    if not os.path.isfile(manifest):
        # 退化：尝试 PRE-0 manifest（保证脚本可执行）
        manifest = os.path.join(_REPO, "pre0", "protocol", "split_manifest.json")
    import json
    m = json.load(open(manifest, encoding="utf-8"))
    splits = m.get("split", m)
    train_ids = splits.get("train", splits.get("train", []))
    val_ids = splits.get("val", splits.get("val", []))
    test_ids = splits.get("test", splits.get("test", []))
    from pre0.source.dataset.scene_loader import scenes_with_files
    tr = scenes_with_files(args.data_root, train_ids)
    va = scenes_with_files(args.data_root, val_ids)
    te = scenes_with_files(args.data_root, test_ids)
    if not tr:
        print("无训练数据；需先在 P1-04 协议下生成数据。")
        return

    model = PROBES[args.probe]().to(device)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    out_ck = os.path.join(_REPO, "p1", "checkpoints")
    out_log = os.path.join(_REPO, "p1", "logs")
    out_res = os.path.join(_REPO, "p1", "probes")
    os.makedirs(out_ck, exist_ok=True); os.makedirs(out_log, exist_ok=True); os.makedirs(out_res, exist_ok=True)
    os.makedirs(os.path.join(out_res, args.probe), exist_ok=True)
    log_rows = []
    best_si = float("inf")
    t0 = time.time()
    train_b = SceneBatcher(tr, args.batch, True, device)
    val_b = SceneBatcher(va, 1, False, device)
    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for imgs, albedo, depth, mask, sh, normal in train_b:
            # 训练子集选择：mode=varN → 随机 N∈[3,15]；mode=fixed5 → 固定前 5
            K = imgs.shape[1]
            if args.mode == "varN":
                N = np.random.default_rng(SEED + ep * 1000 + nb).integers(3, 16)
                N = min(N, K)
                sub = sorted(np.random.default_rng(SEED + ep * 1000 + nb + 1)
                              .choice(K, N, replace=False).tolist())
            else:
                sub = list(range(min(5, K)))
            imgs_sub = imgs[:, sub]
            sh_sub = sh[:, sub]
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                d, a, sh_p = model(imgs_sub)
                n = depth_to_normal(d.float(), mask)
                shading = sh_shading(n, sh_p.float())
                recon = a.float() * shading
                imgs_k = imgs_sub[:, :, 0]
                m = mask
                l_recon = F.l1_loss(recon * m, imgs_k * m)
                l_alb = (a.float() - albedo).abs()[m > 0.5].mean()
                l_dep = (d.float() - depth).abs()[m > 0.5].mean()
                l_sh = F.mse_loss(sh_p.float(), sh_sub)
                loss = l_recon + 0.5 * l_alb + 0.5 * l_dep + 0.2 * l_sh
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach()); nb += 1
        if ep % 2 == 0 or ep == args.epochs:
            met = evaluate(model, val_b, device)
            row = dict(probe=args.probe, mode=args.mode, epoch=ep,
                       loss=ep_loss / max(nb, 1), t=round(time.time() - t0, 1), **met)
            log_rows.append(row)
            print(f"[probe{args.probe}/{args.mode}] ep{ep:02d} loss={row['loss']:.4f} "
                  f"si={met['si_mae']:.4f} normal={met['normal_ang']:.2f}° "
                  f"psnr={met['recon_psnr']:.2f}")
            torch.save({"model": model.state_dict(), "epoch": ep, "params": n_params},
                       os.path.join(out_ck, f"probe_{args.probe}_{args.mode}_last.pth"))
            if met["si_mae"] < best_si:
                best_si = met["si_mae"]
                torch.save({"model": model.state_dict(), "epoch": ep, "params": n_params,
                            "val_si_mae": best_si},
                           os.path.join(out_ck, f"probe_{args.probe}_{args.mode}_best.pth"))
            with open(os.path.join(out_log,
                                   f"probe_{args.probe}_{args.mode}_trainlog.csv"),
                      "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
                w.writeheader(); w.writerows(log_rows)
    # test 一次评估
    ck = torch.load(os.path.join(out_ck, f"probe_{args.probe}_{args.mode}_best.pth"),
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    test_b = SceneBatcher(te, 1, False, device)
    met = evaluate(model, test_b, device)
    out = dict(probe=args.probe, mode=args.mode, params=n_params,
               best_epoch=ck.get("epoch"), best_val_si_mae=best_si, test=met)
    json.dump(out, open(os.path.join(out_res, args.probe, f"{args.mode}_summary.json"),
                        "w", encoding="utf-8"), indent=2)
    print(f"[probe{args.probe}/{args.mode}] TEST: si={met['si_mae']:.4f} "
          f"normal={met['normal_ang']:.2f}° psnr={met['recon_psnr']:.2f}")


if __name__ == "__main__":
    main()
