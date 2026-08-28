"""PRE-03 · 统一预算 Probe 训练（不调参竞赛，三 probe 完全同配置）。

预算（固定，写入 HANDOFF）：
  数据：synthetic_v3 train 447 场景（split hash 见 protocol），linear 域，256²
  batch=8 场景 × 5 光；epoch=56 iters；epochs=40；Adam lr=2e-4；
  loss = w_recon·Charbonnier(recon, img_lin) + 0.5·L1(albedo, GT)（mask）
        + 0.5·L1(depth, GT)（mask）+ 0.2·L2(sh, GT sh)
  渲染：PhysicsRenderer 语义 I = A ⊙ ReLU(SH(n(depth), c))，
        DepthToNormal(use_edge_aware=False)（与数据定义对齐）
  模型选择：val 49 场景 SI-MAE(albedo) 最低的 epoch 存 best；
  测试：test 124 场景一次评估（best 与 last 都报告）。

用法（repo 根目录）:
  python pre0/source/train/train_probe.py --probe A
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "probe_models")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset")))

from probes import PROBES, count_params  # noqa: E402
from scene_loader import load_scene, list_scenes, scenes_with_files  # noqa: E402

DATA_ROOT = "D:/data/synthetic_v3"
SEED = 20260829
BATCH = 8
EPOCHS = 40
LR = 2e-4
W_RECON, W_ALB, W_DEPTH, W_SH = 1.0, 0.5, 0.5, 0.2
NUM_LIGHTS = 5


def charbonnier(pred, target):
    d = pred - target
    return torch.sqrt(d * d + 1e-6).mean()


class SceneBatcher:
    """简单场景级 batch（无增强——PRE-0 保持最小协议）"""

    def __init__(self, scene_dirs, batch, shuffle, device):
        self.dirs = scene_dirs
        self.batch = batch
        self.shuffle = shuffle
        self.device = device

    def __iter__(self):
        order = np.random.permutation(len(self.dirs)) if self.shuffle else np.arange(len(self.dirs))
        for i in range(0, len(order) - len(order) % self.batch, self.batch):
            scenes = [load_scene(self.dirs[j]) for j in order[i:i + self.batch]]
            imgs = np.stack([s["img_lin"] for s in scenes])[:, :, None]  # [B,K,1,H,W]
            albedo = np.stack([s["albedo"] for s in scenes])         # [B,1,H,W]
            depth = np.stack([s["depth"] for s in scenes])           # [B,1,H,W]
            mask = np.stack([s["mask"] for s in scenes]).astype(np.float32)
            sh = np.stack([s["sh"] for s in scenes])                 # [B,K,9]
            normal = np.stack([s["normal"] for s in scenes])         # [B,3,H,W]
            to = lambda x, dt: torch.from_numpy(x).to(self.device, dt)
            yield (to(imgs, torch.float32), to(albedo, torch.float32),
                   to(depth, torch.float32), to(mask, torch.float32),
                   to(sh, torch.float32), to(normal, torch.float32))

    def __len__(self):
        return len(self.dirs) // self.batch


def depth_to_normal(depth, mask):
    """复刻 render_dataset.sobel_normal / physics_renderer(use_edge_aware=False)。

    depth [B,1,H,W] -> normal [B,3,H,W]；背景置 (0,0,1)。
    """
    p = F.pad(depth, (1, 1, 1, 1), mode="replicate")
    gx = (-p[:, :, :-2, :-2] + p[:, :, 2:, :-2]
          - 2 * p[:, :, :-2, 1:-1] + 2 * p[:, :, 2:, 1:-1]
          - p[:, :, :-2, 2:] + p[:, :, 2:, 2:]) / 4.0
    gy = (p[:, :, :-2, :-2] + 2 * p[:, :, 1:-1, :-2] + p[:, :, 2:, :-2]
          - p[:, :, :-2, 2:] - 2 * p[:, :, 1:-1, 2:] - p[:, :, 2:, 2:]) / 4.0
    n = torch.cat([-gx, -gy, torch.ones_like(gx)], dim=1)
    n = F.normalize(n, dim=1)
    bg = (mask[:, 0:1] < 0.5).expand(-1, 3, -1, -1)
    ones = torch.zeros_like(n)
    ones[:, 2] = 1.0
    n = torch.where(bg, ones, n)
    return n


def sh_shading(normal, sh):
    """与 physics_renderer.SphericalHarmonicsLighting 完全一致（含 ReLU）"""
    C0, C1 = 0.282095, 0.488603
    C2 = [1.092548, 1.092548, 0.315392, 1.092548, 0.546274]
    x, y, z = normal[:, 0:1], normal[:, 1:2], normal[:, 2:3]
    B = torch.cat([
        C0 * torch.ones_like(x), C1 * y, C1 * z, C1 * x,
        C2[0] * x * y, C2[1] * y * z, C2[2] * (3 * z * z - 1),
        C2[3] * x * z, C2[4] * (x * x - y * y)], dim=1)          # [B,9,H,W]
    # sh: [B,K,9] -> shading [B,K,H,W]
    K = sh.shape[1]
    out = torch.einsum("bihw,bki->bkhw", B, sh)
    return F.relu(out)


def si_mae_torch(pred, gt, mask):
    """pred/gt [B,1,H,W]；返回逐场景 SI-MAE 列表"""
    out = []
    for b in range(pred.shape[0]):
        p = pred[b][0][mask[b, 0] > 0.5]
        g = gt[b][0][mask[b, 0] > 0.5]
        denom = (p * p).sum()
        if denom < 1e-8:
            out.append(float("nan"))
            continue
        s = (p * g).sum() / denom
        out.append(float((s * p - g).abs().mean()))
    return out


@torch.no_grad()
def evaluate(model, batcher, device, max_scenes=0):
    model.eval()
    dirs = batcher.dirs[: max_scenes] if max_scenes else batcher.dirs
    single = SceneBatcher(dirs, 1, False, device)
    si, dl1, sherr, psnrs, nang = [], [], [], [], []
    for imgs, albedo, depth, mask, sh, normal in single:
        depth_p, albedo_p, sh_p = model(imgs)
        n_p = depth_to_normal(depth_p, mask)
        shading = sh_shading(n_p, sh_p)
        recon = albedo_p * shading
        m = mask > 0.5
        si += si_mae_torch(albedo_p, albedo, mask)
        dl1.append(float((depth_p - depth).abs()[m].mean()))
        sherr.append(float(((sh_p - sh) ** 2).mean() ** 0.5))
        mse = float((((recon - imgs) * mask) ** 2).sum() / mask.sum())
        psnrs.append(10 * math.log10(1.0 / max(mse, 1e-12)))
        dot = (n_p * normal).sum(1).clamp(-1, 1)[m[:, 0]]
        nang.append(float(torch.rad2deg(torch.acos(dot)).mean()))
    model.train()
    return dict(si_mae=float(np.nanmean(si)), depth_l1=float(np.mean(dl1)),
                sh_rmse=float(np.mean(sherr)), recon_psnr=float(np.mean(psnrs)),
                normal_ang=float(np.mean(nang)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=["A", "B", "C"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    manifest = os.path.join(_REPO, "pre0", "protocol", "split_manifest.json")
    tr = scenes_with_files(DATA_ROOT, list_scenes(manifest, "train"))
    va = scenes_with_files(DATA_ROOT, list_scenes(manifest, "val"))
    te = scenes_with_files(DATA_ROOT, list_scenes(manifest, "test"))

    model = PROBES[args.probe]().to(device)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"[probe{args.probe}] params={n_params/1e6:.2f}M train={len(tr)} val={len(va)} test={len(te)}")

    out_ck = os.path.join(_REPO, "pre0", "checkpoints")
    out_log = os.path.join(_REPO, "pre0", "logs")
    os.makedirs(out_ck, exist_ok=True)
    os.makedirs(out_log, exist_ok=True)
    log_rows = []
    best_si = float("inf")
    t0 = time.time()

    train_b = SceneBatcher(tr, args.batch, True, device)
    val_b = SceneBatcher(va, 1, False, device)
    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for imgs, albedo, depth, mask, sh, normal in train_b:
            opt.zero_grad()
            depth_p, albedo_p, sh_p = model(imgs)
            n_p = depth_to_normal(depth_p, mask)
            shading = sh_shading(n_p, sh_p)
            recon = albedo_p * shading                 # [B,K,H,W]
            imgs_k = imgs[:, :, 0]                     # [B,K,H,W]
            m = mask
            l_recon = charbonnier(recon * m, imgs_k * m)
            l_alb = (albedo_p - albedo).abs()[m > 0.5].mean()
            l_dep = (depth_p - depth).abs()[m > 0.5].mean()
            l_sh = F.mse_loss(sh_p, sh)
            loss = W_RECON * l_recon + W_ALB * l_alb + W_DEPTH * l_dep + W_SH * l_sh
            loss.backward()
            opt.step()
            ep_loss += float(loss)
            nb += 1
        met = evaluate(model, val_b, device)
        row = dict(probe=args.probe, epoch=ep, loss=ep_loss / max(nb, 1),
                   seconds=round(time.time() - t0, 1), **met)
        log_rows.append(row)
        print(f"[probe{args.probe}] ep{ep:02d} loss={row['loss']:.4f} "
              f"val si_mae={met['si_mae']:.4f} depth={met['depth_l1']:.4f} "
              f"normal={met['normal_ang']:.2f} psnr={met['recon_psnr']:.2f} "
              f"t={row['seconds']:.0f}s")
        torch.save({"model": model.state_dict(), "epoch": ep, "params": n_params},
                   os.path.join(out_ck, f"probe_{args.probe}_last.pth"))
        if met["si_mae"] < best_si:
            best_si = met["si_mae"]
            torch.save({"model": model.state_dict(), "epoch": ep, "params": n_params,
                        "val_si_mae": best_si},
                       os.path.join(out_ck, f"probe_{args.probe}_best.pth"))
        with open(os.path.join(out_log, f"probe_{args.probe}_trainlog.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader(); w.writerows(log_rows)

    # test 一次评估（best）
    ck = torch.load(os.path.join(out_ck, f"probe_{args.probe}_best.pth"),
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    test_b = SceneBatcher(te, 1, False, device)
    met_test = evaluate(model, test_b, device)
    print(f"[probe{args.probe}] TEST(best ep{ck.get('epoch')}) "
          f"si_mae={met_test['si_mae']:.4f} depth={met_test['depth_l1']:.4f} "
          f"normal={met_test['normal_ang']:.2f} psnr={met_test['recon_psnr']:.2f}")
    with open(os.path.join(_REPO, "pre0", "probe_results",
                           f"probe_{args.probe}_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(probe=args.probe, params=n_params,
                       best_epoch=ck.get("epoch"), best_val_si_mae=best_si,
                       test=met_test), f, indent=2)


if __name__ == "__main__":
    main()
