"""PRE-04 · Evidence Accumulation 诊断协议（对三个 Probe 用完全相同的评价）+
PRE-05 · Held-out illumination（oracle-query-light 主协议）。

A. N curve           N∈{1,2,3,5}（真实 5 光），N=1 仅作 stress test，主分析 N≥3
B. Cross-subset      同场景两个随机 3 光子集 S_a/S_b -> 输出差异 D_A/D_n + 与 GT 误差
C. Novel-light gain  S={1,2,3} + 新光(4或5) vs + 重复光(1)，Δ_new vs Δ_dup
D. Permutation test  固定集合改顺序 -> 输出差异（纯数值误差量级）
E. Fusion sensitivity ||∂y/∂F_k|| 分布（y=canonical albedo 总和通道），比较 N 与模型
PRE-05 oracle-query-light: support S 估 (A,n)，query 光 L_q^GT 由评估器提供，
      Î_q = A ⊙ ReLU(SH(n, L_q))；HO-PSNR/HO-SSIM/HO-MAE；residual 全关。
      predicted-query-light 只允许单独列，禁止混入主指标。

固定子集（跨 probe 共用）：SUBSETS 与 pre02 相同（real5 规则），随机子集
用 seed=20260829 生成一次并落盘 subsets_used.json。

用法（repo 根目录）:
  python pre0/source/evaluate/pre04.py --probes A B C
输出: pre0/evidence_accumulation/{ncurve_*.csv, cross_subset.csv,
  novel_vs_duplicate_probe.csv, permutation_test.csv, fusion_sensitivity.csv}
      pre0/heldout_relighting/{heldout_*.csv, heldout_summary.json}
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "probe_models")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train")))

from probes import PROBES  # noqa: E402
from scene_loader import load_scene, list_scenes, scenes_with_files  # noqa: E402
from train_probe import depth_to_normal, sh_shading  # noqa: E402

DATA_ROOT = "D:/data/synthetic_v3"
SEED = 20260829
SUBSETS = {1: [0], 2: [0, 2], 3: [0, 2, 4], 5: [0, 1, 2, 3, 4]}


def si_mae_np(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    d = (p * p).sum()
    if d < 1e-12:
        return float("nan")
    s = (p * g).sum() / d
    return float(np.abs(s * p - g).mean())


def load_model(probe, device):
    ck_path = os.path.join(_REPO, "pre0", "checkpoints", f"probe_{probe}_best.pth")
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model = PROBES[probe]().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck.get("epoch")


@torch.no_grad()
def predict(model, sc, subset, device):
    """给定场景与光照子集 -> (albedo [H,W], depth [H,W], normal [3,H,W], sh_hat [N,9])"""
    imgs = torch.from_numpy(sc["img_lin"][subset][:, None]).float().to(device)  # [N,1,H,W]
    depth, albedo, sh = model(imgs[None])                                        # [1,N,...]
    depth = depth[0, 0].cpu().numpy()
    albedo = albedo[0, 0].cpu().numpy()
    sh = sh[0].cpu().numpy()
    mask = sc["mask"][0:1].astype(np.float32)
    n = depth_to_normal(torch.from_numpy(depth[None, None]).float().to(device),
                        torch.from_numpy(mask[None]).float().to(device))[0].cpu().numpy()
    return albedo, depth, n, sh


@torch.no_grad()
def predict_grad(model, sc, subset, device):
    """同 predict 但保留 autograd（Fusion sensitivity 用）"""
    imgs = torch.from_numpy(sc["img_lin"][subset][:, None]).float().to(device).requires_grad_(True)
    depth, albedo, sh = model(imgs[None])
    return imgs, depth, albedo, sh


def normal_ang(n_hat, n_gt, mask):
    d = np.clip((n_hat * n_gt).sum(0), -1, 1)
    ang = np.degrees(np.arccos(d))
    return ang[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data_root", default=DATA_ROOT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_ev = os.path.join(_REPO, "pre0", "evidence_accumulation")
    out_ho = os.path.join(_REPO, "pre0", "heldout_relighting")
    os.makedirs(out_ev, exist_ok=True)
    os.makedirs(out_ho, exist_ok=True)

    manifest = os.path.join(_REPO, "pre0", "protocol", "split_manifest.json")
    ids = list_scenes(manifest, args.split)
    scene_dirs = scenes_with_files(args.data_root, ids)
    if args.limit:
        scene_dirs = scene_dirs[: args.limit]
    rng = np.random.default_rng(SEED)
    json.dump({"subsets_real5": {str(k): v for k, v in SUBSETS.items()},
               "random_seed": SEED},
              open(os.path.join(out_ev, "subsets_used.json"), "w"))

    rows_nc, rows_cs, rows_nvd, rows_perm, rows_fus, rows_ho = [], [], [], [], [], []

    for probe in args.probes:
        model, best_ep = load_model(probe, device)
        print(f"[pre04] probe {probe} (best ep{best_ep}) scenes={len(scene_dirs)}")
        for si, sd in enumerate(scene_dirs):
            sc = load_scene(sd)
            mask = sc["mask_bool"]
            A_gt = sc["albedo"][0]
            n_gt = sc["normal"]

            # ---- A. N curve（albedo/depth/normal/lighting 误差）----
            for N, sub in SUBSETS.items():
                albedo, depth, n_hat, sh_hat = predict(model, sc, sub, device)
                # lighting 误差：对每光与 GT SH 做 shape 对比（去掉全局尺度歧义）
                sh_gt = sc["sh"][sub]
                scale = (sh_hat * sh_gt).sum() / max((sh_hat * sh_hat).sum(), 1e-9)
                sh_err = float(np.linalg.norm(scale * sh_hat - sh_gt))
                rows_nc.append(dict(probe=probe, scene=sc["scene"], N=N,
                                    si_mae_A=si_mae_np(albedo, A_gt, mask),
                                    depth_l1=float(np.abs(depth - sc["depth"][0])[mask].mean()),
                                    normal_mae=float(normal_ang(n_hat, n_gt, mask).mean()),
                                    sh_err=sh_err))
                # ---- PRE-05 oracle-query-light（support=sub，query=不在 sub 的光）----
                q = [k for k in range(5) if k not in sub]
                if q:
                    qk = q[0]
                    # 用 GT 光渲染拟合法线：Î_q = albedo ⊙ ReLU(SH(n_hat, L_q^GT))
                    n_t = torch.from_numpy(n_hat[None]).float().to(device)
                    shq = torch.from_numpy(sc["sh"][qk][None, None]).float().to(device)
                    s_q = sh_shading(n_t, shq)[0, 0].cpu().numpy()
                    ih = albedo * s_q
                    iq = sc["img_lin"][qk]
                    mse = float(((ih - iq)[mask] ** 2).mean())
                    rows_ho.append(dict(probe=probe, scene=sc["scene"], N=N,
                                        q_light=qk + 1,
                                        ho_psnr=10 * math.log10(1 / max(mse, 1e-12)),
                                        ho_mae=float(np.abs(ih - iq)[mask].mean())))

            # ---- B. Cross-subset consistency（两个随机 3 光子集）----
            subs = []
            for _ in range(2):
                subs.append(sorted(rng.choice(5, 3, replace=False).tolist()))
            outs = []
            for sub in subs:
                albedo, depth, n_hat, sh_hat = predict(model, sc, sub, device)
                outs.append((albedo, n_hat))
            d_A = float(np.abs(outs[0][0] - outs[1][0])[mask].mean())
            dot = np.clip((outs[0][1] * outs[1][1]).sum(0), -1, 1)
            d_n = float(np.degrees(np.arccos(dot))[mask].mean())
            rows_cs.append(dict(probe=probe, scene=sc["scene"],
                                S_a=",".join(map(str, subs[0])),
                                S_b=",".join(map(str, subs[1])),
                                D_albedo=d_A, D_normal_deg=d_n,
                                si_mae_a=si_mae_np(outs[0][0], A_gt, mask),
                                si_mae_b=si_mae_np(outs[1][0], A_gt, mask)))

            # ---- C. Novel vs duplicate（probe 版）----
            S3 = [0, 2, 4]
            def err_for(sub):
                albedo, _, _, _ = predict(model, sc, sub, device)
                return si_mae_np(albedo, A_gt, mask)
            e3 = err_for(S3)
            en = err_for(S3 + [1])
            ed = err_for(S3 + [0])
            rows_nvd.append(dict(probe=probe, scene=sc["scene"], E_S3=e3,
                                 E_S3_new=en, E_S3_dup=ed,
                                 d_new=e3 - en, d_dup=e3 - ed))

            # ---- D. Permutation test（N=5 全序 vs 两个随机置换）----
            albedo0, _, _, _ = predict(model, sc, SUBSETS[5], device)
            perms = [list(rng.permutation(5)), list(rng.permutation(5))]
            dmax = 0.0
            for pm in perms:
                albedo_p, _, _, _ = predict(model, sc, pm, device)
                dmax = max(dmax, float(np.abs(albedo0 - albedo_p)[mask].max()))
            rows_perm.append(dict(probe=probe, scene=sc["scene"],
                                  max_diff_albedo=dmax))

            # ---- E. Fusion sensitivity（对第 k 图输入扰动的响应）----
            # ||∂(sum albedo)/∂img_k|| / ||∂(sum albedo)/∂img_total|| 近似：
            # 对每张输入图计算梯度范数（autograd on albedo sum）
            grads = []
            imgs_t = torch.from_numpy(sc["img_lin"]).float().to(device)
            for k in range(5):
                im = imgs_t.clone().requires_grad_(True)
                inp = torch.stack([im[k] if j == k else im[j].detach()
                                   for j in range(5)])[:, None]
                _, albedo_o, _ = model(inp[None])
                albedo_o.sum().backward()
                grads.append(float(im.grad.abs().mean()))
            rows_fus.append(dict(probe=probe, scene=sc["scene"], N=5,
                                 g0=grads[0], g1=grads[1], g2=grads[2],
                                 g3=grads[3], g4=grads[4],
                                 g_min_over_max=float(min(grads) / max(max(grads), 1e-12))))

            if (si + 1) % 30 == 0:
                print(f"  {si+1}/{len(scene_dirs)}")

    def dump(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    dump(os.path.join(out_ev, "ncurve_all_probes.csv"), rows_nc)
    dump(os.path.join(out_ev, "cross_subset.csv"), rows_cs)
    dump(os.path.join(out_ev, "novel_vs_duplicate_probe.csv"), rows_nvd)
    dump(os.path.join(out_ev, "permutation_test.csv"), rows_perm)
    dump(os.path.join(out_ev, "fusion_sensitivity.csv"), rows_fus)
    dump(os.path.join(out_ho, "heldout_oracle_query_light.csv"), rows_ho)

    # 汇总 json
    summ = {}
    for probe in args.probes:
        rs = [r for r in rows_ho if r["probe"] == probe]
        summ[probe] = {str(N): dict(
            ho_psnr=float(np.mean([r["ho_psnr"] for r in rs if r["N"] == N])),
            ho_mae=float(np.mean([r["ho_mae"] for r in rs if r["N"] == N])))
            for N in SUBSETS if any(r["N"] == N for r in rs)}
    json.dump(summ, open(os.path.join(out_ho, "heldout_summary.json"), "w",
                         encoding="utf-8"), indent=2)
    print("[pre04/05] done")


if __name__ == "__main__":
    main()
