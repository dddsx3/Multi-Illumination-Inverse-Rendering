"""P1-16 · 新的 Learnability Gate C1-C5。

对已训练的 P1 Probe（varN / fixed5）做 5 项 Gate 评估：

  C1 · Factor N curve：N={3,5,8,15,24} 报告 normal/albedo/lighting 误差
  C2 · Novel vs duplicate：Δ_new > Δ_dup（cardinality-control）
  C3 · Diversity effect：固定 N 下 high-diversity 子集应优于 low-diversity
  C4 · Cross-subset consistency：D_A(S_a,S_b) / D_n 随 N 增大应下降
  C5 · Oracle-query-light held-out：A ⊙ ReLU(Σ c Y(n), L_q^GT) vs query image
        （residual 全关；physics-only）

用法：
  python p1/source/evaluation/learnability_gate.py --probes A varN A fixed5
输出：
  p1/heldout/{C1_ncurve.csv, C2_cardinality.csv, C3_diversity.csv,
              C4_cross_subset.csv, C5_heldout.csv, gate_summary.json, REPORT.md}
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "physics")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pre0", "source", "probe_models")))

from probes import PROBES  # noqa: E402
from pre0.source.train.train_probe import SceneBatcher, depth_to_normal, sh_shading  # noqa: E402
from pre0.source.dataset.scene_loader import list_scenes, scenes_with_files  # noqa: E402
from p1.source.physics.sh import sh_basis_npy  # noqa: E402

NS = [3, 5, 8, 15, 24]


def load_probe(probe, mode, device):
    ck_path = os.path.join(_REPO, "p1", "checkpoints", f"probe_{probe}_{mode}_best.pth")
    if not os.path.isfile(ck_path):
        return None
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    m = PROBES[probe]().to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m


@torch.no_grad()
def predict(model, sc, subset, device):
    imgs = torch.from_numpy(sc["img_lin"][subset][:, None]).float().to(device)
    d, a, sh = model(imgs[None])
    mask = sc["mask"][0:1].astype(np.float32)
    n = depth_to_normal(d, mask)
    return (a[0,0].cpu().numpy(), d[0,0].cpu().numpy(),
            n[0].cpu().numpy(), sh[0].cpu().numpy())


def si_mae_np(p, g, m):
    p_, g_ = p[m], g[m]
    d = (p_*p_).sum()
    if d < 1e-12: return float("nan")
    s = (p_*g_).sum() / d
    return float(np.abs(s*p_ - g_).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="+", default=["A varN", "A fixed5"])
    ap.add_argument("--data_root", default="D:/data/synthetic_v4")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(_REPO, "p1", "heldout"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_models = {}
    for tag in args.probes:
        p, m = tag.split()
        mdl = load_probe(p, m, device)
        if mdl is None:
            print(f"[skip] {tag} no checkpoint")
            continue
        probe_models[tag] = mdl
    if not probe_models:
        print("无可用 probe checkpoint；先跑 p1/source/probes/train_probe_p1.py")
        return
    manifest = os.path.join(_REPO, "p1", "protocol", "split_manifest.json")
    if not os.path.isfile(manifest):
        manifest = os.path.join(_REPO, "pre0", "protocol", "split_manifest.json")
    import json
    m = json.load(open(manifest, encoding="utf-8"))
    scenes = scenes_with_files(args.data_root, m.get("split", m).get(args.split, []))
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    print(f"[learnability] scenes={len(scenes)} probes={list(probe_models.keys())}")

    rows_c1, rows_c2, rows_c3, rows_c4, rows_c5 = [], [], [], [], []
    from pre0.source.renderer.oracle import recover_light_dir, make_sphere_grid
    grid = make_sphere_grid()
    for tag, model in probe_models.items():
        probe, mode = tag.split()
        for sd in scenes:
            sc = np.load if False else None
            import os as _os
            from p1.source.calibration.oracle_gate import load_scene as load_p1
            sc = load_p1(sd)
            if sc["imgs_lin"].shape[0] < max(NS):
                continue
            mask = sc["mask"]
            A_gt = sc["albedo"]
            n_gt = sc["n_mesh"]
            light_dirs = np.stack([recover_light_dir(sc["sh_irr"][k], grid)[0]
                                    for k in range(sc["imgs_lin"].shape[0])])

            # ---- C1 N curve ----
            for N in NS:
                sub = list(range(N))
                alb, dep, n_hat, sh_hat = predict(model, sc, sub, device)
                rows_c1.append(dict(probe=tag, scene=sc["name"], N=N,
                                    si_mae=si_mae_np(alb, A_gt, mask),
                                    depth_l1=float(np.abs(dep - sc["depth"][0])[mask].mean()),
                                    sh_err=float(np.linalg.norm(
                                        sh_hat - sc["sh_irr"][sub]))))

            # ---- C2 Novel vs duplicate（cardinality-control）----
            S3 = list(range(3))
            out_set = [i for i in range(sc["imgs_lin"].shape[0]) if i not in S3]
            if out_set:
                sub_new = S3 + [out_set[0]]
                sub_dup = S3 + [S3[0]]
                e3 = si_mae_np(predict(model, sc, S3, device)[0], A_gt, mask)
                e_new = si_mae_np(predict(model, sc, sub_new, device)[0], A_gt, mask)
                e_dup = si_mae_np(predict(model, sc, sub_dup, device)[0], A_gt, mask)
                rows_c2.append(dict(probe=tag, scene=sc["name"],
                                    E_S3=e3, E_S3_new=e_new, E_S3_dup=e_dup,
                                    d_new=e3 - e_new, d_dup=e3 - e_dup))

            # ---- C3 Diversity effect（固定 N=5）----
            # 构造两个 5-光子集：diverse = 沿半球均匀 5 个；near = 与 fib_dirs 0 接近的 5 个
            center = light_dirs[0]
            ang_to_center = np.array([math.degrees(math.acos(np.clip(d @ center, -1, 1)))
                                       for d in light_dirs])
            near_idx = np.argsort(ang_to_center)[:5]
            # diverse：每 30° 取最远方向
            order = np.argsort(-ang_to_center)   # 从最远离 center 开始
            diverse_idx = []
            cur = center
            for cand in order:
                if len(diverse_idx) == 5: break
                d = light_dirs[cand]
                if not diverse_idx or all(math.degrees(math.acos(np.clip(d @ light_dirs[k], -1, 1))) > 30
                                        for k in diverse_idx):
                    diverse_idx.append(cand)
            sub_div = sorted(diverse_idx)
            sub_near = sorted(near_idx.tolist())
            e_div = si_mae_np(predict(model, sc, sub_div, device)[0], A_gt, mask)
            e_near = si_mae_np(predict(model, sc, sub_near, device)[0], A_gt, mask)
            rows_c3.append(dict(probe=tag, scene=sc["name"],
                                E_high_div=e_div, E_low_div=e_near,
                                diff=e_near - e_div,   # 正 = high-diversity 更好
                                div_spread=ang_to_center[sub_div].mean(),
                                near_spread=ang_to_center[sub_near].mean()))

            # ---- C4 Cross-subset consistency ----
            rng = np.random.default_rng(0)
            subs = [sorted(rng.choice(sc["imgs_lin"].shape[0], 3, replace=False).tolist())
                    for _ in range(2)]
            outs = []
            for sub in subs:
                alb, dep, n_hat, sh_hat = predict(model, sc, sub, device)
                outs.append((alb, n_hat))
            d_A = float(np.abs(outs[0][0] - outs[1][0])[mask].mean())
            dot = np.clip((outs[0][1] * outs[1][1]).sum(0), -1, 1)
            d_n = float(np.degrees(np.arccos(dot))[mask].mean())
            rows_c4.append(dict(probe=tag, scene=sc["name"],
                                D_albedo=d_A, D_normal_deg=d_n))

            # ---- C5 Oracle-query-light held-out（仅 first 3 light subset → query 第 4 盏）----
            S = list(range(min(3, sc["imgs_lin"].shape[0] - 1)))
            q = max(S) + 1 if max(S) + 1 < sc["imgs_lin"].shape[0] else (0 if 0 not in S else 1)
            alb, dep, n_hat, sh_hat = predict(model, sc, S, device)
            n_t = torch.from_numpy(n_hat[None]).float().to(device)
            shq = torch.from_numpy(sc["sh_irr"][q][None, None]).float().to(device)
            s_q = sh_shading(n_t, shq)[0, 0].cpu().numpy()
            ih = alb * s_q
            iq = sc["imgs_lin"][q]
            mse = float(((ih - iq)[mask] ** 2).mean())
            rows_c5.append(dict(probe=tag, scene=sc["name"],
                                N=len(S), q_light=q,
                                ho_psnr=10 * math.log10(1 / max(mse, 1e-12)),
                                ho_mae=float(np.abs(ih - iq)[mask].mean())))

    def dump(p, rows):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    dump(os.path.join(args.out_dir, "C1_ncurve.csv"), rows_c1)
    dump(os.path.join(args.out_dir, "C2_cardinality.csv"), rows_c2)
    dump(os.path.join(args.out_dir, "C3_diversity.csv"), rows_c3)
    dump(os.path.join(args.out_dir, "C4_cross_subset.csv"), rows_c4)
    dump(os.path.join(args.out_dir, "C5_heldout.csv"), rows_c5)

    # Summary
    summary = {}
    for tag in probe_models:
        c1 = [r for r in rows_c1 if r["probe"] == tag]
        c2 = [r for r in rows_c2 if r["probe"] == tag]
        c3 = [r for r in rows_c3 if r["probe"] == tag]
        c4 = [r for r in rows_c4 if r["probe"] == tag]
        c5 = [r for r in rows_c5 if r["probe"] == tag]
        summary[tag] = dict(
            C1_ncurve={str(N): float(np.mean([r["si_mae"] for r in c1 if r["N"] == N]))
                       for N in NS if any(r["N"] == N for r in c1)},
            C2_cardinality=dict(d_new=float(np.mean([r["d_new"] for r in c2])) if c2 else None,
                                d_dup=float(np.mean([r["d_dup"] for r in c2])) if c2 else None,
                                d_new_gt_dup=(np.mean([r["d_new"] for r in c2]) >
                                              np.mean([r["d_dup"] for r in c2])) if c2 else None),
            C3_diversity=dict(E_high=float(np.mean([r["E_high_div"] for r in c3])) if c3 else None,
                               E_low=float(np.mean([r["E_low_div"] for r in c3])) if c3 else None,
                               high_better=(np.mean([r["E_high_div"] for r in c3]) <
                                            np.mean([r["E_low_div"] for r in c3])) if c3 else None),
            C4_cross_subset=dict(D_albedo=float(np.mean([r["D_albedo"] for r in c4])) if c4 else None,
                                 D_normal_deg=float(np.mean([r["D_normal_deg"] for r in c4])) if c4 else None),
            C5_heldout=dict(ho_psnr=float(np.mean([r["ho_psnr"] for r in c5])) if c5 else None,
                             ho_mae=float(np.mean([r["ho_mae"] for r in c5])) if c5 else None),
        )
    json.dump(summary, open(os.path.join(args.out_dir, "gate_summary.json"),
                            "w", encoding="utf-8"), indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
