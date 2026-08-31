"""R4″ Task C · Noise-floor 实验（任务书 §7；Day 1 最高优先级）。

目标：判定 N=8 究竟是科学上的 saturation，还是 correlation 偶然变弱。
       并为 Task B 的绝对收敛判据提供数值尺度标定数据。

方差分解：
    σ²_solver = Var over solver seeds      | 固定 (scene, N, subset, render)
    σ²_render = Var over render realizations| 固定 (scene, N, subset, seed)
    σ²_repeat = σ²_solver + σ²_render
    σ²_subset = Var over subsets            | 固定 (scene, N)，先对 seed/render 取均值
    R_signal  = σ_subset / σ_repeat

Scene 选择（§C1，覆盖 G 三档 + 两类形状族，**不看旧 ρ 正负**）：
    conf_cube_axis        sh_gram_rank=4  n_eff=1.00  low     sparse-normal
    conf_prism8           sh_gram_rank=5  n_eff=1.18  low     sparse-normal
    conf_cylinder_r03_d12 sh_gram_rank=6  n_eff=1.20  low-med ruled
    conf_cone_r04_d12     sh_gram_rank=9  n_eff=1.24  med     ruled
    conf_egg              sh_gram_rank=9  n_eff=2.16  high    smooth
    conf_icosphere_sub3   sh_gram_rank=9  n_eff=2.27  high    smooth

用法：
    python r4pp_noise_floor.py --stage config                    # 冻结矩阵（只做一次）
    python r4pp_noise_floor.py --stage run --subsets 2 --seeds 2  # H2.6 标定 pilot（~10min）
    python r4pp_noise_floor.py --stage run                        # H3.4 全量（~51min）
    python r4pp_noise_floor.py --stage summarize                  # 方差分解
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "information_audit"))
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "physics"))

# 注意：torch 只在 --stage run 内部 lazy import。
# 本机 Windows commit 配额常被占满，torch 加载 cublas DLL 会抛
# OSError WinError 1455（页面文件太小）；config / summarize 是纯 numpy 阶段，
# 不应被 GPU 依赖阻塞。load_scene 改用 gauge_fisher_v2 的 numpy 版。
from gauge_fisher_v2 import load_scene  # noqa: E402
from sh import sh_basis_npy  # noqa: E402


def si_mae_np(pred, gt, mask):
    """scale-invariant MAE（与 information_audit_v2.si_mae_np 逐字等价，纯 numpy）。"""
    p, g = pred[mask], gt[mask]
    d = (p * p).sum()
    if d < 1e-12:
        return float("nan")
    s = (p * g).sum() / d
    return float(np.abs(s * p - g).mean())

DATA_ROOT = os.path.join(_REPO, "p1", "calibration_set", "data_sun_confirmatory")
OUT_DIR = os.path.join(_REPO, "r4pp")
CFG_DIR = os.path.join(OUT_DIR, "config")
TRACE_DIR = os.path.join(OUT_DIR, "traces")
CFG = os.path.join(CFG_DIR, "noise_floor_matrix.json")
RUNS = os.path.join(OUT_DIR, "02_noise_floor.csv")
SUMM = os.path.join(OUT_DIR, "02_noise_floor_summary.csv")

SCENES = ["conf_cube_axis", "conf_prism8", "conf_cylinder_r03_d12",
          "conf_cone_r04_d12", "conf_egg", "conf_icosphere_sub3"]
NS = [2, 3, 5, 8]                 # D-3 已批准纳入 N=2
N_SUBSETS = 4                     # §C3 每 (scene,N) 4 个固定 subset
N_SEEDS = 5                       # §C4 每 subset 5 个独立 solver seed
MATRIX_SEED = 20260901            # 生成 subset 列表用（与渲染 seed 无关）
SOLVER_SEED_BASE = 40000          # 与旧 20260830 明确区分
RENDER_REALIZATION = 0            # §C5 的 render repeat 由独立脚本追加（realization>0）


def build_config():
    """生成并冻结 (scene, N, subset, seed) 矩阵。已存在则拒绝改写。"""
    os.makedirs(CFG_DIR, exist_ok=True)
    if os.path.exists(CFG):
        cfg = json.load(open(CFG, encoding="utf-8"))
        print(f"[config] 已存在且冻结：{CFG}")
        print(f"[config] cells={len(cfg['cells'])} runs={cfg['n_runs_planned']}")
        return cfg
    rng = np.random.default_rng(MATRIX_SEED)
    cells = []
    for scn in SCENES:
        sc = load_scene(os.path.join(DATA_ROOT, scn))
        K = sc["imgs_lin"].shape[0]
        for N in NS:
            subs, seen = [], set()
            while len(subs) < N_SUBSETS:
                s = tuple(sorted(rng.choice(K, N, replace=False).tolist()))
                if s in seen:
                    continue
                seen.add(s)
                subs.append(list(s))
            cells.append(dict(scene=scn, N=N,
                              subsets=[",".join(map(str, s)) for s in subs],
                              seeds=[SOLVER_SEED_BASE + j for j in range(N_SEEDS)]))
    cfg = dict(frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               matrix_seed=MATRIX_SEED, solver_seed_base=SOLVER_SEED_BASE,
               scenes=SCENES, N_values=NS, n_subsets=N_SUBSETS, n_seeds=N_SEEDS,
               solver=dict(restarts=1, base_iters=800, lr=1e-2, lam_tv=0.03,
                           note="restarts=1：seed 效应即 σ_solver，不能取 best-of-restarts"),
               cells=cells,
               n_runs_planned=len(cells) * N_SUBSETS * N_SEEDS)
    json.dump(cfg, open(CFG, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[config] 冻结 -> {CFG}")
    print(f"[config] {len(SCENES)} scene × {len(NS)} N = {len(cells)} cell, "
          f"{cfg['n_runs_planned']} runs planned")
    return cfg


def ho_psnr_oracle(sc, A_hat, subset):
    """oracle-query-light held-out relighting：q = 子集外最小索引。"""
    K = sc["imgs_lin"].shape[0]
    q = [k for k in range(K) if k not in subset][0]
    mask = sc["mask"]
    n_cam = sc["n_mesh"].transpose(1, 2, 0)
    Y = sh_basis_npy(n_cam[mask])
    s_q = np.maximum(Y @ sc["sh_irr"][q], 0.0)
    ih = A_hat[mask] * s_q
    iq = sc["imgs_lin"][q][mask]
    mse = float(((ih - iq) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-12)), q


def stage_run(subset_limit=None, seed_limit=None, save_traces=True):
    from information_audit_v2 import joint_solve   # lazy：仅此阶段需要 torch
    cfg = json.load(open(CFG, encoding="utf-8"))
    os.makedirs(TRACE_DIR, exist_ok=True)
    done = set()
    if os.path.exists(RUNS):
        for r in csv.DictReader(open(RUNS, encoding="utf-8")):
            if r.get("N", "").lstrip("-").isdigit():
                done.add((r["scene"], int(r["N"]), r["subset"],
                          int(r["solver_seed"]), int(r["render_realization"])))
    f = open(RUNS, "a", newline="", encoding="utf-8")
    wr = None
    if os.path.exists(RUNS) and os.path.getsize(RUNS) > 0:
        wr = csv.DictWriter(f, fieldnames=_FIELDS)
    n_new = 0
    t0 = time.time()
    cache = {}
    for cell in cfg["cells"]:
        scn, N = cell["scene"], cell["N"]
        subs = cell["subsets"][:subset_limit] if subset_limit else cell["subsets"]
        seeds = cell["seeds"][:seed_limit] if seed_limit else cell["seeds"]
        todo = [(s, sd) for s in subs for sd in seeds
                if (scn, N, s, sd, RENDER_REALIZATION) not in done]
        if not todo:
            continue
        if scn not in cache:
            cache.clear()
            cache[scn] = load_scene(os.path.join(DATA_ROOT, scn))
        sc = cache[scn]
        tc = time.time()
        for s, sd in todo:
            sub = [int(x) for x in s.split(",")]
            r = joint_solve(sc, sub, restarts=1, seed=sd,
                            return_trace=save_traces,
                            **{k: v for k, v in cfg["solver"].items()
                               if k in ("base_iters", "lr", "lam_tv")})
            err = si_mae_np(r["A_hat"], sc["albedo"], sc["mask"])
            ho, q = ho_psnr_oracle(sc, r["A_hat"], sub)
            tag = f"{scn}_N{N}_{s.replace(',', '-')}_s{sd}_r{RENDER_REALIZATION}"
            if save_traces:
                np.save(os.path.join(TRACE_DIR, tag + ".npy"),
                        r["loss_trace"].astype(np.float32))
            rec = dict(scene=scn, N=N, subset=s, solver_seed=sd,
                       render_realization=RENDER_REALIZATION,
                       reconstruction_error=err, ho_psnr=ho, query_light=q,
                       final_objective=r["final_loss"], grad_norm=r["grad_norm"],
                       proj_grad_norm=r["proj_grad_norm"],
                       tail_rel_change=r["tail_rel_change"],
                       tail_range_abs=None,
                       conv_finite=int(r["conv_finite"]),
                       old_style_converged=int(r["success"]),
                       iters=r["iters"], trace_file=tag + ".npy" if save_traces else "")
            if wr is None:
                wr = csv.DictWriter(f, fieldnames=_FIELDS)
                wr.writeheader()
            wr.writerow(rec)
            n_new += 1
        f.flush()
        print(f"  [nf] {scn:24s} N={N}: {len(todo)} runs in {time.time()-tc:.0f}s "
              f"({(time.time()-tc)/len(todo):.1f}s/run)", flush=True)
    f.close()
    print(f"[nf] +{n_new} runs in {time.time()-t0:.0f}s -> {RUNS}")


_FIELDS = ["scene", "N", "subset", "solver_seed", "render_realization",
           "reconstruction_error", "ho_psnr", "query_light",
           "final_objective", "grad_norm", "proj_grad_norm", "tail_rel_change",
           "tail_range_abs", "conv_finite", "old_style_converged", "iters",
           "trace_file"]


def stage_summarize():
    import pandas as pd
    df = pd.read_csv(RUNS)
    df = df[pd.to_numeric(df["N"], errors="coerce").notna()].copy()
    df["N"] = df["N"].astype(int)
    gram = {r["scene"]: r for r in csv.DictReader(
        open(os.path.join(_REPO, "p1", "information_audit", "diagnostics",
                          "r4p_scene_gram_spectrum.csv"), encoding="utf-8"))}
    rows = []
    for (scn, N), g in df.groupby(["scene", "N"]):
        # σ_solver：每 (subset, render) 内跨 seed 的 sd，再对 subset 取 RMS 汇总
        v_solver, v_render = [], []
        for (s, rr), gg in g.groupby(["subset", "render_realization"]):
            if gg["solver_seed"].nunique() >= 2:
                v_solver.append(gg["reconstruction_error"].var(ddof=1))
        for (s, sd), gg in g.groupby(["subset", "solver_seed"]):
            if gg["render_realization"].nunique() >= 2:
                v_render.append(gg["reconstruction_error"].var(ddof=1))
        s_solver = math.sqrt(float(np.mean(v_solver))) if v_solver else float("nan")
        s_render = math.sqrt(float(np.mean(v_render))) if v_render else 0.0
        # σ_subset：先对 seed/render 取均值，再跨 subset 的 sd
        per_sub = g.groupby("subset")["reconstruction_error"].mean()
        s_subset = float(per_sub.std(ddof=1)) if per_sub.size >= 2 else float("nan")
        s_repeat = math.sqrt((0.0 if math.isnan(s_solver) else s_solver ** 2)
                             + (0.0 if math.isnan(s_render) else s_render ** 2))
        R = s_subset / s_repeat if s_repeat > 0 else float("inf")
        gm = gram.get(scn, {})
        rows.append(dict(
            scene=scn, N=N, n_runs=int(len(g)),
            n_subsets=int(g["subset"].nunique()),
            n_seeds=int(g["solver_seed"].nunique()),
            n_renders=int(g["render_realization"].nunique()),
            err_mean=float(g["reconstruction_error"].mean()),
            err_median=float(g["reconstruction_error"].median()),
            sigma_solver=s_solver, sigma_render=s_render,
            sigma_repeat=s_repeat, sigma_subset=s_subset, R_signal=R,
            cv_solver=s_solver / max(g["reconstruction_error"].mean(), 1e-12),
            sh_gram_rank=float(gm.get("sh_gram_rank_1em8", "nan")),
            normal_cov_eff_rank=float(gm.get("normal_cov_eff_rank", "nan")),
            conv_finite_rate=float(g["conv_finite"].mean()),
            old_style_conv_rate=float(g["old_style_converged"].mean()),
            proj_grad_norm_median=float(g["proj_grad_norm"].median()),
            tail_rel_change_median=float(g["tail_rel_change"].median()),
        ))
    out = pd.DataFrame(rows).sort_values(["sh_gram_rank", "normal_cov_eff_rank",
                                          "scene", "N"])
    out.to_csv(SUMM, index=False)
    print(f"[nf] -> {SUMM}  ({len(out)} cell)\n")
    cols = ["scene", "N", "n_runs", "err_median", "sigma_solver", "sigma_subset",
            "R_signal", "sh_gram_rank", "normal_cov_eff_rank"]
    print(out[cols].to_string(index=False,
                              float_format=lambda v: f"{v:.4g}"))
    print("\n=== §8 预先裁决规则（operational threshold）===")
    for N in sorted(out["N"].unique()):
        g = out[out["N"] == N]
        hi = g[g["sh_gram_rank"] >= 9]
        frac_gt1 = float((g["R_signal"] > 1).mean())
        frac_gt2 = float((g["R_signal"] > 2).mean())
        hi_gt2 = float((hi["R_signal"] > 2).mean()) if len(hi) else float("nan")
        print(f"  N={N}: R_signal median={g['R_signal'].median():.3f} | "
              f">1: {frac_gt1:.2f} | >2: {frac_gt2:.2f} | high-G scene >2: {hi_gt2:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["config", "run", "summarize"])
    ap.add_argument("--subsets", type=int, default=None, help="只跑前 k 个 subset（pilot 用）")
    ap.add_argument("--seeds", type=int, default=None, help="只跑前 k 个 seed（pilot 用）")
    ap.add_argument("--no_traces", action="store_true")
    args = ap.parse_args()
    if args.stage == "config":
        build_config()
    elif args.stage == "run":
        build_config()
        stage_run(args.subsets, args.seeds, save_traces=not args.no_traces)
    else:
        stage_summarize()


if __name__ == "__main__":
    main()
