"""R5 Compute-Aware Campaign · Route B1 · 随机像素降采样排名保真

问题: GSIQ 排名是否只需要一小部分像素?
  对同一批光照子集, 分别用 P_c (少像素) 和 P_ref (多像素) 计算 GSIQ,
  看两者的排名是否一致 (Spearman rho + top-10% 重叠)。

GO 门槛 (任务书 Route B1, 冻结):
  强GO:  median rho(P=512,  P_ref) >= 0.95
  中GO:  median rho(P=1024, P_ref) >= 0.95
  FAIL:  P=2048 仍 rho < 0.9

说明:
  - P_ref 默认 2048 (本地内存可承受的"高精度参照")。
    可选 --deep 用真实全分辨率 mask (~6178) 只跑 20 个子集做抽检。
  - 只算 O 路径 (P1-A smoke 已证明 O 与 A 排名 rho=1.0, 二者等价)。

用法 (本地 Windows 直接跑, 无需 GPU):
  python r5_ca_02_pixel_coreset.py                          # 默认 3 scene × 200 subsets
  python r5_ca_02_pixel_coreset.py --n_subsets 50 --p_ref 1024   # 快速版
输出:
  r5_compute_audit/ranking/pixel_coreset.csv
  r5_compute_audit/decision_reports/B1_verdict.md           <- 大白话结论
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent
sys_path = str(REPO / "p1" / "source" / "information_audit")
import sys  # noqa: E402
sys.path.insert(0, sys_path)

from gauge_fisher_v2 import ga_isi_v2_scores, load_scene, sh_basis_npy  # noqa: E402

DATA_ROOT = REPO / "p1" / "calibration_set" / "data_sun_confirmatory"
OUT_DIR = REPO / "r5_compute_audit"
DEFAULT_SCENES = ["conf_sphere_r05", "conf_cube_axis", "conf_egg"]


def score_at(a, n_pts, C, P, retries=3):
    """在指定像素集合上算 GSIQ (O 路径)。a/n_pts 已是子采样后的数组。

    Windows commit 配额偶发抖动: 失败时 gc + 等待后重试。
    """
    rms = np.sqrt((a * a).mean())
    a_fx = a / max(rms, 1e-9)
    Y = sh_basis_npy(n_pts)
    for attempt in range(retries):
        try:
            r = ga_isi_v2_scores(a_fx, Y, C)
            return float(r["full_logdet_pos_norm"]), r["structural_status"], r["d_extra_null"]
        except MemoryError:
            gc.collect()
            time.sleep(2.0 * (attempt + 1))
    raise MemoryError(f"score_at P={P}: {retries} 次重试后仍 OOM (系统 commit 配额不足)")


def mem_guard():
    """启动前内存体检: 可用内存不足时给出大白话提示。"""
    import psutil
    vm = psutil.virtual_memory()
    avail_gb = vm.available / 1e9
    print(f"[内存体检] 可用 RAM = {avail_gb:.1f} GB")
    if avail_gb < 4.0:
        print("  !! 警告: 可用内存不足 4 GB, 大概率中途 OOM。")
        print("  !! 建议: 关闭浏览器/其他程序后再跑; 或重启电脑后第一件事跑本脚本。")


def main():
    mem_guard()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--n_subsets", type=int, default=200)
    ap.add_argument("--p_cores", default="128,256,512,1024")
    ap.add_argument("--p_ref", type=int, default=2048)
    ap.add_argument("--deep", action="store_true",
                    help="额外抽检: 1 个 scene 用真实全分辨率 mask 跑 20 个子集")
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    p_cores = [int(x) for x in args.p_cores.split(",")]
    (OUT_DIR / "ranking").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "decision_reports").mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "ranking" / "pixel_coreset.csv"
    jsonl_path = OUT_DIR / "raw_profile"
    jsonl_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("R5-Campaign · Route B1 像素降采样排名保真")
    print(f"scenes={args.scenes}  N={args.n}  subsets/scene={args.n_subsets}")
    print(f"P_cores={p_cores}  P_ref={args.p_ref}")
    print("=" * 70)

    K = 32
    rng = np.random.default_rng(args.seed)
    # 预生成子集池 (所有 scene 共用同一批子集, 便于跨 scene 汇总)
    subsets = []
    seen = set()
    while len(subsets) < args.n_subsets:
        cand = tuple(sorted(rng.choice(K, args.n, replace=False).tolist()))
        if cand not in seen:
            seen.add(cand)
            subsets.append(cand)

    rows = []
    per_scene_rho = {P: [] for P in p_cores}
    per_scene_top = {P: [] for P in p_cores}

    for scene in args.scenes:
        t0 = time.time()
        sc = load_scene(str(DATA_ROOT / scene))
        idx_all = np.argwhere(sc["mask"])
        P_ref = min(args.p_ref, len(idx_all))
        C_all = sc["sh_irr"].astype(np.float64)
        a_full = sc["albedo"]
        n_full = sc["n_mesh"].transpose(1, 2, 0)

        scores = {P: [] for P in p_cores + [P_ref]}
        for si, sub in enumerate(subsets):
            C = C_all[list(sub)]
            # P_ref 像素先采, P_c 是它的子集 (嵌套采样, 语义正确)
            sel_ref = rng.choice(len(idx_all), P_ref, replace=False)
            idx_ref = idx_all[sel_ref]
            a_ref = a_full[idx_ref[:, 0], idx_ref[:, 1]]
            n_ref = n_full[idx_ref[:, 0], idx_ref[:, 1]]
            for P in p_cores + [P_ref]:
                if P == P_ref:
                    idx_p, a_p, n_p = idx_ref, a_ref, n_ref
                else:
                    sel = rng.choice(P_ref, P, replace=False)
                    idx_p = idx_ref[sel]
                    a_p = a_ref[sel]
                    n_p = n_ref[sel]
                I, status, d_extra = score_at(a_p, n_p, C, P)
                scores[P].append(I)
            gc.collect()
            if (si + 1) % 50 == 0:
                print(f"  [{scene}] {si+1}/{len(subsets)} subsets "
                      f"({time.time()-t0:.0f}s)", flush=True)

        # 排名诊断
        ref = np.array(scores[P_ref])
        for P in p_cores:
            cur = np.array(scores[P])
            rho = float(spearmanr(cur, ref)[0])
            k = max(1, int(round(0.10 * len(ref))))
            top_ref = set(np.argsort(-ref)[:k].tolist())
            top_cur = set(np.argsort(-cur)[:k].tolist())
            ov = len(top_ref & top_cur) / k
            per_scene_rho[P].append(rho)
            per_scene_top[P].append(ov)
            rows.append(dict(scene=scene, N=args.n, P_c=P, P_ref=P_ref,
                             n_subsets=len(ref), rho=round(rho, 4),
                             top10_overlap=round(ov, 3),
                             median_abs_shift=round(float(np.median(np.abs(cur - ref))), 4)))
            print(f"  [{scene}] P={P:5d}: rho={rho:.4f}  top10={ov:.3f}")
        del sc, C_all
        gc.collect()

    # ---- deep 抽检 (真实全分辨率) ----
    deep_note = ""
    if args.deep:
        scene = args.scenes[0]
        sc = load_scene(str(DATA_ROOT / scene))
        idx_all = np.argwhere(sc["mask"])
        P_true = len(idx_all)
        C_all = sc["sh_irr"].astype(np.float64)
        a_full = sc["albedo"]
        n_full = sc["n_mesh"].transpose(1, 2, 0)
        n_deep = 20
        deep_subsets = subsets[:n_deep]
        s_core, s_true = [], []
        print(f"\n[deep 抽检] {scene}: P_true={P_true}, {n_deep} 子集 (较慢, 每个 ~30-60s)")
        for sub in deep_subsets:
            C = C_all[list(sub)]
            sel_ref = rng.choice(len(idx_all), min(2048, P_true), replace=False)
            idx_ref = idx_all[sel_ref]
            I_core, _, _ = score_at(a_full[idx_ref[:, 0], idx_ref[:, 1]],
                                    n_full[idx_ref[:, 0], idx_ref[:, 1]], C, 2048)
            I_true, _, _ = score_at(a_full[idx_all[:, 0], idx_all[:, 1]],
                                    n_full[idx_all[:, 0], idx_all[:, 1]], C, P_true)
            s_core.append(I_core)
            s_true.append(I_true)
            gc.collect()
        rho_deep = float(spearmanr(np.array(s_core), np.array(s_true))[0])
        deep_note = (f"\n[deep 抽检] P=2048 vs 真实全分辨率 P={P_true} "
                     f"({n_deep} 子集): rho={rho_deep:.4f}\n")
        print(deep_note)
        rows.append(dict(scene=scene + "(deep)", N=args.n, P_c=2048, P_ref=P_true,
                         n_subsets=n_deep, rho=round(rho_deep, 4),
                         top10_overlap="", median_abs_shift=""))

    # ---- 写 CSV ----
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---- 大白话裁决 (通用逻辑: 按实际测过的 P 判定, 不写死 512/1024) ----
    med = {P: float(np.median(v)) for P, v in per_scene_rho.items()}
    med_top = {P: float(np.median(v)) for P, v in per_scene_top.items()}
    speed_est = {P: (args.p_ref / P) ** 3 for P in p_cores}
    tested = sorted(p_cores)

    # 任务书门槛: 强GO = ≤512 档 ρ≥0.95 且 top10 重合≥0.8; 中GO = ≤1024 档同标准
    # (top10 必须同时达标: rho 高但 top10 低 = 排名大体对但"选最好的"会选错, 不算过)
    strong = [P for P in tested if P <= 512 and med[P] >= 0.95 and med_top[P] >= 0.8]
    mid = [P for P in tested if P <= 1024 and med[P] >= 0.95 and med_top[P] >= 0.8]
    any_ge_09 = [P for P in tested if med[P] >= 0.9 and med_top[P] >= 0.8]

    if strong:
        verdict_key = "强GO"
        chosen = min(strong)
    elif mid:
        verdict_key = "中GO"
        chosen = min(mid)
    elif any_ge_09:
        verdict_key = "CONDITIONAL"
        chosen = max(any_ge_09)
    else:
        verdict_key = "FAIL"
        chosen = max(tested)

    lines = []
    lines.append("# Route B1 像素降采样 · 大白话结论\n")
    lines.append(f"- 场景: {args.scenes} · 灯数 N={args.n} · 每场景 {args.n_subsets} 个子集\n")
    lines.append(f"- 参照精度: P_ref={args.p_ref} 像素\n\n")
    lines.append("| 用多少像素 | 排名一致度 rho | 前10%重合 | 大约提速倍数 |\n|---|---|---|---|\n")
    for P in p_cores:
        sp = speed_est[P]
        lines.append(f"| {P} | {med[P]:.4f} | {med_top[P]:.3f} | ~{sp:.0f}x |\n")
    lines.append(deep_note + "\n")
    if verdict_key == "强GO":
        lines.append(f"## 裁决: **强GO** — 只用 {chosen} 像素即可, 排名与高精度几乎一样\n")
        lines.append(f"> 结论: GSIQ 排名不需要全部像素。用 {chosen} 像素 ≈ 提速 {speed_est[chosen]:.0f} 倍, "
                     "排名保持。论文价值: 中等 (计算加速), 且为 Route D (预算感知) 打开大门。\n")
    elif verdict_key == "中GO":
        lines.append(f"## 裁决: **中GO** — 需要 {chosen} 像素才能保排名\n")
        lines.append(f"> 结论: 可用但提速有限 (≈{speed_est[chosen]:.0f} 倍)。优先继续 Route C 看更大空间。\n")
    elif verdict_key == "CONDITIONAL":
        lines.append(f"## 裁决: **边缘** — 最大测试档 {chosen} 像素 ρ≥0.9 但 <0.95\n")
        lines.append("> 结论: 加大像素档或换自适应采样 (B2) 再测; 同时继续 Route C。\n")
    else:
        lines.append(f"## 裁决: **FAIL** — 降到 {chosen} 像素排名就保不住\n")
        lines.append("> 结论: 像素降采样路线停止, 不要再投入。重心放 Route C。\n")
    # 任务书 Stop Rule 1 (只在真的测过 512 档时才判)
    if 512 in med and med[512] < 0.9:
        lines.append("\n**Stop Rule 1 触发**: P=512 rho<0.9 -> B/D 两条路线停止。\n")
    (OUT_DIR / "decision_reports" / "B1_verdict.md").write_text("".join(lines), encoding="utf-8")
    print("\n" + "".join(lines))

    # JSON 审计
    for P in p_cores:
        rec = dict(scene=",".join(args.scenes), N=args.n, method=f"pixel_coreset_P{P}",
                   runtime="", peak_memory="", score=med[P], rank_score=med[P],
                   selection_error="", status=verdict_key)
        with open(jsonl_path / "campaign.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"产物: {csv_path}\n      {OUT_DIR / 'decision_reports' / 'B1_verdict.md'}")


if __name__ == "__main__":
    main()
