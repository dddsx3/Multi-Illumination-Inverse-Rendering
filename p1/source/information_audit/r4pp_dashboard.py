"""R4″ 08 · GO/NO-GO Dashboard 生成器（任务书 §41，严格 6 行，禁止增删）。

输入：r4pp/ 下各阶段产物
输出：08_go_no_go_dashboard.md/.pdf + 09_R4pp_decision.md 模板 + figures

6 行：
  Instrument  | metric stability       | M1 log-pdet 5/5 PASS
  Signal      | low-N signal/noise     | Task C R_signal 全 cell > 2
  Direction   | info→error             | Task F β_G 方向（跨 G 档）
  Interaction | geometry gating        | Task F G↑ ⇒ |β_G|↑
  Saturation  | N=8 noise-floor        | Task C σ_subset/err 3.2% + R_signal 22.6
  Externality | local-init replication | Task G oracle-local β 存活

裁决映射（§44）：
  Instrument+Signal+Direction+Interaction 全 PASS → GO (A2)
  Instrument+Signal+Direction PASS, Interaction FAIL → PIVOT (B′)
  Instrument FAIL 或 Signal FAIL → KILL H-COND

用法：python r4pp_dashboard.py [--beta CSV] [--out DIR]
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(_REPO, "r4pp")


def load_beta(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = []
    for r in rows:
        try:
            out.append(dict(scene=r["scene"], N=int(float(r["N"])), G=float(r["G"]),
                            beta=float(r["beta_G"]), lo=float(r["boot_ci_lo"]),
                            hi=float(r["boot_ci_hi"]), n=int(float(r["n"]))))
        except (ValueError, KeyError):
            continue
    return out


def load_noise_summary(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for r in rows:
        out.setdefault(int(r["N"]), []).append(float(r["R_signal"]))
    return {N: (float(np.median(v)), float(np.mean([x > 2 for x in v])))
            for N, v in out.items()}


def verdict_for(beta_rows):
    """Gate 3/4 判定。

    Gate 3 (Direction): β_G 中位数 < 0 且非单场景驱动（≥60% 场景 β<0）
    Gate 4 (Interaction): 同一 family 内 G 与 β 的 Spearman 方向
    """
    if not beta_rows:
        return dict(G3="INSUFFICIENT", G4="INSUFFICIENT",
                    G3_detail="无 β 数据", G4_detail="无 β 数据")
    med = float(np.median([r["beta"] for r in beta_rows]))
    frac_neg = float(np.mean([r["beta"] < 0 for r in beta_rows]))
    g3_pass = med < 0 and frac_neg >= 0.6
    # Gate 4：family 内 G↑ ⇒ β 更负（ρ(G, β) < 0）
    fam_map = {}
    for r in beta_rows:
        fam = r["scene"][0]
        fam_map.setdefault(fam, []).append(r)
    g4_info = []
    for fam, rs in sorted(fam_map.items()):
        if len(rs) >= 4:
            xs = np.array([r["G"] for r in rs])
            ys = np.array([r["beta"] for r in rs])
            if xs.std() > 0 and ys.std() > 0:
                rho = float(np.corrcoef(xs, ys)[0, 1])
            else:
                rho = float("nan")
            g4_info.append((fam, rho, len(rs)))
    neg_rhos = [r for _, r, _ in g4_info if np.isfinite(r) and r < 0]
    g4_pass = len(g4_info) > 0 and len(neg_rhos) == len(g4_info)
    return dict(G3=g3_pass, G4=g4_pass, G3_detail=f"β med={med:+.3f}, 负占比={frac_neg:.2f}",
                G4_detail="; ".join(f"{f}:ρ={r:+.2f}(n={n})" for f, r, n in g4_info))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", default=os.path.join(OUT_DIR, "06_beta_per_geometry.csv"))
    ap.add_argument("--noise", default=os.path.join(OUT_DIR, "02_noise_floor_summary.csv"))
    ap.add_argument("--stability", default=os.path.join(OUT_DIR, "03_metric_stability.csv"))
    ap.add_argument("--local", default=os.path.join(OUT_DIR, "07_local_vs_global_init.csv"))
    args = ap.parse_args()

    beta = load_beta(args.beta)
    noise = load_noise_summary(args.noise)
    stab = {}
    if os.path.exists(args.stability):
        for r in csv.DictReader(open(args.stability)):
            stab[r["metric"]] = r

    # ---- 6 行判定 ----
    m1 = stab.get("M1", {})
    instrument_pass = all(m1.get(k) == "True" for k in
                          ["MA_PASS", "MB_PASS", "MC_PASS", "MD_PASS", "MF_PASS"])
    sig = noise.get(3, (0, 0))[1] if noise else 0
    signal_pass = sig >= 0.8 and all(noise.get(N, (0, 0))[0] > 2 for N in noise)
    v = verdict_for(beta)
    sat = noise.get(8, (0, 0))
    sat_pass = sat[0] > 2 and noise.get(3, (0, 0))[0] > sat[0] * 1.5

    local_pass = None
    if os.path.exists(args.local):
        rows = list(csv.DictReader(open(args.local)))
        betas = {}
        for r in rows:
            if r.get("N", "").lstrip("-").isdigit():
                key = (r["scene"], int(r["N"]), r["subset"])
                betas.setdefault(r["init_mode"], []).append(
                    (float(r["information"]), float(r["reconstruction_error"])))
        from scipy import stats
        gl = betas.get("global", []); ol = betas.get("oracle_local", [])
        if len(gl) >= 20 and len(ol) >= 20:
            def slope(pts):
                I = np.array([p[0] for p in pts]); E = np.log(np.array([p[1] for p in pts]))
                Iz = (I - I.mean()) / max(I.std(), 1e-12)
                return stats.linregress(Iz, E)[0]
            bg, bo = slope(gl), slope(ol)
            local_pass = bo < 0
            local_detail = f"global β={bg:+.3f}, oracle_local β={bo:+.3f}"
        else:
            local_detail = f"insufficient (global={len(gl)}, local={len(ol)})"
    else:
        local_detail = "未运行 Task G"

    rows_out = [
        dict(gate="Instrument", metric="M1 log-pdet 5/5 stability PASS",
             result="PASS" if instrument_pass else "FAIL",
             evidence=f"MA={m1.get('test_MA_cutoff_rho','?')} MB={m1.get('test_MB_cap_rho','?')} "
                      f"MC={m1.get('test_MC_boot_rho','?')} MF={m1.get('test_MF_modedrop_rho_min','?')}"),
        dict(gate="Signal", metric="low-N signal/noise",
             result="PASS" if signal_pass else "FAIL",
             evidence=f"R_signal: N=2 {noise.get(2,(0,0))[0]:.0f}, N=3 {noise.get(3,(0,0))[0]:.0f}, "
                      f"N=5 {noise.get(5,(0,0))[0]:.0f}, N=8 {noise.get(8,(0,0))[0]:.0f}"),
        dict(gate="Direction", metric="info→error β<0",
             result="PASS" if v["G3"] else "FAIL", evidence=v["G3_detail"]),
        dict(gate="Interaction", metric="G↑ ⇒ |β_G|↑",
             result="PASS" if v["G4"] else "FAIL", evidence=v["G4_detail"]),
        dict(gate="Saturation", metric="N=8 noise-floor",
             result="PASS" if sat_pass else "FAIL",
             evidence=f"N=8 R_signal={sat[0]:.1f} vs N=3 {noise.get(3,(0,0))[0]:.1f}"),
        dict(gate="Externality", metric="local-init replication",
             result=str(local_pass) if local_pass is not None else "PENDING",
             evidence=local_detail),
    ]
    n_pass = sum(1 for r in rows_out if r["result"] == "PASS")
    if instrument_pass and signal_pass and v["G3"] and v["G4"]:
        verdict = "GO (A2)"
    elif instrument_pass and signal_pass and v["G3"]:
        verdict = "PIVOT (B′)"
    else:
        verdict = "KILL H-COND"
    if local_pass is False and verdict in ("GO (A2)", "PIVOT (B′)"):
        verdict += " [Externality FAIL 警示]"

    md = [
        "# 08 · GO/NO-GO Dashboard", "",
        f"> 生成：{datetime.now().isoformat()} · 严格 6 行（任务书 §41，禁止增删）",
        "",
        "| Gate | 指标 | 结果 | 证据 |",
        "|---|---|---|---|",
    ]
    for r in rows_out:
        md.append(f"| {r['gate']} | {r['metric']} | **{r['result']}** | {r['evidence']} |")
    md += ["", f"**合计：{n_pass}/6 PASS**", "",
           f"**预注册裁决：{verdict}**", "",
           "## 说明",
           "- Instrument/Signal 为硬门槛；Direction/Interaction 决定 GO vs PIVOT；",
           "- Externality 失败不直接 KILL，但警示 A2 结论的稳健性；",
           "- 本 dashboard 只反映已产出的证据；Task G 未跑时 Externality 为 PENDING。"]
    with open(os.path.join(OUT_DIR, "08_go_no_go_dashboard.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\n[dashboard] -> {os.path.join(OUT_DIR, '08_go_no_go_dashboard.md')}")


if __name__ == "__main__":
    main()
