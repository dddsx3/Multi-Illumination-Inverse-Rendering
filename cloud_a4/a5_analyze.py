"""A5 frozen statistical analysis (final task book, section 5) — zero design.

Executes exactly the preregistered statistics on the assembled uos_table.csv:
  primary   Delta_random = median_o(U_o,mode - U_o,random)  (<0 = benefit)
            paired object-level bootstrap B=10000, seed=20260910;
            all 11 object-level paired differences reported;
  secondary Delta_E/A/D same machinery + sign count; per-regime;
  grades    A/B/C per the frozen rules (strong/neutral/negative wording).

Writes: allocation_summary.json, allocation_deltas.csv,
        allocation_object_pairs.csv (all 11 paired differences x regime x policy).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BUDGETS = [0.1, 0.2, 0.4, 0.6, 0.8]
REGIMES = [10, 100]
POLICIES = ["mode_aware", "e_opt", "a_opt", "d_opt", "random"]
DET_POLICIES = POLICIES[:-1]
BOOTSTRAP_SEED = 20260910
B = 10000

GRADES = {
    "strong": ("mode-aware stably beats random and at least one classical baseline",
               "mode-resolved calibration analysis can support allocation decisions "
               "that improve reconstruction under a fixed calibration budget."),
    "neutral": ("mode-aware beats random but is comparable to E/A/D-opt",
                "mode-derived allocation improves over random allocation, while "
                "classical optimal-design criteria achieve comparable aggregate "
                "performance; the mode decomposition additionally exposes which "
                "vulnerable directions drive the decision."),
    "negative": ("mode-aware not stably better than random",
                 "mode-resolved vulnerability predicts degradation but did not "
                 "produce a robust downstream allocation advantage under the "
                 "tested policy."),
}


def trapezoid_auc(Es):
    Es = np.asarray(Es, float)
    bs = np.asarray(BUDGETS, float)
    return float(np.sum((Es[:-1] + Es[1:]) / 2.0 * np.diff(bs)) / (0.8 - 0.1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    out = Path(args.results)
    table = {}
    with open(out / "uos_table.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            table[(r["object"], int(r["regime"]), r["policy"], float(r["budget"]))] = \
                float(r["E_osb"])

    summary = {"regimes": {}, "note": "lower E_osb / lower AUC is better; "
               "Delta < 0 means the mode-aware policy improved reconstruction"}
    lines = ["regime,policy,median_delta,ci_lo,ci_hi,benefit_significant,improved_of_11"]
    pair_lines = ["regime,policy,object,U_policy,U_random,delta,improved"]
    for regime in REGIMES:
        objs = sorted({o for (o, r, _p, _b) in table if r == regime})
        assert len(objs) == 11, f"expected 11 objects, got {len(objs)}"
        Uos = {o: {p: trapezoid_auc(
            [table[(o, regime, p, b)] for b in BUDGETS]) for p in POLICIES}
            for o in objs}
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = len(objs)
        draws = [rng.integers(0, n, n) for _ in range(B)]   # shared resamples
        res = {}
        for p in DET_POLICIES:
            d = np.array([Uos[o][p] - Uos[o]["random"] for o in objs])
            boots = [float(np.median(d[idx])) for idx in draws]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            res[p] = dict(median=float(np.median(d)),
                          ci95=[float(lo), float(hi)],
                          benefit_significant=bool(hi < 0),
                          per_object={o: float(x) for o, x in zip(objs, d)},
                          improved=int((d < 0).sum()))
            for o, x in zip(objs, d):
                pair_lines.append(f"{regime},{p},{o},{Uos[o][p]:.6g},"
                                  f"{Uos[o]['random']:.6g},{x:.6g},"
                                  f"{int(x < 0)}")
            lines.append(f"{regime},{p},{res[p]['median']:.6g},{lo:.6g},{hi:.6g},"
                         f"{res[p]['benefit_significant']},{res[p]['improved']}/11")
        m = res["mode_aware"]
        strong = m["benefit_significant"] and any(
            res[p]["benefit_significant"] for p in ("e_opt", "a_opt", "d_opt"))
        neutral = m["benefit_significant"] and not strong
        grade = "strong" if strong else ("neutral" if neutral else "negative")
        summary["regimes"][regime] = dict(
            policy_deltas=res, grade=grade, objects=objs,
            Uos=Uos, grade_condition=GRADES[grade][0],
            grade_wording=GRADES[grade][1])
        print(f"[a5] regime {regime}x: grade {grade.upper()}")
        print(f"     condition: {GRADES[grade][0]}")
        for p in DET_POLICIES:
            r = res[p]
            print(f"     Delta_{p}: median {r['median']:+.4g} "
                  f"CI [{r['ci95'][0]:+.4g}, {r['ci95'][1]:+.4g}] "
                  f"improved {r['improved']}/11")

    (out / "allocation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "allocation_deltas.csv").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
    (out / "allocation_object_pairs.csv").write_text(
        "\n".join(pair_lines) + "\n", encoding="utf-8")
    print("[a5] wrote allocation_summary.json / allocation_deltas.csv / "
          "allocation_object_pairs.csv")
    print("A5 = DONE")


if __name__ == "__main__":
    main()
