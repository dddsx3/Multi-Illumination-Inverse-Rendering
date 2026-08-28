"""
M7: 10 epoch reproducibility audit (read-only, no training).

Per INC-0010 §3.1: the project self-claims "val_loss=0.027 after fix" for
the 10-epoch run. We do NOT retrain; instead we ask the equivalent
counterfactual question:

  "If a 100-epoch run were truncated at epoch 9/19/29/.../99, what
  would its val_loss have been, and how stable is that proxy of
  '10-epoch reproducibility' across runs?"

We use the val/total scalar curves already recorded in TensorBoard by
the three 100-epoch runs:

  - p2_t22_f_n5gray_20260825  (F-N5-gray, the one most relevant to the
                               INC-0007 fix narrative)
  - p2_r0_v3gray_20260825     (R0-v3gray baseline)
  - p2_t25_f_resA             (F-resA ablation, residual switched off)

The 10-epoch self-claim is then judged by:
  1. val_loss distribution at epoch 9 across runs (mean, std, range)
  2. epoch of first "stable" val_loss (defined as std over a sliding
     5-epoch window < 5% of the median) — does epoch 9 already qualify?
  3. epoch of best (lowest) val_loss — when is the 10-epoch
     "convergence" actually attained?
  4. epoch where the run reaches within 10% of its final best val_loss
     — i.e. when has it "essentially converged"?

Pass criterion (strict, INC-0010 spirit):
  - If epoch 9 is NOT in the stable region AND the run continues to
    improve by > 10% afterwards, the 10-epoch claim is FALSIFIED.
  - If epoch 9 IS stable and within 10% of best, the claim is
    CONSISTENT.
  - Mixed case (stable at 9 but far from best) is FLAGGED.

This script does NOT load checkpoints or run eval — the val_loss
trajectory in TensorBoard is the recorded ground truth of what the
training loop measured.
"""
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_ROOT = REPO_ROOT / "logs"

TARGET_EPOCHS = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

RUNS = {
    "F-N5-gray": "p2_t22_f_n5gray_20260825",
    "R0-v3gray": "p2_r0_v3gray_20260825",
    "F-resA":    "p2_t25_f_resA",
}


def load_val_total(log_dir: Path):
    """Aggregate val/total across all event files in a run dir."""
    scalars = []
    for ef in sorted(log_dir.glob("events.out.tfevents.*")):
        ea = EventAccumulator(str(ef))
        ea.Reload()
        for s in ea.Scalars("val/total"):
            scalars.append((s.step, s.value))
    scalars.sort(key=lambda x: x[0])
    # Deduplicate by step (keep first occurrence)
    seen = {}
    for step, val in scalars:
        seen.setdefault(step, val)
    return sorted(seen.items())


def first_stable_epoch(series, window=5, rel_tol=0.05):
    """First epoch whose 5-epoch window std / median < rel_tol."""
    vals = [v for _, v in series]
    if len(vals) < window:
        return None
    for i in range(window - 1, len(vals)):
        w = vals[i - window + 1: i + 1]
        med = statistics.median(w)
        if med <= 0:
            continue
        sd = statistics.pstdev(w)
        if sd / med < rel_tol:
            return series[i][0]
    return None


def first_within_frac_of_best(series, frac=0.10):
    """First epoch whose val_loss is within `frac` of the best."""
    best = min(v for _, v in series)
    threshold = best * (1.0 + frac)
    for step, v in series:
        if v <= threshold:
            return step, v, best
    return None, None, best


def main():
    out = {
        "agent": "G",
        "experiment": "M7",
        "question": (
            "INC-0010 §3.1 self-claim 'val_loss=0.027 after fix at 10 "
            "epochs' — is it reproducible / interpretable as '10 epoch "
            "is enough'?"
        ),
        "method": (
            "Read-only. Aggregate val/total scalar curves from 3 "
            "delivered 100-epoch runs in TensorBoard. Extract val_loss "
            "at TARGET_EPOCHS; compute stability window, best epoch, "
            "and 'within-10%-of-best' epoch."
        ),
        "no_training": True,
        "runs": {},
    }

    summary_rows = []

    for run_name, sub in RUNS.items():
        log_dir = LOGS_ROOT / sub
        if not log_dir.is_dir():
            out["runs"][run_name] = {"error": f"missing {log_dir}"}
            continue
        series = load_val_total(log_dir)
        if not series:
            out["runs"][run_name] = {"error": "no val/total scalars"}
            continue

        lookup = dict(series)
        row = {"epochs_recorded": len(series)}
        for ep in TARGET_EPOCHS:
            row[f"val_loss@{ep:02d}"] = (
                round(lookup[ep], 6) if ep in lookup else None
            )
        best_step, best_val = min(series, key=lambda x: x[1])
        row["best_epoch"] = int(best_step)
        row["best_val_loss"] = round(float(best_val), 6)
        row["first_stable_epoch_5pct"] = first_stable_epoch(series, 5, 0.05)
        row["first_within_10pct_of_best"] = first_within_frac_of_best(
            series, 0.10
        )[0]

        out["runs"][run_name] = row
        summary_rows.append(row)

    # Cross-run statistics at epoch 9
    e9_vals = [
        r["val_loss@09"] for r in summary_rows
        if r.get("val_loss@09") is not None
    ]
    if e9_vals:
        out["cross_run_epoch9"] = {
            "n": len(e9_vals),
            "mean": round(statistics.mean(e9_vals), 6),
            "std":  round(statistics.pstdev(e9_vals), 6) if len(e9_vals) > 1 else 0.0,
            "min":  round(min(e9_vals), 6),
            "max":  round(max(e9_vals), 6),
            "range_over_mean": (
                round((max(e9_vals) - min(e9_vals)) / statistics.mean(e9_vals), 4)
                if e9_vals else None
            ),
        }

    # Reproducibility verdict
    # "10 epoch claim 0.027" is consistent if:
    #   - the run actually reaches ~0.027 at some point AND
    #   - epoch 9 is NOT in the stable-flat region (i.e. 10 epochs is
    #     NOT enough to be 'done').
    # We test the latter as the load-bearing falsifier.
    falsifier_holds = []
    for r in summary_rows:
        stable = r.get("first_stable_epoch_5pct")
        best_ep = r.get("best_epoch")
        within10 = r.get("first_within_10pct_of_best")
        e9 = r.get("val_loss@09")
        e99 = r.get("val_loss@99")

        # Falsifier: training continues to improve substantially AFTER
        # epoch 9 → 10 epochs is NOT enough.
        if e9 is not None and e99 is not None and e99 > 0:
            improvement_pct = (e9 - e99) / max(e9, 1e-9) * 100.0
        else:
            improvement_pct = None
        falsifier_holds.append({
            "run": r.get("run_name", ""),
            "val_loss@9":  e9,
            "val_loss@99": e99,
            "improvement_9_to_99_pct": (
                round(improvement_pct, 2) if improvement_pct is not None
                else None
            ),
            "best_epoch": best_ep,
            "first_stable_epoch_5pct": stable,
            "first_within_10pct_of_best": within10,
            "verdict": None,  # filled below
        })

    for entry in falsifier_holds:
        imp = entry["improvement_9_to_99_pct"]
        stable = entry["first_stable_epoch_5pct"]
        if imp is None:
            entry["verdict"] = "INSUFFICIENT_DATA"
        elif imp < 5.0 and (stable is None or stable <= 9):
            entry["verdict"] = "CONSISTENT (10-epoch near-final)"
        elif imp < 5.0 and stable is not None and stable > 9:
            entry["verdict"] = (
                "FLAT_LATE (10-epoch not stable, but no further gain)"
            )
        elif imp >= 5.0 and (stable is None or stable > 9):
            entry["verdict"] = "FALSIFIED (10-epoch too early, large late gain)"
        elif imp >= 5.0 and stable is not None and stable <= 9:
            entry["verdict"] = (
                "MIXED (stable early, but late improvement contradicts "
                "stability)"
            )
        else:
            entry["verdict"] = "AMBIGUOUS"

    out["per_run_verdict"] = falsifier_holds

    # Aggregate verdict
    verdicts = [e["verdict"] for e in falsifier_holds]
    n_falsified = sum(1 for v in verdicts if "FALSIFIED" in v)
    n_consistent = sum(1 for v in verdicts if "CONSISTENT" in v and "FLAT" not in v)
    n_flat = sum(1 for v in verdicts if "FLAT" in v)
    n_mixed = sum(1 for v in verdicts if "MIXED" in v)
    n_ambiguous = sum(1 for v in verdicts if v == "AMBIGUOUS")

    if n_falsified >= 2:
        out["aggregate_verdict"] = "10-EPOCH-CLAIM FALSIFIED (majority of runs continue improving past epoch 9)"
        out["pass"] = False
    elif n_falsified == 1 and n_mixed + n_ambiguous == 0:
        out["aggregate_verdict"] = (
            "MIXED EVIDENCE: 1/3 run falsifies; need more data"
        )
        out["pass"] = False
    elif n_consistent >= 2:
        out["aggregate_verdict"] = (
            "10-EPOCH-CLAIM CONSISTENT (most runs near-converged by "
            "epoch 9)"
        )
        out["pass"] = True
    else:
        out["aggregate_verdict"] = (
            f"INCONCLUSIVE: falsified={n_falsified} consistent="
            f"{n_consistent} flat_late={n_flat} mixed={n_mixed} "
            f"ambiguous={n_ambiguous}"
        )
        out["pass"] = False

    out["counts"] = {
        "falsified": n_falsified,
        "consistent": n_consistent,
        "flat_late": n_flat,
        "mixed": n_mixed,
        "ambiguous": n_ambiguous,
    }

    # Write to fixed audit output location
    out_path = (
        REPO_ROOT / "tests_audit" / "out_M7_10epoch_reproducibility.json"
    )
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[M7] wrote {out_path}")
    print(f"[M7] aggregate_verdict: {out['aggregate_verdict']}")
    print(f"[M7] pass={out['pass']}")
    return out


if __name__ == "__main__":
    main()
