"""
M8: Early-stopping criteria vs real convergence.

INC-0007 added a 4-metric "convergence" gate (trainer.py:1393-1398):

    is_qualified = (
        shading_var > 0.01
        and 0.8 <= sh0_mean <= 1.2
        and albedo_corr > 0.7
        and lambertian_ratio > 0.95
    )

plus "10 consecutive qualified epochs ⇒ stop training" (trainer.py:1411).

This script answers: would the gate have triggered in the 100-epoch
runs? If so, when, and how does that compare with the true val_loss
minimum?

KEY FACT (corrected after M8 v1):
  The 4 gold metrics ARE recorded — just not in TensorBoard. They
  are PRINTED to the training log (stdout) on every epoch. We parse
  them out of the training-log text files:

    F-N5-gray: repo/_train_f_n5gray_log.txt
    R0-v3gray: repo/_train_p2_r0_log.txt
    F-resA:    repo/_arm_p2_t25_f_resA_log.txt

  Each epoch appears as either
    ✅ Epoch N 达标！
    ❌ Epoch N 未达标
  followed by a metrics line:
    指标: Shading Var=0.0253, SH[0]=0.6632, Albedo Corr=0.9791, Lambertian Ratio=0.5370

This is the most reliable ground-truth for the gate's behavior.
"""
import json
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_ROOT = REPO_ROOT / "logs"
REPO_DIR  = REPO_ROOT / "repo"

# Run -> training log file (stdout capture)
LOG_FILES = {
    "F-N5-gray": REPO_DIR / "_train_f_n5gray_log.txt",
    "R0-v3gray": REPO_DIR / "_train_p2_r0_log.txt",
    "F-resA":    REPO_DIR / "_arm_p2_t25_f_resA_log.txt",
}

# Pattern to match
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+(达标|未达标).*?\n\s*指标:\s*"
    r"Shading Var=([-\d.eE+]+),\s*"
    r"SH\[0\]=([-\d.eE+]+),\s*"
    r"Albedo Corr=([-\d.eE+]+),\s*"
    r"Lambertian Ratio=([-\d.eE+]+)"
)


def parse_log(path: Path):
    """Return list of dicts: epoch, qualified, shading_var, sh0_mean,
    albedo_corr, lambertian_ratio."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    out = []
    for m in EPOCH_RE.finditer(text):
        ep = int(m.group(1))
        is_qualified_str = m.group(2)  # "达标" or "未达标"
        qualified = (is_qualified_str == "达标")
        out.append({
            "epoch": ep,
            "qualified_text": is_qualified_str,
            "shading_var": float(m.group(3)),
            "sh0_mean":    float(m.group(4)),
            "albedo_corr": float(m.group(5)),
            "lambertian_ratio": float(m.group(6)),
        })
    out.sort(key=lambda x: x["epoch"])
    return out


def gate(m):
    """Apply trainer.py:1393-1398."""
    return (
        m["shading_var"] > 0.01
        and 0.8 <= m["sh0_mean"] <= 1.2
        and m["albedo_corr"] > 0.7
        and m["lambertian_ratio"] > 0.95
    )


def first_consecutive_true(flags, k=10):
    n_true = 0
    for i, f in enumerate(flags):
        if f:
            n_true += 1
            if n_true >= k:
                return i
        else:
            n_true = 0
    return None


def main():
    report = {
        "agent": "G",
        "experiment": "M8",
        "question": (
            "Is the 4-metric early-stop gate consistent with true "
            "val_loss convergence? When would it have fired in the "
            "delivered 100-epoch runs?"
        ),
        "method": (
            "Parse the 4 gold metrics (Shading Var, SH[0], Albedo "
            "Corr, Lambertian Ratio) out of each run's training log "
            "(they are printed per-epoch, not in TensorBoard). "
            "Re-evaluate trainer.py:1393-1398 gate per epoch; find "
            "the first 10-consecutive-True run. Compare with the "
            "val_loss minimum from TensorBoard."
        ),
        "no_training": True,
        "gate_formula": (
            "is_qualified = (shading_var > 0.01) AND "
            "(0.8 <= sh0_mean <= 1.2) AND (albedo_corr > 0.7) AND "
            "(lambertian_ratio > 0.95)"
        ),
        "early_stop_rule": (
            "stop when 10 consecutive epochs are qualified "
            "(trainer.py:1411)"
        ),
        "runs": {},
    }

    for run_name, log_path in LOG_FILES.items():
        log_dir = LOGS_ROOT / run_name.replace("F-N5-gray", "p2_t22_f_n5gray_20260825") \
                                     .replace("R0-v3gray", "p2_r0_v3gray_20260825") \
                                     .replace("F-resA", "p2_t25_f_resA")
        # val_loss from TensorBoard
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
        val_total = {}
        for ef in sorted(log_dir.glob("events.out.tfevents.*")):
            ea = EventAccumulator(str(ef))
            ea.Reload()
            for s in ea.Scalars("val/total"):
                val_total.setdefault(s.step, s.value)
        best_epoch, best_val = min(val_total.items(), key=lambda x: x[1])

        parsed = parse_log(log_path)
        if not parsed:
            report["runs"][run_name] = {
                "error": f"could not parse {log_path}",
                "log_path": str(log_path),
            }
            continue

        # Re-evaluate gate from metrics
        for r in parsed:
            r["gate_pass"] = gate(r)
        # Replace text-based qualified with computed
        flags = [r["gate_pass"] for r in parsed]
        # How many text-claimed 'qualified' epochs match computed
        n_text_qualified  = sum(1 for r in parsed if r["qualified_text"] == "达标")
        n_gate_pass       = sum(flags)
        n_both_agree      = sum(1 for r in parsed
                                if r["qualified_text"] == "达标" and r["gate_pass"])
        n_text_only       = n_text_qualified - n_both_agree
        n_computed_only   = n_gate_pass - n_both_agree

        trigger_idx = first_consecutive_true(flags, 10)
        trigger_epoch = parsed[trigger_idx]["epoch"] if trigger_idx is not None else None

        if trigger_epoch is None:
            verdict = "GATE_NEVER_TRIGGERS (100 epochs necessary under gate logic)"
            budget_necessary = True
        elif trigger_epoch < 70:
            verdict = (
                f"GATE_TRIGGERS_EARLY at epoch {trigger_epoch} < 70 "
                f"(val_loss minimum at epoch {best_epoch}); 100 "
                f"epochs is WASTEFUL under the gate's own logic"
            )
            budget_necessary = False
        elif trigger_epoch < 90:
            verdict = (
                f"GATE_TRIGGERS_MID at epoch {trigger_epoch} "
                f"(val_loss minimum at epoch {best_epoch}; "
                f"gap={best_epoch - trigger_epoch})"
            )
            budget_necessary = True
        else:
            verdict = (
                f"GATE_TRIGGERS_LATE at epoch {trigger_epoch} >= 90 "
                f"(val_loss minimum at epoch {best_epoch}; "
                f"gap={trigger_epoch - best_epoch})"
            )
            budget_necessary = True

        # Per-metric first-pass epoch (forensic)
        first_pass = {}
        for metric in ("shading_var", "sh0_mean", "albedo_corr", "lambertian_ratio"):
            for r in parsed:
                if metric == "shading_var" and r["shading_var"] > 0.01:
                    first_pass[metric] = r["epoch"]; break
                if metric == "sh0_mean" and 0.8 <= r["sh0_mean"] <= 1.2:
                    first_pass[metric] = r["epoch"]; break
                if metric == "albedo_corr" and r["albedo_corr"] > 0.7:
                    first_pass[metric] = r["epoch"]; break
                if metric == "lambertian_ratio" and r["lambertian_ratio"] > 0.95:
                    first_pass[metric] = r["epoch"]; break
            else:
                first_pass[metric] = None

        report["runs"][run_name] = {
            "log_path": str(log_path.relative_to(REPO_ROOT)),
            "n_epochs_parsed": len(parsed),
            "n_text_qualified_epochs": n_text_qualified,
            "n_computed_gate_pass_epochs": n_gate_pass,
            "agreement": {
                "both_agree": n_both_agree,
                "text_only": n_text_only,
                "computed_only": n_computed_only,
            },
            "first_metric_pass_epoch": first_pass,
            "would_trigger_epoch": trigger_epoch,
            "best_val_epoch": int(best_epoch),
            "best_val_loss": round(float(best_val), 6),
            "verdict": verdict,
            "budget_necessary_under_gate_logic": budget_necessary,
            "per_epoch": parsed,
        }

    # Aggregate
    verdicts = [r.get("verdict", "") for r in report["runs"].values()
                if "verdict" in r]
    n_early  = sum(1 for v in verdicts if "EARLY" in v)
    n_late   = sum(1 for v in verdicts if "LATE" in v or "NEVER" in v)
    n_mid    = sum(1 for v in verdicts if "MID" in v)
    report["aggregate"] = {
        "n_runs_parsed": sum(1 for r in report["runs"].values()
                             if "verdict" in r),
        "n_early": n_early,
        "n_late_or_never": n_late,
        "n_mid": n_mid,
        "verdict": (
            "GATE TRIGGERS EARLY in majority → 100 epochs WASTEFUL"
            if n_early > n_late
            else "GATE TRIGGERS LATE / NEVER in majority → 100 "
                 "epochs NECESSARY (gate not binding)"
        ),
    }

    # Slim copy for storage
    slim = {
        "agent": report["agent"],
        "experiment": report["experiment"],
        "question": report["question"],
        "method": report["method"],
        "no_training": report["no_training"],
        "gate_formula": report["gate_formula"],
        "early_stop_rule": report["early_stop_rule"],
        "aggregate": report["aggregate"],
        "runs_summary": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_epoch"}
            for k, v in report["runs"].items() if "verdict" in v
        },
    }
    out_path = REPO_ROOT / "tests_audit" / "out_M8_earlystop_validity.json"
    out_path.write_text(json.dumps(slim, indent=2, ensure_ascii=False))
    # Also write the full per-epoch version
    full_path = REPO_ROOT / "tests_audit" / "out_M8_earlystop_validity_full.json"
    full_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[M8] wrote {out_path}")
    print(f"[M8] wrote {full_path}")
    print(f"[M8] aggregate: {report['aggregate']['verdict']}")
    return report


if __name__ == "__main__":
    main()
