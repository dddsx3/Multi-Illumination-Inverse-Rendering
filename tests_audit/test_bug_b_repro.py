"""
Experiment M2: Bug B (Lambertian Ratio numerical stability) reproducibility
audit.
Location under audit: trainer.py:987-994
Claim: The old ratio (rendered**2).mean() / (images**2).mean() is
       numerically unstable:
         - rendered == 0  ->  0/positive = 0   (false negative: still 0.0)
         - rendered == 10 * images  ->  100   (extreme positive)
         - one RGB channel = 0        ->  NaN  (0/0 if images that ch also 0)
       The new formula 1 - mean(|dE|/E) is in [0, 1] and never NaN/Inf for
       strictly-positive images; in the 'all-zero images' edge case the
       eps keeps the denominator finite.

We re-implement BOTH formulas in isolation (no trainer.py import) and
sweep a 5-case x 2-modality matrix.  Each cell is asserted to match the
documented expectation.  Trainer.py is read but not modified.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
TRAINER = REPO_ROOT / "trainer.py"
CSV_PATH = SCRIPT_PATH.parent / "test_bug_b_results.csv"


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


# --- New formula ported from trainer.py:987-994 (verbatim semantics) -------
def new_lambertian_ratio(rendered: torch.Tensor,
                         images: torch.Tensor,
                         eps: float = 1e-6) -> float:
    """1 - mean_channel( |E_r - E_i| / (E_i + eps) ), clipped to [0, 1]."""
    rendered_ch = rendered.detach().abs().mean(dim=(0, 2, 3))
    images_ch = images.detach().abs().mean(dim=(0, 2, 3))
    ch_rel_diff = ((rendered_ch - images_ch).abs()
                   / (images_ch + eps))
    energy_match = 1.0 - ch_rel_diff.mean().item()
    return max(0.0, min(1.0, energy_match))


# --- Old formula, as the docstring describes -------------------------------
def old_lambertian_ratio(rendered: torch.Tensor,
                         images: torch.Tensor) -> float:
    """(rendered**2).mean() / (images**2).mean()  -- unbounded, NaN/Inf-prone.

    Returns NaN if division-by-zero happens (we use math.nan rather than
    raising so the audit can record the crash point)."""
    import math
    num = (rendered ** 2).mean().item()
    den = (images ** 2).mean().item()
    if den == 0:
        return math.nan
    return num / den


def make_inputs(case: str, *, seed: int = 42, modality: str = "rgb"):
    rng = np.random.default_rng(seed)
    if modality == "rgb":
        shape = (2, 3, 32, 32)
    else:  # gray
        shape = (2, 1, 32, 32)
    if case == "perfect":
        x = torch.from_numpy(rng.uniform(0.2, 0.9, size=shape)).float()
        return x.clone(), x.clone()
    if case == "collapse":
        x = torch.from_numpy(rng.uniform(0.2, 0.9, size=shape)).float()
        return torch.zeros_like(x), x
    if case == "explode":
        x = torch.from_numpy(rng.uniform(0.2, 0.9, size=shape)).float()
        return 10.0 * x, x
    if case == "channel_imbalance":
        x = torch.from_numpy(rng.uniform(0.2, 0.9, size=shape)).float()
        r = x.clone()
        if modality == "rgb":
            r[:, 0, :, :] = 0.0
            r[:, 2, :, :] = 2.0 * x[:, 2, :, :]
        else:
            r = torch.zeros_like(x)
        return r, x
    if case == "zero_image":
        x = torch.from_numpy(rng.uniform(0.2, 0.9, size=shape)).float()
        return x, torch.zeros_like(x)
    raise ValueError(f"unknown case {case!r}")


def _is_finite(x: float) -> bool:
    return (x == x) and (x not in (float("inf"), float("-inf")))


def main():
    cases = ["perfect", "collapse", "explode", "channel_imbalance", "zero_image"]
    mods = ["gray", "rgb"]
    rows = []
    overall_pass = True
    for case in cases:
        for mod in mods:
            r, i = make_inputs(case, seed=42, modality=mod)
            old = old_lambertian_ratio(r, i)
            new = new_lambertian_ratio(r, i)
            row = {
                "case": case,
                "modality": mod,
                "old_ratio": old,
                "new_ratio": new,
                "old_finite": _is_finite(old),
                "new_finite": _is_finite(new),
                "new_in_unit": (0.0 <= new <= 1.0 + 1e-9),
            }
            # case-specific asserts
            if case == "perfect":
                ok = (abs(new - 1.0) < 1e-5) and (abs(old - 1.0) < 1e-5)
                row["expectation"] = "old~1 new=1"
            elif case == "collapse":
                # Old formula = 0.  New formula = 1 - |0 - E_i|/(E_i+eps) per ch
                #                  = 1 - 1*E_i/(E_i+eps) -> ~0 (since E_i > 0)
                ok = (abs(old) < 1e-9) and _is_finite(new) and (0.0 <= new <= 1.0)
                row["expectation"] = "old=0 new in [0,1]"
            elif case == "explode":
                # Old = 100.  New = 1 - |10-1| = -8 -> clipped to 0
                ok = (old > 50) and _is_finite(new) and (abs(new) < 1e-9)
                row["expectation"] = "old>>1 new clipped to 0"
            elif case == "channel_imbalance":
                if mod == "rgb":
                    expected_new = 1.0 / 3.0
                    ok = (abs(new - expected_new) < 1e-3) and _is_finite(new)
                    row["expectation"] = f"new~1/3 ({expected_new:.4f})"
                else:
                    ok = (abs(old) < 1e-9) and _is_finite(new) and (new < 0.05)
                    row["expectation"] = "old=0 new~0"
            elif case == "zero_image":
                # Old formula divides by 0 -> NaN.  New formula has eps floor
                # and remains finite + in [0,1].  We assert the new is
                # well-behaved; the old's NaN is *expected breakage* and is
                # documented separately.
                ok = _is_finite(new) and (0.0 <= new <= 1.0) and (not _is_finite(old))
                row["expectation"] = "old=NaN new=0 (eps keeps new finite)"
            else:
                ok = False
            row["sample_pass"] = bool(ok)
            rows.append(row)
            if not ok:
                overall_pass = False

    # CSV
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    trainer_hash = file_sha256(TRAINER)
    script_hash = file_sha256(SCRIPT_PATH)

    print("=" * 78)
    print("EXPERIMENT M2 -- Bug B (Lambertian Ratio numerical stability)")
    print("=" * 78)
    print(f"  script_path : {SCRIPT_PATH}")
    print(f"  script_hash : sha256:{script_hash}")
    print(f"  trainer.py  : {TRAINER}  sha256:{trainer_hash}")
    print(f"  csv_path    : {CSV_PATH}")
    print()
    print(f"  {'case':>20} {'mod':>5} {'old_ratio':>14} {'new_ratio':>14} "
          f"{'old_fin':>8} {'new_fin':>8} {'new∈[0,1]':>10} {'pass':>6}")
    for r in rows:
        print(f"  {r['case']:>20} {r['modality']:>5} "
              f"{r['old_ratio']:>14.4e} {r['new_ratio']:>14.6f} "
              f"{str(r['old_finite']):>8} {str(r['new_finite']):>8} "
              f"{str(r['new_in_unit']):>10} {('Y' if r['sample_pass'] else 'N'):>6}")
    print()
    print("  Crash points (old formula):")
    for r in rows:
        if (not r["old_finite"]) or r["old_ratio"] > 50 or r["old_ratio"] < 0:
            print(f"    case={r['case']:>20} mod={r['modality']:>3} "
                  f"old={r['old_ratio']!r:>20}  (BROKEN)")
    print()
    print("  FALSIFIER (what would overturn the conclusion):")
    print("    - If new_ratio(perfect, *) is not within 1e-5 of 1.0, the new")
    print("      formula does not preserve the 'matched=1' invariant.")
    print("    - If new_ratio returns NaN or Inf on any case (incl. zero_image")
    print("      or channel_imbalance), the eps term is mis-tuned.")
    print("    - If new_ratio exceeds 1.0 on any case, the clipping is wrong.")
    print("    - If old_ratio on the 'perfect' case deviates from 1.0, we have")
    print("      a re-implementation bug, not a real audit.")
    print()
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}  "
          f"({sum(r['sample_pass'] for r in rows)}/{len(rows)} samples)")

    (SCRIPT_PATH.parent / "test_bug_b_summary.json").write_text(
        json.dumps({
            "all_pass": overall_pass,
            "rows": rows,
            "script_sha256": script_hash,
            "trainer_sha256": trainer_hash,
        }, indent=2)
    )


if __name__ == "__main__":
    main()
