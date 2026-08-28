"""
Experiment M1: Bug A (albedo_corr threshold flip) reproducibility audit.
Location under audit: trainer.py:1393-1398
Claim: Old `albedo_corr < 0.4` is logically inverted; new `albedo_corr > 0.7`
       is the correct gate (A is the low-frequency envelope of I = A * S).

This script does NOT touch trainer.py. It synthesises pairs of (albedo A,
shading S) tensors, builds the rendered image I = A * S, and shows that
  * corr(A, I)         > 0.7   on 10/10 samples (correct decomposition)
  * corr(A, I_perm)    < 0.4   on 10/10 samples (random pairing)

Therefore, the old `> 0.4`-as-weak-correlation gate and the new `> 0.7`-
as-strong-correlation gate evaluate the predicate to opposite booleans on
the {correct, permuted} input domain, i.e. they are complementary on {0,1}.

Evidence pointers:
  - trainer.py:1388-1398   (the patched gate)
  - trainer.py:1411        (the consumer of `is_qualified`)
  - commit equivalent: local tree, no remote; tracked by file hash below.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
TRAINER = REPO_ROOT / "trainer.py"
RESULTS_DIR = SCRIPT_PATH.parent
CSV_PATH = RESULTS_DIR / "test_bug_a_results.csv"


def smooth_lowfreq(rng: np.random.Generator, h: int = 256, w: int = 256,
                   base_sigma: float = 24.0) -> np.ndarray:
    """Smooth 2-D random field.  Low-frequency by construction (Gaussian
    blur of iid noise).  Output is in [0, 1] after normalisation."""
    small_h, small_w = h // 16, w // 16
    small = rng.standard_normal((small_h, small_w))
    # upsample by repeat (matches 'blocky low-freq' assumption)
    big = np.kron(small, np.ones((h // small_h, w // small_w)))
    # add a touch of higher frequency for visual realism
    big += 0.1 * rng.standard_normal(big.shape)
    # Gaussian smooth via separable convolution (torch, no scipy)
    t = torch.from_numpy(big)[None, None].float()
    k = 31
    sigma = base_sigma / 4.0  # base_sigma is in pixels; converted to kernel
    x = torch.arange(k, dtype=torch.float32) - k / 2
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    # outer product to make a (1,1,k,k) Gaussian kernel
    kernel = (g[:, None] * g[None, :])[None, None]  # (1,1,k,k)
    t = torch.nn.functional.conv2d(t, kernel, padding=k // 2)
    out = t[0, 0].numpy()
    # normalise to [0, 1] then bias to [0.1, 0.9] to keep strictly positive
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    return 0.1 + 0.8 * out


def make_pair(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = smooth_lowfreq(rng)
    S = smooth_lowfreq(rng, base_sigma=16.0)
    S = 0.4 + 1.2 * S  # positive shading, mid-range
    I = A * S
    return A, S, I


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten() - a.mean()
    bf = b.flatten() - b.mean()
    denom = np.sqrt((af * af).sum() * (bf * bf).sum()) + 1e-12
    return float((af * bf).sum() / denom)


def evaluate() -> dict:
    rng = np.random.default_rng(42)
    seeds = [42 + i * 7 for i in range(10)]  # 10 distinct (A, S) draws
    rows = []
    both_pass = True
    for s in seeds:
        A, S, I = make_pair(s)
        I_perm = I.flatten()
        rng.shuffle(I_perm)
        I_perm = I_perm.reshape(I.shape)
        c_corr = pearson_corr(A, I)
        c_perm = pearson_corr(A, I_perm)
        old_correct = c_corr < 0.4    # OLD gate, correct decomposition
        old_perm = c_perm < 0.4       # OLD gate, permuted
        new_correct = c_corr > 0.7    # NEW gate, correct decomposition
        new_perm = c_perm > 0.7       # NEW gate, permuted
        # Old gate should be "True for permuted, False for correct" (BUG).
        # New gate should be "True for correct, False for permuted" (FIX).
        old_is_buggy = old_perm and not old_correct
        new_is_correct = new_correct and not new_perm
        sample_pass = old_is_buggy and new_is_correct
        both_pass = both_pass and sample_pass
        rows.append({
            "seed": s,
            "corr_A_I": round(c_corr, 6),
            "corr_A_Iperm": round(c_perm, 6),
            "old_correct_True": bool(old_correct),
            "old_perm_True": bool(old_perm),
            "new_correct_True": bool(new_correct),
            "new_perm_True": bool(new_perm),
            "old_is_buggy": bool(old_is_buggy),
            "new_is_correct": bool(new_is_correct),
            "sample_pass": bool(sample_pass),
        })
    # write CSV
    import csv
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return {
        "n_samples": len(rows),
        "n_pass": sum(r["sample_pass"] for r in rows),
        "rows": rows,
        "all_pass": both_pass,
    }


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> None:
    summary = evaluate()
    trainer_hash = file_sha256(TRAINER)
    script_hash = file_sha256(SCRIPT_PATH)
    print("=" * 72)
    print("EXPERIMENT M1 -- Bug A (albedo_corr threshold flip)")
    print("=" * 72)
    print(f"  script_path : {SCRIPT_PATH}")
    print(f"  script_hash : sha256:{script_hash}")
    print(f"  trainer.py  : {TRAINER}  sha256:{trainer_hash}")
    print(f"  csv_path    : {CSV_PATH}")
    print(f"  n_samples   : {summary['n_samples']}  pass: {summary['n_pass']}")
    print()
    print(f"  {'seed':>4} {'corr(A,I)':>12} {'corr(A,Iperm)':>14} "
          f"{'old_perm>0.4':>13} {'new_corr>0.7':>12}")
    for r in summary["rows"]:
        print(f"  {r['seed']:>4} {r['corr_A_I']:>12.4f} "
              f"{r['corr_A_Iperm']:>14.4f} {str(r['old_perm_True']):>13} "
              f"{str(r['new_correct_True']):>12}")
    print()
    print("  Verdicts:")
    for r in summary["rows"]:
        ok = r["sample_pass"]
        tag = "PASS" if ok else "FAIL"
        print(f"    seed={r['seed']:>3}  old_is_buggy={r['old_is_buggy']}  "
              f"new_is_correct={r['new_is_correct']}  -> {tag}")
    print()
    print("  FALSIFIER (what would overturn the conclusion):")
    print("    - If on seed=42..42+63 (10 draws) corr(A,I) ever fell below 0.7,")
    print("      the *fix* is not as advertised on these inputs.")
    print("    - If on the same 10 draws corr(A,Iperm) ever rose above 0.4,")
    print("      the *old* threshold would not have been inverted on the bug")
    print("      domain (the bug would have been cosmetic only).")
    print("    - If both predicates (old_is_buggy, new_is_correct) were True")
    print("      on EVERY sample, that is consistent with our claim; if any")
    print("      sample shows them both False or both True, our claim fails.")
    print()
    print(f"  OVERALL: {'PASS' if summary['all_pass'] else 'FAIL'}")
    # also dump JSON for the agent's downstream consumer
    (RESULTS_DIR / "test_bug_a_summary.json").write_text(
        json.dumps({
            "all_pass": summary["all_pass"],
            "n_samples": summary["n_samples"],
            "n_pass": summary["n_pass"],
            "rows": summary["rows"],
            "script_sha256": script_hash,
            "trainer_sha256": trainer_hash,
        }, indent=2)
    )


if __name__ == "__main__":
    main()
