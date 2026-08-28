"""
Experiment M3: Bug C (4:5:1 stage-adaptive config) reproducibility audit.
Location under audit: trainer.py:298-325
Claim:
  if s1_cfg + s2_cfg < total -> (stage1_end, stage2_end) = (s1, s1+s2)
  else                       -> compress to 4:5:1 of (total-1)
                                 with stage3 = total-1-floor((total-1)*9/10) >= 1
  and at every (s1, s2, total) we have stage1_end < stage2_end < total.

We DO NOT import or instantiate the trainer.  We read the relevant source
text (read-only) and re-implement the *core* branch as a pure function,
then sweep a (s1, s2, total) grid.

Backward-compat: s1_cfg=30, s2_cfg=30, total=100 -> (30, 60).
"""
from __future__ import annotations
import csv
import hashlib
import json
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
TRAINER = REPO_ROOT / "trainer.py"
CSV_PATH = SCRIPT_PATH.parent / "test_bug_c_results.csv"


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


# ---------- pure-function port of the core branch in trainer.py:307-325 ----
def define_stage_configs(s1_cfg: int, s2_cfg: int, total: int) -> tuple[int, int]:
    if s1_cfg + s2_cfg < total:
        return s1_cfg, s1_cfg + s2_cfg
    s1_len = max(1, (total - 1) * 4 // 10)
    s2_len = max(1, (total - 1) * 5 // 10)
    if s1_len + s2_len >= total:
        s1_len = max(1, (total - 1) // 2)
        s2_len = max(1, total - 1 - s1_len)
    return s1_len, s1_len + s2_len


# ---------- verifier: read trainer.py and ensure the branch text matches ---
def verify_source_matches():
    src = TRAINER.read_text(encoding="utf-8")
    needed = [
        "s1_cfg + s2_cfg < total",
        "stage1_end = s1_cfg",
        "stage2_end = s1_cfg + s2_cfg",
        "(total - 1) * 4 // 10",
        "(total - 1) * 5 // 10",
    ]
    missing = [n for n in needed if n not in src]
    return missing, src


def main():
    missing, src = verify_source_matches()
    print("=" * 78)
    print("EXPERIMENT M3 -- Bug C (4:5:1 stage-adaptive)")
    print("=" * 78)
    print(f"  script_path : {SCRIPT_PATH}")
    print(f"  trainer.py  : {TRAINER}  sha256:{file_sha256(TRAINER)}")
    print(f"  csv_path    : {CSV_PATH}")
    print()
    if missing:
        print("  [WARN] trainer.py source does not contain expected patterns:")
        for m in missing:
            print(f"    - missing: {m!r}")
        print("  The pure function port may be stale relative to trainer.py.")
    else:
        print("  [OK] trainer.py:298-325 contains the expected branch patterns.")

    totals = [10, 20, 30, 50, 70, 100]
    s1_cfgs = [30, 20, 10]
    s2_cfgs = [30, 20, 10]

    rows = []
    overall_pass = True
    for total in totals:
        for s1 in s1_cfgs:
            for s2 in s2_cfgs:
                s1_end, s2_end = define_stage_configs(s1, s2, total)
                branch = "default" if s1 + s2 < total else "compressed"
                ok_order = (1 <= s1_end) and (s1_end < s2_end) and (s2_end < total)
                # also require s1_end + s2_len == s2_end and s2_len >= 1
                s2_len = s2_end - s1_end
                ok_len = (s1_end >= 1) and (s2_len >= 1) and (s1_end + s2_len < total)
                ok = ok_order and ok_len
                if not ok:
                    overall_pass = False
                rows.append({
                    "total": total,
                    "s1_cfg": s1,
                    "s2_cfg": s2,
                    "s1_plus_s2": s1 + s2,
                    "branch": branch,
                    "stage1_end": s1_end,
                    "stage2_end": s2_end,
                    "stage3_len": total - s2_end,
                    "s1_end_lt_s2_end": int(s1_end < s2_end),
                    "s2_end_lt_total": int(s2_end < total),
                    "stage3_at_least_1": int(total - s2_end >= 1),
                    "ok": int(ok),
                })

    # Backward-compat check
    bc_s1, bc_s2 = define_stage_configs(30, 30, 100)
    bc_ok = (bc_s1 == 30 and bc_s2 == 60)
    if not bc_ok:
        overall_pass = False

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"  {'total':>6} {'s1':>4} {'s2':>4} {'s1+s2':>6} {'branch':>10} "
          f"{'s1_end':>7} {'s2_end':>7} {'s3_len':>7} {'ok':>4}")
    for r in rows:
        print(f"  {r['total']:>6} {r['s1_cfg']:>4} {r['s2_cfg']:>4} "
              f"{r['s1_plus_s2']:>6} {r['branch']:>10} "
              f"{r['stage1_end']:>7} {r['stage2_end']:>7} {r['stage3_len']:>7} "
              f"{r['ok']:>4}")
    print()
    print(f"  Backward-compat (s1=30, s2=30, total=100): "
          f"({bc_s1}, {bc_s2})  -> {'PASS' if bc_ok else 'FAIL'}")
    print()
    fails = [r for r in rows if r["ok"] == 0]
    if fails:
        print("  FAILED samples:")
        for f in fails:
            print(f"    total={f['total']} s1={f['s1_cfg']} s2={f['s2_cfg']} "
                  f"-> (s1_end={f['stage1_end']}, s2_end={f['stage2_end']})")
    else:
        print("  No failed samples in the 6x3x3=54-cell sweep.")
    print()
    print("  FALSIFIER (what would overturn the conclusion):")
    print("    - If s2_end == total for any (s1,s2,total) cell, stage3 is")
    print("      starved (0 epochs) -- the compressed branch's floor")
    print("      '(total-1)*4//10 + (total-1)*5//10' would have to be < total")
    print("      for this to happen.  4+5=9, so 9//10 of (total-1) is always")
    print("      < total for total>=2.  Verify: total=10 -> s1_end=3, s2_end=7.")
    print("    - If backward-compat cell returns != (30, 60), the original")
    print("      semantic for 100-epoch runs is broken.")
    print("    - If 'default' branch is selected for s1+s2 >= total (i.e. when")
    print("      s1_cfg=30, s2_cfg=30, total=10, the default branch would emit")
    print("      s1_end=30 > total=10).  The check must fall to 'compressed'.")
    print()
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}  "
          f"({sum(r['ok'] for r in rows)}/{len(rows)} samples, "
          f"bc={'PASS' if bc_ok else 'FAIL'})")

    (SCRIPT_PATH.parent / "test_bug_c_summary.json").write_text(
        json.dumps({
            "all_pass": overall_pass,
            "backward_compat": {"s1": bc_s1, "s2": bc_s2, "ok": bc_ok},
            "rows": rows,
            "missing_source_patterns": missing,
            "script_sha256": file_sha256(SCRIPT_PATH),
            "trainer_sha256": file_sha256(TRAINER),
        }, indent=2)
    )


if __name__ == "__main__":
    main()
