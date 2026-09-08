#!/usr/bin/env python3
"""audit_claims: wording-ban CI (semantic pattern family; F2 / audit finding 1).

Usage: python scripts/audit_claims.py --manuscript paper
Exit: 0 = clean, 1 = banned phrase hit.
"""
import argparse
import re
import sys
from pathlib import Path

BANNED_PATTERNS = [
    r"structurally blind",
    r"universally superior",
    r"always beats",
    r"all scalar[^.]{0,40}(fail|blind)",
    r"has not been studied",
    r"remains unexplored",
    r"first to (study|address|propose)",
    r"agrees quantitatively after an affine scale",
    r"real-world.{0,30}(validated|prediction)",
    r"(?<!claim no )(?<!no )new (information geometry|Cram|CRB)",
    r"universal (new )?(theorem|threshold)",
    r"robust to.{0,20}misspec",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default="paper")
    ap.add_argument("--registry", default="CLAIMS_REGISTRY.yaml")
    args = ap.parse_args()
    mroot = Path(args.manuscript)
    files = sorted(set(
        f for f in [mroot / "manuscript_v0.8.md", *mroot.glob("manuscript_v0.*.md")]
        if f.exists()))
    hits = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for pat in BANNED_PATTERNS:
            for m in re.finditer(pat, text, re.I):
                line_no = text[: m.start()].count(chr(10)) + 1
                ctx = text[max(0, m.start() - 40): m.end() + 30].replace(chr(10), " ")
                hits.append((str(f), line_no, m.group(0), ctx))
    if hits:
        for f, ln, g, ctx in hits:
            print(f"FAIL {f}:{ln} [{g}] ...{ctx}...")
        sys.exit(1)
    print(f"audit_claims: {len(files)} manuscript files x "
          f"{len(BANNED_PATTERNS)} semantic patterns -> 0 hits")


if __name__ == "__main__":
    main()
