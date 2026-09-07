#!/usr/bin/env python3
"""make_tables · 论文表一键重建（卡 C04 空壳，Table I–VII 接口冻结）。

宪法 §11：Table 只从 artifacts/frozen/ 重建。C20 前逐表实现。
"""

from __future__ import annotations

import argparse

TABLES = {f"Table {roman}": i for i, roman in
          enumerate(["I", "II", "III", "IV", "V", "VI", "VII"], start=1)}


def make_table(n, out_dir="paper/tables"):
    raise NotImplementedError(
        f"Table {n} 在卡 C20 实现（接口冻结：--table {n}，输入=artifacts/frozen/）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=int, required=True, choices=range(1, 8))
    args = ap.parse_args()
    make_table(args.table)


if __name__ == "__main__":
    main()
