"""
生成数据集三向划分清单（train / val / test）—— 验收条件 C1 落地。

规则（v2 数据集，冻结）：
1. test := 旧版 80/20 划分（seed42）的验证子集，共 127 场景。
   该子集从未参与 T1.6 基线模型的训练，且已用于其评估 => 冻结为正式测试集，
   保证与 Phase 1 基线可比；Phase 2 起只准评估、不准调参。
2. val := 从旧训练子集中按种子 43 再抽 10%（50 场景），供 Phase 2 模型选择。
3. train := 旧训练子集其余部分。

用法:
  python make_split_manifest.py --data_root D:/data/synthetic_v2 \
      --out splits/synthetic_v2.json
"""
import argparse
import datetime
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import MultiLightingDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="splits/synthetic_v2.json")
    ap.add_argument("--legacy_split", type=float, default=0.8,
                    help="旧版 train 比例（决定 test 冻结子集，勿改）")
    ap.add_argument("--legacy_seed", type=int, default=42,
                    help="旧版划分种子（勿改，须与 T1.6 基线一致）")
    ap.add_argument("--val_frac_of_train", type=float, default=0.10,
                    help="从旧训练子集中切出的新验证比例")
    ap.add_argument("--carve_seed", type=int, default=43)
    args = ap.parse_args()

    probe = MultiLightingDataset(root_dir=args.data_root, is_training=False,
                                 load_gt=False)
    all_names = sorted(probe.valid_scenes)

    g = torch.Generator().manual_seed(args.legacy_seed)
    perm = torch.randperm(len(all_names), generator=g).tolist()
    n_legacy_train = max(int(args.legacy_split * len(all_names)), 1)
    legacy_train = sorted(all_names[i] for i in perm[:n_legacy_train])
    test = sorted(all_names[i] for i in perm[n_legacy_train:])

    g2 = torch.Generator().manual_seed(args.carve_seed)
    perm2 = torch.randperm(len(legacy_train), generator=g2).tolist()
    n_val = max(int(args.val_frac_of_train * len(legacy_train)), 1)
    val = sorted(legacy_train[i] for i in perm2[:n_val])
    val_set = set(val)
    train = sorted(s for s in legacy_train if s not in val_set)

    assert not (set(train) & set(val)) and not (set(train) & set(test)) \
        and not (set(val) & set(test)), "split 交集不为空"
    assert len(train) + len(val) + len(test) == len(all_names)

    manifest = {
        "dataset_root": os.path.abspath(args.data_root),
        "created": datetime.date.today().isoformat(),
        "frozen": True,
        "rule": (
            "test = 旧版80/20划分(seed={ls})的验证子集，未参与T1.6基线训练且已被"
            "其评估 -> 冻结为正式测试集（只准评估、不准调参、不准改动）；"
            "val = 旧训练子集的10%(seed={cs})，供模型选择；train = 其余。"
            "来源：验收裁决书 C1。"
        ).format(ls=args.legacy_seed, cs=args.carve_seed),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "train": train,
        "val": val,
        "test": test,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("manifest written:", args.out)
    print("  train =", len(train))
    print("  val   =", len(val))
    print("  test  =", len(test), "(frozen)")


if __name__ == "__main__":
    main()
