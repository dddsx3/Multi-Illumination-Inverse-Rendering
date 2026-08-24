"""划分清单（split manifest）工具 —— Phase 1 验收条件 C1 的落地。

清单 JSON 格式：
{
  "dataset_root": "...",
  "created": "YYYY-MM-DD",
  "rule": "冻结规则说明",
  "frozen": true,
  "train": ["scene_a", ...],
  "val":   [...],
  "test":  [...]
}

test 集受冻结规则保护：只准评估、不准调参、不准改动清单中的 test 列表。
"""
import json
from typing import Dict, List


def load_manifest(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_split(path: str, split: str) -> List[str]:
    """读取指定 split 的场景名列表。test 冻结规则由使用方遵守。"""
    m = load_manifest(path)
    if split not in m or not isinstance(m[split], list):
        available = [k for k in m if isinstance(m.get(k), list)]
        raise KeyError(f"manifest 中不存在 split '{split}'（可用: {available}）")
    return sorted(m[split])
