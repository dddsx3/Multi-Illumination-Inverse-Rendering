import sys, json
sys.path.insert(0, ".")
from split_manifest import load_split
from data_loader import split_scene_names

root = "D:/data/synthetic_v2"
manifest_test = set(load_split("splits/synthetic_v2.json", "test"))
tr, val_now = split_scene_names(root, 0.8, 42)
val_now_set = set(val_now)
print("manifest_test:", len(manifest_test))
print("fresh_val_now :", len(val_now_set))
print("identical     :", manifest_test == val_now_set)

# v1 数据集的划分对照（若 T1.6 误用 v1 根，其 val 应为 121 场景）
try:
    tr1, val1 = split_scene_names("D:/data/synthetic", 0.8, 42)
    print("v1_val_count:", len(val1))
    v1set = set(val1)
    print("old_csv_121 与 v1_val 交集:", len(v1set & _old121) if False else "n/a")
except Exception as e:
    print("v1 probe fail:", e)