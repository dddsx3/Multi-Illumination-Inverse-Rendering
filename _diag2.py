import sys
sys.path.insert(0, ".")
from data_loader import MultiLightingDataset
ds = MultiLightingDataset(root_dir="D:/data/synthetic_v3", is_training=False, load_gt=False)
print("valid:", len(ds.valid_scenes))

# 复现 make_split_manifest 的调用方式
import inspect
sig = inspect.signature(MultiLightingDataset.__init__)
print("params:", list(sig.parameters))