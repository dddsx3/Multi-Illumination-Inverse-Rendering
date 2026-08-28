"""用修复后的评估重算三个 probe 的 test 指标（覆盖 *_summary.json）。"""
import json
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "probe_models")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train")))

from probes import PROBES  # noqa: E402
from scene_loader import list_scenes, scenes_with_files  # noqa: E402
from train_probe import SceneBatcher, evaluate  # noqa: E402

DATA_ROOT = "D:/data/synthetic_v3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifest = os.path.join(_REPO, "pre0", "protocol", "split_manifest.json")
te = scenes_with_files(DATA_ROOT, list_scenes(manifest, "test"))

for probe in "ABC":
    ck_path = os.path.join(_REPO, "pre0", "checkpoints", f"probe_{probe}_best.pth")
    if not os.path.isfile(ck_path):
        print(f"probe {probe}: no checkpoint, skip")
        continue
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model = PROBES[probe]().to(device)
    model.load_state_dict(ck["model"])
    test_b = SceneBatcher(te, 1, False, device)
    met = evaluate(model, test_b, device)
    out = dict(probe=probe, params=ck.get("params"), best_epoch=ck.get("epoch"),
               best_val_si_mae=ck.get("val_si_mae"), test=met,
               note="recon_psnr 修复广播后重算；评估域=linear")
    path = os.path.join(_REPO, "pre0", "probe_results", f"probe_{probe}_summary.json")
    old = {}
    if os.path.isfile(path):
        old = json.load(open(path, encoding="utf-8"))
        if "best_val_si_mae" not in out or out["best_val_si_mae"] is None:
            out["best_val_si_mae"] = old.get("best_val_si_mae")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print(probe, json.dumps(met))
