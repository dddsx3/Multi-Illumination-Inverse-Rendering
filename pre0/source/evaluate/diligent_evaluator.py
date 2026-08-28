"""PRE-07 · DiLiGenT 评估器（合同实现，见 benchmark/DILIGENT_CONTRACT.md）。

功能：
  1. 生成并落盘固定 subsets（所有模型共用，禁止各自抽图）；
  2. 加载指定物体/子集的线性域 luma 图像（模型输入）与 GT 法线（evaluator 专用）；
  3. 对"模型预测法线 npy"计算标准指标（MAE/median/P90/<11.25°%）；
  4. mask 使用 |Normal_gt|>0 代理（数据缺官方 mask，见合同 §2/§5）。

用法（repo 根目录）:
  python pre0/source/evaluate/diligent_evaluator.py --make_subsets
  python pre0/source/evaluate/diligent_evaluator.py --eval_object bearPNG \
      --pred_npy <path> --subset_json <path> --n 10 --k 1
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)

DILIGENT_ROOT = "D:/data/DiLiGenT/pmsData"
SEED = 20260829
NS = [3, 5, 10, 25, 50, 96]
K = 5
OBJECTS = ["ballPNG", "bearPNG", "buddhaPNG", "catPNG", "cowPNG",
           "gobletPNG", "harvestPNG", "pot1PNG", "pot2PNG", "readingPNG"]


def srgb_to_linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def load_object(obj):
    """返回 dict: imgs_lin [96,H,W]（线性 luma）、light_dirs [96,3]、
    light_int [96,3]、normal_gt [3,H,W]、mask [H,W] bool（gt-normal 代理）"""
    d = os.path.join(DILIGENT_ROOT, obj)
    names = [ln.strip() for ln in open(os.path.join(d, "filenames.txt"),
                                       encoding="utf-8") if ln.strip()]
    dirs = np.loadtxt(os.path.join(d, "light_directions.txt"))
    ints = np.loadtxt(os.path.join(d, "light_intensities.txt"))
    imgs = []
    for i, fn in enumerate(names):
        p = os.path.join(d, fn)
        if not os.path.isfile(p):
            p = os.path.join(d, f"{i+1:03d}.png")
        rgb = np.asarray(Image.open(p), dtype=np.float32) / 255.0
        lin = srgb_to_linear(np.clip(rgb, 0, 1))
        luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
        imgs.append(luma.astype(np.float32))
    imgs = np.stack(imgs)
    import scipy.io as sio
    m = sio.loadmat(os.path.join(d, "Normal_gt.mat"))["Normal_gt"]  # [H,W,3]
    normal = m.transpose(2, 0, 1).astype(np.float32)
    mask = np.linalg.norm(normal, axis=0) > 1e-6
    return dict(imgs_lin=imgs, light_dirs=dirs, light_int=ints,
                normal_gt=normal, mask=mask, n_lights=len(names))


def make_subsets():
    """固定种子生成所有 (object, N, k) 子集并落盘"""
    subsets = {}
    for obj in OBJECTS:
        nl = 96
        for N in NS:
            if N == nl:
                subsets[f"{obj}|N{N}|k0"] = list(range(nl))
                continue
            import zlib
            for k in range(K):
                rng = np.random.default_rng([SEED, zlib.crc32(obj.encode()), N, k])
                subsets[f"{obj}|N{N}|k{k}"] = sorted(
                    rng.choice(nl, N, replace=False).tolist())
    path = os.path.join(_REPO, "pre0", "benchmark", "diligent_subsets.json")
    json.dump({"seed": SEED, "rule": "rng([20260829, crc32(object), N, k]) 无放回抽样",
               "subsets": subsets}, open(path, "w"), indent=1)
    print("written", path, f"({len(subsets)} subsets)")


def normal_metrics(pred, gt, mask):
    d = np.clip((pred * gt).sum(0), -1, 1)
    ang = np.degrees(np.arccos(d))[mask]
    return dict(mae=float(ang.mean()), median=float(np.median(ang)),
                p90=float(np.percentile(ang, 90)),
                good_11_25=float((ang < 11.25).mean() * 100))


def eval_pred(obj, pred_path, subset_json, N, k):
    o = load_object(obj)
    sub = json.load(open(subset_json))["subsets"][f"{obj}|N{N}|k{k}"]
    pred = np.load(pred_path)                       # [3,H,W] 相机系单位法线
    pred = pred / np.maximum(np.linalg.norm(pred, axis=0, keepdims=True), 1e-9)
    met = normal_metrics(pred, o["normal_gt"], o["mask"])
    print(json.dumps(dict(object=obj, N=N, k=k, **met), indent=2))
    return met


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_subsets", action="store_true")
    ap.add_argument("--eval_object")
    ap.add_argument("--pred_npy")
    ap.add_argument("--subset_json",
                    default=os.path.join(_REPO, "pre0", "benchmark",
                                         "diligent_subsets.json"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--k", type=int, default=0)
    a = ap.parse_args()
    if a.make_subsets:
        make_subsets()
    elif a.eval_object and a.pred_npy:
        eval_pred(a.eval_object, a.pred_npy, a.subset_json, a.n, a.k)
    else:
        print("用法: --make_subsets 或 --eval_object <obj> --pred_npy <npy> --n N --k k")
