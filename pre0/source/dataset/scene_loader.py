"""PRE-0 公共场景加载器（numpy，无 torch 依赖）。

唯一定义三套图像域（来源见 pre0/protocol/pre0_protocol.yaml）：
  train 域:  (uint8/255)^(1/2.2)   —— data_loader.py 的历史约定（网络输入/重建目标所在域）
  linear 域: 精确 sRGB 反变换       —— 物理域（GT albedo / SH / 渲染方程所在域）
  raw 域:    uint8/255             —— 磁盘原值

其余 GT 与 data_loader/validate_dataset 语义一致：depth/normal/albedo/mask
不增强不裁剪，全部 256×256 原始分辨率。
"""
import glob
import os

import numpy as np
from PIL import Image

GT_CORE = ("depth.npy", "albedo.npy", "normal.npy", "mask.npy", "sh_coeffs.npy")


def _srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """精确 sRGB 反变换（IEC 61966-2-1），v∈[0,1]"""
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def load_scene(scene_dir: str, num_lights: int = 5) -> dict:
    """读取单场景全部通道，返回 numpy dict（原始分辨率，不裁剪）。"""
    raws, train_dom, lin_dom = [], [], []
    for k in range(1, num_lights + 1):
        p = os.path.join(scene_dir, f"light_{k:03d}.png")
        v = np.asarray(Image.open(p), dtype=np.float32) / 255.0
        raws.append(v)
        train_dom.append(np.power(v, 1.0 / 2.2))       # data_loader 约定
        lin_dom.append(_srgb_to_linear(v).astype(np.float32))
    out = {
        "scene": os.path.basename(scene_dir),
        "img_raw": np.stack(raws),          # [K,H,W] 磁盘原值
        "img_train": np.stack(train_dom),   # [K,H,W] 训练域
        "img_lin": np.stack(lin_dom),       # [K,H,W] 线性域
        "sh": np.load(os.path.join(scene_dir, "sh_coeffs.npy")),      # [K,9]
    }
    for key, name in (("depth", "depth.npy"), ("albedo", "albedo.npy"),
                      ("normal", "normal.npy"), ("mask", "mask.npy")):
        out[key] = np.load(os.path.join(scene_dir, name))
    out["mask_bool"] = out["mask"][0] > 0
    missing = [f for f in GT_CORE if not os.path.isfile(os.path.join(scene_dir, f))]
    if missing:
        raise FileNotFoundError(f"{scene_dir} 缺少 {missing}")
    return out


def list_scenes(split_manifest: str, split: str) -> list:
    import json
    m = json.load(open(split_manifest, encoding="utf-8"))
    return sorted(m[split])


def scenes_with_files(root: str, scene_ids: list, num_lights: int = 5) -> list:
    ok = []
    for s in scene_ids:
        sd = os.path.join(root, s)
        if all(os.path.isfile(os.path.join(sd, f)) for f in GT_CORE) and \
           os.path.isfile(os.path.join(sd, f"light_{num_lights:03d}.png")):
            ok.append(sd)
    return ok
