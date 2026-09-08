"""B3 · DiLiGenT loader（收敛单一入口 + manifest 接入）。

来源：REPO_MIGRATION B3 行 1 —— eval_diligent/ + evaluate_diligent.py 收敛。
口径（与 exp8r 系列逐位一致）：灰度图、light_intensities 归一（禁峰值）、
mask>128、Normal_gt 截取到 mask 内。CI05 只承担 real sanity（宪法 §4.5）。
manifest 契约：make_manifest 产出对象清单 + checksum（sha256）——先写 manifest 再跑实验（宪法 §11）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_object(d):
    """单对象加载：返回 dict(dirs (96,3), normals_gt (P,3), I_norm (96,P), mask (H,W) bool)。

    数值口径与 exp8r_diligent_discrimination_v3.load_object 逐位一致：
    I_norm[k, p] = gray_k[mask] / light_intensities[k]（归一化，禁峰值缩放）。
    """
    d = Path(d)
    dirs = np.loadtxt(d / "light_directions.txt")
    ints = np.loadtxt(d / "light_intensities.txt")[:, 0]
    N_gt = sio.loadmat(d / "Normal_gt.mat")["Normal_gt"]
    mask = np.array(Image.open(d / "mask.png").convert("L")) > 128
    imgs = np.stack([np.array(Image.open(str(d / f"{i:03d}.png")).convert("L")).astype(float)
                     for i in range(1, 97)])
    I = imgs[:, mask] / ints[:, None]
    return dict(dirs=dirs, normals_gt=N_gt[mask], I_norm=I, mask=mask)


def make_manifest(data_root, objects=None):
    """对象清单 manifest（先写 manifest 再跑实验，宪法 §11）。

    返回 dict：objects[{name, n_lights, n_pixels, files{key: {sha256, bytes}}}]、
    data_root、created_utc。CI05 使用前把本 dict 落盘并 checksum 冻结。
    """
    import datetime
    root = Path(data_root)
    names = objects or sorted(x.name for x in root.iterdir() if x.is_dir())
    entries = []
    for name in names:
        d = root / name
        files = {}
        for key, fn in [("light_directions", "light_directions.txt"),
                        ("light_intensities", "light_intensities.txt"),
                        ("normal_gt", "Normal_gt.mat"),
                        ("mask", "mask.png")]:
            p = d / fn
            files[key] = dict(sha256=sha256_file(p), bytes=p.stat().st_size)
        obj = load_object(d)
        entries.append(dict(name=name, n_lights=int(obj["dirs"].shape[0]),
                            n_pixels=int(obj["normals_gt"].shape[0]), files=files))
    return dict(data_root=str(root), objects=entries,
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
