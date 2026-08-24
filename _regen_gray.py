"""
INC-0004 修复执行：统一 v3 数据集灰度推导。
从已存档的彩色 PNG（权威源）按编码域 BT.709 luma 重导灰度 PNG，
消除僵尸渲染实例造成的混合推导不一致。
"""
import os
import numpy as np
from PIL import Image

ROOT = "D:/data/synthetic_v3"
fixed = skipped = 0
dirs = sorted(os.listdir(ROOT))
for name in dirs:
    sd = os.path.join(ROOT, name)
    if not os.path.isdir(sd) or name.startswith("_"):
        continue
    changed = False
    for k in range(1, 6):
        rgb_p = os.path.join(sd, f"light_{k:03d}_rgb.png")
        gray_p = os.path.join(sd, f"light_{k:03d}.png")
        if not os.path.isfile(rgb_p):
            continue
        rgb8 = np.asarray(Image.open(rgb_p))
        new_gray = np.round(
            0.2126 * rgb8[..., 0].astype(np.float32)
            + 0.7152 * rgb8[..., 1].astype(np.float32)
            + 0.0722 * rgb8[..., 2].astype(np.float32)).astype(np.uint8)
        old = np.asarray(Image.open(gray_p)) if os.path.isfile(gray_p) else None
        if old is None or old.shape != new_gray.shape or (old != new_gray).any():
            Image.fromarray(new_gray, mode="L").save(gray_p)
            changed = True
        else:
            skipped += 1
    if changed:
        fixed += 1
    else:
        skipped += 1

print(f"scenes regenerated: {fixed}, already-consistent: {skipped}")