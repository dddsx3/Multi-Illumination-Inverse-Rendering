"""B3/C14 · OpenIllumination loader（HF `OpenIllumination/OpenIllumination`，CC BY 4.0）。

数据形态（2026-09-08 侦察后降采样选择，见 manifest）：
  - 完整单对象 ~41 GB（RAW 传感器转储为主）>> 主控计划书 2GB 触发线；
  - 选择 = 取缩略图层：`OLAT/<obj>/Lights/<NNN>/com_masked_thumbnail/<CAM>.png`
    （合成已掩码，200×273 RGBA，背景黑）+ `output/com_masks/<CAM>.png`（对象掩码）；
  - GT 光照 = 数据集根 `light_pos.npy`（142, 3）灯位（全局共享，灯球半径 ~1m，
    对象位于原点）→ 方向 = −pos/‖pos‖（灯指向对象）；
  - 单相机视图 CAM="A1"（多视图按需幂等补下，snapshot_download）。

先写 manifest 再跑实验：`make_manifest` 逐文件 sha256 + 降采样选择 + 对象清单；
development/test 清单冻结后不可改（dev_selection.json，C14 已冻结 8 对象）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

N_LIGHTS = 142
CAMERA = "A1"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_object(data_root, obj_name, camera=CAMERA, data_meta=None):
    """单对象加载。

    返回 dict(images (142,H,W,3) float[0,1]、mask (H,W) bool、
    light_positions (142,3)、light_directions (142,3) 单位向量（−pos/‖pos‖）、
    meta（对象名/相机/降采样选择）)。
    """
    root = Path(data_root)
    odir = root / "OLAT" / obj_name
    lpos_path = root / "light_pos.npy"
    if not lpos_path.exists():
        lpos_path = Path(data_meta or root.parent / "OpenIllumination_meta") / "light_pos.npy"
    light_pos = np.load(lpos_path)
    assert light_pos.shape == (N_LIGHTS, 3), f"GT 灯位形状异常: {light_pos.shape}"
    norm = np.linalg.norm(light_pos, axis=1, keepdims=True)
    light_dir = -light_pos / norm                      # 灯指向对象（对象在原点口径）

    imgs, alphas, missing = [], [], []
    for i in range(N_LIGHTS):
        p = odir / "Lights" / f"{i:03d}" / "com_masked_thumbnail" / f"{camera}.png"
        if not p.exists():
            missing.append(i)
            continue
        arr = np.array(Image.open(p))
        alphas.append(arr[..., 3])
        imgs.append(arr[..., :3].astype(np.float64) / 255.0)
    if missing:
        raise FileNotFoundError(f"{obj_name} 缺 {len(missing)} 张灯位图（前 5: {missing[:5]}）")
    images = np.stack(imgs)                            # (142, H, W, 3)
    # 掩码 = thumbnail alpha 通道（com_masked 合成层自带；与图像 200×273 精确对齐。
    # output/com_masks 是 4096×3000 全分辨率原始掩码，需降采样且含相机视野差异——
    # 只作交叉校验不作主掩码，避免重采样插值污染）
    alpha0 = alphas[0]
    mask = alpha0 > 128
    for al in alphas[1:]:
        if not np.array_equal(al > 128, mask):
            raise ValueError(f"{obj_name} 各灯位图 alpha 掩码不一致——检查数据完整性")
    return dict(images=images, mask=mask,
                light_positions=light_pos, light_directions=light_dir,
                meta=dict(obj_name=obj_name, camera=camera,
                          resolution=[int(images.shape[2]), int(images.shape[1])],
                          mask_frac=float(mask.mean()),
                          mask_source="com_masked_thumbnail alpha channel（200x273 精确对齐；"
                                      "output/com_masks 全分辨率仅交叉校验用）",
                          downsampling_verdict="thumbnail 层（完整对象 41GB >> 2GB 触发线；"
                                               "RAW/全分辨率不取，多视图按需幂等补下）"))


def make_manifest(data_root, objects, camera=CAMERA, data_meta=None,
                  selection_note=None):
    """数据合同 manifest（先写 manifest 再跑实验）。"""
    import datetime
    root = Path(data_root)
    lpos_path = root / "light_pos.npy"
    if not lpos_path.exists():
        lpos_path = Path(data_meta or root.parent / "OpenIllumination_meta") / "light_pos.npy"
    entries = []
    for name in objects:
        odir = root / "OLAT" / name
        files = {}
        for i in range(N_LIGHTS):
            p = odir / "Lights" / f"{i:03d}" / "com_masked_thumbnail" / f"{camera}.png"
            files[f"light_{i:03d}"] = dict(sha256=sha256_file(p), bytes=p.stat().st_size)
        mp = odir / "Lights" / "000" / "com_masked_thumbnail" / f"{camera}.png"
        files["mask_source"] = dict(sha256=sha256_file(mp), bytes=mp.stat().st_size)
        entries.append(dict(name=name, files=files))
    return dict(
        data_root=str(root), camera=camera, n_lights=N_LIGHTS,
        lighting_gt=dict(path=str(lpos_path), sha256=sha256_file(lpos_path),
                         shape=[N_LIGHTS, 3]),
        objects=entries,
        downsampling_verdict="单对象全量 ~41GB >> 2GB 触发线 → 取 com_masked_thumbnail "
                             "(200x273) + com_masks；RAW 与全分辨率不取",
        selection_note=selection_note or "",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def save_manifest(path, **kwargs):
    mf = make_manifest(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mf, ensure_ascii=False, indent=1), encoding="utf-8")
    return mf
