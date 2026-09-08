"""C14 绑定测试 · OpenIllumination loader + manifest。

绑定：REPO_MIGRATION B3 行 2 —— 已知对象光照 GT 数值 sanity。
真数据依赖（D:/data/OpenIllumination，C14 已下载 development 8 对象）：
不存在时整文件 skip（数据合同由 manifest 冻结，宪法 §11）。
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from calibinfo.datasets.openillumination import load_object, make_manifest

DATA_ROOT = "D:/data/OpenIllumination"
META = "D:/data/OpenIllumination_meta"
HAS_DATA = Path(DATA_ROOT, "OLAT", "obj_01_egg", "Lights", "000",
                "com_masked_thumbnail", "A1.png").exists()

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="本机无 OpenIllumination development 数据")


def test_light_gt_reproduces_official_values():
    """验收：loader 复现官方 GT 光照数值——light_pos.npy 逐位一致 (142,3)，
    方向 = −pos/‖pos‖ 单位化。"""
    obj = load_object(DATA_ROOT, "obj_01_egg", data_meta=META)
    official = np.load(Path(META) / "light_pos.npy")
    assert obj["light_positions"].shape == (142, 3)
    assert np.array_equal(obj["light_positions"], official)
    assert np.allclose(np.linalg.norm(obj["light_directions"], axis=1), 1.0)
    assert np.allclose(obj["light_directions"],
                       -official / np.linalg.norm(official, axis=1, keepdims=True))


def test_images_mask_alignment():
    """142 图 + alpha 掩码对齐 + 与全分辨率掩码覆盖率交叉校验（±0.02）。"""
    obj = load_object(DATA_ROOT, "obj_01_egg", data_meta=META)
    assert obj["images"].shape == (142, 273, 200, 3)
    assert obj["images"].min() >= 0 and obj["images"].max() <= 1
    frac_thumb = obj["meta"]["mask_frac"]
    m_full = np.array(Image.open(Path(DATA_ROOT) / "OLAT" / "obj_01_egg"
                                 / "output" / "com_masks" / "A1.png"))
    frac_full = float((m_full > 128).mean())
    assert abs(frac_thumb - frac_full) < 0.02


def test_manifest_frozen_and_complete():
    """manifest：8 development 对象逐文件 sha256 + 降采样裁决记录（宪法 §11）。"""
    mf = make_manifest(DATA_ROOT, json.loads(
        (Path(META) / "dev_selection.json").read_text(encoding="utf-8"))["selection"],
        data_meta=META)
    assert len(mf["objects"]) == 8
    assert "downsampling_verdict" in mf and "41GB" in mf["downsampling_verdict"]
    for e in mf["objects"]:
        assert len(e["files"]) == 143          # 142 灯 + 掩码源
        for meta in e["files"].values():
            assert len(meta["sha256"]) == 64

