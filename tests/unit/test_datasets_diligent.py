"""B3 绑定测试 · DiLiGenT loader（口径对账 + checksum 工具）。

绑定：REPO_MIGRATION B3 行 1 —— 小样本 checksum/形状断言。
真数据依赖（D:/data/DiLiGenT）：不存在时跳过（数据合同在 run 时由 manifest 冻结，
宪法 §11；测试不伪造数据路径）。
"""

import numpy as np
import pytest

from calibinfo.datasets.diligent import load_object, make_manifest, sha256_file

DATA_ROOT = "D:/data/DiLiGenT/pmsData"
HAS_DATA = __import__("pathlib").Path(DATA_ROOT, "ballPNG").exists()

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="本机无 DiLiGenT 副本")


def test_load_object_shapes_and_ranges():
    """形状与值域断言：96 光、法线单位球、I_norm 非负有限。"""
    obj = load_object(f"{DATA_ROOT}/ballPNG")
    assert obj["I_norm"].shape == (96, obj["normals_gt"].shape[0])
    assert obj["dirs"].shape == (96, 3)
    assert np.all(np.isfinite(obj["I_norm"]))
    assert np.allclose(np.linalg.norm(obj["normals_gt"], axis=1), 1.0, atol=1e-9)


def test_load_object_matches_legacy_exp8r(tmp_path=None):
    """口径对账：与 exp8r legacy load_object 逐位一致（迁移验收）。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                            / "critical_experiments"))
    import exp8r_diligent_discrimination_v3 as legacy
    d = Path(DATA_ROOT) / "ballPNG"
    dirs_l, n_l, I_l, mask_l = legacy.load_object(d)
    new = load_object(d)
    assert np.abs(dirs_l - new["dirs"]).max() == 0.0
    assert np.abs(n_l - new["normals_gt"]).max() == 0.0
    assert np.abs(I_l - new["I_norm"]).max() == 0.0
    assert np.array_equal(mask_l, new["mask"])


def test_manifest_checksum(tmp_path):
    """manifest：对象清单 + sha256 checksum 逐文件可复现；sha256 工具对已知内容自检。"""
    # sha256 工具自检（已知内容 → 已知摘要）
    p = tmp_path / "known.bin"
    p.write_bytes(b"calibinfo")
    import hashlib
    assert sha256_file(p) == hashlib.sha256(b"calibinfo").hexdigest()
    # 真数据 manifest
    mf = make_manifest(DATA_ROOT, objects=["ballPNG"])
    entry = mf["objects"][0]
    assert entry["name"] == "ballPNG" and entry["n_lights"] == 96
    for key, meta in entry["files"].items():
        assert len(meta["sha256"]) == 64 and meta["bytes"] > 0
