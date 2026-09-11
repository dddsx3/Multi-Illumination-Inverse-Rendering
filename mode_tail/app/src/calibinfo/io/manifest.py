"""io · run_manifest。

每个 run 生成 manifest：git SHA、dirty flag、Python/NumPy/SciPy 版本、config hash、
seed、dataset checksum、机器信息、开始/结束时间。
绑定测试：tests/unit/test_io_manifest.py（字段齐全 + config hash 对空白差异敏感）。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_info(repo_root):
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return sha, dirty
    except Exception:
        return None, None


def config_hash(config):
    """config → sha256（规范化 JSON，键排序；对空白/键序差异不敏感，对内容差异敏感）。"""
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True,
                           default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_manifest(config, repo_root=None, seed=None, dataset_checksums=None,
                 experiment=None, started_utc=None):
    """构造一个 run 的 manifest dict（调用方负责落盘）。"""
    repo_root = Path(repo_root or Path(__file__).resolve().parents[3])
    sha, dirty = _git_info(repo_root)
    import numpy as np
    import scipy
    return dict(
        experiment=experiment,
        git_sha=sha,
        git_dirty=bool(dirty),
        python=sys.version.split()[0],
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
        hostname=platform.node(),
        config=config,
        config_hash=config_hash(config),
        seed=seed,
        dataset_checksums=dataset_checksums or {},
        started_utc=started_utc,
        written_utc=datetime.now(timezone.utc).isoformat(),
    )


def write_manifest(path, **kwargs):
    """构造并落盘 manifest.json（UTF-8, ensure_ascii=False）。"""
    mf = run_manifest(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mf, ensure_ascii=False, indent=1), encoding="utf-8")
    return mf
