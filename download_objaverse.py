"""
Objaverse (allenai/objaverse, HF) 子集下载器

- 从 160 个分组的 glbs/{group}/ 中按种子随机抽样 N 个 .glb
- 体积过滤 [MIN_MB, MAX_MB]，跳过损坏/过小文件
- 输出 models_list（绝对路径，供 render_dataset.py 直接以 glb 接入）

用法: python download_objaverse.py --count 40 --out D:/data/objaverse_raw --list_file obj_list.txt
"""
import argparse
import gzip
import io
import json
import os
import random
import sys
import time
import urllib.request

BASE = "https://huggingface.co/datasets/allenai/objaverse"
API_BASE = "https://huggingface.co/api/datasets/allenai/objaverse"
HEADERS = {"User-Agent": "mir-benchmark/0.1"}


def http_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def http_download(url, dst, min_bytes, max_bytes):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if not (min_bytes <= len(data) <= max_bytes):
        return False, len(data)
    with open(dst, "wb") as f:
        f.write(data)
    return True, len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--out", default="D:/data/objaverse_raw")
    ap.add_argument("--list_file", default="obj_models_list.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_mb", type=float, default=0.2)
    ap.add_argument("--max_mb", type=float, default=60.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    folders = http_json(API_BASE + "/tree/main/glbs?limit=1000")
    folders = [f["path"] for f in folders if f.get("type") == "directory"]
    print(f"folders available: {len(folders)}")

    picked_folders = rng.sample(folders, min(len(folders), max(args.count // 3, 8)))
    candidates = []
    for fo in picked_folders:
        listing = http_json(f"{API_BASE}/tree/main/{fo}?limit=1000")
        for entry in listing:
            if entry.get("type") == "file" and entry["path"].endswith(".glb"):
                candidates.append(entry)
        if len(candidates) >= args.count * 6:
            break
    rng.shuffle(candidates)
    print(f"candidates: {len(candidates)}")

    min_b, max_b = int(args.min_mb * 1e6), int(args.max_mb * 1e6)
    saved, tried = 0, 0
    t0 = time.time()
    for entry in candidates:
        if saved >= args.count:
            break
        tried += 1
        rel = entry["path"]
        sha = os.path.splitext(os.path.basename(rel))[0]
        dst = os.path.join(args.out, sha + ".glb")
        if os.path.exists(dst):
            saved += 1
            continue
        url = f"{BASE}/resolve/main/{rel}"
        try:
            ok, nbytes = http_download(url, dst, min_b, max_b)
            if ok:
                saved += 1
                print(f"  [{saved}/{args.count}] {sha[:12]} {nbytes/1e6:.1f}MB")
        except Exception as exc:
            print(f"  skip {sha[:12]}: {exc}")
    print(f"downloaded {saved} models ({tried} tried) in {time.time()-t0:.0f}s -> {args.out}")

    with open(args.list_file, "w", encoding="ascii") as f:
        for name in sorted(os.listdir(args.out)):
            if name.endswith(".glb"):
                f.write(os.path.abspath(os.path.join(args.out, name)) + "\n")
    print(f"list -> {args.list_file}")


if __name__ == "__main__":
    main()