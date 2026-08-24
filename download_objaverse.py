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


def http_head_size(url):
    """HEAD 预检体积，避免把超大/极小文件整个读进内存"""
    req = urllib.request.Request(url, method="HEAD", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None


def http_download_stream(url, dst):
    """流式分块写盘（1MB chunks），恒定内存占用；先写 .part 再原子改名"""
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers=HEADERS)
    nbytes = 0
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            nbytes += len(chunk)
    os.replace(tmp, dst)
    return nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--out", default="D:/data/objaverse_raw")
    ap.add_argument("--list_file", default="obj_models_list.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_mb", type=float, default=0.2)
    ap.add_argument("--max_mb", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max_seconds", type=int, default=14400,
                    help="安全熔断：超过该秒数优雅收尾（已下载文件保留，可续跑）")
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
    max_seconds = int(args.max_seconds)
    t0 = time.time()

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lock = threading.Lock()
    state = {"saved": 0, "tried": 0}

    def worker(entry):
        rel = entry["path"]
        sha = os.path.splitext(os.path.basename(rel))[0]
        dst = os.path.join(args.out, sha + ".glb")
        url = f"{BASE}/resolve/main/{rel}"
        try:
            hsize = None
            try:
                hsize = http_head_size(url)
            except Exception:
                pass
            if hsize is not None and not (min_b <= hsize <= max_b):
                return ("size_skip", sha, 0.0)
            tid = threading.get_ident()
            tmp = dst + f".part{tid}"
            req = urllib.request.Request(url, headers=HEADERS)
            nbytes = 0
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    nbytes += len(chunk)
            os.replace(tmp, dst)
            return ("ok", sha, float(nbytes))
        except Exception as exc:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return ("fail", sha, 0.0)

    def need_more():
        with lock:
            return state["saved"] < args.count and time.time() - t0 < max_seconds

    existing = len([n for n in os.listdir(args.out) if n.endswith(".glb")])
    state["saved"] = existing
    print(f"already on disk: {existing}", flush=True)

    chunk_size = 30
    idx = 0
    pool = ThreadPoolExecutor(max_workers=max(args.workers, 1))
    while need_more() and idx < len(candidates):
        batch = candidates[idx: idx + chunk_size]
        idx += chunk_size
        futures = [pool.submit(worker, entry) for entry in batch]
        for fut in as_completed(futures):
            status, sha, nbytes = fut.result()
            with lock:
                state["tried"] += 1
                if status == "ok":
                    state["saved"] += 1
                done, tried_n = state["saved"], state["tried"]
            if status == "ok":
                print(f"  [{done}/{args.count}] {sha[:12]} {nbytes/1e6:.1f}MB", flush=True)
            elif status == "fail":
                print(f"  skip {sha[:12]}", flush=True)
        el = time.time() - t0
        rate = tried_n / max(el, 1)
        print(f"  ... saved={done}/{args.count} tried={tried_n} elapsed={el:.0f}s "
              f"({rate:.2f} tries/s)", flush=True)

    final_saved = len([n for n in os.listdir(args.out) if n.endswith(".glb")])
    print(f"downloaded -> {final_saved} models on disk "
          f"({state['tried']} tried this run, {time.time()-t0:.0f}s)")

    with open(args.list_file, "w", encoding="ascii") as f:
        for name in sorted(os.listdir(args.out)):
            if name.endswith(".glb"):
                f.write(os.path.abspath(os.path.join(args.out, name)) + "\n")
    print(f"list -> {args.list_file}")


if __name__ == "__main__":
    main()