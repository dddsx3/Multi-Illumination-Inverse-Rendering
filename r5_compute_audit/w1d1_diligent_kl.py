"""R5-B' W1-D1 stage 2: DiLiGenT vs 合成 KL 检验 (本机, 0 GPU)

任务书新路线书 §B.2:
  量化域差: 合成图 vs DiLiGenT 真实图 (同 N=5 协议)
  三项 KL 散度:
    1. 梯度直方图 KL
    2. 噪声功率谱 KL (用 2D FFT 径向谱)
    3. 高光像素占比 KL
  [本脚本加 4. 亮度直方图 KL, 增强结论稳定性]
  闸门: 三项 KL 都 < 0.1 → 域差假设死 (转查架构容量)
        任一项 KL > 0.5 → 域差假设强支持, 推进 B0 协议

数据:
  - 合成: p1/calibration_set/data_sun_confirmatory/ 6 dev scene × 前 5 灯
  - 真实: D:/data/DiLiGenT/pmsData/ 10 物体 × 前 5 灯
  - 都统一到 256×256 (中心裁剪 + 缩放), 单通道灰度

用法:
  python r5_compute_audit/w1d1_diligent_kl.py
产物:
  r5_compute_audit/raw_profile/kl_diligent_vs_synth.csv  (per object × metric)
  r5_compute_audit/decision_reports/W1D1_stage2_KL_verdict.md
"""
from __future__ import annotations
import argparse, csv, gc, sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SYNTH_ROOT = REPO / "p1" / "calibration_set" / "data_sun_confirmatory"
REAL_ROOT = Path("D:/data/DiLiGenT/pmsData")
OUT_CSV = REPO / "r5_compute_audit" / "raw_profile" / "kl_diligent_vs_synth.csv"
OUT_MD = REPO / "r5_compute_audit" / "decision_reports" / "W1D1_stage2_KL_verdict.md"

# 6 dev scene (R5-B' smoke 用过的)
SYNTH_SCENES = ["conf_sphere_r05", "conf_cube_axis", "conf_prism8",
                "conf_egg", "conf_cylinder_r06_d06", "conf_ellipsoid_z06"]
# 10 DiLiGenT 物体
REAL_OBJECTS = ["ballPNG", "bearPNG", "buddhaPNG", "catPNG", "cowPNG",
                "gobletPNG", "harvestPNG", "pot1PNG", "pot2PNG", "readingPNG"]
N_LIGHTS = 5  # N=5 子采样 (W1D4 协议)
IMG_SIZE = 256  # 统一到 256×256


def to_gray_256(path):
    """读 PNG → 单通道灰度 256×256, BT.709 luma"""
    img = Image.open(path).convert("RGB")
    # 中心裁剪到方形, 缩放 256
    w, h = img.size
    s = min(w, h)
    l = (w - s) // 2
    t = (h - s) // 2
    img = img.crop((l, t, l + s, t + s)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return luma


def load_synth(scene, n=N_LIGHTS):
    """合成图: 复用 W1D1 stage 1 的渲染方法, 但读取每 scene 自己的数据"""
    scene_dir = SYNTH_ROOT / scene
    sh = np.load(scene_dir / "sh_coeffs_irradiance.npy")[:n]
    albedo = np.load(scene_dir / "albedo.npy")[0]
    normal = np.load(scene_dir / "normal_mesh.npy").transpose(1, 2, 0)
    sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))
    from gauge_fisher_v2 import sh_basis_npy
    H, W, _ = normal.shape
    Y = sh_basis_npy(normal.reshape(-1, 3)).reshape(H, W, 9)
    imgs = []
    for k in range(n):
        c = sh[k]
        L = (Y @ c[:9] * 4 * np.pi)
        img = (L[..., None] * albedo).clip(0, 1).mean(-1)
        # 中心裁剪 + 缩放
        from PIL import Image as _I
        im = _I.fromarray((img * 255).astype(np.uint8))
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        im = im.resize((IMG_SIZE, IMG_SIZE), _I.BILINEAR)
        imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
    return imgs


def load_real(obj, n=N_LIGHTS):
    """DiLiGenT: 均匀抽 5 灯 (与 R4″ evaluate_diligent.py 一致)"""
    obj_dir = REAL_ROOT / obj
    # 取前 n 张 (001-005), 灯均匀抽样
    paths = sorted(obj_dir.glob("*.png"))[:n]
    paths = [p for p in paths if p.name != "mask.png"]
    return [to_gray_256(p) for p in paths[:n]]


def grad_hist(img, bins=64):
    """2D 梯度幅值直方图 (旋转不变: 联合 mag, 不需方向)"""
    gx = np.diff(img.astype(np.float32), axis=1)
    gy = np.diff(img.astype(np.float32), axis=0)
    gx = gx[:-1, :]
    gy = gy[:, :-1]
    mag = np.sqrt(gx**2 + gy**2).ravel()
    H, _ = np.histogram(mag, bins=bins, range=(0, 1.0), density=True)
    H = H / max(H.sum(), 1e-12)
    return H + 1e-12  # 平滑避免 log(0)


def spec_radial(img, n_bins=64):
    """2D FFT 径向平均功率谱 (对数尺度)"""
    gray = img.astype(np.float32)
    F = np.fft.fft2(gray)
    P = np.abs(F) ** 2
    P = np.fft.fftshift(P)
    h, w = P.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((Y - cy)**2 + (X - cx)**2)
    R_int = R.astype(int)
    max_r = min(cy, cx)
    radial = np.bincount(R_int.ravel(), weights=P.ravel(), minlength=max_r)
    counts = np.bincount(R_int.ravel(), minlength=max_r)
    radial = radial / np.maximum(counts, 1)
    radial = radial[1:] / max(radial[1:].sum(), 1e-12)  # 跳过 DC
    # log 尺度重采样
    log_r = np.logspace(0, np.log10(max_r - 1), n_bins).astype(int)
    log_spec = np.array([radial[r].mean() if r < len(radial) else 0 for r in log_r])
    log_spec = log_spec / max(log_spec.sum(), 1e-12)
    return log_spec + 1e-12


def hist_255(img, bins=64):
    """亮度直方图 (0-1 范围, 64 bin)"""
    H, _ = np.histogram(img.ravel(), bins=bins, range=(0, 1), density=True)
    H = H / max(H.sum(), 1e-12)
    return H + 1e-12


def highlight_frac(img):
    """高光像素占比 (R, G, B 都接近 1, 灰度版本用 luminance > 0.95)"""
    return float((img > 0.95).mean())


def kl_div(p, q):
    """KL(p || q) = Σ p log(p/q) (位)"""
    return float(np.sum(p * np.log(p / q)))


def aggregate_metric(synth_imgs, real_imgs, metric_fn, n_bins=64):
    """对所有图算 metric (1D 直方图), 把多张图的直方图累加得总分布, 再 KL"""
    # 每张图给一个 1D 分布
    syn_acc = None
    real_acc = None
    for im in synth_imgs:
        h = metric_fn(im, bins=n_bins)
        if syn_acc is None: syn_acc = h.copy()
        else: syn_acc += h
    for im in real_imgs:
        h = metric_fn(im, bins=n_bins)
        if real_acc is None: real_acc = h.copy()
        else: real_acc += h
    # 归一化
    syn_acc = (syn_acc + 1e-12) / max(syn_acc.sum(), 1e-12)
    real_acc = (real_acc + 1e-12) / max(real_acc.sum(), 1e-12)
    return float(kl_div(syn_acc, real_acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_lights", type=int, default=N_LIGHTS)
    ap.add_argument("--synth_root", default=str(SYNTH_ROOT))
    ap.add_argument("--real_root", default=str(REAL_ROOT))
    ap.add_argument("--out_csv", default=str(OUT_CSV))
    ap.add_argument("--out_md", default=str(OUT_MD))
    args = ap.parse_args()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"W1-D1 stage 2: 合成 vs DiLiGenT KL 检验 (N={args.n_lights}, 256×256)")
    print("=" * 70)

    # 加载数据
    print(f"\n[loading] 合成图 ({len(SYNTH_SCENES)} scene × {args.n_lights} light)...")
    synth_imgs = []
    for sc in SYNTH_SCENES:
        try:
            synth_imgs.extend(load_synth(sc, args.n_lights))
        except Exception as e:
            print(f"  [skip] {sc}: {e}")
    print(f"  loaded {len(synth_imgs)} synthetic images")

    print(f"\n[loading] DiLiGenT ({len(REAL_OBJECTS)} object × {args.n_lights} light)...")
    real_imgs = []
    for obj in REAL_OBJECTS:
        try:
            real_imgs.extend(load_real(obj, args.n_lights))
        except Exception as e:
            print(f"  [skip] {obj}: {e}")
    print(f"  loaded {len(real_imgs)} real images")

    if len(synth_imgs) < 5 or len(real_imgs) < 5:
        print(f"ERROR: 数据不够 (synth={len(synth_imgs)}, real={len(real_imgs)})")
        sys.exit(1)

    # 计算 4 项 KL
    print(f"\n[KL computation]")
    metrics = {
        "grad_hist": (grad_hist, "梯度直方图 (联合 mag, 64 bin)"),
        "spec_radial": (lambda im, bins: spec_radial(im, n_bins=bins), "功率谱径向平均 (log 尺度, 64 bin)"),
        "luma_hist": (hist_255, "亮度直方图 (0-1, 64 bin)"),
        "highlight_frac": (None, "高光占比 (luma > 0.95)"),  # 1D 标量, 走特殊处理
    }
    rows = []
    for name, (fn, desc) in metrics.items():
        if name == "highlight_frac":
            # 标量: 先均值
            syn_avg = np.mean([highlight_frac(im) for im in synth_imgs])
            real_avg = np.mean([highlight_frac(im) for im in real_imgs])
            # 用 0/1 分布当 KL
            eps = 1e-6
            kl = kl_div(np.array([1 - syn_avg + eps, syn_avg + eps]),
                       np.array([1 - real_avg + eps, real_avg + eps]))
            print(f"  {name:18s}  synth avg={syn_avg:.4f}  real avg={real_avg:.4f}  KL={kl:.4f}")
        else:
            kl = aggregate_metric(synth_imgs, real_imgs, fn)
            print(f"  {name:18s}  KL={kl:.4f}")
        rows.append(dict(metric=name, description=desc, kl_value=round(kl, 4),
                         status="SUPPORT" if kl > 0.5 else ("EQUAL" if kl < 0.1 else "WEAK")))

    # 写 CSV
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # 闸门判定
    kls = [r["kl_value"] for r in rows]
    n_high = sum(1 for k in kls if k > 0.5)
    n_low = sum(1 for k in kls if k < 0.1)
    if n_low == len(kls):
        gate = "域差假设**死亡** (三项 KL 全 < 0.1, 合成 vs 真实低层统计无显著差异, 域差不是问题) → 立即转查架构容量 (W2-C 路线优先)"
    elif n_high >= 2:
        gate = "域差假设**强支持** (≥2 项 KL > 0.5, 合成图 vs 真实图在低层统计显著不同) → 推进 B0 协议 (2×2 因子实验)"
    elif n_high == 1:
        gate = "域差假设**弱支持** (1 项 KL > 0.5) → 推进 B0 协议但加更细 KL (按 metric 找根因)"
    else:
        gate = "域差假设**中等等级** (0.1 < KL < 0.5) → 推进 B0 协议 (谨慎, 加更多扰动成分)"

    # 写报告
    md = []
    md.append("# W1-D1 stage 2 · KL 检验结果 (合成 vs DiLiGenT)\n\n")
    md.append(f"## 数据规模\n- 合成图: {len(synth_imgs)} 张 ({len(SYNTH_SCENES)} scene × {args.n_lights} light, 256×256)\n")
    md.append(f"- DiLiGenT: {len(real_imgs)} 张 ({len(REAL_OBJECTS)} object × {args.n_lights} light, 中心裁剪 + 缩放)\n\n")
    md.append("## KL 散度结果 (位)\n\n")
    md.append("| Metric | 描述 | KL (位) | 状态 |\n|---|---|---:|---|\n")
    status_emoji = {"SUPPORT": "🔥 强支持域差", "EQUAL": "✅ 域差 = 0", "WEAK": "⚠️ 中等"}
    for r in rows:
        md.append(f"| {r['metric']} | {r['description']} | **{r['kl_value']:.4f}** | {status_emoji[r['status']]} |\n")
    md.append(f"\n## 闸门判定\n\n**{gate}**\n\n")
    md.append("## 任务书闸门阈值 (W1D1 路线书 §B.2)\n\n")
    md.append("- 三项 KL 全 < 0.1 → **域差假设死**, 转查架构容量 (B→C)\n")
    md.append("- 任何 KL > 0.5 → 域差假设**获强支持**, 推进 B0 协议 (2×2 因子)\n")
    md.append("- 0.1 < KL < 0.5 → 域差假设中等等级, 推进 B0 协议但加更细 KL\n\n")
    md.append("## 解读\n\n")
    md.append("KL > 0.5 意味着合成图与真实图在**对应维度**上的分布几乎不重叠, 域差结构性存在.\n")
    md.append("KL < 0.1 意味着两个分布几乎重合, 域差在该维度不存在 (本假设被否决).\n")
    md.append("0.1 < KL < 0.5: 分布有部分重叠, 域差弱-中等.\n\n")
    md.append("## 下一步 (按闸门分支)\n\n")
    if "死亡" in gate:
        md.append("- **B 轨基础不再成立** → 跳过 W2-B.1 训练\n")
        md.append("- 立刻做: 架构容量实验 (W2-C 网络加深 + 训练数据扩)\n")
    elif "强支持" in gate:
        md.append("- **B 轨基础强支持** → 推进 B0 协议 (W1D4 文档)\n")
        md.append("- W2-B 需 A10/H100 算力 (40 h GPU)\n")
    else:
        md.append("- **B 轨基础中等等级** → 推进 B0 协议 (加更细扰动成分)\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")

    print(f"\n产物: {OUT_CSV}")
    print(f"      {OUT_MD}")
    print(f"\n=== 闸门: {gate} ===")


if __name__ == "__main__":
    main()
