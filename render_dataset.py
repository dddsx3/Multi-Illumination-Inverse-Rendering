"""
Phase 1 合成数据集渲染脚本（BlenderProc 2.8.0 API）

输出协议（与 evaluate.py 模块 docstring 完全一致），每场景 10 个文件：
  light_001..K.png  多光照灰度图（sRGB 编码，K=5）
  depth.npy         视空间深度 [1,H,W] float32（近大远小正数）
  albedo.npy        线性反照率 [1,H,W] float32（无光照 base color，[0,1]）
  normal.npy        相机空间法线 [3,H,W] float32（单位向量，面朝相机 z>0）
  mask.npy          有效像素掩码 [1,H,W] uint8（0/1，1=前景）
  sh_coeffs.npy     每光照二阶实 SH 系数 [K,9] float32

关键设计（Phase 1 审计确认，勿改动语义）：
1. SH 系数用与 physics_renderer.SphericalHarmonicsLighting 完全一致的
   基常数与基顺序（C0=0.282095, C1=0.488603, C2 五值），点光源按远场
   方向光近似：c = I * Y_lm(d)，d 为光源指向场景中心的方向。
2. 法线从 BlenderProc EXR 输出解码（n*2-1，BlenderProc 写入时做了
   n*0.5+0.5 编码与 G/B 交换），再做方向自检：与深度导数法线
   （numpy 复刻 physics_renderer 的 n=normalize([-dz/dx,-dz/dy,1]) 约定）
   对比平均夹角，若 >90° 则整体翻转并告警——保证落盘法线满足
   "面朝相机 z>0"，与网络输出监督语义一致。
3. PNG 编码：线性 RGB -> ITU-R BT.709 灰度 -> 标准 sRGB OETF 存 uint8。
   data_loader 端 img^(1/2.2) 近似解码与之互逆（暗部偏差 ~2%）。
   若改此处编码，必须同步改 data_loader 的解码，否则全量数据重建损失
   带系统性偏置（Phase 1 门禁 G 会抓出）。
4. 深度输出为视空间 z（近大远小正数），与 physics_renderer 深度语义一致。

用法：
  blenderproc run render_dataset.py --obj_list models.txt --out_dir D:/data/synthetic --size 256 --gpu
  obj_list: 每行一个 .obj 绝对路径；支持 --start/--count 分段并行（多开实例分模型子集）

依赖：blenderproc>=2.6（pip install blenderproc，首次运行自动下载 Blender），
      numpy, Pillow。渲染用 Blender 的 Cycles（GPU 优先，CPU 可跑但慢）。
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image

import blenderproc as bproc

# ---------------------------------------------------------------------------
# SH 常量（与 physics_renderer.SphericalHarmonicsLighting 完全一致，勿改）
# ---------------------------------------------------------------------------
C0 = 0.282095   # 1/2 * sqrt(1/pi)
C1 = 0.488603   # sqrt(3/4pi)
C2 = [1.092548, 1.092548, 0.315392, 1.092548, 0.546274]


def sh_basis(d):
    """d: [3] 单位方向向量 -> 9 个基值，顺序与 compute_sh_basis 完全一致
    [Y0, Y1_n1(y), Y1_0(z), Y1_p1(x), Y2_n2(xy), Y2_n1(yz), Y2_0(3z^2-1), Y2_p1(xz), Y2_p2(x^2-y^2)]"""
    x, y, z = d
    return np.array([
        C0,
        C1 * y, C1 * z, C1 * x,
        C2[0] * x * y, C2[1] * y * z, C2[2] * (3.0 * z * z - 1.0),
        C2[3] * x * z, C2[4] * (x * x - y * y),
    ], dtype=np.float32)


def point_light_sh(direction, intensity):
    """点光源 -> 二阶 SH 系数（远场方向光近似）：c = I * Y(d)"""
    return intensity * sh_basis(direction)


# ---------------------------------------------------------------------------
# 图像工具
# ---------------------------------------------------------------------------
def linear_to_srgb(img):
    """线性 float [0,1] -> sRGB 编码 uint8（标准 sRGB OETF）"""
    img = np.clip(img, 0.0, 1.0)
    out = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(img, 1.0 / 2.4) - 0.055)
    return (out * 255.0 + 0.5).astype(np.uint8)


def sobel_normal(depth, h, w):
    """numpy 复刻 physics_renderer.DepthToNormal（use_edge_aware=False）：
    n = normalize([-dz/dx, -dz/dy, 1])，面朝相机 z>0。
    用于法线方向自检（逐像素角度误差统计）。"""
    pad = np.pad(depth, 1, mode='edge')
    gx = (-pad[:-2, :-2] + pad[2:, :-2] - 2 * pad[:-2, 1:-1] + 2 * pad[2:, 1:-1]
          - pad[:-2, 2:] + pad[2:, 2:]) / 4.0
    gy = (pad[:-2, :-2] + 2 * pad[1:-1, :-2] + pad[2:, :-2]
          - pad[:-2, 2:] - 2 * pad[1:-1, 2:] - pad[2:, 2:]) / 4.0
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def look_at(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """OpenGL 风格 look-at 4x4（-Z 指向目标，Blender 世界 Z-up）"""
    cam_pos, target, up = np.array(cam_pos, float), np.array(target, float), np.array(up, float)
    z_axis = cam_pos - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    m = np.eye(4)
    m[:3, 0] = x_axis
    m[:3, 1] = y_axis
    m[:3, 2] = z_axis
    m[:3, 3] = cam_pos
    return m


def normalize_scene(obj, target_longest=1.6):
    """按包围盒最长边缩放到 target_longest，并置于原点"""
    bbox = np.array(obj.get_bound_box())
    size = bbox.max(axis=0) - bbox.min(axis=0)
    s = target_longest / max(size.max(), 1e-6)
    obj.set_scale([s, s, s])
    bbox = np.array(obj.get_bound_box())
    obj.set_location([0, 0, 0])
    # 把几何中心移到原点
    center = (bbox.max(axis=0) + bbox.min(axis=0)) / 2.0
    obj.set_location(-center)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def render_one(obj_path, out_dir, size, num_lights, light_energy, fov_deg,
               cam_dist, samples):
    scene_name = os.path.splitext(os.path.basename(obj_path))[0]
    scene_dir = os.path.join(out_dir, scene_name)
    if os.path.exists(os.path.join(scene_dir, "sh_coeffs.npy")):
        print(f"[skip] {scene_name} 已存在")
        return "skip"

    # 清理上一场景的全部对象/光照（保留相机设置，后面重建）
    bproc.clean_up(clean_up_camera=True)

    obj = bproc.loader.load_obj(obj_path)[0]
    bproc.object.manifold_cleanup(obj)
    normalize_scene(obj)

    # 相机：前视图（+X 方向 30° 俯视），look-at 原点
    cam_pos = [cam_dist * math.cos(math.radians(30.0)), 0.0, cam_dist * math.sin(math.radians(30.0))]
    pose = look_at(cam_pos, [0, 0, 0], up=[0, 0, 1])

    # 5 个光照帧：方位角 72° 步进、高度角 50°，半径固定
    frames = []
    light_dirs = []
    for k in range(num_lights):
        az = k * (360.0 / num_lights)
        el = 50.0
        lx = R * math.cos(math.radians(el)) * math.cos(math.radians(az))
        ly = R * math.cos(math.radians(el)) * math.sin(math.radians(az))
        lz = R * math.sin(math.radians(el))
        light_dirs.append(np.array([lx, ly, lz], float))
        frames.append(k)

    bproc.camera.set_resolution(size, size)
    bproc.camera.set_fov(math.radians(fov_deg))
    for k in frames:
        bproc.camera.add_camera_pose(pose, frame=k)

    # 光照：每帧一个点光源（其余帧无光）
    bproc.world.set_world_background(np.zeros(3))
    for k in frames:
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(light_dirs[k].tolist(), frame=k)
        light.set_energy(light_energy, frame=k)

    # G-buffer 输出
    bproc.renderer.set_render_devices(use_only_cpu=not args.gpu)
    bproc.renderer.set_max_amount_of_samples(samples)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_diffuse_color_output()
    bproc.renderer.enable_segmentation_output(map_by="instance")
    bproc.utility.set_keyframe_render_interval(frames[0], frames[-1])

    data = bproc.renderer.render(return_data=True)

    os.makedirs(scene_dir, exist_ok=True)

    # 1) 光照图：线性 RGB -> BT.709 灰度 -> sRGB 编码
    grayscale_srgb = []
    for k in frames:
        rgb = data["colors"][k][:, :, :3]                      # 线性 float
        gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        gray = np.clip(gray, 0.0, 1.0)
        img8 = linear_to_srgb(gray)
        Image.fromarray(img8, mode="L").save(os.path.join(scene_dir, f"light_{k + 1:03d}.png"))
        grayscale_srgb.append(img8.astype(np.float32) / 255.0)

    # 2) 深度（视空间 z，近大远小正数）
    depth = data["depth"][0][:, :, 0].astype(np.float32)       # [H,W]
    np.save(os.path.join(scene_dir, "depth.npy"), depth[None])

    # 3) 反照率（线性、无光照）
    albedo = data["diffuse"][0][:, :, :3].astype(np.float32)
    albedo_gray = 0.2126 * albedo[:, :, 0] + 0.7152 * albedo[:, :, 1] + 0.0722 * albedo[:, :, 2]
    albedo_gray = np.clip(albedo_gray, 0.0, 1.0)
    np.save(os.path.join(scene_dir, "albedo.npy"), albedo_gray[None])

    # 4) 掩码（segmap: 0=背景）
    seg = data["segmap"][0].astype(np.int64)
    mask = (seg != 0).astype(np.uint8)
    np.save(os.path.join(scene_dir, "mask.npy"), mask[None])

    # 5) 法线：解码 + 方向自检（与 depth 导数法线对比）
    # BlenderProc 把法线以 n*0.5+0.5 编码写入 EXR；若加载结果含负值或 >1
    # 说明该版本已返回解码值，跳过。判断依据：编码值必然全在 [0,1]。
    normal = data["normals"][0][:, :, :3].astype(np.float32)   # BlenderProc 已做 G/B 交换
    if not (normal.min() < -0.01 or normal.max() > 1.01):
        normal = normal * 2.0 - 1.0                            # 解码 n*0.5+0.5 编码
    n_gt = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    # 双保险：解码状态错误时单位范数自检会偏离
    mask_bool = mask[0] > 0
    n_scale = np.linalg.norm(normal[mask_bool], axis=-1).mean()
    if n_scale < 0.5 or n_scale > 1.5:
        print(f"  [warn] {scene_name}: 法线范数异常 ({n_scale:.2f})，请检查 BlenderProc 版本编码行为")
    n_sobel = sobel_normal(depth, size, size)
    dot = np.clip((n_gt * n_sobel).sum(axis=-1), -1.0, 1.0)
    mean_angle = np.degrees(np.arccos(dot[mask[0] > 0])).mean()
    if mean_angle > 90.0:
        print(f"  [warn] {scene_name}: 法线与深度导数方向相反 (夹角 {mean_angle:.1f}°)，整体翻转")
        n_gt = -n_gt
    np.save(os.path.join(scene_dir, "normal.npy"), n_gt.transpose(2, 0, 1))

    # 6) SH 系数：点光源按远场方向光近似，强度取"参考距离处的辐照度"
    #    I_eff = energy / (4*pi*R^2)，使 GT SH 量级与训练中网络从图像学到的
    #    亮度量级一致（否则 sh_l2 先验会与重建损失打架）
    R_light = float(np.linalg.norm(light_dirs[0]))
    i_eff = light_energy / (4.0 * math.pi * R_light ** 2)
    sh = np.stack([point_light_sh(d / np.linalg.norm(d), i_eff) for d in light_dirs])
    np.save(os.path.join(scene_dir, "sh_coeffs.npy"), sh.astype(np.float32))

    # 7) 自检统计写入场景日志
    with open(os.path.join(scene_dir, "render_stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"scene: {scene_name}\n")
        f.write(f"normal_sobel_angle_deg: {mean_angle:.2f}\n")
        f.write(f"depth_range: {depth[mask[0] > 0].min():.4f} ~ {depth[mask[0] > 0].max():.4f}\n")
        f.write(f"mask_coverage: {mask.mean():.3f}\n")
        f.write(f"albedo_range: {albedo_gray[mask[0] > 0].min():.4f} ~ {albedo_gray[mask[0] > 0].max():.4f}\n")

    print(f"[done] {scene_name}  (法线-导数夹角 {mean_angle:.1f}°, 掩码覆盖 {mask.mean():.3f})")
    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Phase 1 合成数据集渲染（BlenderProc）")
    parser.add_argument("--obj_list", required=True, help="每行一个 .obj 绝对路径的文本文件")
    parser.add_argument("--out_dir", required=True, help="数据集输出根目录")
    parser.add_argument("--size", type=int, default=256, help="图像分辨率（默认 256）")
    parser.add_argument("--num_lights", type=int, default=5, help="光照数量 K（默认 5）")
    parser.add_argument("--light_energy", type=float, default=100.0, help="点光源能量 W（默认 100，Blender 默认量级）")
    parser.add_argument("--fov_deg", type=float, default=50.0, help="相机视场角（默认 50°）")
    parser.add_argument("--cam_dist", type=float, default=2.6, help="相机距离（默认 2.6）")
    parser.add_argument("--samples", type=int, default=128, help="Cycles 采样数（默认 128）")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU 渲染（默认 CPU）")
    parser.add_argument("--start", type=int, default=0, help="从列表第 N 个模型开始")
    parser.add_argument("--count", type=int, default=-1, help="渲染数量（默认全部）")
    args = parser.parse_args()

    with open(args.obj_list, encoding="utf-8") as f:
        objs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    objs = objs[args.start:]
    if args.count > 0:
        objs = objs[: args.count]

    global R
    R = args.cam_dist * 1.15  # 光源轨道半径

    bproc.init()
    for i, obj_path in enumerate(objs):
        print(f"[{i + 1}/{len(objs)}] {os.path.basename(obj_path)}")
        try:
            render_one(obj_path, args.out_dir, args.size, args.num_lights,
                       args.light_energy, args.fov_deg, args.cam_dist, args.samples)
        except Exception as e:
            print(f"[fail] {obj_path}: {e}")
            import traceback
            traceback.print_exc()
    print("全部完成。")


if __name__ == "__main__":
    main()
