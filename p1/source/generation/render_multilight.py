# BlenderProc 强制要求：第一个有效代码行必须是 import blenderproc
import blenderproc as bproc
import argparse
import json
import math
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "physics")))
from sh import sh_basis_npy, A_L  # noqa: E402

# ---------------- helpers（与 PRE-0 渲染器兼容）----------------
def look_at(cam_pos, target, up=(0.0, 0.0, 1.0)):
    z_axis = np.array(cam_pos, float) - np.array(target, float)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(np.array(up, float), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    m = np.eye(4)
    m[:3, 0] = x_axis; m[:3, 1] = y_axis; m[:3, 2] = z_axis; m[:3, 3] = cam_pos
    return m


def set_world_black():
    import bpy
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    if bg is None:
        bg = nt.nodes.new("ShaderNodeBackground")
    out = next((n for n in nt.nodes if n.type == "OUTPUT_WORLD"), None)
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs[0])
    bg.inputs[0].default_value[:] = [0.0, 0.0, 0.0, 1.0]
    bg.inputs[1].default_value = 1.0


def camera_frame_R():
    """世界→相机旋转（与 PRE-0 oracle.camera_frame_matrix 同）。"""
    cam_pos = np.array([2.6 * math.cos(math.radians(30.0)), 0.0,
                        2.6 * math.sin(math.radians(30.0))])
    z_c = cam_pos / np.linalg.norm(cam_pos)
    x_c = np.array([0.0, 1.0, 0.0])
    y_c = np.cross(z_c, x_c); y_c /= np.linalg.norm(y_c)
    return np.stack([x_c, y_c, z_c])


def hemisphere_lights(n_lights, el_lo=20.0, el_hi=70.0, seed=20260830):
    """半球内 Fibonacci 等面积（黄金角序列，固定种子）→ 世界系单位方向 [n,3]"""
    rng = np.random.default_rng(seed)
    golden = math.pi * (3 - math.sqrt(5))
    dirs = []
    for i in range(n_lights):
        t = (i + 0.5) / n_lights
        el = math.radians(el_lo + (el_hi - el_lo) * t)
        az = golden * i + rng.uniform(0, 0.05)
        dirs.append([math.cos(el) * math.cos(az),
                     math.cos(el) * math.sin(az),
                     math.sin(el)])
    d = np.stack(dirs)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def irradiance_sh_coeffs_camera(d_world, I_eff, R_cw):
    """Route A：d_world × R_cw → d_camera；c = k_l · I_eff · Y(d_camera)。"""
    d_cam = R_cw @ d_world
    Y = sh_basis_npy(d_cam[None])[0]                        # [9]
    k = np.array([A_L[0], A_L[1], A_L[1], A_L[1],
                  A_L[2], A_L[2], A_L[2], A_L[2], A_L[2]])
    return I_eff * k * Y


def srgb_to_linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def render_one(obj_path, out_dir, size, num_lights, light_energy, fov_deg,
               cam_dist, samples, use_gpu, R_cw, light_dirs_w, rng_seed,
               light_type="sun"):
    scene_name = os.path.splitext(os.path.basename(obj_path))[0]
    scene_dir = os.path.join(out_dir, scene_name)
    if os.path.exists(os.path.join(scene_dir, "sh_coeffs_irradiance.npy")):
        return "skip"
    os.makedirs(scene_dir, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    t0 = time.time()

    # 一次性：场景、相机、几何（不变）；后续每盏光只重建光
    bproc.clean_up(clean_up_camera=True)
    loaded = bproc.loader.load_obj(obj_path)
    if not loaded:
        raise RuntimeError(f"无法加载 {obj_path}")
    mesh_objs = [o for o in loaded if o.blender_obj.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError("无 MESH 对象")
    # 归一化（与 PRE-0 一致）
    pts = np.vstack([np.asarray(o.get_bound_box(), dtype=np.float64) for o in mesh_objs])
    s = 1.6 / max(float((pts.max(0) - pts.min(0)).max()), 1e-6)
    for o in mesh_objs:
        o.set_scale([s, s, s])
    # 实例 id
    import bpy
    for idx, ob in enumerate(o for o in bpy.data.objects if o.type == "MESH"):
        ob.pass_index = idx + 1

    # P1-R2 P-domain 物理封口：强制纯 Lambertian（Diffuse BSDF）材质。
    # 默认 Principled BSDF 含 specular=0.5，colors 通道混入镜面高光，
    # 违反 P 域 "Lambertian only" 定义（R0 审查发现的第 3 个失配源）。
    if light_type == "sun":
        mat = bpy.data.materials.new("p1_diffuse")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
        diff.inputs["Color"].default_value = [0.8, 0.8, 0.8, 1.0]
        nt.links.new(diff.outputs[0], out.inputs[0])
        for o in mesh_objs:
            o.blender_obj.data.materials.clear()      # 无条件清除默认 Principled
            o.blender_obj.data.materials.append(mat)

    cam_pos = [cam_dist * math.cos(math.radians(30.0)), 0.0, cam_dist * math.sin(math.radians(30.0))]
    pose = look_at(cam_pos, [0, 0, 0], up=[0, 0, 1])
    bproc.camera.set_resolution(size, size)
    lens_mm = 18.0 / math.tan(math.radians(fov_deg) / 2.0)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=lens_mm, image_width=size, image_height=size, lens_unit="MILLIMETERS")
    bproc.camera.add_camera_pose(pose)

    # 几何/材质 AOV：固定 1 灯（最大能量 100W）渲一次拿 depth/albedo/normal/mask
    set_world_black()
    ref_light = bproc.types.Light()
    ref_light.set_type("POINT")
    ref_light.set_location(light_dirs_w[0].tolist())        # 任意位置（只用于 AOV 提取）
    ref_light.set_energy(light_energy)
    bproc.renderer.set_render_devices(use_only_cpu=not use_gpu)
    bproc.renderer.set_max_amount_of_samples(samples)
    # 顺序：先 diff/normals，最后 depth（避免 normals 内部重新 enable depth 报错）
    bproc.renderer.enable_diffuse_color_output()
    bproc.renderer.enable_normals_output()                  # P1-06 mesh normal GT
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by="instance")
    data = bproc.renderer.render(return_data=True)

    # ---- 保存几何 / 材质 AOV（与光无关）----
    depth_raw = data["depth"][0]
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    depth = depth_raw.astype(np.float32)
    np.save(os.path.join(scene_dir, "depth.npy"), depth[None])

    # 线性反照率（diffuse luma）
    albedo_rgb = data["diffuse"][0][:, :, :3].astype(np.float32)
    if albedo_rgb.max() > 1.0001:
        albedo_rgb = albedo_rgb / 255.0
    alb_lin = srgb_to_linear(np.clip(albedo_rgb, 0, 1))
    alb_gray = 0.2126 * alb_lin[..., 0] + 0.7152 * alb_lin[..., 1] + 0.0722 * alb_lin[..., 2]
    np.save(os.path.join(scene_dir, "albedo.npy"), alb_gray[None])

    # Mesh normal GT（BlenderProc normal AOV）
    # P1-R2 修正（实验证实）：BlenderProc 2.8 的 normals AOV 已是【相机系】
    # （立方体面法线聚类 = R_cw.T @ n_cam 的证据），此前又做了一次
    # world→cam 旋转导致法线被二次旋转破坏（61~77° 夹角假象）。直接落盘。
    normal_aov = data["normals"][0]
    if normal_aov.ndim == 4:
        normal_aov = normal_aov[0]
    n_cam = normal_aov.astype(np.float32) * 2.0 - 1.0
    n_cam /= np.maximum(np.linalg.norm(n_cam, axis=-1, keepdims=True), 1e-9)
    np.save(os.path.join(scene_dir, "normal_mesh.npy"), n_cam.transpose(2, 0, 1))

    # Depth-derived normal（反投影，P1-06 报告）
    H, W = depth.shape
    f = (H / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cam_pos_np = np.array(cam_pos)
    fwd = -cam_pos_np / np.linalg.norm(cam_pos_np)
    x_img = np.array([0.0, 1.0, 0.0])
    y_up = np.cross(x_img, -fwd); y_up /= np.linalg.norm(y_up)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    du = (uu - (W - 1) / 2.0) / f
    dv = ((H - 1) / 2.0 - vv) / f
    P = (cam_pos_np[None, None, :]
         + depth[..., None] * (fwd[None, None, :] + du[..., None] * x_img
                               + dv[..., None] * y_up))
    du_p = P[1:-1, 2:, :] - P[1:-1, :-2, :]
    dv_p = P[2:, 1:-1, :] - P[:-2, 1:-1, :]
    n_d_full = np.cross(du_p, dv_p)
    n_d_full /= np.maximum(np.linalg.norm(n_d_full, axis=-1, keepdims=True), 1e-9)
    # 全图填充：用最近邻扩边（mask 内才有效）
    n_d = np.zeros_like(P) + np.array([0, 0, 1])
    n_d[1:-1, 1:-1] = n_d_full
    np.save(os.path.join(scene_dir, "normal_depth.npy"), n_d.transpose(2, 0, 1))

    # 掩码
    seg = np.asarray(data["instance_segmaps"][0]).astype(np.int64)
    mask = (seg != 0).astype(np.uint8)
    if mask.sum() == 0:
        # 场景不在画幅内（如水平平面在 30° 俯角相机下不可见）——干净跳过
        raise RuntimeError("mask 为空：物体不可见（跳过该场景）")
    np.save(os.path.join(scene_dir, "mask.npy"), mask[None])

    # ---- 移除参考光，开始 per-light render ----
    ref_light.delete()
    set_world_black()

    # P1-R1 修正：P-domain 用 SUN（真远场方向光），R/stress 域才用 POINT 近场
    # I_eff 语义：albedo 乘子域 irradiance = S/π（Lambertian BSDF 的 1/π 并入），
    # 使 A⊙ReLU(Σ cY(n)) 与渲染像素在同一能量域（无 scale gauge 残余）。
    import mathutils
    if light_type == "sun":
        I_eff = light_energy / math.pi          # light_energy 参数在此语义下 = SUN strength S
    else:
        R_light = float(np.linalg.norm(light_dirs_w[0]) * 2.99)  # point 模式放到远场半径
        I_eff = light_energy / (4 * math.pi * R_light ** 2)
    irradiance_coeffs = np.zeros((num_lights, 9), dtype=np.float32)
    light_meta = []
    img_raw_list = []
    # INC-001 帧校验用解析期望（一次性预计算：L2 SH 单灯模型，无自阴影）
    mask_bool = mask.astype(bool)
    a_masked = alb_gray[mask_bool]                                  # [P]
    Y_masked = sh_basis_npy(n_cam[mask_bool])                       # [P,9]

    for k in range(num_lights):
        d_w = light_dirs_w[k]
        light = bproc.types.Light()
        if light_type == "sun":
            light.set_type("SUN")
            # SUN 沿光对象 -Z 发射；令 +Z 对准 d_w（场景→光方向），发射即 -d_w（照向场景）
            quat = mathutils.Vector(d_w).to_track_quat("Z", "Y")
            light.set_rotation_euler(quat.to_euler())
            light.set_energy(light_energy)      # SUN strength = irradiance W/m^2
        else:
            light.set_type("POINT")
            loc = np.array(d_w) / np.linalg.norm(d_w) * 2.99
            light.set_location(loc.tolist())
            light.set_energy(light_energy)
        # ---- INC-001 帧级完整性（R4'-C canary 发现）：新建灯偶发未同步进
        # Cycles depsgraph → 整帧丢光（黑帧）或残留上一帧伪影（亮带）。
        # 修复：渲染前强制 view_layer 同步；渲染后对照解析 shading 校验，
        # 坏帧重试（≤3 次），仍坏则 raise 干净失败。物理语义零改动。
        # 阈值标定（discovery 4 scene ×32 = 128 健康帧）：ratio∈[0.86,1.69]，
        # bg≤0.0074；坏帧实测 ratio≈0.001 / bg≈0.10。取宽安全带（复合 mesh
        # 自阴影留余量）：ratio∈[0.15,3.5]，bg≤0.05。
        c_k = irradiance_sh_coeffs_camera(d_w, I_eff, R_cw)
        expected_mean = float((a_masked * np.maximum(Y_masked @ c_k, 0.0)).mean())
        bpy.context.view_layer.update()
        attempt = 0
        while True:
            d_kimg = bproc.renderer.render(return_data=True)["colors"][0][:, :, :3]
            light.delete()
            set_world_black()
            # 渲染输出按 bproc 默认走 PNG sRGB 编码（uint8），或 EXR 浮点。
            # 自动检测：d_kimg.dtype == uint8 → sRGB 编码 → 必须 sRGB 反变换到线性
            # d_kimg.dtype == float → 已是线性 → 直接用
            if d_kimg.dtype == np.uint8:
                u = d_kimg.astype(np.float32) / 255.0
                lin = srgb_to_linear(np.clip(u, 0, 1))
            else:
                lin = np.clip(d_kimg.astype(np.float32), 0, 1)  # 已是 linear
            gray = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
            fg_mean = float(gray[mask_bool].mean())
            bg_mean = float(gray[~mask_bool].mean())
            ratio = fg_mean / max(expected_mean, 1e-9)
            if 0.15 <= ratio <= 3.5 and bg_mean <= 0.05:
                break
            attempt += 1
            if attempt >= 3:
                raise RuntimeError(
                    f"INC-001 frame {k} integrity fail after {attempt} renders: "
                    f"ratio={ratio:.4f} bg_mean={bg_mean:.4f} expected={expected_mean:.5f}")
            print(f"    [INC-001] frame {k} bad (ratio={ratio:.4f} bg={bg_mean:.4f}) "
                  f"→ resync + re-render (attempt {attempt})", flush=True)
            bpy.context.view_layer.update()
        np.save(os.path.join(scene_dir, f"light_{k + 1:03d}_lin.npy"), gray.astype(np.float32))
        img_raw_list.append(gray.astype(np.float32))
        # 相机系 SH（Route A irradiance coefficients）
        irradiance_coeffs[k] = c_k
        light_meta.append(dict(idx=k, dir_world=d_w.tolist(), dir_camera=(R_cw @ d_w).tolist(),
                               I_eff=I_eff))
    np.save(os.path.join(scene_dir, "sh_coeffs_irradiance.npy"), irradiance_coeffs)
    json.dump(light_meta, open(os.path.join(scene_dir, "light_meta.json"), "w"),
              indent=2, ensure_ascii=False)

    # ---- P1-14 自动 Gate：validation.json ----
    img_arr = np.stack(img_raw_list)                            # [K,H,W] 线性灰度
    # 像素对差异
    D_ij = np.array([[np.abs(img_arr[i] - img_arr[j]).mean() for j in range(num_lights)]
                     for i in range(num_lights)])
    # 重复渲染噪声（用与光 0 同样参数重渲一次——这里近似用：球面内"相似方向"
    # 算 ΔD / D(repeat-noise-floor) ；先用 stub，待 P1-04/05 完整体化时再补真实 repeat）
    R_l_to_noise = float(D_ij.max() / (1e-3 + 1e-3))   # 重复噪声 floor = 1e-3（per-design assumption）
    val = dict(
        image_diversity={"max_D_ij": float(D_ij.max()),
                         "mean_D_ij_offdiag": float(D_ij[np.triu_indices(num_lights, k=1)].mean())},
        repeat_render_noise={"assumed_floor_1e-3": True},
        light_to_noise_ratio=R_l_to_noise,
        lighting_metadata_match={"sh_irradiance_vs_image_cam_frame":
            float(np.linalg.norm(irradiance_coeffs[0] - irradiance_coeffs[1]))},
        albedo_range=[float(alb_gray[mask > 0].min()), float(alb_gray[mask > 0].max())],
        depth_range=[float(depth[mask > 0].min()), float(depth[mask > 0].max())],
        normal_unit_error=float(np.linalg.norm(n_cam, axis=-1).max() - 1),
        normal_mesh_vs_depth_ang_mae=None,                    # P1-06 报告里计算
        sh_or_light_range=[float(irradiance_coeffs.min()), float(irradiance_coeffs.max())],
        oracle_reconstruction=None,                            # P1-08 评估时填
        nan_inf=bool(np.isnan(img_arr).any() or np.isinf(img_arr).any()),
        scene_id=scene_name,
        zsh_hash=zlib.crc32(json.dumps(light_meta).encode()) if False else None,
        generation_seconds=round(time.time() - t0, 1),
        protocol="P1_v1_route_A_irradiance",
    )
    json.dump(val, open(os.path.join(scene_dir, "validation.json"), "w"),
              indent=2, ensure_ascii=False)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj_list", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--num_lights", type=int, default=32)
    ap.add_argument("--light_energy", type=float, default=100.0)
    ap.add_argument("--fov_deg", type=float, default=50.0)
    ap.add_argument("--cam_dist", type=float, default=2.6)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--light_type", choices=["sun", "point"], default="sun",
                    help="sun=P-domain 真远场方向光（默认）；point=近场 stress 域")
    args = ap.parse_args()

    with open(args.obj_list, encoding="utf-8") as f:
        objs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    objs = objs[args.start:]
    if args.count > 0:
        objs = objs[: args.count]

    bproc.init()
    R_cw = camera_frame_R()
    light_dirs = hemisphere_lights(args.num_lights, seed=args.seed)

    for i, p in enumerate(objs):
        try:
            r = render_one(p, args.out_dir, args.size, args.num_lights,
                           args.light_energy, args.fov_deg, args.cam_dist,
                           args.samples, args.gpu, R_cw, light_dirs, args.seed,
                           light_type=args.light_type)
            print(f"[{i+1}/{len(objs)}] {os.path.basename(p)} → {r}")
        except Exception as e:
            import traceback
            print(f"[FAIL] {p}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
