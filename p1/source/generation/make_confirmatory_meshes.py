import blenderproc as bproc
# P1-R4'-C 确认集 mesh 家族生成器（确定性参数化；objaverse 原始资产已删，
# 按《P1 下一阶段执行任务书 v1.0》T4'.1 用"便宜但多样"的参数化家族）。
# 注意：BlenderProc 要求第一有效行必须是 import blenderproc（无 docstring）。
# 用法：
#   blenderproc run p1/source/generation/make_confirmatory_meshes.py --out_dir p1/calibration_set/meshes_confirmatory
# 设计：
#   - 平滑法线族（F_k 满秩预期）：球/椭球/环面/环面结/平截半球
#   - 簇状法线族（F_k 低秩预期——conditioning regime 多样性）：
#     轴对齐/旋转立方体、四棱/六棱锥、八棱柱、变径圆柱、变径圆锥
#   - 复合族（遮挡 + 混合法线）：雪人、球叠立方、双球、柱顶球、立方顶锥、三立方
#   - 与 Discovery Set（cube/cylinder/hemisphere/sphere 原版）不重复：
#     纯球保留 1 个作平滑基线；其余全部改参数/旋转/复合。
#   - 渲染端会把 mesh 归一到 1.6 max-extent，故绝对尺寸只影响族内相对比例。
import bpy
import argparse
import os


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for o in bpy.context.scene.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = None


def write_obj(path, mesh, name="prim"):
    """自写 OBJ（Blender 4.2 无 export_scene.obj）。matrix_world 需 depsgraph 已更新。"""
    bpy.context.view_layer.update()
    mw = mesh.matrix_world
    verts = [tuple(mw @ v.co) for v in mesh.data.vertices]
    loops = mesh.data.loops
    polys = mesh.data.polygons
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Exported by P1-R4'C make_confirmatory_meshes.py\n")
        f.write(f"o {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for p in polys:
            idx = [loops[i].vertex_index + 1 for i in p.loop_indices]
            f.write("f " + " ".join(map(str, idx)) + "\n")
    return os.path.isfile(path) and os.path.getsize(path) > 0


def add_prim(spec):
    """spec = (prim, kwargs)；返回 active object。"""
    prim, kw = spec
    if prim == "uv_sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(**kw)
    elif prim == "ico_sphere":
        bpy.ops.mesh.primitive_ico_sphere_add(**kw)
    elif prim == "cube":
        bpy.ops.mesh.primitive_cube_add(**kw)
    elif prim == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(**kw)
    elif prim == "cone":
        bpy.ops.mesh.primitive_cone_add(**kw)
    elif prim == "torus":
        bpy.ops.mesh.primitive_torus_add(**kw)
    elif prim == "torus_knot":
        bpy.ops.mesh.primitive_torus_knot_add(**kw)
    else:
        raise ValueError(prim)
    return bpy.context.view_layer.objects.active


def bisect_top(obj):
    """半球裁切（Z>0 保留），复制自 make_calibration_meshes。"""
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.bisect(plane_co=(0, 0, 0.001), plane_no=(0, 0, 1),
                        use_fill=True, clear_inner=False, clear_outer=True)
    bpy.ops.object.mode_set(mode="OBJECT")


def build(name, specs, rot=None, scale=None, bisect=False):
    """specs: 1~2 个 primitive；>1 时 join 成单对象。"""
    clear_scene()
    objs = [add_prim(s) for s in specs]
    if len(objs) > 1:
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    if rot is not None:
        obj.rotation_euler = rot
    if scale is not None:
        obj.scale = scale
    if bisect:
        bisect_top(obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    bproc.init()

    Z30 = 30 * 3.14159265 / 180
    Z45, X30 = 45 * 3.14159265 / 180, 30 * 3.14159265 / 180
    items = [
        # —— 平滑法线族（9）
        ("conf_sphere_r05",        dict(specs=[("uv_sphere", dict(radius=0.5, segments=64, ring_count=32))])),
        ("conf_icosphere_sub3",    dict(specs=[("ico_sphere", dict(radius=0.55, subdivisions=3))])),
        ("conf_ellipsoid_z06",     dict(specs=[("uv_sphere", dict(radius=0.55, segments=64, ring_count=32))], scale=(1.0, 1.0, 0.6))),
        ("conf_ellipsoid_x13z07",  dict(specs=[("uv_sphere", dict(radius=0.55, segments=64, ring_count=32))], scale=(1.3, 1.0, 0.7))),
        ("conf_torus_R05_r02",     dict(specs=[("torus", dict(major_radius=0.5, minor_radius=0.2))])),
        ("conf_torus_R06_r035",    dict(specs=[("torus", dict(major_radius=0.6, minor_radius=0.35))])),
        ("conf_torusknot",         dict(specs=[("torus_knot", dict())])),
        ("conf_hemisphere_sq",     dict(specs=[("uv_sphere", dict(radius=0.6, segments=64, ring_count=16))], scale=(1.0, 1.0, 0.75), bisect=True)),
        ("conf_egg",               dict(specs=[("uv_sphere", dict(radius=0.5, segments=64, ring_count=32))], scale=(0.85, 0.85, 1.2))),
        # —— 簇状法线族（11）
        ("conf_cube_axis",         dict(specs=[("cube", dict(size=0.9))])),
        ("conf_cube_rot30z",       dict(specs=[("cube", dict(size=0.9))], rot=(0, 0, Z30))),
        ("conf_cube_rot45z30x",    dict(specs=[("cube", dict(size=0.9))], rot=(X30, 0, Z45))),
        ("conf_pyramid4",          dict(specs=[("cone", dict(radius1=0.6, radius2=0.0, depth=0.9, vertices=4))])),
        ("conf_pyramid4_rot30z",   dict(specs=[("cone", dict(radius1=0.6, radius2=0.0, depth=0.9, vertices=4))], rot=(0, 0, Z30))),
        ("conf_pyramid6",          dict(specs=[("cone", dict(radius1=0.55, radius2=0.0, depth=0.8, vertices=6))])),
        ("conf_prism8",            dict(specs=[("cylinder", dict(radius=0.45, depth=0.9, vertices=8))])),
        ("conf_cylinder_r03_d12",  dict(specs=[("cylinder", dict(radius=0.3, depth=1.2, vertices=64))])),
        ("conf_cylinder_r06_d06",  dict(specs=[("cylinder", dict(radius=0.6, depth=0.6, vertices=64))])),
        ("conf_cone_r08_d06",      dict(specs=[("cone", dict(radius1=0.8, radius2=0.0, depth=0.6, vertices=48))])),
        ("conf_cone_r04_d12",      dict(specs=[("cone", dict(radius1=0.4, radius2=0.0, depth=1.2, vertices=48))])),
        # —— 复合族（6；多 primitive join 成单 OBJ）
        ("conf_snowman", dict(specs=[
            ("uv_sphere", dict(radius=0.42, segments=48, ring_count=24, location=(0, 0, 0.30))),
            ("uv_sphere", dict(radius=0.28, segments=48, ring_count=24, location=(0, 0, 0.80))),
        ])),
        ("conf_sphere_on_cube", dict(specs=[
            ("cube", dict(size=0.75, location=(0, 0, -0.15))),
            ("uv_sphere", dict(radius=0.28, segments=48, ring_count=24, location=(0, 0, 0.35))),
        ])),
        ("conf_two_spheres_row", dict(specs=[
            ("uv_sphere", dict(radius=0.33, segments=48, ring_count=24, location=(-0.38, 0, 0))),
            ("uv_sphere", dict(radius=0.33, segments=48, ring_count=24, location=(0.38, 0, 0))),
        ])),
        ("conf_cyl_plus_sphere", dict(specs=[
            ("cylinder", dict(radius=0.3, depth=0.7, vertices=48, location=(0, 0, -0.2))),
            ("uv_sphere", dict(radius=0.34, segments=48, ring_count=24, location=(0, 0, 0.35))),
        ])),
        ("conf_cube_plus_cone", dict(specs=[
            ("cube", dict(size=0.7, location=(0, 0, -0.2))),
            ("cone", dict(radius1=0.3, radius2=0.0, depth=0.5, vertices=32, location=(0, 0, 0.35))),
        ])),
        ("conf_three_cubes", dict(specs=[
            ("cube", dict(size=0.42, location=(-0.5, 0, 0))),
            ("cube", dict(size=0.42, location=(0, 0, 0.0), rotation=(0, 0, Z30))),
            ("cube", dict(size=0.42, location=(0.5, 0, 0), rotation=(0, 0, Z45))),
        ])),
    ]

    written = []
    for name, spec in items:
        p = os.path.join(args.out_dir, name + ".obj")
        try:
            obj = build(name, **spec)
            ok = write_obj(p, obj, name=name)
        except Exception as e:  # noqa: BLE001 —— 单 mesh 失败不拖垮整批
            print(f"  {name} FAIL: {e}")
            ok = False
        size = os.path.getsize(p) if os.path.isfile(p) else 0
        print(f"  {name} {'OK' if ok else 'FAIL'} ({size} B)")
        if ok:
            written.append(p)

    list_path = os.path.join(args.out_dir, "confirmatory_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in written:
            f.write(p + "\n")
    print(f"list -> {list_path} ({len(written)} items)")


if __name__ == "__main__":
    main()
