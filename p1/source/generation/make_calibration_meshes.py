# BlenderProc 强制要求：第一个有效代码行必须是 import blenderproc
import blenderproc as bproc
import bpy
import argparse
import os
import math


def clear_scene():
    """删除场景中所有对象（不重置 settings，避免覆盖 active context）"""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for o in bpy.context.scene.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = None


def write_obj(path, mesh, name="prim"):
    """自写 OBJ（顶点/面/winding 翻转）。Blender 4.2 已移除 export_scene.obj。"""
    # 局部变换：Blender 物体 mesh.data 在 bmesh 之前用 obj.matrix_world 应用
    mw = mesh.matrix_world
    verts = [tuple(mw @ v.co) for v in mesh.data.vertices]
    loops = mesh.data.loops
    polys = mesh.data.polygons
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Exported by P1-05 make_calibration_meshes.py\n")
        f.write(f"o {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for p in polys:
            idx = [loops[i].vertex_index + 1 for i in p.loop_indices]
            f.write("f " + " ".join(map(str, idx)) + "\n")
    return os.path.isfile(path) and os.path.getsize(path) > 0


def make_and_export(prim_name, path, **kwargs):
    clear_scene()
    if prim_name == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(**kwargs)
    elif prim_name == "plane":
        bpy.ops.mesh.primitive_plane_add(**kwargs)
    elif prim_name == "cube":
        bpy.ops.mesh.primitive_cube_add(**kwargs)
    elif prim_name == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(**kwargs)
    elif prim_name == "cone":
        bpy.ops.mesh.primitive_cone_add(**kwargs)
    else:
        raise ValueError(prim_name)
    obj = bpy.context.view_layer.objects.active
    # 半球裁切（Z>0）
    if prim_name == "sphere" and "hemisphere" in path:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.bisect(plane_co=(0, 0, 1), plane_no=(0, 0, 1),
                            use_fill=True, clear_inner=False, clear_outer=True)
        bpy.ops.object.mode_set(mode="OBJECT")
    return write_obj(path, obj, name=os.path.splitext(os.path.basename(path))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    bproc.init()
    items = [
        ("sphere.obj",    "sphere",   dict(radius=0.6, segments=64, ring_count=32)),
        ("plane.obj",     "plane",    dict(size=1.2)),
        ("hemisphere.obj", "sphere",   dict(radius=0.6, segments=64, ring_count=16)),
        ("cube.obj",      "cube",     dict(size=0.9)),
        ("cylinder.obj",  "cylinder", dict(radius=0.4, depth=1.0, vertices=64)),
        ("cone.obj",      "cone",     dict(radius1=0.5, radius2=0.0, depth=1.0, vertices=48)),
    ]
    list_path = os.path.join(args.out_dir, "calibration_list.txt")
    written = []
    for name, prim, kwargs in items:
        p = os.path.join(args.out_dir, name)
        ok = make_and_export(prim, p, **kwargs)
        print(f"  {name} {'OK' if ok else 'FAIL'} ({os.path.getsize(p) if os.path.isfile(p) else 0} B)")
        if ok:
            written.append(p)
    with open(list_path, "w", encoding="utf-8") as f:
        for p in written:
            f.write(p + "\n")
    print(f"list -> {list_path} ({len(written)} items)")


if __name__ == "__main__":
    main()
