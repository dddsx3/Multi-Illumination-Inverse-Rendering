import blenderproc as bproc
# R4″ Task F · controlled geometry mesh 生成器（任务书 §17）
# 两个 family，各 5 level，只有 normal coverage 系统变化（相机/材质/尺度/renderer/光池全固定）：
#   family A prism4→prism8→prism16→prism32→cylinder
#   family B cube→bevel0.05→bevel0.15→bevel0.30→rounded cube
# 首行必须 import blenderproc（踩坑 #1）
import bpy
import argparse
import math
import os


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for o in bpy.context.scene.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = None


def write_obj(path, mesh, name="prim"):
    bpy.context.view_layer.update()
    mw = mesh.matrix_world
    verts = [tuple(mw @ v.co) for v in mesh.data.vertices]
    loops = mesh.data.loops
    polys = mesh.data.polygons
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Exported by R4pp make_controlled_geometry_meshes.py\n")
        f.write(f"o {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for p in polys:
            idx = [loops[i].vertex_index + 1 for i in p.loop_indices]
            f.write("f " + " ".join(map(str, idx)) + "\n")
    return os.path.isfile(path) and os.path.getsize(path) > 0


def make_prism(vertices, radius, depth, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius,
                                        depth=depth, location=location)
    return bpy.context.view_layer.objects.active


def make_cylinder(radius, depth, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius, depth=depth,
                                        location=location)
    return bpy.context.view_layer.objects.active


def make_cube(size, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    return bpy.context.view_layer.objects.active


def bevel_cube(obj, width, segments=2):
    """对 cube 做 bevel（编辑模式）。affect 枚举是 VERTICES。
    segments 递增 → 法线簇连续增多（B 族的 G 区分度来自 segments 而非 offset）。"""
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.bevel(offset=width, segments=segments, affect="VERTICES")
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    bproc.init()

    items = []
    # family A：prism4→cylinder（顶点数递增 = normal coverage 连续增加）
    for nv, tag in [(4, "A_prism4"), (8, "A_prism8"), (16, "A_prism16"),
                    (32, "A_prism32")]:
        items.append((tag, make_prism, dict(vertices=nv, radius=0.55, depth=1.1)))
    items.append(("A_cylinder", make_cylinder, dict(radius=0.55, depth=1.1)))
    # family B：cube→rounded（offset + segments 递增 → 法线簇 6→连续）
    for w, seg, tag in [(0.0, 0, "B_cube"), (0.05, 2, "B_bevel05"),
                        (0.15, 4, "B_bevel15"), (0.30, 8, "B_bevel30"),
                        (0.55, 12, "B_rounded")]:
        items.append((tag, make_cube, dict(size=1.0)))

    written = []
    for name, fn, kw in items:
        p = os.path.join(args.out_dir, name + ".obj")
        try:
            clear_scene()
            obj = fn(**kw)
            if "bevel" in name or "rounded" in name:
                width = float(name.split("bevel")[-1]) if "bevel" in name else 0.55
                seg = 12 if "rounded" in name else (2 if name == "B_bevel05"
                                                    else (4 if name == "B_bevel15" else 8))
                if width > 0:
                    bevel_cube(obj, width, segments=seg)
            ok = write_obj(p, obj, name=name)
        except Exception as e:  # noqa: BLE001
            print(f"  {name} FAIL: {e}")
            ok = False
        size = os.path.getsize(p) if os.path.isfile(p) else 0
        print(f"  {name:14s} {'OK' if ok else 'FAIL'} ({size} B)")
        if ok:
            written.append(p)
    list_path = os.path.join(args.out_dir, "controlled_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in written:
            f.write(p + "\n")
    print(f"list -> {list_path} ({len(written)} items)")


if __name__ == "__main__":
    main()
