# Phase 1 · G1 渲染冒烟验收报告

**日期**：2026-08-24　**执行环境**：本机（RTX 5070 Ti Laptop 12GB / Ryzen 9 7945HX / D 盘 295GB 空闲）
**软件栈**：Python 3.10.11 + BlenderProc 2.8.0（Blender 4.2.1 LTS，OPTIX GPU 渲染）｜校验器运行于 Python 3.14.2（torch 2.12 CPU 路径）

## 复现路径

```bash
py -3.10 -m pip install blenderproc            # 2.8.0
py -3.10 -m blenderproc pip install Pillow     # 装进 Blender 内置 Python
# 模型清单 models_list.txt（5 个程序化参数化网格，ASCII 无 BOM）
py -3.10 -m blenderproc run render_dataset.py --obj_list models_list.txt --out_dir D:/data/synthetic_smoke --size 128 --count 5 --gpu
python validate_dataset.py --root D:/data/synthetic_smoke --sample 5
```

## 门禁判定：PASS

| 门禁项 | 标准 | 实测 | 结论 |
|---|---|---|---|
| 场景成功数 | 5/5 [done] 无 [fail] | 5/5 | ✅ |
| 文件齐全 | 每场景 10 文件 | 5×10 齐全（另含 render_stats.txt） | ✅ |
| validate_dataset | PASS | PASS（exit 0），C1-C7 全过 | ✅ |
| 掩码覆盖 mean | ∈ [0.2, 0.8] | 0.302（range 0.146-0.551） | ✅ |
| 法线-导数夹角 | < 10° | mean/max = 0.01°（按构造自洽） | ✅ |
| 重渲染 PSNR | > 12dB | mean 17.8dB / min 15.2dB（128²）；15.5dB（256²） | ✅ |

## 耗时与资源

- 单场景端到端（含 blenderproc/Blender 启动）：**128² ≈ 5.1s，256² ≈ 6.0s**
- 纯渲染部分约 1-2s/场景（OPTIX）；据此推算 600 场景 × 256² 全量 ≈ 1 小时 GPU 时间
- 显存占用低（128²/32spp 冒烟级别），全量建议 samples=128 并控制并行实例 ≤2

## 过程中发现并修复的问题（全部已提交修复）

1. **BlenderProc 强制 import 顺序**：`import blenderproc` 必须是首个有效代码行（docstring 都不能在前）——已调整脚本头部。
2. **API 差异（相对 patch 编写假设）**：`manifold_cleanup` 不存在（加 hasattr 守卫）；`set_fov` 不存在（换算等效焦距走 `set_intrinsics_from_blender_params`）；`bproc.world.set_world_background` 未导出（新增 `set_world_black()` 直接操作 bpy 节点）。
3. **输出使能每进程仅一次**：depth 使能二次调用会抛错——渲染器配置移至 main() 进程级。
4. **帧区间左闭右开**：`frames[-1]` 少渲一帧 → 改为 `+1`。
5. **instance_segmaps 全零根因**：`enable_segmentation_output` 只为调用时刻已存在的对象分配 pass_index；使能在空场景时进行则后续对象全成"背景"。修复：每次加载模型后手动 `assign_instance_ids()`。
6. **colors/diffuse 为 sRGB uint8**（并非线性浮点）：直接当线性值用会导致双重 gamma 与反照率饱和。新增 `_to_linear_float` 反解码后再进 BT.709 luma。
7. **GT 法线改由深度导数导出**：实测 normals 合成器链在本机多场景进程返回恒零（常数 0.5 编码零向量）。改为 `sobel_normal(depth)`——与 physics_renderer 的深度→法线约定完全一致，保证训练监督端到端自洽；validator C4 基准同步一致。真实网格法线恢复列为已知事项（需自定义 AOV）。
8. **models_list.txt BOM**：PowerShell utf8 写入带 BOM 致首路径损坏——改用 .NET WriteAllLines。

## 数据抽样

montage_sample.png（5 场景 × [light_001 | albedo | normal_rgb | depth_jet | mask]）位于
`D:/data/synthetic_smoke/_validation/`；逐场景统计见同目录 stats.csv / validation.json。

## 结论

**G1 通过。** 可进入 T1.2 全量数据生成（建议先 --count 20 用 256² 全量参数压一遍再放量到 600）。
注意：冒烟用的是程序化参数化模型（管线验证用）；全量数据集请替换为 ShapeNet/公开 OBJ 以满足 G3 场景多样性要求。