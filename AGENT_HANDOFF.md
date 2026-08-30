# AGENT_HANDOFF · 交接文档（写给下一个执行 agent）

> **更新**：2026-08-31 · commit `6d895d3` · 写作目的：下一份任务书下来时，
> 你（新 agent）能在 15 分钟内建立完整上下文并直接开跑，不重蹈任何已踩过的坑。
> 阅读顺序：§1 状态 → §2 必读文件 → §3 环境 → §4 纪律红线 → §5 踩坑清单
> → §6 快速启动命令 → §7 待办与阻塞 → §8 Git 锚点。

---

## 1. 项目 30 秒现状

**Multi-Illumination Inverse Rendering**：固定单相机，输入同场景 N 张未知
光照图像（N 可变 1~32），feed-forward 联合恢复 canonical albedo +
depth/normal + per-light 9D SH（Route A irradiance coefficients）。

**科学核心（未定核，hypothesis 状态）**：GA-ISI —— 光照子集的 gauge-aware
有效信息（而非基数 N）控制联合分解可辨识性。
R4 定核 Gate：**G1 PASS**（固定 N 内 ρ(λ⁺_min, 误差)=-0.42~-0.86，全 p<1e-4）
/ **G2 FAIL**（超出 N 的解释力 ΔR²=0.002<0.05）→ **不定核、不杀方向**，
R4' 修正清单待做（见 §7）。

**物理协议已封口**：P 域 SUN 远场方向光 + 纯 Diffuse 材质 + 相机系统一 +
线性域；Oracle Gate **PASS（Or1 SI-PSNR = 28.25 dB > 25）**。

**历史包袱（已裁决，勿翻案）**：synthetic_v3 五图同图数据灾难（多光照维度
无效）→ 全部旧多光照结论作废，见 `docs/verdicts/PRE0_VERDICT.md`。

## 2. 必读文件（按此顺序）

| 文件 | 内容 | 何时读 |
|---|---|---|
| `p1/protocol/CLAIM_REGISTRY.md` | 论文三句话宪法 + 降级路径 | **任何工作之前**（一切实验必须服务于它） |
| `p1/P1_R0_STOP_LINE.md` | R 轮停线决议 + R1-R4 全部执行结果 | 之前 |
| `p1/HANDOFF.md` | P1 阶段 15 问逐答（部分数字已被 R 轮更新覆盖） | 之前 |
| `p1/protocol/LIGHTING_MODEL.md` | SH 语义（Â=[π,2π/3,π/4]）+ 单位/gauge/ReLU 政策 | 写代码碰 SH 时 |
| `p1/protocol/DATASET_CONTRACT.md`（pre0 同名） | 数据合同 + 缺陷通告 | 碰数据时 |
| `p1/protocol/EXPERIMENT_CONTRACT.md` | E1-E9 实验契约 + C0 Gate | 跑实验前 |
| `p1/protocol/IDENTIFIABILITY.md` | GA-ISI 理论骨架（Schur 补推导） | 写 §5 章节时 |
| `p1/literature/RELATED_WORK_MATRIX_v2.md` + `closest_prior_verified.md` | 新颖性边界（哪些 claim 已被占） | 写 Related Work / 新 claim 前 |
| `docs/verdicts/PRE0_VERDICT.md` | 历史结论冻结（勿引用作废数字） | 之前 |
| `docs/P1_任务书.md` / `docs/PRE0_任务书.md` | 两份任务书原文 | 之前 |
| `EXPERT_BRIEFING.md` | 专家视角一页纸 | 需要外部求助时 |

## 3. 环境与运行方式

- **GPU**：RTX 5070 Ti Laptop 12GB（CUDA 可用，torch 2.12+cu128，主 python 3.14）。
- **BlenderProc 2.8.0 + Blender 4.2.1** 在 **Python 3.10 环境**，只能用 CLI 调：
  ```
  /c/Users/35702/AppData/Local/Programs/Python/Python310/Scripts/blenderproc run <script.py> [args]
  ```
  主 python（3.14）**import 不了 blenderproc**（只装在 3.10）。
- 数据：`D:\data\synthetic_v3`（**invalid**，仅单光照参考）、
  `D:\data\DiLiGenT`（真实基准）、
  `p1/calibration_set/data_sun/`（4 场景 SUN 32 灯，P 域现行）、
  `p1/calibration_set/data/`（**nearfield_stress**，只许 robustness 用）。
- 交付物输出习惯：zip 打到 `D:\MIR_Archive_20260829\`，报告 md 放归档根 +
  仓库内双份；每次里程碑 commit+push（远端 = github.com/dddsx3/Multi-Illumination-Inverse-Rendering）。

## 4. 纪律红线（违反 = 返工）

1. **CLAIM_REGISTRY 三句话是宪法**——新实验先问"服务哪一句"；
   "超出 N 解释力"未证实前，禁止把 H-COND 写成结论。
2. PRE-0/P1 任务书的禁止事项持续有效：不设计主网络、不引入 attention/FiLM、
   不跑 100 epoch 正式实验、不恢复旧 FusionUNet、不用 PRE-0 Probe 结果判断
   aggregation 优劣。
3. 禁止引用作废数字："75% SH 误差 / 需升 L=4"、"22.25dB 归因 SH 截断"、
   "synthetic_v3 N 曲线"、"旧管线 N 敏感性结论"。
4. 一切物理量 **camera frame + linear 域**；albedo 评估只用 SI-MAE
   （scale gauge）；法线 GT 口径 = mesh normal（normal_depth 仅对照）。
5. 图像 PNG 落盘 = sRGB 编码；任何与线性域混算前必须精确 sRGB 反变换
   （**禁止 `x**2.2` 简易式**，旧管线双重 gamma 事故的根源）。
6. 新 claim 措辞用安全式："Closest works address X/Y, while none of the
   works examined here studies…"（R5 规则）。

## 5. 踩坑清单（每条都真实付出过代价，别再踩）

**BlenderProc / 生成器**
1. `blenderproc run` 的脚本**第一有效行必须是 `import blenderproc`**（docstring
   都不能在前面）；主 python 里 import 它会直接 RuntimeError。
2. `bproc.renderer.enable_depth_output()` 等 enable 是**进程级一次性**——
   多场景复用同进程会 "can not be called twice"。**每场景独立进程跑**（bash for 循环）。
3. **BlenderProc normals AOV 已是相机系**——不要再做 world→cam 旋转
   （二次旋转曾把 oracle 打到 15 dB，修复后 26.7+）。
4. **SUN 能量语义**：strength=100 会全图饱和（线性域裁到 1.0，形状信息全毁）。
   用 `--light_energy 3.0`（I_eff=S/π≈0.95）。渲染后必查饱和占比=0。
5. **P 域必须纯 Diffuse BSDF**（默认 Principled 有 specular=0.5，colors 通道
   混高光）；且 `materials.clear()` 要无条件执行（OBJ 无材质时 if 判断会跳过 append）。
6. BlenderProc frame animation（`Light.set_location(..., frame=k)`）**不可信**
   ——synthetic_v3 五图同图灾难根源；**每灯独立 render call**。
7. Blender 4.2 没有 `bpy.ops.export_scene.obj`（用脚本自带 write_obj）。
8. 空掩码场景（如水平 plane 在 30° 俯角下不可见）要 raise 干净跳过。
9. Windows git-bash **不支持 process substitution** `<(echo ...)` 当 obj_list——
   用临时文件；且别把 .obj 本身当 list 传（每行 OBJ 内容会变成"场景名"建垃圾目录）。
10. 渲染输出 dtype 自动检测：uint8→sRGB 反变换；float→已是线性。

**SH / 物理**
11. 卷积系数用 `A_L=[π,2π/3,π/4]`（`sh.py`），**不是** K_L 旧值；
    E_L2 解析式 = 0.25+0.5μ+0.3125·P₂(μ)。
12. 全 (0,0,1) 法线初始化是**鞍点**（SH x/y 项梯度恒 0）——联合优化必须
    随机小扰动 + c 小噪声破对称（exp2 教训）。
13. 存在全局旋转 gauge（法线+光照同旋转图像不变）——法线角误差必须
    Kabsch 对齐后再报告；albedo/light 尺度 gauge → SI-MAE。

**训练 / 求解器**
14. 训练渲染用 `use_edge_aware=False`（与数据定义一致；True 是旧 config 遗留冲突）。
15. GPU 双进程并行会互相拖慢数倍（小 kernel 串行化）——**串行跑大任务**。
16. 后台跑长任务：stdout 会缓冲看不到进度，用 `python -u` 或查
    checkpoint/csv 落盘时间戳；`blenderproc run -u` 不认 -u（会传给脚本）。
17. 系统内存可能被残留 python 进程吃满（出现过 2.2MB 都分配失败的 OOM）——
    跑大任务前 `Get-Process python` 清点并清理。
18. `SceneBatcher._CACHE` 类内缓存跨实例共享——注意多 loader 场景的内存。

**评估 / 统计**
19. pooled 回归跨场景比较必须先按场景 z-score 归一（场景尺度效应会淹没
    子集效应）——R4 G2 的教训；且别只用最弱指标（logdet）下结论。
20. solver 收敛判据（tail-loss<1e-7 & grad<1e-3）过严会 0% success——
    对比实验前先标定判据；对比只用收敛 trials（P1-10 纪律）。
21. 评估代码里 `[B,K,H,W]` vs `[B,K,1,H,W]` 广播是复发 bug 源
    （recon PSNR 曾错 20dB）——shape 断言先写。

## 6. 快速启动命令（全部从仓库根目录）

```bash
REPO=/d/MIR_Archive_20260829/Multi-Illumination-Inverse-Rendering
BP=/c/Users/35702/AppData/Local/Programs/Python/Python310/Scripts/blenderproc

# SH/坐标系单元测试（秒级，先跑确认环境）
python p1/tests/test_sh_physics.py && python p1/tests/test_coordinate_frames.py

# P 域渲染一个场景（~8 分钟 GPU；注意 --light_type sun --light_energy 3.0）
echo p1/calibration_set/meshes/sphere.obj > /tmp/one.txt
$BP run p1/source/generation/render_multilight.py --obj_list /tmp/one.txt \
  --out_dir p1/calibration_set/data_sun --num_lights 32 --gpu --size 128 \
  --samples 32 --light_type sun --light_energy 3.0

# Oracle Gate（SI-PSNR 口径）
python p1/source/calibration/oracle_gate.py --data_root p1/calibration_set/data_sun

# GA-ISI 分数（CPU 秒级）
python p1/source/information_audit/gauge_fisher.py --data_root p1/calibration_set/data_sun \
  --out_csv p1/information_audit/ga_isi.csv --ns 3 5 8 12 --subsets_per_N 20

# 定核 Gate 扫描（~2h GPU，320 runs）
python -u p1/source/information_audit/defining_gate_r4.py --subsets_per_N 20 --restarts 1

# P1-13 全量生成（200×32，~27h；务必 --light_type sun）
# $BP run p1/source/generation/render_multilight.py --obj_list <200场景列表> \
#   --out_dir D:/data/synthetic_v4 --num_lights 32 --gpu --size 256 --samples 64 \
#   --light_type sun --light_energy 3.0   # 分片跑，每场景独立进程

# Information Audit v2 / Probe 训练 / Learnability Gate（数据就位后）
python p1/source/information_audit/information_audit_v2.py --data_root D:/data/synthetic_v4 --restarts 2 --exps 1 2 3 4
python p1/source/probes/train_probe_p1.py --probe A --mode varN   # + fixed5；A/B/C
python p1/source/evaluation/learnability_gate.py --probes "A varN" "A fixed5"
```

## 7. 待办与阻塞（优先级序）

| # | 任务 | 状态 | 依赖 |
|---|---|---|---|
| 1 | **R4' 修正**：场景尺度规范 + 扩场景数（cube/sphere/hemisphere/cylinder 之外加 mesh）+ restarts≥2 收敛控制 → 重测 G2 | 脚本就绪，半天 | 无（可用现有 calibration 加 2-3 个新 mesh） |
| 2 | **P1-13 全量 200×32**（~27h GPU，分片） | 脚本就绪 | R4' 若 PASS 则优先级↑；mesh 资产需先补（objaverse 已删，校准 mesh 是脚本生成的） |
| 3 | repeat-render noise 实测标定（替 G1 占位 floor 1e-3） | 脚本接口已留 | 同一场景同灯渲 5 次 |
| 4 | normal_depth 反投影边缘修复（当前 65° 假夹角） | 半天 | 无 |
| 5 | P1-15/16 Probe varN/fixed5 + C0-C5 Gate | 脚本就绪 | #2 |
| 6 | 论文 Draft 0（`paper_draft_outline.md`：60-70% 不依赖训练） | 骨架就绪 | 无（可与 #1 并行） |
| 7 | mesh 资产补充：objaverse 原始 GLB 已删，需按 `patches_asset_lists/obj_models_list.txt`（在归档根）跑 `download_objaverse.py` | 半天+下载 | 无 |

## 8. Git 锚点与回滚

```
2c23026  重置前最后提交（Phase 2，含全部旧产物；历史结论已冻结）
a9f9526  PRE-0 起点（bundle 恢复 + 空提交验证推送）
2872550  PRE-0 交付（数据灾难发现 + PRE-0 全套）
ad9d183  P1 交付（基础设施 + calibration 15.3→28.25 前夜）
20d2ba8  专家审查轮起点（EXPERT_BRIEFING）
506e248  R0-R4 交付（SH 修正 + SUN + GA-ISI + 定核 Gate G1 PASS/G2 FAIL）★当前基线
6d895d3  当前 HEAD（briefing 补 R 轮）
```

完整历史离线恢复：`git clone git_history_full.bundle`（195MB bundle 在
`D:\MIR_Archive_20260829\`，若无则从任意 commit `git bundle create` 现做）。

## 9. 给下一棒的三句话

1. **先读 CLAIM_REGISTRY 再动手**——项目的全部价值押在"gauge-aware
   illumination-set information"这一条上，做任何事前问它服务哪一句。
2. **物理已封口（28.25dB），别再动协议**——要动就先读 STOP_LINE 并写新的
   停线文档；§5 的 21 条坑是真实代价换来的。
3. **R4' 是当前唯一挡在"定核/杀方向"分叉前的事**——小数据能做完，
   不要先烧 27h 全量生成。

---

*写作者：ZCode agent · 2026-08-31 · 本文档随每个里程碑更新；过期内容以
verdict/STOP_LINE 类"裁决型文档"为准。*
