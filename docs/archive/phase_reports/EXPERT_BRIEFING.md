# EXPERT_BRIEFING · 外部专家求助导航（2026-08-30）

> 本文件为向外部专家求助/咨询准备的一页式导航。
> 读者按 §1→§2→§3 顺序 10 分钟可建立完整背景；§4 是建议重点咨询的问题。

## 1. 项目一页纸现状（TL;DR）

**Multi-Illumination Inverse Rendering**：固定单相机，输入同一场景 N 张
未知光照图像（N 可变 1~32），联合分解 canonical albedo + depth/normal +
per-light 二阶 SH 光照；核心研究问题是 **variable-cardinality evidence
accumulation**（多光照证据累积的可辨识性与可学习性）。

| 时间线 | 事件 | 结论文档 |
|---|---|---|
| ~2026-08-27 | Phase 2 中期（FusionUNet + 6 变体消融），v2 best: PSNR 37.25 dB | `docs/Phase2_验收报告初稿.md` |
| 2026-08-28 | 项目完全重置（本地全删，bundle 备份）；PRE-0 任务书发布 | `docs/PRE0_任务书.md` |
| 2026-08-29 | **PRE-0 执行：发现数据灾难**——synthetic_v3 的 5 张"不同光照"图实为同一张图（BlenderProc 帧动画失效），多光照维度无效；Gate B FAIL | **`pre0/oracle_renderer/ORACLE_AUDIT.md`（必读）** |
| 2026-08-30 晚 | **专家审查触发的 R 轮修正**：①SH Lambertian 卷积系数错误（√π 系 → Â=[π,2π/3,π/4]，MC MAE 0.0309 与专家核算一致，"需升 L=4"作废）②生成器三连修（normals 二次旋转/材质 specular/SUN 饱和）→ **Oracle Gate PASS 28.25 dB** ③GA-ISI gauge-aware Fisher 实现 + 320 子集定核 Gate：**G1 PASS（固定 N 内 ρ=-0.42~-0.86）/ G2 FAIL（ΔR²=0.002）→ 不定核不杀方向** | **`p1/P1_R0_STOP_LINE.md`（附录含全部结果）** |
| 2026-08-30 | **P1 基础设施封口**：生成器修复（每灯独立 render call）+ SH 物理重构（Route A irradiance）+ calibration 5 mesh × 32 灯实证 N 曲线 2.9× 改善（N≥8 饱和）+ Oracle Gate 22.25 dB | **`p1/HANDOFF.md`（15 问逐答）** |
| 当前 | 待跑：P1-13 全量 200×32 数据生成（~27h GPU）→ 正式 Information Audit → Probe 重训 → C1-C5 Learnability Gate | `p1/HANDOFF.md §门禁状态` |

**当前总裁决**：多光照信息在修复后的真实渲染数据中确凿存在（首次实证）；
下一阶段唯一下注假设 **H-COND**（光照子集质量通过局部 Fisher 条件数控制
逆渲染可辨识性），见 `p1/HANDOFF.md` Q15。

## 1.5 最新一轮（R 轮）关键裁决

- **`_archive/P1-R_定核Gate轮交付报告.md`**（包内）或仓库根交付报告——R0~R4 全过程
- 当前 H-COND 状态：G1（固定 N 内子集质量预测误差）强支持；G2（超出 N 解释力）未通过 → **不定核、不杀方向**，R4' 修正清单见 STOP_LINE 附录

## 2. 必读文件（按优先级）

1. **`p1/HANDOFF.md`** — 15 问逐答 + 门禁状态（当前状态的最权威描述）
2. **`pre0/oracle_renderer/ORACLE_AUDIT.md`** — 数据灾难发现全过程 + 物理协议审计
3. **`docs/verdicts/PRE0_VERDICT.md`** — 历史结论永久冻结（哪些旧数字作废）
4. **`p1/protocol/LIGHTING_MODEL.md`** — SH 语义（Route A）与截断误差实测
5. **`p1/literature/closest_prior_verified.md`** — 8 篇最接近工作 + novelty 风险
6. `p1/information_audit/INFORMATION_AUDIT_v2.md` — N 信息量受控实验
7. `pre0/protocol/DATASET_CONTRACT.md` / `p1/protocol/split_manifest.json` — 数据合同

## 3. 事故记录索引（INC-0001 ~ 0012，全部在 `docs/incidents/`）

| 编号 | 一句话 | 对当前的意义 |
|---|---|---|
| INC-0001 | 训练 NaN 发散 | 数值稳定性经验（fp16/bf16 选择） |
| INC-0002 | Blender 连渲崩溃 | 单进程长渲染的脆弱性 → P1-04 每灯独立 call 的动机之一 |
| INC-0003 | 冒烟覆盖生产 checkpoint | 产物隔离纪律 |
| INC-0004 | T16 基线误用未冻结子集 | split 冻结纪律的由来 |
| INC-0005 | 断点恢复阶段状态不同步 | 续跑状态机 |
| INC-0006 | 续跑 off-by-one 与 n5rgb 污染定性 | 编排器续跑风险 |
| INC-0007 | fp16 缺 unscale 与 T4 回退 | 混合精度陷阱 |
| INC-0008/0009 | spawn/num_workers 编排问题 | Windows 多进程训练陷阱 |
| INC-0010（3 份：执行版/v2/v3） | 数学底层三重偏差 + 审计裁决 v2/v3 | 指标定义审计的完整案例 |
| INC-0011 | F-resA 单 run 反转与 seed 噪声 | 单 run 结论不可信 |
| INC-0012 | 物理约束补建与 albedo_smooth 二次风险 | 物理约束模块 |
| （未编号） | **synthetic_v3 五图同图数据灾难**（2026-08-29） | 本项目最大事故，见 `ORACLE_AUDIT.md §7` |

## 4. 建议向专家咨询的问题（附我方现有答案）

1. **SH 阶数决策**：L=2 单方向光截断误差实测 ~75% 相对 MAE
   （`p1/tests/test_sh_physics.py`），P 域 oracle 22.25 dB。是否值得
   直接升级 L=4（卷积系数 k_3/k_4 需补推导），还是保留 L=2 用
   "多光叠加降误差" 论证？
2. **H-COND 假设评估**：per-light Fisher 有效秩 4.6/9 且与 N 无关 →
   我们主张"子集联合设计矩阵的条件数（而非 N）是可辨识性主控量"
   （`p1/HANDOFF.md` Q15）。这个 framing 在 PS/逆渲染文献中是否已有
   等价工作？（对照 `p1/literature/closest_prior_verified.md`）
3. **P 域 vs R 域双数据集策略**（`p1/protocol/split_manifest.json`）：
   mechanism 验证（无阴影无间接光）与 robustness（Cycles 全效果）
   分开报告——审稿人是否接受这种"物理干净域先行"的论证结构？
4. **variable-N Probe 的监督设计**：per-light SH 头是否会让网络退化成
   "单图解 + 逐光拟合"（PRE-0 已观察到该退化）？variable-N sampling
   （N~U{3..15}）+ GT albedo/depth 监督是否足以避免？
5. **与 SCPS-NIR（ECCV22，联合 mesh+albedo+光照分解）的差异防御**：
   variable-N + 9 维可查询 SH + 2.5D feed-forward 三点是否足够
   （`closest_prior_verified.md` §5）？

## 5. 仓库结构速览

```
├── README.md                    ← 项目简介 + PRE-0/P1 摘要
├── EXPERT_BRIEFING.md           ← 本文件
├── docs/
│   ├── incidents/               ← INC-0001~0012 事故记录（13 份）
│   ├── verdicts/PRE0_VERDICT.md ← 历史结论冻结
│   ├── P1_任务书.md / PRE0_任务书.md
│   └── Phase1/Phase2 报告、HANDOFF_20260828（旧交接）
├── pre0/                        ← PRE-0 全套（协议/审计/信息量实验/Probe/文献）
│   ├── HANDOFF.md（12 问，已标 superseded header）
│   └── oracle_renderer/ORACLE_AUDIT.md ← 数据灾难发现
└── p1/                          ← P1 全套（当前活跃）
    ├── HANDOFF.md（15 问）      ← 当前状态最权威
    ├── protocol/（LIGHTING_MODEL / split_manifest）
    ├── source/（physics/generation/evaluation/calibration/information_audit/probes）
    ├── tests/（SH 物理 + 坐标系，全过）
    ├── calibration_set/（5 mesh × 32 灯实测数据 + Oracle Gate 22.25dB）
    ├── information_audit/（N 曲线 2.9× + conditioning）
    └── literature/closest_prior_verified.md
```

## 6. 数据与可复现性

- 训练/评估数据：`D:\data\synthetic_v3`（**invalid_for_multi_illumination_claims**，
  仅作单光照参考）与 `D:\data\DiLiGenT`（0.9GB，真实基准）；
  新数据 synthetic_v4（200×32）尚未生成（P1-13）。
- 所有实验脚本在 `pre0/source/` 与 `p1/source/`，一键复现命令见
  `p1/HANDOFF.md` 末尾与 `P1_交付报告.md` §6。
- Git 历史：`2c23026`（重置前最后提交）→ `a9f9526`（PRE-0 起点）→
  `2872550`（PRE-0 交付）→ `ad9d183`（P1 交付，当前 main）。
