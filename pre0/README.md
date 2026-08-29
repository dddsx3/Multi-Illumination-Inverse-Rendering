<!--
```yaml
protocol_status: superseded
valid_for_multi_illumination_claims: false
reason: duplicated illumination frames
supersedes: synthetic_v3 (commit 2c23026, before 2026-08-29)
verdict: docs/verdicts/PRE0_VERDICT.md
next_valid_artifact: p1/calibration_set/, p1/physics_clean/
```
-->

# pre0/ · Multi-Illumination Inverse Rendering · PRE-0 前置证据交付

> 按 [`../docs/PRE0_任务书.md`](../docs/PRE0_任务书.md) 执行。
> **总裁决**：`./HANDOFF.md` —— Gate B FAIL（synthetic_v3 多光照维度无效，
> 数据须先修复）；Gate A 有条件 PASS / Gate C BLOCKED / Gate D 有条件 PASS。

## 目录（与任务书 §10 交付包结构对应）

```
pre0/
├── HANDOFF.md                 ← 12 问逐答 + 4 门禁裁决（最重要的单文件）
├── README.md                  ← 本文件
├── protocol/
│   ├── pre0_protocol.yaml     ← 唯一实验协议（v1.0）
│   ├── DATASET_CONTRACT.md    ← 数据合同 + 缺陷通告
│   └── split_manifest.json    ← split hash 98e91fc5…
├── source/
│   ├── dataset/scene_loader.py     ← 三图像域/GT 加载
│   ├── renderer/oracle.py          ← PRE-01（GT oracle + 近场点光 + 帧审计）
│   ├── renderer/relight.py         ← 解析补光（15 光）
│   ├── information_audit/pre02.py  ← PRE-02（exp 1-4）
│   ├── probe_models/probes.py      ← PRE-03（ProbeA/B/C，~0.71M，共享 encoder/decoder）
│   ├── train/train_probe.py        ← PRE-03 训练（统一预算，RAM 缓存 + bf16）
│   ├── evaluate/pre04.py           ← PRE-04/05（N曲线/交叉子集/novel-dup/置换/融合敏感度 + oracle-query-light）
│   ├── evaluate/resummarize_probes.py  ← 修复广播后重算 test 指标
│   └── evaluate/diligent_evaluator.py  ← PRE-07（合同+260 固定子集）
├── oracle_renderer/           ← PRE-01 输出（ORACLE_AUDIT.md + csv + png）
├── information_audit/         ← PRE-02 输出（INFORMATION_AUDIT.md + csv + png）
├── probe_results/             ← PRE-03 输出（_summary.json + 训练曲线）
├── checkpoints/               ← PRE-03 模型（probe_{A,B,C}_{best,last}.pth）
├── logs/                      ← PRE-03 训练日志
├── evidence_accumulation/     ← PRE-04 输出
├── heldout_relighting/        ← PRE-05 输出
├── literature/                ← PRE-06（matrix 43 篇 + CLOSEST_PRIOR_WORK.md）
└── benchmark/                 ← PRE-07（合同 + diligent_subsets.json）
```

## 跑哪条就能得到什么

| 想验证 | 命令 | 产物 |
|---|---|---|
| 物理协议渲染器能否解释数据 | `python pre0/source/renderer/oracle.py --split test` | `oracle_renderer/` 整套 |
| N 在数据/解析域中提供多少信息 | `python pre0/source/information_audit/pre02.py --exp 1 3 4 --domains analytic15_sh` 然后 `--exp 2 --exp2_scenes 32 --domains analytic15_sh` | `information_audit/` 整套 |
| 三个最小 Probe 的差异 | `python pre0/source/train/train_probe.py --probe {A\|B\|C}` | `checkpoints/` `probe_results/` `logs/` |
| 完整 PRE-04/05 诊断 + held-out | `python pre0/source/evaluate/pre04.py --probes A B C` | `evidence_accumulation/` `heldout_relighting/` |
| DiLiGenT 评估器 | `python pre0/source/evaluate/diligent_evaluator.py --make_subsets` 首次落盘；之后接 `--eval_object` | `benchmark/diligent_subsets.json` |

## 数字速查

| 项 | 数值 | 文件 |
|---|---|---|
| GT oracle PSNR（linear 域，124 scenes） | 14.88 dB (P10 8.28) | `oracle_renderer/oracle_summary.json` |
| GT oracle PSNR（train 域 = 双重 gamma） | **6.31 dB** ← 旧管线域错位 | 同上 |
| 解析 15 光 N 曲线（SH 族，TV 正则，124 场景） | SI-MAE: 0.077 (N=1) → 0.020 (N≥5) | `information_audit/subset_results.csv` |
| 三 Probe test SI-MAE | A 0.0545 / B 0.0545 / C 0.0547 | `probe_results/probe_*_summary.json` |
| held-out PSNR（任何 N） | 15.5 dB（A/B）/ 16.9 dB（C） | `heldout_relighting/heldout_summary.json` |
| 光照方向世界系 vs 相机系 | 0.92° vs 69.6°（帧错位是二阶） | `oracle_renderer/lighting_convention.csv` |
| 跨 5 光图像差异 | 0.000~0.049/255（数据缺陷） | 在 ORACLE_AUDIT §7 与 INFORMATION_AUDIT §0 |

## 已知遗留（数据修复后必须重做的实验）

1. PRE-02 真实域全部（exp1-4 + exp2）：当前 N=1 噪声地板反映数据缺陷；
2. PRE-04 全部（probe 的 N 曲线、cross-subset 一致性、novel-dup、held-out）；
3. PRE-07 DiLiGenT 评估：当前 0 个 probe 法线 .npy 待生成；
4. 解析域 exp4 应让优化预算随 N 缩放（restarts≥2, iters↑）以排除"预算-可辨识性"混杂。

## 不在 PRE-0 范围内

- 任何新模型架构设计、3-seed 主训练、消融矩阵、论文图表；
- 旧 FusionUNet 训练流程恢复（PRE-0 §1 禁止）；
- 数据集重渲（数据修复任务，另立项）。
