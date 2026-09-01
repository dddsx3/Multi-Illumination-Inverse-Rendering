# R5 Compute-Aware Campaign · 汇总裁决报告 (2026-09-01)

> **执行者**: ZCode agent (本地 Windows, RTX 5070 Ti Laptop, 无云主机)
> **任务书**: R5-B′ Compute-Aware Branch (五路线审计)
> **一句话总结**: **B 和 C 两条路线都通过了各自门槛, 论文升级通道 (Route D) 已解锁。**

---

## 1. 已完成的三个 Gate (Day 0-1 计划, 实际一天完成)

### Gate A0 · 基线瓶颈画像 → **Case A: 进入 Matrix-free**

正式配置: conf_sphere_r05 × P{2000, 3000, 5000} × N{3, 5, 8}

| P | N=3 eigh占比 | N=5 | N=8 | 单次总耗时 |
|---:|---:|---:|---:|---:|
| 2000 | 80% | 77% | 72% | 0.44-0.63s |
| 3000 | 83% | 78% | 71% | 1.16-1.36s |
| 5000 | **91%** | **88%** | **86%** | 5.8-7.9s |

- 特征分解 (eigh) 占总时间 **80%** (P 越大占比越高, P=5000 达 91%) → **>70% 门槛, Case A 触发**
- 内存泄漏检测: 无 (5 次连续调用 RSS 增长 -0.2%)
- 内存占用: P=5000 单次峰值仅 407 MB → 本地 16GB 机器可跑 P=5000

**大白话**: 时间都花在"把大矩阵做完整分解"这一步上。不存大矩阵、只做乘法的算法 (Route C) 正中要害。

### Gate B1 · 像素降采样排名保真 → **强GO**

正式配置: 3 scene (sphere / cube_axis / egg) × 100 subsets × N=3, 参照 P_ref=2048

| 用多少像素 | 排名一致度 ρ (3 scene 中位) | 前10%重合 (中位) | 提速 |
|---:|---:|---:|---:|
| 128 | 0.9806 | 0.800 | ~4096× |
| 256 | 0.9902 | 0.800 | ~512× |
| **512** | **0.9964** | **1.000** | **~64×** |
| 1024 | 0.9988 | 0.900 | ~8× |

- 冻结门槛判定: **强GO** (512 像素档 ρ=0.996 ≥ 0.95 且 top10 重合 100%)
- **稳健推荐档位: P=512** — 三个场景 top10 全部 1.000;
  P=128 虽过冻结线但 egg 场景 top10 只有 0.5 (会选错最好的子集), 不建议用于 selection
- 像素降采样对 rho 的影响极小: GSIQ 排名由"谱形状"主导, 不需要全部像素

### Gate C1+C2 · Matrix-free → **GO (强)**

正式配置: C1 @ P{500, 2000} × 100 向量; C2 @ P=1000 × 40 光照子集 × 30 探针

| 项 | 结果 | 门槛 | 判定 |
|---|---|---|---|
| C1 matvec 相对误差 | **1e-14** (机器精度) | < 1e-5 | ✅ PASS |
| C2 SLQ ρ (10 步) | **0.9944** | ≥ 0.95 | ✅ |
| C2 SLQ ρ (50 步) | **0.9921** | ≥ 0.95 | ✅ PASS |
| C2 SLQ τ (50 步) | **0.9410** | ≥ 0.9 推荐 | ✅ |
| C2 提速 (10 步, P=1000) | **13.3×** | > 5× (B 路线标准) | ✅ |

- **ρ 在 10 步就饱和到 0.994** — 不需要 100 步; 少步数 + 多探针 (30) 是正确配方
- C2 期间发现并修复一个实现 bug (循环未换光照子集), 修复后结果如上; 修复前单矩阵
  SLQ 精度诊断已确认估计器本身正确 (err 0.003-0.012)

---

## 2. 对照任务书决策树

```
Baseline (A0): eigh 占 80% > 70%  → Case A → Matrix-free  ✅ 触发
Pixel512 ρ>0.95?               → 0.996 ≥ 0.95             ✅ 强GO (B 路线成立)
Matrix-free τ>0.9?             → 0.941 ≥ 0.9              ✅ GO   (C 路线成立)
Selection preserved?           → 未测 (C3, 需要 solver GPU) ⏳ 下一决策点
```

**两条技术路线 (B 降维 + C 无矩阵) 同时成立, 且可叠加**
(P=512 SLQ 组合下, GSIQ 单次成本 ≈ 0.54s/64/13 ≈ **<0.001s 量级**, 即论文全量 11.9 万次
调用从 ~18 小时压到 **~2-5 分钟** 量级 — 待 C3 验证选择保持后即可写进论文)。

---

## 3. 论文状态 (给非专业读者的直接结论)

| 问题 | 回答 |
|---|---|
| 论文是否升级? | **预备升级**。如果 C3 (下一步) 通过, 论文从"解释性指标"升级为"可部署的预算感知信息选择方法" (Route D / Budget-aware GSIQ) |
| 论文是否降级? | 没有。没有触发任何 Stop Rule (1-4 全部未触发) |
| 遇到什么阻塞? | C3 (选择保持验证) 需要 solver 跑重构误差 — 本机 GPU 被 Windows commit 配额锁死, **需要一小块 GPU 算力** (T4 20h 就够, 或任何能跑 torch CUDA 的 Linux 实例 ~2h) |
| 花了多少钱? | 0 元 (全部本地完成) |

---

## 4. 下一步 (按优先级)

1. **C3 selection preservation** (决定论文最终档次): 用 P=512 + SLQ-10 步打分选 top-10% 子集,
   与 oracle/random 对比 solver 重构误差。需要: ~2h GPU (T4 即可)。
2. **Route D 骨架** (B+C 叠加的 accuracy-speed Pareto 曲线): 数据基本已有 (A0/B1/C2 的
   runtime+score 表), 补一个画图脚本即可, 无需新算力。
3. **B2 adaptive coreset** (可选加分项): 只有想再挖 10-20% 提升才做。

---

## 5. 产物索引

| 文件 | 内容 |
|---|---|
| `r5_compute_audit/raw_profile/baseline_profile.csv` | A0 正式画像 9 配置 |
| `r5_compute_audit/raw_profile/baseline_profile.jsonl` | Phase 0 统一审计 JSON |
| `r5_compute_audit/raw_profile/campaign.jsonl` | B/C 路线统一审计 JSON |
| `r5_compute_audit/ranking/pixel_coreset.csv` | B1 全部 (scene, P) 排名诊断 |
| `r5_compute_audit/runtime/matrixfree.csv` | C1+C2 全部数值 |
| `r5_compute_audit/decision_reports/A0_verdict.md` | A0 大白话裁决 |
| `r5_compute_audit/decision_reports/B1_verdict.md` | B1 大白话裁决 |
| `r5_compute_audit/decision_reports/C_verdict.md` | C 大白话裁决 |
| `r5_ca_01_baseline_profile.py` | A0 脚本 |
| `r5_ca_02_pixel_coreset.py` | B1 脚本 |
| `r5_ca_03_matrixfree.py` | C1+C2 脚本 |

---

*作者: ZCode agent · 2026-09-01 · 全部实验本地完成, 0 云算力消耗*
*GSIQ 定义未做任何修改 (CLAIM_REGISTRY v0.4 不变); 本 campaign 只验证"同一指标的低成本算法"*
