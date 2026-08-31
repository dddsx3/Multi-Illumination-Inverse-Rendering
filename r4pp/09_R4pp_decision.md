# 09 · R4″ Final Decision · 2026-08-31

> **裁决：PIVOT (B′)**
> **依据**：R4″ 任务书 §28 三分支裁决 + dashboard `r4pp/08_go_no_go_dashboard.md`（6 行）
> **基线 commit**：`2af42c0`（Task F 完成点）

## 1. 6 行 Gate 终裁

| Gate | 指标 | 结果 | 关键证据 |
|---|---|---|---|
| Instrument | M1 log-pdet 5/5 stability | **PASS** | MA ρ=0.999, MB=0.993, MC=0.996, MF=0.998 |
| Signal | low-N signal/noise | **PASS** | R_signal 全部 24 cell > 2（median 27.2, min 9.0）|
| Direction | info→error β<0 | **PASS** | β median −0.348, 81% 负号 |
| Interaction | G↑ ⇒ \|β_G\|↑ | **FAIL** | A N=3 ρ=+0.29, A N=5 ρ=+0.72（**反向**）|
| Saturation | N=8 noise-floor | **PASS** | N=8 R_signal 22.6 vs N=3 43.0, σ_subset/err 3.2% |
| Externality | local-init replication | **PENDING** | Task G 因本机 WinError 1455（页面文件太小）无法执行 |

**4/6 PASS**。

## 2. 裁决映射

按任务书 §44：

```
Instrument PASS + Signal PASS + Direction PASS + Interaction FAIL  →  PIVOT (B′)
```

不进入 GO (A2)：Gate 4 明确失败——family A 内**未观察到 G↑⇒|β_G|↑ 的连续趋势**（N=3 ρ=+0.29、N=5 ρ=+0.72 方向相反）。

不进入 KILL H-COND：Instrument（信息度量本身稳健）+ Signal（信息效应远高于噪声）+ Direction（β 中位数显著为负）三条硬门槛全部通过。

Externality 标记 PENDING 而非 FAIL：因本机硬约束（commit 配额 + WinError 1455）无法执行 Task G 进程；任务书 §28 规定"Externality 失败不直接 KILL，但警示 A2 结论的稳健性"——同理未执行也不直接升级裁决。

## 3. PIVOT (B′) 后的新科学问题

按任务书 §28 任务书原文：

> **B′ 主线**：
> *Beyond Cardinality: Effective Information at Fixed Illumination Budget*
> 说明 illumination information 本身有效，但 geometry interaction 没有稳定机制证据。

这等价于**放弃"大 unified regime theory"**（illumination × geometry 联合模型），回到**在固定 illumination budget 下的有效信息度量**。具体可辩护的新题：

1. **在所有 G 档上 information 主效应都成立**（Direction PASS）—— 这是 H-COND
   的**弱化版**（弱化条件：去掉"随 G 系统增强"的部分）
2. **N=8 是真 saturation**（Saturation PASS）—— 这是新发现
3. **M1 (log pdet) 是 robust 信息度量**（Instrument PASS）—— 工具已就绪

## 4. 与 R4′ 旧 R4prime_frozen 数据的差别

| 维度 | R4′（冻结） | R4″（本次）|
|---|---|---|
| Primary metric | `λ_min⁺`（退化）| `M1 log pdet`（5/5 PASS）|
| 收敛判据 | P75×P75（内生筛选）| `finite ∧ abs_step200<1e-6 ∧ loss<3e-4`（无内生）|
| Solver | 写死 seed, 无 trace | 显式 seed/theta0/return_trace + proj_grad_norm |
| Noise floor | 未测 | 480 run 全量标定 |
| Geometry gating | 未测 | 8 scene controlled pilot |
| **R_signal** | — | **> 2 全部 24 cell** |
| **H-COND 状态** | 不可判 | **Direction PASS, Interaction FAIL** |

## 5. Task G 限制声明

Task G（local-vs-global）**因本机环境硬约束无法执行**：
- 现象：`OSError: [WinError 1455] 页面文件太小，无法完成操作` 发生在 torch 加载
  `cufft64_11.dll` 阶段（commit 配额 ~94% 已用）
- 后果：Task G 进程在 import 阶段崩溃，0 trial 产出
- 不影响裁决：任务书 §28 规定 Externality 失败不直接 KILL
- 改进路径：换算力环境（≥32GB commit + ≥4GB 虚拟内存）即可重跑 Task G
- 预算：240 runs × 1.5h ≈ 6h GPU

## 6. 提交产物

| commit | 内容 |
|---|---|
| `b6ea30f` | H0.1 冻结归档 + failure audit |
| `73b3e53` | H1.0 C-1 solver 改动 + 22/22 单测 |
| `3b17c81` | H1.7 master table + P0-3 修正 |
| `6a8be9c` | H2.6 收敛判据冻结 + 05 报告 |
| `c6f5cff` | H3.4 Task C noise floor 480/480 |
| `f79972c` | Task D bake-off + M1 冻结 |
| `1aa45f3` | Task F mesh/渲染/INC-001 修复 |
| `2af42c0` | Task F solve 400/400 + β_G 分析 |

## 7. 后续 30 天行动建议

1. **优先**（基于 PIVOT B′ 方向）：
   - 写 *Beyond Cardinality: Effective Information at Fixed Budget* 论文草稿
   - 核心贡献 = M1 信息度量（1/d logdet）+ N=8 saturation 发现 + 噪声地板方法
   - Venue：CVPR/ICCV/NeurIPS（具体见论文发表顾问输入）
2. **如换算力**：重跑 Task G 验证 Externality；3 个 B 族场景（bevel30/rounded +
   prism4）补齐；可能改变 PIVOT/GO 方向（但 +6% 概率）
3. **如不换算力**：以 PIVOT 方向投低风险会议；不再追加 synthetic scenes 救 H-COND

## 8. 签发

本决策已冻结；所有相关材料在 `r4pp/` 与 `archive/R4prime_frozen/`。

签发：R4″ sprint · 2026-08-31 · 基线 `2af42c0`
