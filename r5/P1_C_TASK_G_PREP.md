# R5-P1-C · Task G Linux H100 Prep Note

> **Date**: 2026-09-01
> **状态**: 准备就绪，等 Linux H100 算力窗口
> **脚本**: `p1/source/information_audit/r4pp_local_vs_global.py`（已存在，无需修改）
> **依据**: 任务书 §21（P1-C）+ HANDOFF §3 / §4.1（本机 WinError 1455）

## 算力要求

| 项 | 要求 | 原因 |
|---|---|---|
| OS | Linux | 本机 WinError 1455 反复阻断 torch 加载（HANDOFF §3） |
| GPU | H100（或同等 A100 / RTX 4090+） | batched solver + dense eigh |
| commit quota | ≥32 GB | Python 进程峰值 ~14 GB（HANDOFF §3 实测） |
| 页面文件 | ≥4 GB | 同上 |
| CUDA / torch | 2.12.0.dev+cu128（同本机） | 与本机产数据兼容 |

## 脚本不变

- `p1/source/information_audit/r4pp_local_vs_global.py` 已是 R4″ 冻结版本，2026-08-30 commit
- 不需要任何代码改动
- 输出: `r4pp/07_local_vs_global_init.csv`（增量式，自动 skip 已 done 行）

## 任务量

```
SCENES = ["conf_cube_axis", "conf_prism8", "conf_cylinder_r03_d12",
          "conf_cone_r04_d12", "conf_egg", "conf_icosphere_sub3"]
NS = [3, 5]
N_SUBSETS = 10
init_mode ∈ {global, oracle_local}

Total: 6 scenes × 2 N × 10 subset × 2 init_mode = 240 runs
```

按 R4″ 实测 ~10 s/run @ base_iters=800 × single restart ≈ 40 min。

如果算力紧张可降到 base_iters=400 或 fewer scenes（`--limit N` 已有）。

## 输出解释（任务书 §R5-P1-C Case 1/2/3）

比较 β_global vs β_oracle_local（from linear fit: log E = α + β I + ε）：

- **Case 1**：两者都 β < 0 ⇒ 信息效应不仅在 global 难优化时成立，最佳
- **Case 2**：global β<0, oracle_local β≈0 或符号翻正 ⇒ claim 改成
  *"predicts practical optimization recoverability"*，**不再**claim
  *intrinsic identifiability→error*
- **Case 3**：两者都消失 ⇒ 停止强 claim

## 不动什么

- ❌ 不动 solver 接口（`joint_solve(theta0=...)` 已是 R4″ 末态）
- ❌ 不动 scene list（与 P4pp 6 line Gate 的 controlled geometry pilot 同源）
- ❌ 不动 PERTURB_A / PERTURB_C（R4″ 已冻结 5% RMS / 5% norm）
- ❌ 不动 csv schema（incremental resume 依赖其字段稳定）

## 与 P1-A / P1-B 的并行性

- Task G 与 P1-A / P1-B **输出文件不同**（07_*.csv vs r5/r5_p1_*.csv）
- Task G 与 P1-A / P1-B **scene set 有交集**（cube_axis / prism8 / egg 三个都在两边），
  但不是必须同时跑；可以串行也可以并行
- 建议算力分配（如果同时拿到 H100）：
  - H100 #1: Task G (240 runs, ~40 min)
  - H100 #2: P1-A full + P1-B (~1.5–2 h GPU)
  - 总计 ~2.5 h
- 如果只有一块 GPU：先 Task G（更短 + 回答 Q2），再 P1-A/B（回答 Q1/Q3）

## 不变量（per CLAIM_REGISTRY v0.4 字面禁词）

- ❌ 仍不写"joint recoverability"
- ❌ 仍不写"M1 uniquely stable"
- ❌ Task G 输出 csv 字段 `init_mode ∈ {global, oracle_local}`，**不**称
  "intrinsic vs. optimization"；称 "global" 与 "oracle_local" 是中性 wording
- P1-C 的 wording 在 Case 1/2/3 三档下分别冻结（任务书 §R5-P1-C Case 1/2/3）

---

## 启动 checklist（H100 到位后）

```
[ ] git clone / pull to H100
[ ] pip install -r requirements.txt (torch 2.12.0.dev+cu128)
[ ] ulimit -v 33554432  # 32 GB commit quota
[ ] sysctl vm.overcommit_memory=1
[ ] cd p1/source/information_audit
[ ] python r4pp_local_vs_global.py --limit 6  # 默认就跑全部 6 scenes × 240 runs
[ ] 监控 r4pp/07_local_vs_global_init.csv 增量
[ ] 完成后输出到 r5/r5_taskG_local_global.csv（手动 copy + 加 R5-B′ 标注）
[ ] 用本文 §"输出解释" 写 P1-C 报告
```