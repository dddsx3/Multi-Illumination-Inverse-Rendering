# F2 · 手稿禁词/锁死句审计报告（只检不改，2026-09-09）

> 性质：**检测报告**。逐句替换（Gate D 句、λ⋆ 两段式、misspec 双读数、novelty 三段式）
> 属写作会话（F4 手稿正文同批执行）；本报告给出精确到行的替换清单。
> 权威措辞来源 = `CLAIMS_REGISTRY.yaml::frozen_sentences_20260908`。

## 1. 禁词 grep 结果（manuscript_v0.8.md，0 命中 ✓）

| 禁词 | 命中 |
|---|---|
| "agrees quantitatively after an affine scale" | 0 |
| "real-world" | 0 |
| "robust to" | 0 |
| "has not been studied / remains unexplored / first to" | 0 |
| "1.82" | 0 |

豁免清单：`docs/任务布置_投稿闭口阶段_20260908_卡F1–F6.md` 与 `docs/论文状态盘点_20260908_投稿闭口阶段.md`
出现禁词均为**引用规则原文**（治理文档非论文正文）。

## 2. 写作会话替换清单（精确到行，按 frozen_sentences 逐字执行）

| 行号 | 现文（v0.8） | 动作 |
|---|---|---|
| L24（摘要） | "Spearman 0.728, 95% CI [0.668, 0.783], 11 held-out objects" | 保留数字；后续若引用退化结论须用 Gate D 锁死句式 |
| L165–168（§VIII） | "**Result (Gate D)**: Spearman 0.728, ... the claim is rank-and-scale, reported with its sensitivity band" | **替换为专家 Gate D 原句**（registry::gate_D）；log-log 斜率 1.82 当前未出现于稿中（好），写作时只入 Limitations |
| §IX L175 | "magnitude-level real-data agreement is bounded by Σ_c_real and model mismatch" | 保留；补 λ⋆ 两段式（合成 quantitative + real-scene diagnostic remains finite/meaningful） |
| Abstract L24 | "Spearman 0.728" 句 | 升 v0.9 时按 Gate D 锁死句重写从句 |
| 全文 | "robust to Λ misspecification" 泛化句 | 0 命中 ✓（无需删除） |
| 新增 | — | Novelty 段落按 F1 matrix §2 差异句 + 三段式（Intro 第一页） |

## 3. 结论

- 禁词现状干净（0 命中），registry 已含全部锁死句；
- 写作会话（F4 + 手稿 v0.9）按 §2 清单执行四处替换与 Novelty 段落；
- 版本升 v0.9 时 CHANGELOG 记"措辞冻结轮"。
