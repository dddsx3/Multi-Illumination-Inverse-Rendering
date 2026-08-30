# EXPERIMENT_CONTRACT · 论文实验契约（结果出来前冻结，防"按结果改指标"）

> 版本 v0.1 · 2026-08-30 · 依据：P1 任务书 + 专家 R6 建议。
> 任何偏离本契约的实验必须在论文 Limitations 或 rebuttal 中声明。

## E1 · N curve
- 数据：synthetic_v4 P-domain（140/30/30），test 30 场景；
- N ∈ {1,2,3,5,8,15,24}（前 N 光固定顺序子集）；
- 指标：SI-MAE(A)、normal 角误差（mesh GT 口径）、per-light SH 误差、
  held-out HO-PSNR；
- 判读：主要因子随 N 单调改善 + 饱和点。

## E2 · same-N different-conditioning（**定核第一图**）
- 固定 N ∈ {3,5,8,12}，每场景随机抽 ≥100 子集；
- 每子集：GA-ISI 分数（λ⁺_min / logdet⁺ / A-opt）+ solver 误差
  （受控：多 restart、收敛 flag、只比较收敛 trials）；
- 统计：固定 N 内 Spearman 相关 + 跨场景 pooled + 置信区间（bootstrap）；
- PASS：符号稳定且 p<0.01。

## E3 · matched-conditioning different-N（**定核第二图**）
- 取不同 N 但 GA-ISI 分数相近的子集对（分位数匹配）；
- 判读：恢复误差相近 → "N curve 是 conditioning curve 的投影"。

## E4 · novel vs duplicate（cardinality-control）
- S₃ + 新光 vs S₃ + 重复光；Δ_new > Δ_dup 需 bootstrap 95% CI 支持；
- 另做 diversity-control：S₃+互补光 vs S₃+冗余光。

## E5 · conditioning-aware subset selection
- 用 GA-ISI 贪心选 N 光 vs 随机 N 光 vs 均匀几何采样；
- 判读：GA-ISI 选择 ≥ 随机（误差或 held-out）。

## E6 · cross-subset consistency
- 同场景两个 N-光子集 → (A,n) 输出差 D_A/D_n 随 N 下降。

## E7 · held-out relighting（oracle-query-light）
- support 光估 (A,n)，query 光 L_q^GT 由评估器提供；
- residual 全关；HO-PSNR/HO-SSIM/HO-MAE；predicted-query-light 只许单列。

## E8 · variable-N vs fixed-N probe
- Probe A/B/C 各训 varN(N~U{3..15}) 与 fixed5 两个版本（同预算）；
- 判读：varN 在 N≠5 上泛化优于 fixed5。

## E9 · P-domain → R-domain robustness
- P 域训练的模型零样本测 R 域（Cycles 全效果）；
- 报告 degradation 分解（阴影/间接/材质）。

## 新增 C0 Gate（专家建议，优先于 C1-C5）

**C0 · Conditioning Predictivity Gate**：神经模型的恢复误差在固定 N 内
随 GA-ISI 改善、且回归中 GA-ISI 解释力显著超出 N。
没有 C0，C1 的 N curve 只是性能曲线；有了 C0 才是论文发现。

## 统一纪律

- 一切误差指标在 mesh normal / 线性域口径下计算（P1-06/07）；
- solver 类对比只使用收敛 trials（记录 restarts/iters/grad-norm/gap）；
- 全部表格同时给 N 与 GA-ISI 两行上下文，防止选择性报告。
