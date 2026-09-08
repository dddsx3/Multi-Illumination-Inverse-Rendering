# CI04-R 最终收口 memo · 2026-09-09（卡 CI04-R · Branch B）

## 裁决（脚本自动生成，branch_decision.json）

**Branch B — PASS-A 且 S0**（任务书预判的"最可能、最稳"分支）。

| 量 | 数字 |
|---|---|
| R2-A（within-cell mode ranking，第一承重墙） | **R_A = 0.90**，cluster 95% CI [0.90, 0.95]；**66/66 cells 正、11/11 objects 正**（≥9/11 达标）→ **PASS-A** |
| R2-B stratified（fixed-level severity，co-primary） | mode 0.536；best scalar = **E-min 0.536**（Δ = +0.00，cluster CI [−0.33, 0.00]）→ **S0** |
| pooled（descriptive 降级） | 0.728，object-cluster CI **[0.705, 0.754]**（替换旧 330-point CI [0.668, 0.783]） |
| R2-C（secondary） | top-1 hit：mode = E-min = 0.909（within-cell 设计下 ĵ_emin ≡ ĵ_mode 构造性一致） |

## 结构性发现（如实，不作为新主张）

1. **P_emin ≡ P_mode（逐位恒等）**：CI04 的 tracked modes 定义为同一 retention 谱的 bottom-5，
   E-min 的 min-eig 方向即 bottom tracked mode 的特征向量 → 两个 predictor 数值恒等。
   这是 CI04 设计的结构性事实（非新发现），意味着本轮 severity 对照的实际信息量在
   mode/E-min vs trace/logdet 之间（mode 0.536 vs logdet 0.418 vs trace 0.400）。
2. **T5.3 重算门 4.4e-12**：预测侧确定性重算与 frozen artifact 位级一致（ rng 流重放精确）。
3. **n=5 离散性**：within-cell Spearman 只取 0.1 网格值（已声明）；undefined cell = 0。

## 事故与修复（执行期，全部修复后重跑）

1. P3 随机测试两轮输入构造错误（行置零产生非 PSD S；参考路线正则化在精确零方向注入泄漏）
   ——改为 SVD 截断 PSD 构造 + G 正定断言 + 双口径容差（rel 1e-12×scale + abs 地板）。
2. strat_of scalar 分支收集错 predictor 值（所有 scalar 被算成 P_mode → Δ 退化 0.0/CI(0,0)）
   ——修正为 d["sp"][lv][which] 后 P_trace 0.400/P_logdet 0.418/P_emin 0.536 与手算一致。
3. 手稿补丁因换行不匹配三处未落——按实际现文锚点重打（全部落盘后 grep 终验）。

## 修复的既有错误（R0.2 允许范围）

- pooled 0.728 降级 descriptive + cluster CI 替换（T9.1）；
- log-log 斜率 1.82 旧解释撤销（T9.2，手稿现文无此句，正文slope句已入 limitations 口径）；
- λ⋆ "0.00000" accuracy 数字全部删除/降档 model-internal diagnostic（T10.3，grep=0）；
- CI05 taxonomy "comparable" 改 descriptive 大小对比句（T10.5：55.9–1131.4×，中位 273×，
  从 frozen ci05 重导出复现成功）；
- Prop 1 四条件 + "not a universal bound" 限定句（T10.1，手稿+theory note 双落）；
- 并联和 Anderson–Duffin 一般式入 theory note + 手稿，[S⁺+Λ⁻¹]⁻¹ 一般式清零（T10.2）。

## 红队 RT-F1–F12

**12/12 PASS**（artifacts/ci04r/rt_f12_verdict.json）——其中 RT-F3/F4 的首轮 FAIL 为
检查器字符串口径（表行标签 corruption magnitude vs 字面 P_corr），文件本身合规，修正检查器后通过。

## 状态字

```
THEORY:               FROZEN
EXPERIMENTAL PROGRAM: CLOSED
EVIDENCE:             FROZEN
CLAIMS:               FROZEN TO BRANCH B
MANUSCRIPT:           FINALIZATION
SUBMISSION BLOCKERS:  TCI LATEX/PAGE LIMIT + AUTHOR REVIEW（异机 smoke 已由 F3 覆盖）
```

**永久禁句**（grep 入 CI）：all scalar information criteria fail / scalar criteria are
structurally blind / mode-resolved is universally superior / real-world λ⋆ prediction validated /
agrees quantitatively after an affine scale。
