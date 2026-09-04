# T1 · FW（Fisher-Weighted）融合权重设计文档（A1 设计冻结包 · 条例 T1-1/T1-2）

> 状态：**设计初稿（DESIGN DRAFT，未实现未冻结）** · 归属：任务书 T v2.0 条例 T1-1/T1-2 + v2.2 §4-1（审计输入：保留 FiLM（R-M）；albedo 压缩若 EX-04 未恢复 → 值域压缩机制调查进设计考量）。
> 日期：2026-09-04 · 符号基准：IDENTIFIABILITY_v3 §2–§3（F_ll,k、B_k、F_eff 逐符号对齐）。
> 里程碑：EX-06 评估前交付设计冻结包（v2.2 §4-1）；实现（T1-4/T1-5）与评审（T1-6）在 A3-3 启动前完成。

---

## 1. 数学形式推导（条例 T1-1）

### 1.1 从逐光 Fisher 到逐光信息量

**基准符号（v3 §2，逐符号一致）**：

- `z_kp = Y_pᵀ c_k`，`s_kp = ReLU(z_kp)`，`h_kp = 1[z_kp>0]`，`Y_p = SH_basis(n_p) ∈ R^9`（SH-2 冻结口径）；
- 逐光 Fisher（Gauss-Newton 块）：`F_ll,k = Σ_p a_p² h_kp Y_p Y_pᵀ ∈ R^{9×9}`（v3 §2 式 1）；
- 交叉块：`B_k[p,:] = a_p · s_kp · h_kp · Y_pᵀ`（v3 §2 式 2）；
- 消去逐光系数后的有效 Fisher：`F_eff = F_ss − Σ_k B_k F_ll,k† B_kᵀ`（v3 §3 式 1），其中 `F_ss = diag_p(Σ_k s_kp²)`。

**逐光信息量定义（由上式直接得到）**：光 k 对 albedo 有效信息量的贡献是它从 F_eff 中"消除"的交叉耦合：

```
I_k := tr( B_k F_ll,k† B_kᵀ )          [标量，≥0]
     = Σ_p a_p² s_kp² h_kp · Y_pᵀ F_ll,k† Y_p    （展开式，用于逐像素实现）
```

由 F_eff 的定义（v3 §3 式 1）直接得到：`tr(F_eff) = tr(F_ss) − Σ_k I_k`，即 **I_k 就是光 k 经 Schur 消去贡献的信息量份额**。此定义自动满足：
- 可辨识性对齐：F_ll,k 病态（近零谱 → 机制 (a)）时 F_ll,k† 截断 → I_k 自动变小（弱光少投票）；
- gauge 不变：I_k 不依赖全局尺度（a→λa, c→c/λ 下 B_k F† Bᵀ 的迹按 λ² 缩放，归一化后消去）。

**权重公式（明确、可微）**：

```
w_k = softmax_k( β · log Ĩ_k ),    Ĩ_k = I_k / (Σ_j I_j + ε)
```

- β 为可学习标量（初始 1.0）；log-域归一化避免 I_k 量级随场景漂移；
- softmax 保证 Σw_k=1 且处处可微；ε=1e-8；
- **边界（分母为零）**：若 Σ_j I_j < ε（全暗/全 inactive），w_k → 均匀权重 1/N（T1-5 数值测试覆盖此分支）。

### 1.2 误差-下界理论斜率 E_min(N)（写清坐标）

在 homoscedastic 高斯噪声（σ² 已知）下，albedo 估计的 CRB（v3 §1.1 假设内）：

```
E_min(N) = (σ² / P) · tr( F_eff^proj† )     [每像素平均 albedo MSE 下界]
```

其中 F_eff^proj = Π_g F_eff Π_g（gauge 投影，gauge_fisher_v2 主策略）。**坐标声明：斜率取 E_min 对 N 的线性坐标相对斜率**（log-log 会把近饱和段压扁，G1 判据需感知幅度）：

```
slope := (E_min(N=5) − E_min(N=1)) / E_min(N=1)   （负值，G1 判据"≥60% 理论预测"即 |slope|）
```

理论预测（数值预演见 T1_slope_preview.md）：N=1→5 降 **78–84%**（多场景中位）。FW 权重的理论作用 = 在同样 N 下选高 I_k 子集 → E_min 进一步下降；预测机制在 §3 验收判据草案量化。

**每步溯源**：I_k 由 v3 §2 式 1/2 与 §3 式 1 的定义得到（F_eff 定义本身）；E_min 由 CRB 定义（F_eff† 的迹）得到；坐标选择与斜率定义为本设计文档 §1.2 明示。

## 2. 权重注入实现设计（条例 T1-2）

### 2.1 现有结构（读 fusion_unet.py 后的挂载点）

SetTransformerLite 聚合链：逐光 token → 注意力池化 → 聚合特征 → FiLM(γ,β) 调制主干。**最小改造**：在"逐光 token → 聚合"之间插入逐光标量权重 w_k 乘到 token 上（或乘到池化贡献上），一处注入，不动主干。

### 2.2 改造方案（文件级）

| 改动点 | 文件:位置 | 内容 | 为什么 |
|---|---|---|---|
| 1 | `fusion_unet.py` · SetTransformerLite 前向 | 新增 `PerLightFisherWeight` nn.Module：输入逐光特征图统计量（见 2.3），输出 w_k∈R^N（softmax） | 权重生成器独立成模块，可单测 |
| 2 | `fusion_unet.py` · 聚合处 | token 加权：`token_k ← w_k · token_k`（池化前） | 单点注入，diff < 60 行（R5 精神：能小改不大改） |
| 3 | `config.py` | 新旗标 `--fw_weight on/off`（默认 off，A3-3 开启） | 一臂一变量：FW 臂唯一差异变量 |
| 4 | `tests/test_permutation_invariance.py` | 追加权重分支用例（T1-4） | 置换不变性是红线 |

### 2.3 代理特征（网络内如何近似 I_k）

训练时无法拿到解析 F_ll,k（c_k 未知）。**代理统计量**（逐光自身、可从特征图读出）：
`u_k = [mean(s_k), var(s_k), active_frac_k, ||grad_k||]`——光照头输出 s_k 与 active 占比是 I_k 的一阶代理（s_kp²·h_kp 正是 I_k 展开式的主项）。PerLightFisherWeight 是 2 层 MLP：u_k → w_k。**训练初期 c_k 未学好 → w_k 接近均匀（MLP 偏置初始化为零向量 → softmax 输出 1/N），不破坏 A3-0 已有收敛路径**。

**置换兼容性论证（T1-2 验收点）**：w_k 只依赖 u_k（光 k 自身的统计量），不依赖跨光排序/其它光的索引——对输入光照序列的任意置换 π，有 w_π(k) = w_k 且 token 同步置换，加权池化结果不变。这就是"等变权重 + 不变聚合"结构，置换测试（T1-4）将数值验证 max_diff<1e-5。

### 2.4 与 FiLM 的关系（审计输入 R-M）

FW 不替换 FiLM：FiLM 是"聚合特征 → 主干调制"的条件生成路径（A3-1 已证重建头依赖它）；FW 是"聚合前的逐光证据加权"。两者正交，A3-3 主臂 = A3-0 + FW 单变量（保留 FiLM）。

### 2.5 albedo 值域压缩的接口预留（审计输入 ②）

若 EX-04（lowSmooth）未恢复 albedo range（0.252 → >0.30），FW 设计需自查：权重 w_k 集中到少数强光 → 弱光像素的 albedo 证据被降权 → 值域进一步压缩。**预留检查**：A3-3 判据包 4 之外，加观测指标 phys_albedo_range（与 EX-04 同口径对照）。若 FW 臂 albedo range 显著低于 A3-0（>10% 相对差），回本设计 §1.1 检查 I_k 对 a_p² 的依赖是否需要除以像素数归一（消除大 albedo 区域的过度投票）。

### 2.6 评估协议冻结

沿用 evaluate_model.py 13+2 指标 + --num_lights 子集协议（T1-2 步骤 ④ 原文），不加新指标；判据包 6 项以 v2.0 A3-3 原文为准（判据 1 斜率 ≥60% 理论预测为 G1 守门员前身）。

## 3. 验收判据草案（供 T1-6 评审）

1. **置换不变**：权重分支 on/off 两种模式 max_diff < 1e-5（T1-4）；
2. **数值边界**：全暗子集 → 均匀权重，无 NaN（T1-5）；
3. **理论斜率可实现**：E_min(N) N=1→5 单调降 >5%（预演已证 78–84%，T1-3 ✓）；
4. **A3-3 门禁**（判据包 6 项摘引）：normal MAE ≤ A3-0±2.0°、albedo si-MAE ≤0.065、phys 0%、判据 1 斜率 ≥60% 理论预测、PSNR 不劣于 A3-0−2dB、N 曲线行为不劣化；
5. **albedo 值域检查**（§2.5 预留）。

## 4. 状态与下一步

- T1-1 ✓（本文档 §1）· T1-2 ✓（§2）· T1-3 ✓（`T1_slope_preview.md`）；
- T1-4/T1-5（实现+测试）：**不在本批执行**——v2.2 §4-1 里程碑是"设计冻结包"，实现排 EX-06 后、EX-07 前（避免与 EX-04~06 的 GPU 窗口抢工）；
- T1-6（评审）：设计文档发导师，§3 判据被接受/修订后冻结。

*v0.1 草稿 · 2026-09-04 · 待 T1-6 评审冻结；符号对齐 IDENTIFIABILITY_v3 §2–§3；实现 diff 计划见 §2.2 表。*
