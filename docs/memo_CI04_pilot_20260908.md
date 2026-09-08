# CI04 pilot · 1 页 memo（2026-09-08 · 卡 C15）

**判定：pilot 协议链跑通，Gate D 主判据（秩相关）预注册口径下 spearman=0.676
[CI 0.590–0.757] > 0.5 = pass；但量级层有 R-A/R-D 已知风险的实证形态（下文§3），
protocol freeze 按"秩相关 + 全局仿射尺度"主张冻结，量级差如实入 sensitivity 带。**
pilot=`artifacts/frozen/ci04_pilot_summary.json`（dev 8 对象 × 6 强度档 × 8 seeds）。

## 1. 协议（估计器侧 corruption，预注册冻结）

- 真值 = GT 灯位（OpenIllumination light_pos.npy，校准 GT 齐备）；数据固定；
  corruption 注入**估计器的假设校准** δc~N(0,Σ_c)（物理单位 φ 空间：log²I、rad²），
  即理论量化的校准不确定度本体；
- nominal（corruption 前，GT 计算）：标定朗伯 PS（GT 灯已知，逐像素交替 LSQ）→
  (ρ̂,n̂)；线性化 A_k=D(ŝ_k)、B_k=ρ̂h·[∂/∂logI,∂/∂θ,∂/∂ψ]；白化（异方差 a+bI）→
  逐灯 Λ'=Σ_φ⁻¹ → Σ_k delta_f（V1b 块对角恒等式）→ ΔF → 弱 5 模式退化比
  deg_pred = λ_j(F∞)/λ_j(ΔF)；
- empirical：注入 → 固定 n̂ 的白化逐像素 GLS 重估 ρ̃ → 尺度 gauge 对齐（ρ·gain
  双解的已知自由度）→ 投影弱模式 → 相对**对照臂**（残差 bootstrap 重抽，零
  corruption）方差退化比 deg_emp。

## 2. pilot 数字

| level（σ_logI=σ_deg） | 0.10 | 0.20 | 0.35 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|
| 组内 spearman | +0.80 | +0.63 | +0.74 | +0.82 | +0.85 | +0.69 |
| pred p50 | 1.52 | 1.60 | 1.66 | 1.75 | 1.95 | 2.16 |

全池 spearman **0.676**（bootstrap95 [0.590, 0.757]，500 重抽）> 预注册 0.5 ✅。

## 3. 实证形态与风险归属（pilot 的本职——发现，不硬解释）

1. **量级差 ~10²–10³**：0.1° 旋转档 corrupted 模式方差已 100–200×对照臂，而线性
   预测只 1.5–34×。分解：纯方向 0.1° 伤害（std 9.2e-5）≪ 纯强度伤害（std 5.8e-3，
   mean/std≈0.81 即主要被尺度 gauge 吸收——对齐后残伤害仍 ~6e-3）。
   真实残差 |I−ŝρ̂|≈0.024（非朗伯/阴影失配）与方向/强度扰动**耦合放大**：
   R-D（模型失配主导）+ R-A（Σ_c_real 叠加）风险的实证形态。
   → 处置：主 claim 冻结为**秩相关 + 一个全局仿射尺度**（宪法 CI04 协议原文允许）；
   量级差作为 Σ_c_eff 敏感性带如实报告，不宣称绝对量级吻合。
2. **强度维非线性**：首轮 level=4（log-gain 跨度 e^8≈3×10³ 倍）远超线性化域，
   spearman 崩到 0.23——C13 包络纪律（<10% 域边界）的直接应用：档位重设
   0.1–1.0（gain 跨度 ≤e²），回收后 spearman 0.23→0.68。
3. **对照臂构造**：δc=0 的 GLS 是解析恒等解（方差 0 → 除法爆炸，首轮 emp=1e29 假
   读数）；对照臂 = 残差 bootstrap 重抽（保留失配/噪声量级，破坏空间相关的副作用
   已记录）。

## 4. Protocol freeze（交 C16 正式 run 的冻结清单）

- 判据：spearman(pred_deg, emp_deg) > 0.5 且 bootstrap95 下界 > 0（不变）；
  主张措辞按 CLAIMS_REGISTRY C6（秩/尺度相关；禁只凭 MAE 单调宣称成功）；
- levels=[0.1, 0.2, 0.35, 0.5, 0.75, 1.0]（C13 线性化域内），seeds ≥20，
  test 对象清单 C16 时冻结（当前未下载、未选）；
- 读数：弱 5 模式退化比 + 尺度 gauge 对齐 + 残差 bootstrap 对照臂；
- 已知限制（进论文 limitations）：绝对量级失配 10²（Σ_c_real + 模型失配），
  以 sensitivity 带呈现。
