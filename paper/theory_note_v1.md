# Theory Note v1 · 冻结理论骨架手稿版（卡 C08 · 2026-09-07）

> 依据：宪法 §3（每条命题的严格证明计划）+ 三轮红队 R1–R4 修订入版 + CI01 正式 run
> （Gate A 通过，`artifacts/frozen/ci01_formal_summary.json`）。
> 用途：TCI 论文 Section IV–VI 的手稿底稿 + supplement 证明骨架。
> 措辞纪律：全部命题用 recall / instantiate / verify 定位，**无 "novel theorem" 措辞**；
> λ⋆ 全文只在绝对信息单位语境出现（R2 修订）；每条命题附四元组（假设→结论→新颖性权重→对应实验）。

---

## 0. 记号与假设（一次性定义，正文不再重复）

**模型（白化形式）**：局部线性化观测

> y = A x + B δc + ε,  ε ~ N(0, σ²I),  δc ~ N(0, Σ_c)，

其中 x ∈ Rⁿ 为场景参数（确定性），δc ∈ R^q 为校准 nuisance（随机），
A ∈ R^{m×n}、B ∈ R^{m×q}。若原始噪声协方差为 Σ_y，先以 W = Σ_y^{-1/2} 白化
A、B、残差（`whiten_system`），后续公式一律在白化空间陈述。定义

> Λ = σ² Σ_c⁻¹,  M(Λ) = I − B(BᵀB+Λ)⁻¹Bᵀ,  ΔF(Λ) = Aᵀ M(Λ) A,  F∞ = AᵀA。

物理解释：continuum 横轴不是"裸先验精度"，而是**校准信息相对图像噪声水平** Λ；
参数化用 δc = Σ₀^{1/2} z、z~N(0, τ⁻¹I)，τ 为主横轴，Λ 仅作计算记号（宪法 §2.1）。

**假设清单**（每条首次出现处标注）：
- (H1) Λ ≻ 0（可逆域；Λ=0 用 Moore–Penrose 极限单独处理，见 Lemma 1 remark 2）；
- (H2) Σ_c 物理参数化：δc = Jφ·φ、φ~N(0,Σ_φ)（Σ_c = JφΣ_φJφᵀ，可秩亏——
  此时一律在 φ 有效空间工作，c 空间禁 inv(Σ_c)，CI01 已实装）；
- (H3) 白化域：Σ_y ≻ 0（对角或稠密 SPD；非正定噪声模型先修再白化）；
- (H4) F∞ 在 range(F∞) 上正定（可识别子空间；秩亏时限制到 range，见 Lemma 2）。

**两套读出严格分离**（宪法 §2.2）：

| 度量 | 定义 | 用途 | 限制 |
|---|---|---|---|
| 绝对信息谱 | eig(ΔF(τ)) 或指定方向 Rayleigh | gauge lifting、μ_floor、λ⋆ | 可跨 τ 比同一物理参数化；跨场景仅单位一致时谨慎比较 |
| 归一化 retention | R(τ)=F∞^{-1/2}ΔF(τ)F∞^{-1/2}, ρⱼ∈[0,1] | within-scene 保留读出 | 禁用于 λ⋆；跨场景只比统计分布 |

---

## 1. Lemma 1 · Schur/HCRB calibration continuum 与单调性

**陈述**。在 (H1)(H3) 下，ΔF(Λ) = Aᵀ[I − B(BᵀB+Λ)⁻¹Bᵀ]A 是混合信息矩阵
（确定性 x + Gaussian nuisance δc）的 Schur 补；若 0 ≼ Λ₁ ≼ Λ₂（Loewner），则

> ΔF(Λ₁) ≼ ΔF(Λ₂) ≼ F∞ = AᵀA。

Λ→0 由 Moore–Penrose 极限定义（M(0) = I − UUᵀ，thin-SVD B = USVᵀ）；
Λ→∞ 得 F∞。

**证明步骤**（supplement 完整版）：
1. 联合 Hessian/信息块矩阵 H = [[AᵀA, AᵀB],[BᵀA, BᵀB+Λ]]；
2. 对 nuisance 块取 Schur 补：H 对 x 的有效信息 = AᵀA − AᵀB(BᵀB+Λ)⁻¹BᵀA = ΔF(Λ)；
3. 由 Q₁ ≼ Q₂ ⇒ Q₂⁻¹ ≼ Q₁⁻¹（矩阵逆的 Loewner 反序）推出 M(Λ) 与 ΔF(Λ) 单调；
4. Λ=0 秩亏：thin-SVD 展开，M(0) = I − UUᵀ；**禁止**把 full-SVD 零奇异方向写成 0/0
   （红队 N3-fix erratum；零方向由 (I−UUᵀ) 承载，RL-thin-svd 红线）；
5. 一般 W 情形：先白化，或直接写 AᵀWA − AᵀWB(BᵀWB+Λ)⁻¹BᵀWA。

**引用定位**（recall/instantiate）：ΔF 即 hybrid Cramér–Rao 界框架中确定性参数的
有效信息（Noam & Messer 2008，广义高斯线性估计中 HCRB 与经典 CRLB 精确重合）；
Schur 补单调性为教科书结果。本文不宣称新信息几何（CLAIMS_REGISTRY C1 禁令）。

**四元组**：假设 (H1)(H3) → 结论（连续谱 + Loewner 单调 + 两端点）→
新颖性权重：成熟理论实例化 → 对应实验 CI01（双路线/尺度/谱界 100/100 绿）。

---

## 2. Proposition 1 · affine-Gaussian 有限样本协方差恒等式

**陈述**。在 (H1)(H3) 且 ΔF(Λ) 可逆下，profiled joint MAP 估计器

> x̂ = ΔF(Λ)⁻¹ Aᵀ M(Λ) y

在联合系综 δc~N(0,Σ_c)、ε~N(0,σ²I)（**joint sampling**）下满足

> E[x̂] = x,  Cov(x̂) = σ² ΔF(Λ)⁻¹ ——有限样本精确恒等式（非渐近）。

**Corollary（fixed-δc 诊断，R4 修订）**。固定 δc = δ̄ 时：
- bias(δ̄) = ΔF⁻¹AᵀMBδ̄（可闭式预测）；
- 条件协方差 = σ²ΔF⁻¹AᵀM²AΔF⁻¹ ≠ σ²ΔF⁻¹（M 在 Λ>0 是收缩映射、非幂等投影），
  且**系统性低于 marginal 方差**——fixed-δc 系综会低估不确定性，恰是危险方向。
  fixed-δc 只作诊断，**不得**用于 marginal CRB tightness 声明（宪法 §0 禁令）。

**证明步骤**：
1. MAP 目标 ‖y−Ax−Bc‖² + cᵀΛc 对 c 消元 → profiled 正规方程 ΔF x̂ = AᵀMy；
2. 边缘化 δc：y|x ~ N(Ax, σ²I + BΣ_cBᵀ)；
3. Woodbury：V⁻¹ = σ⁻²M（V = σ²I+BΣ_cBᵀ），故 x̂ 即 GLS；
4. GLS 协方差 (AᵀV⁻¹A)⁻¹ = σ²ΔF⁻¹，得有限样本恒等式；
5. Corollary 直接代入条件分布推导。

**引用定位**（recall/verify in our setting）：该恒等式是线性混合模型固定效应
GLS 协方差的特例——Henderson 混合模型方程组（MME）与 Harville (1977) 综述给出
同一结构（y = Xβ + Zu + e, V = ZGZᵀ + R）；亦与 HCRB 联合 ML/MAP 达到性一致
（Noam & Messer 2008）。计量学侧：GUM/JCGM 100 不确定度传播与 GUM Supplement 1
的 Monte Carlo 传播是同一问题的成熟谱系（Related Work 锚点，R1）。

**四元组**：假设 (H1)(H3) + ΔF 可逆 → 结论（E/Cov 恒等式 + fixed-δc 两条闭式）→
新颖性权重：精确性/一致性资产（不占新颖性）→ 对应实验 CI03（Gate B：joint ensemble MC，
V1 回归：median 比 1.0045；条件方差公式比 1.0015）。

---

## 3. Proposition 2 · gauge-lifting 精确谱响应

**陈述**。设 gauge 恒等式 **Aa = Bc̄**（a 为参数空间的 gauge 方向，c̄ 为对应
nuisance 方向），白化各向同性先验 Λ = λI，B = USVᵀ（thin-SVD），α = Vᵀc̄。则

> aᵀΔF(λ)a = Σᵢ αᵢ² sᵢ² λ/(sᵢ² + λ)，

其中 sᵢ 为 B 的正奇异值。两端：λ→0 线性抬升 aᵀΔF a = λΣᵢ(αᵢ², sᵢ>0) + O(λ²)；
λ→∞ 饱和到 Σαᵢ²sᵢ² = ‖Aa‖² = ‖Bc̄‖²。

**证明步骤**：
1. ΔFa = AᵀB(BᵀB+λI)⁻¹λc̄；
2. 左乘 aᵀ 并用 Aa = Bc̄ 化为 c̄ᵀBᵀB(BᵀB+λI)⁻¹λc̄；
3. SVD 对角化 → 逐奇异方向闭式；
4. Taylor 展开两端（sᵢ>0 的方向贡献线性项；零奇异值方向不进和式——thin-SVD 形式）；
5. 各向异性 Λ 仅补充矩阵式 AᵀB(BᵀB+Λ)⁻¹Λc̄，不强求一维闭式。

**引用定位**：称 Proposition（render-specific 解析资产），不称新定理；
新颖性来自逆渲染实例化 + 光度结构解释 + 验证（红队 V2：25 个数量级 λ 网格
max rel err ≤3.9e-8；线性系数与 Σαᵢ² 逐位一致；饱和 0.999948）。

**四元组**：假设 gauge 恒等式 Aa=Bc̄ + (H1)(H3) + Λ=λI →
结论（精确谱响应 + 两端展开）→ 新颖性权重：核心解析资产 →
对应实验 CI02（闭式 vs direct Rayleigh <1e-8 gate；Fig.3）。

---

## 4. Lemma 2 · calibration-retention spectral bound

**陈述**。在 (H4) 的可识别子空间 range(F∞) 上，

> R(Λ) = F∞^{-1/2} ΔF(Λ) F∞^{-1/2} 满足 0 ≼ R ≼ I，故 0 ≤ ρⱼ ≤ 1。

F∞^{-1/2} 为正定平方根（eigh 构造；F∞=diag(s²) 的光度特例才可写 diag(1/s)——
**禁 F∞^{-1/4} 形式**，红队攻击五实测该错误产生假上界 0.716）。

**证明步骤**：
1. 由 Lemma 1：0 ≼ ΔF(Λ) ≼ F∞；
2. 对 Loewner 不等式作合同变换 F∞^{-1/2}(·)F∞^{-1/2}；
3. F∞ 秩亏时先限制到 range(F∞)（eigh 正特征向量基），可识别子空间显式给出。

**四元组**：假设 (H4) → 结论（谱界 0≤ρ≤1）→ 新颖性权重：解释工具 →
对应实验 CI01/CI02（bounds 100/100；retention 双路线 generalized-eig/白化交叉 <1e-10）。

---

## 5. Definition / Diagnostic · 场景条件可观测性交叉 λ⋆

**不是定理**（写作纪律：Definition + Diagnostic，红队报告三攻击三 R2 修订）。

**操作性定义**（绝对信息单位，全程）：
- **μ_floor**：去除 gauge 子空间后，uncalibrated/near-uncalibrated 基线中的
  最小非 gauge Rayleigh/eigen floor（场景本征弱模式地板；实现接口 = gauge.py::mu_floor，
  CI02 卡 C10 绑定预注册阈值）；
- **λ⋆**：gauge 绝对信息曲线（Prop 2）与 μ_floor 的交叉点。

**小 λ 线性预测器**：λ⋆ ≈ μ_floor / Σᵢ(αᵢ², sᵢ>0)。

**适用条件**：λ⋆ ≪ min_{αᵢ≠0} sᵢ²（线性域）；否则用闭式数值求根，不用线性近似。
**报告口径**：log₁₀(λ⋆^pred / λ⋆^scan) 的 median/IQR/90%（按"预测点是否仍在线性域"
分层），不用单一绝对差。红队 V3 单场景实测比 0.74（先验 sanity，禁泛化）。

**R2 度量纪律**：retention（归一化）度量下 gauge 与最弱模式**不交叉**
（亏缺系数 α 加权平均 s² < s_max²，实测 52.4 vs 65.5）——λ⋆ 只存在于绝对信息谱；
任何"R 谱可用于定位 λ⋆"的表述禁止（CLAIMS_REGISTRY V3 新增禁令）。

**四元组**：假设（场景给 gauge (a, c̄) 与非 gauge 谱）→
结论（λ⋆ 可由场景量预测，一阶闭式 + 数值求根）→ 新颖性权重：诊断资产 →
对应实验 CI02（多场景统计，H3 判据：跨场景长期偏差 >1 decade 才失败）。

---

## 6. 引用谱系（R1 修订落实）

| 谱系 | 锚点 | 用途 |
|---|---|---|
| Hybrid CRB | Noam & Messer (2008) | Lemma 1 身份定位 + Prop 1 达到性谱系 |
| 线性混合模型 | Henderson MME；Harville (1977) | Prop 1 特例身份（"自证不免引"，红队报告三攻击一） |
| 计量学传播 | JCGM 100 (GUM)；JCGM 101 (MC 传播) | Related Work 锚点：校准不确定度→被测量传播 |
| 光度立体 | Quéau et al. (CVPR 2017) inaccurate lighting | 切割句：他们做 reconstruction，本文刻画 information |
| 数据 | Shi et al. DiLiGenT (CVPR 2016)；Liu et al. OpenIllumination (NeurIPS 2023) | CI05 sanity / CI04 controlled corruption |

**Related Work 四块**（论文 §II）：calibrated/inaccurate/uncalibrated PS 谱系；
hybrid estimation + mixed linear models；measurement uncertainty propagation（GUM）；
inverse-rendering information/identifiability。

---

## 7. 与 CLAIMS_REGISTRY 的对应（写作时逐句过表）

| 命题 | Claim 行 | 允许表述 | 禁止表述 |
|---|---|---|---|
| Lemma 1 | C1 | "Schur form 实例化 + 单调（CI01 双路线验证）" | "new information geometry" |
| Prop 1 | C2 | "joint ensemble 下 Cov = σ²ΔF⁻¹（Prop 1 + CI03 MC）" | "new estimator" |
| Prop 2 | C3 | "gauge 绝对信息精确谱响应（Prop 2 + CI02）" | "universal new gauge theorem" |
| λ⋆ | C4 | "场景量闭式预测（小 λ 域近似 + 数值求根）" | "universal threshold" |
| Lemma 2 | C5 | "within-scene retention 读出，跨场景只比分布" | 跨场景逐 mode 直接比较 |

## 8. Supplement 证明骨架 checklist

- [ ] Lemma 1：H 分块 → Schur 补 → 逆序单调 → thin-SVD Λ=0 极限（含 (I−UUᵀ) 幂等）→ 白化一般式
- [ ] Prop 1：profile 消元 → marginal 分布 → Woodbury → GLS 协方差；Corollary 两条闭式单独小节
- [ ] Prop 2：四步代数 + 两端 Taylor；各向异性补充式
- [ ] Lemma 2：合同变换 + range 限制
- [ ] λ⋆：μ_floor 操作定义 → 一阶方程 → 适用条件 → 报告口径
- [ ] 数值验证表：V1–V6 判定与 CI01 formal 数字（provenance：commit SHA + manifest hash）

---

*卡 C08 交付 · 2026-09-07 · 依据：宪法 §3 证明步骤逐条成文；红队报告二/三 R1–R4 修订全部入版。*
