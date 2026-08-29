# LIGHTING_MODEL · SH Irradiance Coefficients（Route A）

> **P1 阶段决议**：网络输出的 9 维量是**表面 irradiance SH coefficients**（Route A）。
> 不再使用 `c = I·Y(d)` 然后直接 `Σ cᵢYᵢ(n)` 这种把 radiance coefficients 当
> Lambertian shading 用的"双重角色"做法。
> 签发：P1-01 任务（docs/P1_任务书.md）· 2026-08-30。

## 1. 输入 lighting 的物理量

P1 数据生成支持的两种 light 类型（**每灯独立 render call**，不再用 frame animation）：

### 1.1 Directional / 远场点光

光源在相机系方向 `d̂ ∈ S²`（`‖d̂‖=1`），辐照度强度 `I_eff ∈ ℝ⁺`，单位 `W/m²`。
辐亮度（radiance）：`L(ω) = I_eff · δ(ω − d̂)`。

### 1.2 可变强度（同一方向）

通过 `I_eff` 标量缩放实现。所有 SH 系数线性缩放，**不需要**修改 renderer。

## 2. SH basis 与坐标系

- **Basis**：标准二阶实 SH（3 basis order = `(2+1)²=9` 系数）；
- **坐标系**：**所有 lighting / normal 系数进入网络前必须先转到 camera frame**（P1-03）。
  记世界→相机的旋转为 `R_cw`：`d_c = R_cw · d_w`，`n_c = R_cw · n_w`，
  `c_lm^camera = c_lm^world`（系数本身在旋转下遵循 SH 的 Wigner-D 矩阵，
  对低阶 L≤2 可用实数形式旋转公式，见 P1-03 节 3）。
- **Normalization**：标准实 SH，正交归一于单位球面：
  `∫_S² Y_{lm}(ω) Y_{l'm'}(ω) dω = δ_{ll'} δ_{mm'}`。
- **Basis order**（与 `physics_renderer.SphericalHarmonicsLighting` 一致，**基常数不重设**，
  此处仅为语义统一文档）：

  | lm | Y_lm(ω) | 常数 |
  |---|---|---|
  | 00 | 1 | C₀ = √(1/(4π)) = 0.282095 |
  | 1,-1 | y | C₁ = √(3/(4π)) = 0.488603 |
  | 1, 0 | z | 同上 |
  | 1,+1 | x | 同上 |
  | 2,-2 | xy | C₂[0] = ½√(15/π) = 1.092548 |
  | 2,-1 | yz | C₂[1] = 同上 |
  | 2, 0 | 3z²−1 | C₂[2] = ¼√(5/π) = 0.315392 |
  | 2,+1 | xz | C₂[3] = 同上 |
  | 2,+2 | x²−y² | C₂[4] = ½√(15/π) = 0.546274 |

  顺序（向量索引 0..8）：`[Y00, Y1-1, Y10, Y1+1, Y2-2, Y2-1, Y20, Y2+1, Y2+2]`。

## 3. Directional light → SH radiance coefficients

**推导**（L=0..2 的解析形式，避免任意系数）：

辐亮度 SH 投影：
```
L_lm = ∫_S² L(ω) Y_lm(ω) dω = I_eff · Y_lm(d̂)
```

对 L=0..2：
```
L_00 = I_eff · C₀
L_1m = I_eff · C₁ · (d̂_x, d̂_y, d̂_z) 顺序为 m=+1,−1,0
L_2m = I_eff · C₂[?] · Y_2m(d̂)    （参见上表）
```

## 4. Lambertian convolution（关键：radiance → irradiance）

**只对**"irradiance coefficients"路径（Route A）应用。

Lambertian BRDF：`f(ω_i, ω_o) = ρ/π`，clamped-cosine `max(0, n·ω)` 与 BRDF 卷积给出**入射 irradiance**：
```
E_lm = k_l · L_lm
```
其中 Lambertian convolution coefficient `k_l`（与表面积分相关，对二阶闭式）：

| l | k_l |
|---|---|
| 0 | √π ≈ 1.7724539 |
| 1 | √(π/3) ≈ 1.0233267 |
| 2 | √(π/5) ≈ 0.8862269 |

来源：Ramamoorthi & Hanrahan 2001 "An Efficient Representation for Irradiance
Environment Maps"，Table 1；亦见 Sloan et al. "Simple Environment Map Filtering"。

注意：clamped-cosine convolution 在实 SH 系下的闭式（仅 L≤2）由上表给出；
高阶项在 L≤2 截断下自动为零，这是 SH 近似误差的主要来源（见 §8）。

**结论**：网络最终输出 `c_lm = k_l · L_lm`（irradiance coefficients）。
Renderer 只需 `E(n) = Σ c_lm Y_lm(n)` 直接求值，不再 `k_l` 系数补偿。

### 4.1 替代：Route B（env radiance）不在本阶段使用

如选择 Route B，需要 renderer 显式执行 `E(n) = ∫ L(ω)max(0,n·ω)dω` 或用
预卷积 lookup table。**P1 阶段明确采用 Route A**（更简单、与当前
`physics_renderer.SphericalHarmonicsLighting` 接口兼容、且与
"`Σ cᵢYᵢ(n)` → albedo ⊙ Σ cᵢYᵢ(n)"的旧语义近似等价，
只要把系数从 radiance 改为 irradiance）。

## 5. 单位与 scale gauge

- **Albedo**：线性域 [0,1]（scalar，luma of linear RGB）；
- **SH 系数 `c_lm` 的物理单位**：W/m²（irradiance），与 albedo 相乘后
  `albedo * E(n)` 仍为 W/m²；图像在完美 Lambertian + 相机响应 `C(v)=v`
  假设下直接是图像强度。**任何相机响应函数需独立建模**。
- **Scale gauge ambiguity**：albedo 与光照存在全局乘积歧义
  (`A·E` 与 `(sA)·(E/s)` 不可区分)，在所有 albedo 评估中
  强制 SI-MAE（per-scene 全局尺度最小二乘闭式解；
  同 `evaluate.albedo_metrics` 的定义）。
- **Per-light gauge 不变**：每盏光独立送入网络时，per-light scale gauge
  不在网络内被消除——评估用 SI-MAE 时应**逐灯**与 GT 对比。

## 6. 表面辐照度的非负性

ReLU(`E(n) = Σ c_lm Y_lm(n)`) 截断负值部分（与 `physics_renderer.py` 现行
实现一致）。**已知后果**：在背向光的"半影"区，会引入 ~5-10% 像素的
负值被错误截断为 0；当 L=2 截断本身就有 ringing 时（见 §8），这层
截断会使 L2 截断的"高估"误差被吃掉一部分。

**禁止替代方案**：用 `softplus` 平滑 ReLU——会引入次像素梯度饱和，
对 identifiability 与 oracle 残差都更糟。统一用 ReLU，副作用如实记录。

## 7. 9 维表示的近似误差（实测，见 `tests/test_sh_physics.py`）

**Route A 实测数字**（10k+ 随机 (n, d) 对，radiance 系数路径，无卷积）：

| 量 | 数值 | 含义 |
|---|---|---|
| MAE | **0.377** | L=2 截断下 E_rad 与 max(0, n·d) 平均绝对差 |
| RMSE | 0.411 | |
| P95 | 0.547 | 95% 误差量级 |
| max | 0.548 | 极端 case |
| 误差占比 | 75%（MAE / 0.5） | L=2 截断已损失大部分方向光形状信息 |

> 注：此即"renderer approximation floor"——任何 L=2 SH 9 系数表示，
> 单独方向光 d 都不能精确表达 Lambertian 积分（需要 L≥4 才能 < 5% 截断误差）。
> 但**多光叠加**下（≥5 个 d 跨半球），此 floor 会随 N 增大而下降——
> P1-11 条件数分析即量化这一点。

**单光背向 ringing 实测**：n=−d 时 radiance 路径 E=0.239（理论应=0，
参考 Lambertian clamped=0），ReLU 截断后 = 0.239 ≠ 0；ReLU **不能**修正
正 ringing，只能吃掉负 ringing。这是 Route A 与 Lambertian 之间的
已知 L=2 截断偏置。

**PRE-0 经验数字**：synthetic_v3 上的 0.92° SH 反解方向误差主要由
路径追踪阴影 + 间接光造成，不是 SH 截断本身；新 P 域（无阴影无间接光）
下 L=2 SH 截断的 oracle 重建 floor 应**显著低于 14.88 dB 旧数字**——
P1-08 Calibration Gate 即用来量化这个 floor。

## 8. 参考文献

- Sloan, Jarosz, Goeseele "Simple Environment Map Filtering"（SH 投影 + convolution 系数）
- Ramamoorthi & Hanrahan 2001 "An Efficient Representation for Irradiance Environment Maps"（k_l 闭式）
- Green 2003 "Spherical Harmonic Lighting: The Gritty Details"（L=0..2 实 SH 基常数与 normalization）

## 9. 实现位置

P1 阶段所有 SH 数学常量、basis 顺序、convention 与本文件一致；不直接
修改 `physics_renderer.py`（其 L=0..2 SH 计算方式与 Route A 兼容，
但解释为 irradiance coefficients 而非 radiance coefficients）。
新生成器的 SH 系数写入时应用 §4 卷积（系数 = k_l · L_lm），并明确
存储 metadata `lighting_kind: "irradiance_coefficients_l2"` 以避免与
历史合成数据的 `radiance_coefficients_l2` 混用。
