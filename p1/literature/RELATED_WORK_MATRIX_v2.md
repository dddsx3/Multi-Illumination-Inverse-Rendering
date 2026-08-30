# RELATED_WORK_MATRIX_v2 · P1-R5 新颖性地图更新（2026-08-30）

> **触发**：外部专家审查指出 `closest_prior_verified.md` 的多条概括已过时
> （"universal PS 全部 normal-only"等）。本版按任务书 P1-R5 要求：
> 把所有 "没有工作做 X" 改写为
> "**Closest works address X/Y, while none of the works examined here
> studies gauge-aware subset identifiability for explicit joint
> scene-light decomposition under variable cardinality**"。
> 逐篇细节仍以 `closest_prior_verified.md` 为基础，本文件聚焦增量。

## 1. 新增/更新的高危工作（2025-2026）

| 工作 | venue | 已覆盖 | 对我们的冲击 | 仍开放 |
|---|---|---|---|---|
| **IDArb** | ICLR 2025 | **arbitrary #views × varying illuminations** 的 intrinsic decomposition（normal+material），PS 是其应用之一 | "任意 N + intrinsic 分解"本身不能再当核心贡献 | 无 per-light 显式可查询光照表示、无子集质量理论、无 held-out relighting 协议（待核实原文） |
| **Light of Normals (LINO-UniPS)** | ICLR 2026 | unknown arbitrary illumination + universal PS，官方代码含 **PBR material estimation** | "2026 universal PS 仍 normal-only" 概括**作废** | 同上：联合 depth+per-light SH+relighting 协议未见 |
| **GeoUniPS** | AAAI 2026 | 明确研究 **limited/biased multi-illumination cues**（光照不充分时性能下降） | "并非所有多光照证据同等有用"的直觉不新 | 只有经验观察，无可辨识性度量/理论 |
| **ReLeaPS** | ICCV 2023 | **illumination planning**：选 20 灯配置 ≈ 全灯性能 | "active light selection" 不能当核心 | 选择准则基于 RL 经验，无 gauge-aware 信息度量 |
| **On Optimal Light Configurations** | ICCV 2005 | calibrated Lambertian PS 的光源几何 → normal recovery 不确定度 | "光的 quality 而非数量"是**经典事实** | calibrated + normal-only + 无 per-image lighting 联合恢复 |
| **Basri & Jacobs 系** | ~2001-2011 | unknown lighting + 低阶 SH + shape ambiguity 分析 | "unknown light + SH" 不新 | 非 variable-cardinality、非联合 albedo/depth/per-light SH |

## 2. 修正后的可辩护新颖性表述

**不能再说**：
- ❌ "首个 arbitrary-N 联合逆渲染"（IDArb 已占）
- ❌ "universal PS 尚无 material 输出"（LINO 已有）
- ❌ "光照质量比数量重要"（ICCV05 / GeoUniPS / ReLeaPS 已占直觉）
- ❌ "active light selection"（ReLeaPS 已占）

**仍然可辩（provisional，待 R4 定核后升级）**：
> Closest works address arbitrary-N intrinsic decomposition (IDArb),
> universal PS with material estimation (LINO), and illumination
> planning (ReLeaPS), **while none of the works examined here
> (i) defines a gauge-aware effective-information measure of an
> illumination subset that remains valid under unknown per-image
> lighting and global scale gauge, (ii) shows it predicts joint
> albedo–geometry–per-light-SH recoverability at FIXED cardinality,
> and (iii) ties the N-curve to this measure as its projection.**

## 3. 对 CLAIM_REGISTRY 的约束（R4 之前）

- C1（科学核心）只允许以 hypothesis 形式出现；
- 一切 "first/novel" 措辞限定在 (i)(ii)(iii) 的合取上；
- 每条 claim 后附"若 R4 失败"的降级路径（见 P1_R0_STOP_LINE §失败分叉）。

## 4. 遗留核实项（下次外部检索循环）

- IDArb 原文核实：是否有 per-image lighting 输出与 held-out relighting；
- LINO 官方代码的 material 输出细节（albedo 精度/域）；
- GeoUniPS 的 "limited cues" 是否给出可量化的 conditioning 代理；
- 2025-2026 新出现的 joint UPS + relighting 工作（每季度一轮）。
