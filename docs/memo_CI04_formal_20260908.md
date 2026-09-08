# CI04 正式 run · 1 页 memo（2026-09-08 · 卡 C16 · Gate D）

**判定：Gate D 通过。** test 11 对象（冻结清单，greenhead exclusion 见 §3）× 6 强度档
× 20 seeds = 330 弱模式点：**spearman(pred_deg, emp_deg) = 0.728，bootstrap95 [0.668, 0.783]
> 预注册 0.5，且 CI 下界 ≫ 0**。逐对象 spearman 全部为正且落在 [0.72, 0.80] 窄带
（无 cherry-picking 空间——test 对象在跑前冻结）。log-log 仿射斜率 1.82（emp 增长
快于 pred 的二阶结构，与 pilot 一致，量级差入 sensitivity 带不作主张）。
摘要=`artifacts/frozen/ci04_formal_summary.json`（manifest 同放）。

## 1. 预注册判据（C15 protocol freeze，未改动）

- spearman > 0.5 且 bootstrap95 下界 > 0 → **pass**（H5：显著秩相关，非仅 MAE 单调）；
- 主张措辞按 CLAIMS_REGISTRY C6：**逐模式退化预测与实测退化存在稳定秩相关 + 一个
  全局仿射尺度**；禁止只凭平均 MAE 单调宣称成功（宪法 Gate D 升级条款满足）。

## 2. 结果结构

| 项 | 读数 |
|---|---|
| 全池 spearman | **0.7278** [0.668, 0.783]（1000 bootstrap） |
| 逐对象 spearman | 11/11 为正，min 0.722（ball）max 0.800（dolphin/pine），带内稳定 |
| log-log 仿射斜率 | 1.82（emp 增长快于线性预测的二阶结构） |
| dev pilot 对照 | 0.676 [0.590,0.757] → test 0.728：未见 development 过拟合迹象 |

## 3. 执行层事项（如实记录）

- **test 清单冻结**：12 对象先冻结后下载；`obj_20_greenhead` 下载数据侧确认缺
  com_masked_thumbnail 层（obj_masked 掩码语义不一致：alpha 空占比 0.67 vs com 口径
  0.90）→ 作为 **exclusion** 如实记录（test_selection.json），test 降为 11 对象，
  无静默换数据。
- **首轮正式 run 误用 dev 清单**（config 回退默认），发现后重跑——"先清单后跑"
  的机械防线起效：对象集合断言进 config 显式 objects 字段，dev/test 集合不再隐式。

## 4. Gate D 处置与下一步

- **Gate D 通过 → 真实定量主张成立**（秩/尺度层）；量级层失配（~10²，R-A Σ_c_real +
  R-D 模型失配）作为 sensitivity 带如实进论文 limitations（红队报告三 §7 风险账本
  两条的处置兑现）。
- 下一步：**C17 DiLiGenT 收敛（loader 已迁移，C03）→ C18 CI05 sanity + 失败分类 →
  C19 ablation 附录**（白化 vs raw λI、Λ 4× misspec、异方差、灯数、mask policy）。
