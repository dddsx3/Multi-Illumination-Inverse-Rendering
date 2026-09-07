# legacy_redteam · 红队脚本只读归档

> **状态：只读归档，禁止任何实验/测试/主路径 import 本目录模块。**
> 归档时间：2026-09-07（卡 C02）· 来源：三轮红队会话（ClawsGO Agent）· 种子：20260907（统一种子，勿改）

## 内容清单

| 文件 | 轮次 | 覆盖 | 判定（见对应红队报告） |
|---|---|---|---|
| `redteam_exp.py` | 第一轮（E1–E6） | 针对 v2 专家评审设计文档的攻击 | 部分主张随 v2 设计文档一并废止（GA-ISI 旧主线）；E 系列数字仅历史参考 |
| `redteam2_exp.py` | 第二轮（N1–N5） | calibration-confidence continuum 主线攻击：Schur 身份/单调性/谱形式/秩亏 erratum/soft gauge/Λ 错设 | 新主线全部通过；N3-fix/N4-fix 两条 erratum/新现象入版（红队报告二 §1） |
| `redteam3_exp.py` | 第三轮（V1–V6） | 冻结前理论收口攻击：Prop1 恒等式/gauge 谱闭式/retention 谱界/参数化不变性/尺度/模式旋转 | 全部通过，理论放行冻结；R1–R4 修订入版（红队报告三 §8） |

## 与新结构的关系

- V1–V6 的判定已拆为正式单元测试：`tests/unit/test_v{1..6}_*.py` + `tests/unit/test_rank_deficient_lambda0.py`
  （卡 C02 迁移；共享数值 helper 在 `tests/unit/_redteam_reference.py`）。
- 生产单源实现位于 `src/calibinfo/`（卡 C06 起）；实验脚本不得复制 Fisher 公式（宪法 §6 原则）。
- 三轮报告存档于逆渲染工作区根（`红队报告*_20260907*.md`），主张允许/禁止表述已并入
  仓库根 `CLAIMS_REGISTRY.yaml` 的 `redteam_addenda` 节。

## 运行方式（仅审计复现用）

```bash
python legacy_redteam/redteam3_exp.py   # 纯 CPU，~5-71s，输出与红队报告数字对照
```
