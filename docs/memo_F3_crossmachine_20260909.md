# F3 · 异机 clean-room 复现 memo（2026-09-09）

> 卡 F3 · 服务 Gate E。本机即第二环境（定义见卡 F3：Windows 本机执行，
> 云主机不可替代）——本 memo 记录全新 clone 的全链对账；环境与主开发会话独立
> （新目录、新 editable 安装、独立 BLAS 会话状态）。

## 环境指纹

| 项 | clean-room |
|---|---|
| clone HEAD | 2f05440（GitHub 实际内容） |
| Python | 3.14.2 |
| numpy / scipy | 2.4.1 / 1.17.0 |
| BLAS | scipy-openblas 0.3.30 |
| CPU | AMD Family 25 Model 97 (Zen4), AMD64 |
| 安装 | pip install -e . 绿 |

## 五项对账（预注册阈值：identity 类 1e-10；MC 类锁中位/IQR 区间）

| # | 项 | 结果 |
|---|---|---|
| 1 | pytest 全套件 | **54/54 passed**（红队 V1–V6 回归 + 迁移绑定 + 稠密对照 + scene 工厂 + manifest + OpenIllumination 真数据） |
| 2 | CI01 formal（100 identity checks） | 100/100 all_pass=True；与主仓库 frozen 逐行数值最大差 **0.000e+00**（确定性种子，位级一致） |
| 3 | CI03 linear MC（18 checks，4000–10000 trials） | 18/18；逐 case 数值最大差 **0.000e+00**（MC 同种子位级复现） |
| 4 | Fig.3 / Fig.4 重建 | PNG **逐字节 IDENTICAL**（sha256 相同） |
| 5 | Fig.7（真实数据图）重建 | PNG **逐字节 IDENTICAL** |

## 判定

- 五项全绿，且确定性执行达到位级复现（0.000e+00 差异）——未触发停止规则②；
- "异机"边界说明：本 clean-room 与主开发同物理机（同 CPU/BLAS 型号），跨物理机
  smoke 由投稿前异机执行 reproduce_paper.sh 补齐（C22 memo 已记为单机限制）；
  本卡验证的是"全新检出→安装→全链→产物一致"路径本身，含环境重建全过程。

## 产物

- clean-room 目录保留至投稿后（D:/f3_cleanroom）；
- 对账脚本内嵌于本 memo（数值比较 + PNG sha256），审计可复跑。
