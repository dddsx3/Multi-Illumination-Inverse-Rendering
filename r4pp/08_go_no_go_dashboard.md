# 08 · GO/NO-GO Dashboard

> 生成：2026-09-01T00:17:23.308067 · 严格 6 行（任务书 §41，禁止增删）

| Gate | 指标 | 结果 | 证据 |
|---|---|---|---|
| Instrument | GSIQ / M1 5/5 stability PASS | **PASS** | MA=0.9991149623068079 MB=0.9926632664346688 MC=0.9961007929587933 MF=0.9982353258898602 |
| Signal | low-N signal vs solver-repeat | **PASS** | R_signal: N=2 31, N=3 43, N=5 30, N=8 23 |
| Direction | info→error β<0 | **PASS** | β med=-0.348, 负占比=0.81 |
| Interaction | G↑ ⇒ |β_G|↑ | **FAIL** | A:ρ=+0.29(n=10); B:ρ=-0.23(n=6) |
| Saturation | N=8 selection-leverage compression | **PASS** | N=8 R_signal=22.6 vs N=3 43.0 |
| Externality | local-init replication | **PENDING** | 未运行 Task G |

**合计：4/6 PASS**

**预注册裁决：PIVOT (B′)**

## 说明
- Instrument/Signal 为硬门槛；Direction/Interaction 决定 GO vs PIVOT；
- Externality 失败不直接 KILL，但警示 A2 结论的稳健性；
- 本 dashboard 只反映已产出的证据；Task G 未跑时 Externality 为 PENDING。