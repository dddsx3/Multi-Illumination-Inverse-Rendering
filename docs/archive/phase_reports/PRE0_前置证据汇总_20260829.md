
---

## PRE-0 前置证据获取（2026-08-29）

按 `docs/PRE0_任务书.md` 完成的全协议级前置证据交付，**结论先行**：
**Gate B FAIL** —— 审计发现 `D:\data\synthetic_v3` 中 5 张"不同光照"图像
实为同一张图（BlenderProc 帧动画失效），多光照维度无效。一切
"evidence accumulation" 主故事暂停，先修数据。详见 `pre0/HANDOFF.md`。

- 协议与数据合同：`pre0/protocol/{pre0_protocol.yaml, DATASET_CONTRACT.md, split_manifest.json}`
- 审计报告（最重要的单文件）：`pre0/oracle_renderer/ORACLE_AUDIT.md`
- 12 问逐答：`pre0/HANDOFF.md`
- 三 Probe 训练（统一预算 0.71M × 40ep，linear 域，edge_aware=False）：
  `pre0/checkpoints/probe_{A,B,C}_{best,last}.pth`
- DiLiGenT 合同与 260 固定子集：`pre0/benchmark/`
- 文献矩阵 43 篇 + 最近 10 篇反驳证据：`pre0/literature/`

### 一键复现
```bash
# PRE-01
python pre0/source/renderer/oracle.py --split test
# PRE-02（解析域裁剪版，含数据缺陷屏蔽）
python pre0/source/information_audit/pre02.py --exp 1 3 4 --domains analytic15_sh
python pre0/source/information_audit/pre02.py --exp 2 --exp2_scenes 32 --domains analytic15_sh
# PRE-03 训练
python pre0/source/train/train_probe.py --probe {A|B|C}
# PRE-04/05 评估（要求三 probe 已训练）
python pre0/source/evaluate/pre04.py --probes A B C
# PRE-06 文献已交付；PRE-07 评估器：subsets 已落盘，对接新模型即可
```
