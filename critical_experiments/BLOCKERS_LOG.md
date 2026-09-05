# 阻塞项记录 · BLOCKERS_LOG(2026-09-06)

## B-1 · exp11v3 κ_weighted 扩展(重试 5 次后停止 · 按用户规则)

- 现象: 19 个 conf_ 场景中 18 个在 trace_theta_inv_safe 内抛出 numpy 线性代数异常
  (首测 "SVD did not converge", 修 NaN/mask 前置与双重包装后转为维度/奇异类异常);
  canary 场景可完整运行(n=20, κ_weighted 不显著)。
- 已排除: C/nrm NaN(mask 前置修复 ✓); trace_pair 双重包装(修复 ✓); 签名不匹配(修复 ✓)。
- 残余根因(未穷尽): conf_ 场景与 data/ 场景的网格结构差异(全图 vs 盘内)使逐像素
  2×2 块在部分配置退化, 现有 keep 过滤未覆盖全部退化模式。
- 影响: κ_weighted 扩展验证未完成——cube 信号按预注册回退判定【降级为孤例】;
  exp11v2 负结果族定稿不受影响(其判据本就不依赖本扩展)。
- 处置: 按用户规则(5 次重试上限)停止; 建议后续以 conf_ 场景的独立调试会话解决
  (单独跑通 1 个 conf_ 场景的 trace 管线再批量)。
- 关联产物: exp11v3_kappa_expansion.{py,json}(含逐场景错误记录)
