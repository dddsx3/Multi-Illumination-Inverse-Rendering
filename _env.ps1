# 训练环境变量固化（INC-0010 审计裁决 §3 A4）
# 用途：让 run_arms.py / main.py 启动时自动加载；或 dot-source 之后手动运行
#
# 物理事实：本机 GPU 闲置温度 74°C（实测 nvidia-smi），原 THERMAL_RESUME=70
# 永远等不到——已固化到 75°C。停机阈值 THERMAL_LIMIT=80，HARD_KILL=88 兜底。
# 安全余量：75 → 80 = 5°C 间隔。

$env:THERMAL_RESUME = "75"
$env:THERMAL_LIMIT = "80"
$env:THERMAL_PACE = "2.0"
$env:PYTHONUNBUFFERED = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# 任何运行 run_arms.py / main.py 之前先 dot-source 此文件：
#   . .\repo\_env.ps1
