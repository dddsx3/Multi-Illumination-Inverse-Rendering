# 训练热保护监督器（无需管理员权限）
# 循环：无训练进程且任务未完成时，等待 GPU <70C 后拉起编排器；
#       每 90s 巡检温度，>=84C 强杀训练进程（编排器幂等，断点续跑零损失）。
$repo = "D:\Multi-Illumination Inverse Rendering\repo"
$log = Join-Path $repo "_supervisor_log.txt"
$done = Join-Path $repo "_orchestrator_log.txt"

function Get-GpuTemp {
    $raw = (& nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
    return [int]($raw -replace '[^0-9]', '')
}

function Get-TrainingProcs {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'main\.py|run_phase2_all\.py|evaluate_model\.py|eval_n_curve\.py' }
}

Add-Content $log "supervisor start $(Get-Date)"
while ($true) {
    if ((Test-Path $done) -and (Select-String -Path $done -Pattern "ALL ARMS COMPLETE" -Quiet)) {
        Add-Content $log "ALL ARMS COMPLETE $(Get-Date) -> exit"
        break
    }
    if (-not (Get-TrainingProcs)) {
        # 启动前等温度回落（本机空闲即 ~73C，故阈值取 75）
        while ($true) {
            $t = Get-GpuTemp
            if ($t -le 75) { break }
            Add-Content $log "pre-launch wait t=$t $(Get-Date)"
            Start-Sleep -Seconds 60
        }
        Add-Content $log "launch orchestrator t=$t $(Get-Date)"
        # 训练内热节流：1.5s/批次（两次热保护关机后加严；只改墙钟不碰超参口径）
        $env:THERMAL_PACE = "1.5"
        Start-Process -FilePath "python" `
            -ArgumentList "-u", "run_phase2_all.py" `
            -WorkingDirectory $repo `
            -RedirectStandardOutput $done `
            -RedirectStandardError (Join-Path $repo "_orchestrator_err.txt") `
            -WindowStyle Hidden
        Start-Sleep -Seconds 150
        continue
    }
    $t = Get-GpuTemp
    Add-Content $log "t=$t $(Get-Date)"
    if ($t -ge 80) {
        Add-Content $log "THERMAL GUARD t=$t -> kill training $(Get-Date)"
        Get-TrainingProcs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    }
    Start-Sleep -Seconds 30
}
