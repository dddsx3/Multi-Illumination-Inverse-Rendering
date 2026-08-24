# 分块续跑渲染：每 ChunkSize 个场景重启 Blender（规避长进程累积性崩溃）
# 已完成场景自动跳过；单块失败不影响后续块
param([int]$ChunkSize = 40)
$ErrorActionPreference = "Continue"
Set-Location "D:/Multi-Illumination Inverse Rendering/repo"

$all = Get-Content obj_models_list.txt
$remaining = @()
foreach ($p in $all) {
    $sha = [IO.Path]::GetFileNameWithoutExtension($p)
    if (-not (Test-Path ("D:/data/synthetic/" + $sha + "/sh_coeffs.npy"))) { $remaining += $p }
}
Write-Output ("remaining_scenes=" + $remaining.Count)

$chunkIdx = 0
for ($i = 0; $i -lt $remaining.Count; $i += $ChunkSize) {
    $end = [Math]::Min($i + $ChunkSize - 1, $remaining.Count - 1)
    $batch = @($remaining[$i..$end])
    $chunkIdx++
    $listFile = Join-Path (Get-Location) "_chunk_list.txt"
    [IO.File]::WriteAllLines($listFile, $batch)
    Write-Output ("=== chunk " + $chunkIdx + " (" + $batch.Count + " models) ===")
    py -3.10 -m blenderproc run render_dataset.py --obj_list $listFile --out_dir D:/data/synthetic --size 256 --gpu >> _render_full_log.txt 2>&1
    Write-Output ("chunk" + $chunkIdx + "_exit=" + $LASTEXITCODE)
}
Write-Output "ALL CHUNKS DONE"