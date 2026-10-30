# Re-run the 3 failed baseline runs (head-par line bug, fixed in train_action_cls2.py).
# Launch AFTER run_cls2_chain.ps1 completes (GPU sequential rule).
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'

$py  = 'F:\zjl\envs\yolo\python\python.exe'
$tr  = Join-Path $PSScriptRoot 'train_action_cls2.py'
$root   = Split-Path -Parent $PSScriptRoot
$basket = Split-Path -Parent $root
$logs   = Join-Path $root 'logs'
$runroot = Join-Path $root 'runs\cls2'
$clog = Join-Path $logs 'cls2_chain_log.txt'

$ds1 = Get-ChildItem $basket -Directory | Where-Object {
    Test-Path (Join-Path $_.FullName 'datasets\dataset_action_clip_val35_sliding\train')
} | Select-Object -First 1
$ds = Join-Path $ds1.FullName 'datasets\dataset_action_clip_val35_sliding'

function Run-One($tag, $arch, $seed, $bigru, $batch, $lr) {
    for ($try = 1; $try -le 3; $try++) {
        $best = Join-Path $runroot "$tag\best.pt"
        if (Test-Path $best) { Write-Output "SKIP $tag (best.pt exists)"; return }
        $line = "===== [$((Get-Date -Format 'MM-dd HH:mm:ss'))] RUN $tag arch=$arch seed=$seed try=$try"
        Write-Output $line
        Add-Content -Path $clog -Value $line -Encoding UTF8
        & $py -u $tr --data-root $ds --run $runroot --tag $tag --arch $arch `
            --seed $seed --batch $batch --lr $lr --bigru $bigru 2>&1 | Tee-Object -FilePath $clog -Append
        $code = $LASTEXITCODE
        Add-Content -Path $clog -Value "===== DONE $tag try=$try exit=$code" -Encoding UTF8
        if ($code -eq 0) { return }
        Start-Sleep -Seconds 30      # GPU 驱动重置后稍等再试
    }
}

Write-Output "STAGE A-retry: R(2+1)D-18 multi-seed x3 (baseline)"
Run-One 'r21d_s0' 'r2plus1d18' 0 0 16 '0.0003'
Run-One 'r21d_s1' 'r2plus1d18' 1 0 16 '0.0003'
Run-One 'r21d_s2' 'r2plus1d18' 2 0 16 '0.0003'
Write-Output 'BASELINE RETRY DONE'
