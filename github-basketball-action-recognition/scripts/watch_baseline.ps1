# Baseline watcher: wait for probe_r21d (s0) to finish, rename it, then run s1/s2
# via direct process launch (chain-window Tee pipeline caused silent deaths, probed).
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

function Log($m) { $l = "===== [$((Get-Date -Format 'MM-dd HH:mm:ss'))] $m"; Write-Output $l; Add-Content -Path $clog -Value $l -Encoding UTF8 }

# 1) wait for the probe run (only python expected) to exit
Log 'WATCHER: waiting for probe_r21d python to exit'
while ((Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0) {
    Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 5

# 2) promote probe run to r21d_s0 if it produced a best
$pd = Join-Path $runroot 'probe_r21d'
if (Test-Path (Join-Path $pd 'best.pt')) {
    if (Test-Path (Join-Path $runroot 'r21d_s0')) { Remove-Item (Join-Path $runroot 'r21d_s0') -Recurse -Force }
    Move-Item $pd (Join-Path $runroot 'r21d_s0')
    Log 'PROMOTED probe_r21d -> r21d_s0'
} else {
    Log 'WARN: probe_r21d has no best.pt, will train r21d_s0 from scratch'
}

# 3) run s1, s2 (and s0 if promotion failed) via direct launch (no Tee pipeline)
function Run-Direct($tag, $seed) {
    for ($try = 1; $try -le 3; $try++) {
        $best = Join-Path $runroot "$tag\best.pt"
        if (Test-Path $best) { Log "SKIP $tag (best.pt exists)"; return }
        $out = Join-Path $logs "$tag`_out.txt"
        $err = Join-Path $logs "$tag`_err.txt"
        Log "RUN $tag try=$try (direct launch)"
        $p = Start-Process -FilePath $py `
            -ArgumentList @('-u', $tr, '--data-root', $ds, '--run', $runroot,
                            '--tag', $tag, '--arch', 'r2plus1d18', '--seed', $seed,
                            '--batch', '16', '--lr', '0.0003', '--bigru', '0') `
            -RedirectStandardOutput $out -RedirectStandardError $err `
            -WindowStyle Hidden -PassThru
        $p.WaitForExit()
        Log "DONE $tag try=$try exit=$($p.ExitCode)"
        if ($p.ExitCode -eq 0) { return }
        Start-Sleep -Seconds 30
    }
}

Run-Direct 'r21d_s0' 0
Run-Direct 'r21d_s1' 1
Run-Direct 'r21d_s2' 2
Log 'BASELINE ALL DONE (direct-launch path)'
