# Unattended watchdog for the 12h LGBM paper-trading session.
# Ensures collector + paper trader stay up until DEADLINE_UTC; restarts paper on crash.
#
# Usage (from repo root):
#   powershell -File scripts/watch_paper_8h.ps1

$ErrorActionPreference = "Continue"
$Repo = "C:\Users\Kev\repos\stochastic-spread-modeling"
$Analysis = "C:\Users\Kev\repos\stochastic-spread-modeling-analysis"
$PyCollect = Join-Path $Repo ".venv\Scripts\python.exe"
$PyPaper = Join-Path $Analysis ".venv\Scripts\python.exe"
# Current session: cex_gbm_new Jul25-split model (68 feats, HORIZON=1 / N_LAGS=3 / ZSCORE=300)
$Model = Join-Path $Analysis "statarb\outputs\statarb_lgbm.txt"
$RunDir = Join-Path $Repo "data\statarb\20260801_025316"
$OutDir = Join-Path $Repo "data\paper_trading\July31st_8_hr"
$LogDir = Join-Path $Repo "data\logs"
$DeadlineUtc = [datetime]::Parse("2026-08-01T14:55:33Z").ToUniversalTime()  # ~12h from paper start
$SessionStartUtc = [datetime]::Parse("2026-08-01T02:55:33Z").ToUniversalTime()

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

function Write-Watch([string]$msg) {
    $line = "[{0}] {1}" -f ([datetime]::UtcNow.ToString("o")), $msg
    Add-Content -Path (Join-Path $LogDir "watchdog_paper_8h.log") -Value $line
    Write-Host $line
}

function Get-MatchingPids([string]$pattern) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match $pattern) } |
        Select-Object -ExpandProperty ProcessId
}

function Ensure-Collector {
    $pids = @(Get-MatchingPids "collect_statarb_data")
    if ($pids.Count -gt 0) { return $false }
    $remainingH = [math]::Max(0.05, ($DeadlineUtc - [datetime]::UtcNow).TotalHours)
    Write-Watch "collector DEAD - restarting for remaining ${remainingH}h on $RunDir"
    # Prefer resume into existing run dir
    $args = @(
        "-m", "experiments.collect_statarb_data",
        "--assets", "volatile",
        "--interval", "60",
        "--slow-every", "1",
        "--hours", ("{0:N2}" -f $remainingH),
        "--skip-ohlcv",
        "--resume", $RunDir
    )
    Start-Process -FilePath $PyCollect -ArgumentList $args -WorkingDirectory $Repo `
        -RedirectStandardOutput (Join-Path $LogDir "collector_paper_8h_watchdog.out.log") `
        -RedirectStandardError (Join-Path $LogDir "collector_paper_8h_watchdog.err.log") `
        -WindowStyle Hidden
    return $true
}

function Ensure-Paper {
    $pids = @(Get-MatchingPids "paper_trade_lgbm")
    if ($pids.Count -gt 0) { return $false }
    $remainingH = [math]::Max(0.05, ($DeadlineUtc - [datetime]::UtcNow).TotalHours)
    Write-Watch "paper trader DEAD - restarting for remaining ${remainingH}h"
    $args = @(
        "-m", "experiments.paper_trade_lgbm",
        "--model", $Model,
        "--run-dir", $RunDir,
        "--hours", ("{0:N2}" -f $remainingH),
        "--entry-tau", "0.5",
        "--poll-sec", "20",
        "--output-dir", $OutDir
    )
    Start-Process -FilePath $PyPaper -ArgumentList $args -WorkingDirectory $Analysis `
        -RedirectStandardOutput (Join-Path $LogDir "paper_lgbm_8h_watchdog.out.log") `
        -RedirectStandardError (Join-Path $LogDir "paper_lgbm_8h_watchdog.err.log") `
        -WindowStyle Hidden
    return $true
}

function Write-HealthSnapshot {
    $c = @(Get-MatchingPids "collect_statarb_data").Count
    $p = @(Get-MatchingPids "paper_trade_lgbm").Count
    $sumPath = Join-Path $OutDir "summary.json"
    $sum = if (Test-Path $sumPath) { Get-Content $sumPath -Raw } else { "{}" }
    $spread = Get-ChildItem (Join-Path $RunDir "spread_matrix\*.jsonl") -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $age = if ($spread) { ([datetime]::UtcNow - $spread.LastWriteTimeUtc).TotalSeconds } else { -1 }
    Write-Watch ("health collectors={0} papers={1} spread_age_s={2:N0} summary={3}" -f $c, $p, $age, ($sum -replace "\s+", " ").Substring(0, [Math]::Min(240, ($sum -replace "\s+", " ").Length)))
}

Write-Watch "watchdog started; deadline=$($DeadlineUtc.ToString('o'))"
Write-HealthSnapshot

while ([datetime]::UtcNow -lt $DeadlineUtc) {
    try {
        Ensure-Collector | Out-Null
        Start-Sleep -Seconds 3
        Ensure-Paper | Out-Null
        Write-HealthSnapshot
    } catch {
        Write-Watch "watchdog loop error: $_"
    }
    Start-Sleep -Seconds 60
}

Write-Watch "deadline reached - watchdog exiting"
Write-HealthSnapshot
