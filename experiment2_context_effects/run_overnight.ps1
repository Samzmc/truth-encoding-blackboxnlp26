# Phase 3 overnight Stage B extraction: 70B first, then 8B.
# Run from any PowerShell window (keep the laptop awake / plugged in):
#   powershell -ExecutionPolicy Bypass -File run_overnight.ps1
# Resumable: if it crashes, rerun this script — completed shards are skipped.

# Point this at the python of the NDIF/nnsight environment (see README)
$py = "python"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$env:PYTHONIOENCODING = "utf-8"

$log = Join-Path $here "overnight_log.txt"
"=== overnight run started $(Get-Date -Format s) ===" | Tee-Object -FilePath $log -Append

& $py extract_context_probs.py --stage b --model 70b 2>&1 |
    Tee-Object -FilePath $log -Append
"=== 70B stage b finished $(Get-Date -Format s) ===" | Tee-Object -FilePath $log -Append

& $py extract_context_probs.py --stage b --model 8b 2>&1 |
    Tee-Object -FilePath $log -Append
"=== 8B stage b finished $(Get-Date -Format s) ===" | Tee-Object -FilePath $log -Append

"=== all done $(Get-Date -Format s) ===" | Tee-Object -FilePath $log -Append
