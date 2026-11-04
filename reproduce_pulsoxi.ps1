[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$DataDir,
    [switch]$OnlyV3,
    [string]$OutputDir = "outputs/local_reproduction"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Inference = Join-Path $Root "scripts\run_inference.py"
$Validator = Join-Path $Root "evaluation\validate_submission.py"
$ApplyV4 = Join-Path $Root "scripts\apply_turnover_stabilization.py"
$DataDir = (Resolve-Path $DataDir).Path
$Out = Join-Path $Root $OutputDir
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Out $Stamp
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$StdOut = Join-Path $RunDir "inference.stdout.log"
$StdErr = Join-Path $RunDir "inference.stderr.log"
$Monitor = Join-Path $RunDir "process_monitor.log"
$Report = Join-Path $RunDir "reproduction_report.json"
$V3 = Join-Path $RunDir "submission_v3be2.csv"
$V4 = Join-Path $RunDir "submission_v4_15s.csv"
$V3Diag = Join-Path $RunDir "diagnostics_v3be2.json"

function Require-File([string]$Path) { if (!(Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required file is missing: $Path" } }
Require-File $Python; Require-File $Inference; Require-File $Validator
if (!(Test-Path -LiteralPath $DataDir -PathType Container)) { throw "Data directory is missing: $DataDir" }
$preserved = Join-Path $Root "outputs\local_inference\submission_v3be2.csv"
$preservedV4 = Join-Path $Root "outputs\local_inference\submission_v4_15s.csv"
Require-File $preserved

$existing = Get-Process -Name python,pythonw -ErrorAction SilentlyContinue
if ($existing) { throw "An existing Python process is detected. Stop it or verify it is unrelated before starting another run." }

$start = Get-Date
$args = @($Inference, "--data-dir", $DataDir, "--candidate", "v3be2", "--out", $V3, "--diagnostics", $V3Diag)
"start=$($start.ToString('o')) pid=pending output=$V3" | Set-Content -LiteralPath $Monitor
$proc = Start-Process -FilePath $Python -ArgumentList $args -RedirectStandardOutput $StdOut -RedirectStandardError $StdErr -PassThru
"pid=$($proc.Id)" | Add-Content -LiteralPath $Monitor
while (!$proc.HasExited) {
    $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($p) { "$(Get-Date -Format o) status=running cpu_s=$([math]::Round($p.CPU,2)) rss_mb=$([math]::Round($p.WorkingSet64/1MB,1))" | Add-Content -LiteralPath $Monitor }
    Start-Sleep -Seconds 15
    $proc.Refresh()
}
$proc.WaitForExit()
$proc.Refresh()
$exitCode = [int]$proc.ExitCode
$end = Get-Date
"end=$($end.ToString('o')) exit_code=$exitCode" | Add-Content -LiteralPath $Monitor
if ($exitCode -ne 0) { throw "Inference exited with code $exitCode. See $StdErr" }
Require-File $V3

& $Python $Validator $V3 | Tee-Object -FilePath (Join-Path $RunDir "v3_validator.log")
if ($LASTEXITCODE -ne 0) { throw "V3 submission validation failed." }
$v3Hash = (Get-FileHash $V3 -Algorithm SHA256).Hash.ToLowerInvariant()
$refHash = (Get-FileHash $preserved -Algorithm SHA256).Hash.ToLowerInvariant()
if ($refHash -ne "46e9dc60762d4812d5ae6d7b23d71e54b48fb73c514b0e98349c80f9c1511bfe") { Write-Warning "Preserved V3 checksum differs from the documented reference." }
$v3Match = $v3Hash -eq $refHash

$v4Hash = $null; $v4Match = $null
if (!$OnlyV3) {
    Require-File $ApplyV4; Require-File $preservedV4
    & $Python $ApplyV4 --input $V3 --output $V4 --seconds 1.5 | Tee-Object -FilePath (Join-Path $RunDir "v4_postprocess.log")
    if ($LASTEXITCODE -ne 0) { throw "V4 postprocessing failed." }
    Require-File $V4
    & $Python $Validator $V4 | Tee-Object -FilePath (Join-Path $RunDir "v4_validator.log")
    if ($LASTEXITCODE -ne 0) { throw "V4 submission validation failed." }
    $v4Hash = (Get-FileHash $V4 -Algorithm SHA256).Hash.ToLowerInvariant()
    $v4Match = $v4Hash -eq ((Get-FileHash $preservedV4 -Algorithm SHA256).Hash.ToLowerInvariant())
}

$result = [ordered]@{ start=$start.ToString('o'); end=$end.ToString('o'); exit_code=$exitCode; v3_output=$V3; v3_sha256=$v3Hash; v3_matches_preserved=$v3Match; v4_output=$V4; v4_sha256=$v4Hash; v4_matches_preserved=$v4Match; stdout=$StdOut; stderr=$StdErr; monitor=$Monitor }
$result | ConvertTo-Json | Set-Content -LiteralPath $Report
if (!$v3Match) { throw "V3 completed and validated, but does not match the preserved candidate. See $Report" }
if (!$OnlyV3 -and !$v4Match) { throw "V4 completed and validated, but does not match the preserved candidate. See $Report" }
Write-Host "REPRODUCTION PASS: checksums match preserved candidate(s). Report: $Report"
