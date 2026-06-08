<#
.SYNOPSIS
    One-shot reproduction script for the Adversarial Skill File Injection study.
    Pulls a balanced set of Ollama models (<= 16 GB VRAM), then runs the hardened
    harness and records all results automatically.

.DESCRIPTION
    Run from anywhere:  powershell -ExecutionPolicy Bypass -File .\run_experiment.ps1
    or from the poc/ folder:  .\run_experiment.ps1

    Everything is logged to poc/logs/run_experiment_<timestamp>.log
    Results are written to poc/logs/:
        hardened_results_<ts>.jsonl   (one row per model x skill x case x rep)
        hardened_summary_<ts>.json    (per-model mean +/- SD, CoT-leak, provenance)

    Idempotent: models already pulled are skipped quickly. Safe to re-run.

.PARAMETER Reps          Repetitions per test case (default 5).
.PARAMETER Temperature   Decoding temperature (default 0.7).
.PARAMETER Judge         Optional Ollama model id for a 2nd automated classifier.
.PARAMETER SkipPull      Skip the download phase (use already-pulled models).
.PARAMETER DryRun        Run a single rep (quick end-to-end validation).
#>
param(
    [int]    $Reps        = 5,
    [double] $Temperature = 0.7,
    [int]    $MaxTokens   = 4096,   # generous budget so even verbose reasoning models
                                    # (e.g. Phi-4 Reasoning Plus) finish thinking AND answer
                                    # (512 truncates badly; 2048 still clips the worst case)
    [string] $Judge       = "",
    [switch] $SkipPull,
    [switch] $DryRun,
    [switch] $Resume        # skip models already complete (>=95 recs) in logs/hardened_results_*.jsonl
)

$ErrorActionPreference = "Continue"
$PocDir     = $PSScriptRoot
$HarnessDir = Join-Path $PocDir "harness"
$LogsDir    = Join-Path $PocDir "logs"
$Harness    = Join-Path $HarnessDir "hardened_test.py"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

$ts  = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $LogsDir "run_experiment_$ts.log"
Start-Transcript -Path $log -Append | Out-Null

function Say($m) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) }

if ($DryRun) { $Reps = 1 }

# --- Balanced model plan (all <= ~14 GB at Q4; fits a 16 GB GPU) -------------
# category: "reasoning" (visible chain-of-thought) or "standard"
# 'tags' lists candidate Ollama tags in preference order; the first one that
# pulls/exists (and is not already used by another entry) is selected. This
# makes the run self-healing if a primary tag is absent from the registry.
$Plan = @(
    @{ name = "DeepSeek-R1 14B";     category = "reasoning"; tags = @("deepseek-r1:14b") }
    @{ name = "DeepSeek-R1 8B";      category = "reasoning"; tags = @("deepseek-r1:8b","deepseek-r1:7b") }
    @{ name = "Qwen3 14B";           category = "reasoning"; tags = @("qwen3:14b") }
    @{ name = "Qwen3 8B";            category = "reasoning"; tags = @("qwen3:8b","qwen3:4b") }
    @{ name = "Phi-4 Reasoning";     category = "reasoning"; tags = @("phi4-reasoning:14b","phi4-reasoning:plus","phi4-reasoning:latest") }
    @{ name = "EXAONE Deep 7.8B";    category = "reasoning"; tags = @("exaone-deep:7.8b","exaone-deep:latest") }
    @{ name = "OpenThinker 7B";      category = "reasoning"; tags = @("openthinker:7b","openthinker:latest") }
    @{ name = "Llama 3.1 8B";        category = "standard";  tags = @("llama3.1:8b","llama3.1:latest") }
    @{ name = "Mistral Nemo 12B";    category = "standard";  tags = @("mistral-nemo:12b","mistral-nemo:latest") }
    @{ name = "Gemma 3 12B";         category = "standard";  tags = @("gemma3:12b") }
    @{ name = "Qwen2.5 14B";         category = "standard";  tags = @("qwen2.5:14b","qwen2.5:7b") }
    @{ name = "Phi-4 14B";           category = "standard";  tags = @("phi4:14b","phi4:latest") }
    @{ name = "GPT-OSS 20B";         category = "standard";  tags = @("gpt-oss:20b","gpt-oss:latest") }
    @{ name = "Gemma 4";             category = "standard";  tags = @("gemma4:latest") }
)

Say "=== Adversarial Skill File Injection - reproduction run ==="
Say ("Reps=$Reps  Temperature=$Temperature  MaxTokens=$MaxTokens  Judge='$Judge'  DryRun=$DryRun")

# --- Preflight: ollama + python + deps ---------------------------------------
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Say "ERROR: 'ollama' not found on PATH. Install/start Ollama and re-run."; Stop-Transcript | Out-Null; exit 1
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Say "ERROR: 'python' not found on PATH."; Stop-Transcript | Out-Null; exit 1
}
# ensure Ollama server is up (pull/list auto-start it, but check explicitly)
ollama list *> $null
if ($LASTEXITCODE -ne 0) {
    Say "Starting Ollama server..."; Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}
Say "Checking Python dependencies (openai, pyyaml)..."
python -c "import openai, yaml" *> $null
if ($LASTEXITCODE -ne 0) { Say "Installing openai + pyyaml..."; python -m pip install --quiet openai pyyaml }

# --- Resolve each model: try candidate tags in order, with dedup -------------
# (One loop handles both pulling and presence-checking. -SkipPull only checks.)
if ($SkipPull) { Say "--- SkipPull set: checking existing models only (no downloads) ---" }
else           { Say "--- Pulling/resolving models (candidate tags tried in order) ---" }
$present = @{}
$used = New-Object System.Collections.Generic.HashSet[string]
foreach ($m in $Plan) {
    $resolved = $null
    foreach ($cand in $m.tags) {
        if ($used.Contains($cand)) { continue }
        if ($SkipPull) {
            ollama show $cand *> $null
            $okp = ($LASTEXITCODE -eq 0)
        } else {
            Say ("  pull {0,-20} {1}" -f $m.name, $cand)
            ollama pull $cand
            $okp = ($LASTEXITCODE -eq 0)
            if ($okp) { ollama show $cand *> $null; $okp = ($LASTEXITCODE -eq 0) }
        }
        if ($okp) { $resolved = $cand; break }
        else { Say ("    - candidate unavailable: {0}" -f $cand) }
    }
    if ($resolved) {
        $present[$m.name] = @{ id = $resolved; category = $m.category }
        [void]$used.Add($resolved)
        Say ("  OK      {0,-20} -> {1}  ({2})" -f $m.name, $resolved, $m.category)
    } else {
        Say ("  MISSING {0,-20} (no candidate available; excluded)" -f $m.name)
    }
}
# Expected records per model = 19 conditions x 3 problems x reps (matrix-aware,
# so -Resume works for the multi-problem run, not just the old 19-case matrix).
$ExpectedPerModel = 19 * 3 * $Reps

# --- Resume: drop models already completed in a previous run's JSONL --------
if ($Resume) {
    Say ("--- Resume: scanning existing results for completed models (>= {0} recs) ---" -f $ExpectedPerModel)
    $counts = @{}
    Get-ChildItem -Path (Join-Path $LogsDir "hardened_results_*.jsonl") -ErrorAction SilentlyContinue | ForEach-Object {
        foreach ($line in [System.IO.File]::ReadLines($_.FullName)) {
            if ($line -match '"model"\s*:\s*"([^"]+)"') {
                $n = $Matches[1]
                if ($counts.ContainsKey($n)) { $counts[$n]++ } else { $counts[$n] = 1 }
            }
        }
    }
    foreach ($n in @($present.Keys)) {
        if ($counts.ContainsKey($n) -and $counts[$n] -ge $ExpectedPerModel) {
            $present.Remove($n)
            Say ("  complete ({0} recs), skipping: {1}" -f $counts[$n], $n)
        }
    }
    Say ("  models remaining to run: {0}" -f $present.Count)
    if ($present.Count -eq 0) { Say "Nothing left to run - all models complete."; Stop-Transcript | Out-Null; exit 0 }
}

if (-not $Resume -and $present.Count -lt 2) {
    Say "ERROR: fewer than 2 models available; aborting."; Stop-Transcript | Out-Null; exit 1
}
$nReason = ($present.Values | Where-Object { $_.category -eq "reasoning" }).Count
$nStd    = ($present.Values | Where-Object { $_.category -eq "standard"  }).Count
Say ("Models to run: {0}  (reasoning={1}, standard={2})" -f $present.Count, $nReason, $nStd)
if ($nReason -lt 1) { Say "WARNING: no reasoning models present - CoT-leakage finding cannot be reproduced." }

# --- Write the harness model-config -----------------------------------------
$cfgPath = Join-Path $HarnessDir "models_config_$ts.json"
# Write UTF-8 WITHOUT BOM (Windows PowerShell 5.1 'Out-File -Encoding utf8' adds a
# BOM that breaks Python's json.load); use .NET to guarantee no BOM.
[System.IO.File]::WriteAllText($cfgPath, ($present | ConvertTo-Json -Depth 5), (New-Object System.Text.UTF8Encoding($false)))
Say "Model config written: $cfgPath"

# --- Run the hardened harness ------------------------------------------------
Say "--- Running hardened harness ($($present.Count) models x 6 skill files x 3 problems x 19 conditions x reps=$Reps) ---"
$argsList = @(
    $Harness,
    "--backend", "ollama",
    "--reps", "$Reps",
    "--temperature", "$Temperature",
    "--max-tokens", "$MaxTokens",
    "--models-config", $cfgPath,
    "--out-dir", $LogsDir
)
if ($Judge -ne "") { $argsList += @("--judge", $Judge) }

Push-Location $HarnessDir
python @argsList
$rc = $LASTEXITCODE
Pop-Location

if ($rc -eq 0) {
    Say "=== DONE. Results in: $LogsDir ==="
    Say "  - hardened_results_*.jsonl   (raw records)"
    Say "  - hardened_summary_*.json    (aggregates + provenance)"
    Say "Report this back and the manuscript tables will be regenerated from the summary."
} else {
    Say "Harness exited with code $rc - check the log above: $log"
}

Stop-Transcript | Out-Null
