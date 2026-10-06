# Run in your own PowerShell: & C:\Projects\EchoRender\tool\fix_large_files.ps1
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
function Invoke-RepoGit {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    & git -C $repoPath @GitArgs
    if ($LASTEXITCODE -ne 0) { throw "Git failed: $($GitArgs -join ' ')" }
}
$dirty = Invoke-RepoGit status --porcelain -- . ':(exclude)tool/fix_large_files.ps1'
if ($dirty) { throw 'Working tree must be clean. Commit or stash your changes first.' }
$currentBranch = Invoke-RepoGit branch --show-current
if ($currentBranch -ne 'main') { throw 'Switch to main before running this script.' }
Invoke-RepoGit fetch origin
$base = Invoke-RepoGit rev-parse origin/main
if ($base.Trim() -ne '9a6c069345f1d64712c22dc617668eccd4c50c12') {
    throw 'Remote main changed since inspection. Stop and review before rewriting local commits.'
}
Invoke-RepoGit merge-base --is-ancestor origin/main HEAD
$backupName = 'backup/before-weight-cleanup-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
Invoke-RepoGit branch $backupName HEAD
Write-Host "Original history backed up as $backupName"
# Soft reset preserves all files and stages the combined changes.
Invoke-RepoGit reset --soft origin/main
$trackedWeights = @(Invoke-RepoGit ls-files -- '*.safetensors')
foreach ($weightPath in $trackedWeights) {
    Invoke-RepoGit rm --cached -- $weightPath
}
$ignorePath = Join-Path $repoPath '.gitignore'
$ignoreText = [System.IO.File]::ReadAllText($ignorePath)
if ($ignoreText -notmatch '(?m)^\*\.safetensors\s*$') {
    [System.IO.File]::AppendAllText($ignorePath, "`n*.safetensors`n", (New-Object System.Text.UTF8Encoding($false)))
}
Invoke-RepoGit add -- .gitignore tool/fix_large_files.ps1
Invoke-RepoGit commit -m 'Add configurable student and teacher feature tools; exclude local weights'
$remainingWeights = Invoke-RepoGit ls-files -- '*.safetensors'
if ($remainingWeights) { throw 'Weights are still tracked; refusing to push.' }
# Push only main, never the backup branch that still holds the large files.
Invoke-RepoGit push origin main
Write-Host 'Done. Local weights preserved; remote main contains code only.'
