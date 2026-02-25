param(
  [string]$Message
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Join-Path $PSScriptRoot ".."
Set-Location $repoRoot

$branch = (git branch --show-current).Trim()
if ([string]::IsNullOrWhiteSpace($branch)) {
  throw "Cannot determine current git branch."
}

git add .

$hasStaged = git diff --cached --name-only
if ([string]::IsNullOrWhiteSpace(($hasStaged | Out-String))) {
  Write-Host "No changes to commit."
  exit 0
}

if ([string]::IsNullOrWhiteSpace($Message)) {
  $Message = "blog: update $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
}

git commit -m $Message
git push origin $branch

Write-Host "Pushed to origin/$branch"
