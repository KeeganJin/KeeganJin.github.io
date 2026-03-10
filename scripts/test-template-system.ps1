Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# 测试模板配置文件是否存在
$templateConfigPath = Join-Path $PSScriptRoot "..\templates\template.json"
if (Test-Path $templateConfigPath) {
  Write-Host "模板配置文件存在: $templateConfigPath"
} else {
  Write-Host "模板配置文件不存在: $templateConfigPath"
  exit 1
}

# 测试模板文件是否存在
$templateFiles = @(
  "tech-frontend.md",
  "tech-backend.md",
  "tech-mobile.md",
  "tech-devops.md",
  "life-daily.md",
  "life-travel.md",
  "life-reading.md",
  "project-personal.md",
  "project-team.md"
)

foreach ($templateFile in $templateFiles) {
  $templatePath = Join-Path $PSScriptRoot "..\templates\$templateFile"
  if (Test-Path $templatePath) {
    Write-Host "模板文件存在: $templateFile"
  } else {
    Write-Host "模板文件不存在: $templateFile"
    exit 1
  }
}

# 测试脚本文件是否存在
$scriptPath = Join-Path $PSScriptRoot "new-post-with-template.ps1"
if (Test-Path $scriptPath) {
  Write-Host "脚本文件存在: new-post-with-template.ps1"
} else {
  Write-Host "脚本文件不存在: new-post-with-template.ps1"
  exit 1
}

Write-Host "模板系统测试完成，所有文件都存在！"
Write-Host "你可以通过以下命令使用模板系统创建新帖子："
Write-Host "  .\scripts\new-post-with-template.ps1"
Write-Host "  或者提供参数："
Write-Host '  .\scripts\new-post-with-template.ps1 -Title "你的帖子标题" -Tags @("标签1", "标签2") -Excerpt "帖子摘要"'
