param(
  [string]$Title,
  [string[]]$Tags = @(),
  [string]$Excerpt = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function To-Slug([string]$InputText) {
  $lower = $InputText.ToLowerInvariant()
  $slug = $lower -replace "[^a-z0-9\s-]", ""
  $slug = $slug -replace "\s+", "-"
  $slug = $slug -replace "-{2,}", "-"
  $slug = $slug.Trim("-")
  if ([string]::IsNullOrWhiteSpace($slug)) {
    $slug = "new-post"
  }
  return $slug
}

function Read-TemplateConfig {
  $templateConfigPath = Join-Path $PSScriptRoot "..\templates\template.json"
  if (-not (Test-Path $templateConfigPath)) {
    throw "Template config file not found: $templateConfigPath"
  }
  $jsonContent = Get-Content -Path $templateConfigPath -Encoding UTF8
  return $jsonContent | ConvertFrom-Json
}

function Select-Module($config) {
  Write-Host "请选择模块："
  for ($i = 0; $i -lt $config.modules.Length; $i++) {
    Write-Host "$($i + 1). $($config.modules[$i].name)"
  }
  Write-Host "0. 退出"
  
  do {
    $choice = Read-Host "请输入选项编号"
    $moduleIndex = [int]$choice - 1
  } while ($moduleIndex -lt -1 -or $moduleIndex -ge $config.modules.Length)
  
  if ($moduleIndex -eq -1) {
    return $null
  }
  
  return $config.modules[$moduleIndex]
}

function Select-Submodule($module) {
  Write-Host "请选择子模块："
  for ($i = 0; $i -lt $module.submodules.Length; $i++) {
    Write-Host "$($i + 1). $($module.submodules[$i].name)"
  }
  Write-Host "0. 退出"
  
  do {
    $choice = Read-Host "请输入选项编号"
    $submoduleIndex = [int]$choice - 1
  } while ($submoduleIndex -lt -1 -or $submoduleIndex -ge $module.submodules.Length)
  
  if ($submoduleIndex -eq -1) {
    return $null
  }
  
  return $module.submodules[$submoduleIndex]
}

function Get-TemplateContent($templateName) {
  $templatePath = Join-Path $PSScriptRoot "..\templates\$templateName"
  if (-not (Test-Path $templatePath)) {
    throw "Template file not found: $templatePath"
  }
  return Get-Content -Path $templatePath -Encoding UTF8
}

function Replace-TemplateVariables($content, $title, $date, $tags, $excerpt) {
  $tagsString = if ($tags.Length -gt 0) { ($tags -join ", ") } else { "markdown" }
  $content = $content -replace "{{title}}", $title
  $content = $content -replace "{{date}}", $date
  $content = $content -replace "{{tags}}", $tagsString
  $content = $content -replace "{{excerpt}}", $excerpt
  return $content
}

# 主逻辑
$config = Read-TemplateConfig

$module = Select-Module $config
if ($null -eq $module) {
  Write-Host "退出脚本"
  exit 0
}

$submodule = Select-Submodule $module
if ($null -eq $submodule) {
  Write-Host "退出脚本"
  exit 0
}

# 如果没有提供标题，提示用户输入
if ([string]::IsNullOrWhiteSpace($Title)) {
  $Title = Read-Host "请输入帖子标题"
}

# 如果没有提供标签，提示用户输入
if ($Tags.Length -eq 0) {
  $tagsInput = Read-Host "请输入标签（用逗号分隔）"
  if (-not [string]::IsNullOrWhiteSpace($tagsInput)) {
    $Tags = $tagsInput -split "," | ForEach-Object { $_.Trim() }
  }
}

# 如果没有提供摘要，提示用户输入
if ([string]::IsNullOrWhiteSpace($Excerpt)) {
  $Excerpt = Read-Host "请输入帖子摘要"
}

$date = Get-Date
$datePart = $date.ToString("yyyy-MM-dd")
$timePart = $date.ToString("yyyy-MM-dd HH:mm:ss")
$slug = To-Slug $Title
$fileName = "$datePart-$slug.md"
$postsDir = Join-Path $PSScriptRoot "..\_posts"
$filePath = Join-Path $postsDir $fileName

if (Test-Path $filePath) {
  throw "Post already exists: $filePath"
}

# 获取模板内容并替换变量
$templateContent = Get-TemplateContent $submodule.template
$content = Replace-TemplateVariables $templateContent $Title $timePart $Tags $Excerpt

# 写入文件
Set-Content -Path $filePath -Value $content -Encoding UTF8
Write-Host "Created: $filePath"
Write-Host "模块: $($module.name)"
Write-Host "子模块: $($submodule.name)"
