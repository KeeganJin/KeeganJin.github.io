param(
  [Parameter(Mandatory = $true)]
  [string]$Title,
  [string[]]$Categories = @("notes"),
  [string[]]$Tags = @("markdown")
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

$categoryValue = if ($Categories.Length -gt 0) { ($Categories -join ", ") } else { "notes" }
$tagValue = if ($Tags.Length -gt 0) { ($Tags -join ", ") } else { "markdown" }

$content = @"
---
layout: post
title: "$Title"
date: $timePart +0800
categories: [$categoryValue]
tags: [$tagValue]
excerpt: ""
---

Write your post here.
"@

Set-Content -Path $filePath -Value $content -Encoding UTF8
Write-Host "Created: $filePath"
