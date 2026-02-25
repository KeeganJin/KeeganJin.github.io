# KeeganJin Blog

Personal blog based on GitHub Pages + Jekyll, with Markdown-first writing workflow.

## Daily workflow

1. Create a new post template:

```powershell
.\scripts\new-post.ps1 -Title "My New Post"
```

2. Edit the generated file in `_posts/`.

3. Publish to GitHub (commit + push):

```powershell
.\scripts\publish.ps1
```

Optional custom commit message:

```powershell
.\scripts\publish.ps1 -Message "blog: update post about xxx"
```

## Post file format

Post files are in `_posts/` and use:

`YYYY-MM-DD-title-slug.md`

Front Matter example:

```md
---
layout: post
title: "Article Title"
date: 2026-02-25 10:00:00 +0800
categories: [life, notes]
tags: [markdown, blog]
excerpt: "Optional summary"
---
```

## Local preview (optional)

```bash
bundle install
bundle exec jekyll serve
```

Then open `http://127.0.0.1:4000`.
