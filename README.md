# KeeganJin Blog

Hexo v8 博客，中英双语。GitHub Pages 部署，只有一个 `main` 分支。

## 目录结构

```
├── docs/              # Hexo 构建输出（GitHub Pages 直接 serve 这个目录）
├── jinsblog/          # Hexo 源码
│   ├── source/_posts/
│   │   ├── zh-CN/     # 中文文章
│   │   │   ├── 技术文档/
│   │   │   └── 随笔/
│   │   └── en/        # 英文文章
│   │       ├── tech-docs/
│   │       └── essays/
│   └── _config.yml    # Hexo 配置
└── README.md
```

## 写新文章

```bash
# 中文技术文章
npx hexo new post --path zh-CN/技术文档/你的文章标题 "你的文章标题"
# 中文随笔
npx hexo new post --path zh-CN/随笔/你的文章标题 "你的文章标题"
# 英文技术文章
npx hexo new post --path en/tech-docs/your-title "Your Title"
# 英文随笔
npx hexo new post --path en/essays/your-title "Your Title"
```

或者直接在对应目录下手动创建 `.md` 文件，添加 front-matter：

```markdown
---
title: 文章标题
date: 2026-05-10
tags: [标签1, 标签2]
---
文章内容。
```

## 更新博客

```bash
# 1. 构建
cd jinsblog && npx hexo generate

# 2. 提交并推送
cd .. && git add docs/ && git commit -m "update" && git push
```

等 GitHub Pages 部署完成（1-2 分钟），网站自动更新。

## 本地预览

```bash
cd jinsblog && npx hexo server
```

访问 http://localhost:4000。
