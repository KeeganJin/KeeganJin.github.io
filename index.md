---
layout: default
title: 首页
---

<section class="intro">
  <h1>你好，我是 Keegan。</h1>
  <p>{{ site.description }}</p>
  <p>这个站点以 Markdown 写作为核心，记录技术、生活和长期项目。</p>
</section>

## 最新文章

<ul class="post-list">
  {% for post in site.posts limit: 10 %}
  <li>
    <a href="{{ post.url | relative_url }}"><strong>{{ post.title }}</strong></a>
    <div class="meta">{{ post.date | date: "%Y-%m-%d" }}</div>
    {% if post.excerpt %}
    <div>{{ post.excerpt | strip_html | truncate: 120 }}</div>
    {% endif %}
  </li>
  {% endfor %}
</ul>
