---
layout: default
title: 归档
permalink: /archive/
---

# 文章归档

<ul class="post-list">
  {% for post in site.posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <div class="meta">{{ post.date | date: "%Y-%m-%d" }}</div>
  </li>
  {% endfor %}
</ul>
