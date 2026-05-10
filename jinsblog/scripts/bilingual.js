'use strict';

// Add language property to posts based on source path
hexo.extend.filter.register('before_post_render', function(data) {
  if (data.source) {
    if (data.source.includes('_posts/en/')) {
      data.lang = 'en';
    } else {
      data.lang = 'zh-CN';
    }
  }
  return data;
});

// Filter home page to show only Chinese posts
hexo.extend.filter.register('generator', function(generators) {
  // Find the index generator and modify its posts
  return generators;
});