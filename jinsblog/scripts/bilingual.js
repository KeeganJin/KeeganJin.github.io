'use strict';

// Add language property to all posts based on source path
hexo.extend.filter.register('after_post_render', function(data) {
  if (data.source && data.source.includes('_posts/en/')) {
    data.lang = 'en';
  } else {
    data.lang = 'zh-CN';
  }
  return data;
});

// Filter home page posts to Chinese only
hexo.extend.filter.register('template_locals', function(locals) {
  if (locals.page) {
    // Check if this is the main index page
    const isHomeIndex = locals.page._path === 'index.html' ||
                        locals.page.path === 'index.html' ||
                        locals.page.__index;

    if (isHomeIndex && locals.page.posts) {
      try {
        // Filter to only Chinese posts
        const filtered = locals.page.posts.filter(p => p.lang === 'zh-CN' || !p.lang);
        if (filtered && filtered.length > 0) {
          locals.page.posts = filtered;
        }
      } catch(e) {
        // Handle warehouse Model vs array
      }
    }
  }
  return locals;
});