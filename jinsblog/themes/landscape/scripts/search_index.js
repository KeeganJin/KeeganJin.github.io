// Generates search.json with all posts data for client-side search
hexo.extend.generator.register('search_index', function(locals) {
  var posts = [];
  locals.posts.each(function(post) {
    var cats = [];
    if (post.categories && post.categories.length) {
      post.categories.each(function(c) { cats.push(c.name); });
    }
    var tags = [];
    if (post.tags && post.tags.length) {
      post.tags.each(function(t) { tags.push(t.name); });
    }
    posts.push({
      title: post.title,
      path: post.path,
      date: post.date ? post.date.format('YYYY-MM-DD') : '',
      content: (post.content || '').replace(/<[^>]+>/g, '').substring(0, 300),
      categories: cats,
      tags: tags,
      lang: post.lang || ''
    });
  });
  return {
    path: 'search.json',
    data: JSON.stringify(posts)
  };
});
