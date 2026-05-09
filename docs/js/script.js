(function($){
  // Search
  var $searchWrap = $('#search-form-wrap'),
    $searchInput = $('#local-search-input'),
    $searchResult = $('#local-search-result'),
    isSearchAnim = false,
    searchAnimDuration = 200,
    searchData = null,
    searchLoading = false;

  var startSearchAnim = function(){
    isSearchAnim = true;
  };

  var stopSearchAnim = function(callback){
    setTimeout(function(){
      isSearchAnim = false;
      callback && callback();
    }, searchAnimDuration);
  };

  // Load search index
  var loadSearchData = function() {
    if (searchData !== null) return;
    if (searchLoading) return;
    searchLoading = true;
    $.getJSON('/search.json', function(data) {
      searchData = data;
      searchLoading = false;
    }).fail(function() {
      searchLoading = false;
    });
  };

  // Perform local search
  var doSearch = function(query) {
    if (!searchData) return [];
    query = query.toLowerCase();
    var results = [];
    searchData.forEach(function(post) {
      var match = post.title.toLowerCase().indexOf(query) > -1 ||
                  post.content.toLowerCase().indexOf(query) > -1 ||
                  post.categories.join(' ').toLowerCase().indexOf(query) > -1 ||
                  post.tags.join(' ').toLowerCase().indexOf(query) > -1;
      if (match) {
        results.push(post);
      }
    });
    return results.slice(0, 20);
  };

  // Render search results
  var renderResults = function(results, query) {
    if (results.length === 0) {
      $searchResult.html('<div class="local-search-empty">' + (query ? 'No results' : 'Type to search') + '</div>');
    } else {
      var html = '<ul class="local-search-list">';
      results.forEach(function(post) {
        var title = post.title;
        var idx = title.toLowerCase().indexOf(query);
        if (idx > -1) {
          title = title.substring(0, idx) + '<em>' + title.substring(idx, idx + query.length) + '</em>' + title.substring(idx + query.length);
        }
        html += '<li><a href="' + post.path + '"><span class="local-search-title">' + title + '</span>' +
                '<span class="local-search-meta">' + post.categories.join(' / ') + (post.lang ? ' [' + post.lang + ']' : '') + '</span></a></li>';
      });
      html += '</ul>';
      $searchResult.html(html);
    }
  };

  $('.nav-search-btn').on('click', function(){
    if (isSearchAnim) return;
    startSearchAnim();
    $searchWrap.addClass('on');
    loadSearchData();
    stopSearchAnim(function(){
      $searchInput.focus();
    });
  });

  // Search as you type
  $searchInput.on('input', function() {
    var query = $(this).val().trim();
    if (query.length < 1) {
      $searchResult.empty();
      return;
    }
    if (searchData) {
      renderResults(doSearch(query), query);
    }
  });

  $searchInput.on('keydown', function(e) {
    if (e.key === 'Escape') {
      $searchInput.val('');
      $searchResult.empty();
      $searchInput.blur();
    }
  });

  $('.search-form-submit').on('click', function() {
    var query = $searchInput.val().trim();
    if (query.length > 0 && searchData) {
      renderResults(doSearch(query), query);
    }
  });

  $searchInput.on('blur', function(){
    // Don't close immediately so click on results works
    setTimeout(function() {
      if (!$searchResult.is(':hover')) {
        startSearchAnim();
        $searchWrap.removeClass('on');
        stopSearchAnim();
      }
    }, 200);
  });

  // Share
  $('body').on('click', function(){
    $('.article-share-box.on').removeClass('on');
  }).on('click', '.article-share-link', function(e){
    e.stopPropagation();

    var $this = $(this),
      url = $this.attr('data-url'),
      encodedUrl = encodeURIComponent(url),
      id = 'article-share-box-' + $this.attr('data-id'),
      title = $this.attr('data-title'),
      offset = $this.offset();

    if ($('#' + id).length){
      var box = $('#' + id);

      if (box.hasClass('on')){
        box.removeClass('on');
        return;
      }
    } else {
      var html = [
        '<div id="' + id + '" class="article-share-box">',
          '<input class="article-share-input" value="' + url + '">',
          '<div class="article-share-links">',
            '<a href="https://twitter.com/intent/tweet?text=' + encodeURIComponent(title) + '&url=' + encodedUrl + '" class="article-share-twitter" target="_blank" title="Twitter"><span class="fa fa-twitter"></span></a>',
            '<a href="https://www.facebook.com/sharer.php?u=' + encodedUrl + '" class="article-share-facebook" target="_blank" title="Facebook"><span class="fa fa-facebook"></span></a>',
            '<a href="http://pinterest.com/pin/create/button/?url=' + encodedUrl + '" class="article-share-pinterest" target="_blank" title="Pinterest"><span class="fa fa-pinterest"></span></a>',
            '<a href="https://www.linkedin.com/shareArticle?mini=true&url=' + encodedUrl + '" class="article-share-linkedin" target="_blank" title="LinkedIn"><span class="fa fa-linkedin"></span></a>',
          '</div>',
        '</div>'
      ].join('');

      var box = $(html);

      $('body').append(box);
    }

    $('.article-share-box.on').hide();

    box.css({
      top: offset.top + 25,
      left: offset.left
    }).addClass('on');
  }).on('click', '.article-share-box', function(e){
    e.stopPropagation();
  }).on('click', '.article-share-box-input', function(){
    $(this).select();
  }).on('click', '.article-share-box-link', function(e){
    e.preventDefault();
    e.stopPropagation();

    window.open(this.href, 'article-share-box-window-' + Date.now(), 'width=500,height=450');
  });

  // Caption
  $('.article-entry').each(function(i){
    $(this).find('img').each(function(){
      if ($(this).parent().hasClass('fancybox') || $(this).parent().is('a')) return;

      var alt = this.alt;

      if (alt) $(this).after('<span class="caption">' + alt + '</span>');

      $(this).wrap('<a href="' + this.src + '" data-fancybox=\"gallery\" data-caption="' + alt + '"></a>')
    });

    $(this).find('.fancybox').each(function(){
      $(this).attr('rel', 'article' + i);
    });
  });

  if ($.fancybox){
    $('.fancybox').fancybox();
  }

  // Mobile nav
  var $container = $('#container'),
    isMobileNavAnim = false,
    mobileNavAnimDuration = 200;

  var startMobileNavAnim = function(){
    isMobileNavAnim = true;
  };

  var stopMobileNavAnim = function(){
    setTimeout(function(){
      isMobileNavAnim = false;
    }, mobileNavAnimDuration);
  }

  $('#main-nav-toggle').on('click', function(){
    if (isMobileNavAnim) return;

    startMobileNavAnim();
    $container.toggleClass('mobile-nav-on');
    stopMobileNavAnim();
  });

  $('#wrap').on('click', function(){
    if (isMobileNavAnim || !$container.hasClass('mobile-nav-on')) return;

    $container.removeClass('mobile-nav-on');
  });

  // === Language Switcher ===
  $('#lang-toggle').on('click', function() {
    var currentLang = $('html').attr('lang') || 'zh-CN';
    // Check if post has translation link in front-matter
    var translationLink = $('meta[name="translation"]').attr('content');
    if (translationLink) {
      window.location.href = translationLink;
      return;
    }
    // Otherwise, try to switch language by swapping URL prefixes, fallback to home
    var path = window.location.pathname;
    var newPath = '/';
    if (currentLang === 'zh-CN' || currentLang === 'zh') {
      newPath = path.replace(/^\/zh-CN\//, '/en/');
      if (newPath === path) newPath = '/';
    } else {
      newPath = path.replace(/^\/en\//, '/zh-CN/');
      if (newPath === path) newPath = '/';
    }
    window.location.href = newPath;
  });

  // Update language label based on current page language
  (function() {
    var lang = $('html').attr('lang');
    if (lang) {
      $('#lang-label').text(lang === 'zh-CN' ? 'EN' : '中文');
    }
  })();

  // === Mobile Dropdown Toggle (tap to expand on touch devices) ===
  if ('ontouchstart' in window) {
    $('.dropdown-toggle').on('click', function(e) {
      e.preventDefault();
      $(this).parent('.dropdown-nav-item').toggleClass('open');
    });
  }

  // === Mobile Nav Dropdown Toggle ===
  $('.mobile-nav-toggle').on('click', function(e) {
    e.preventDefault();
    $(this).parent('.mobile-nav-dropdown').toggleClass('open');
  });

  // === Highlight current nav item ===
  (function() {
    var path = window.location.pathname;
    $('.main-nav-link').each(function() {
      var href = $(this).attr('href');
      if (href && path.indexOf(href) === 0 && href !== '/') {
        $(this).addClass('nav-active');
      }
    });
  })();
})(jQuery);