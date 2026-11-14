/* 站内搜索（2026-11-14）：读取 /index.json，前端匹配标题/作者/标签/摘要 */
(function () {
  var input = document.getElementById('q');
  var box = document.getElementById('results');
  var hint = document.getElementById('hint');
  if (!input || !box) return;
  var data = null, loaded = false;
  function load() {
    if (loaded) return;
    loaded = true;
    fetch('/xliu-group/zh/index.json').then(function (r) { return r.json(); }).then(function (d) {
      data = d; hint.textContent = '共 ' + d.length + ' 篇，可检索标题、作者、标签与摘要';
      run();
    }).catch(function () { hint.textContent = '索引加载失败，请刷新重试'; });
  }
  function run() {
    if (!data) return;
    var q = input.value.trim().toLowerCase();
    if (!q) { box.innerHTML = ''; return; }
    var hits = data.filter(function (x) {
      return [x.title, x.author, x.tags, x.text].join(' ').toLowerCase().indexOf(q) >= 0;
    }).slice(0, 30);
    if (!hits.length) { box.innerHTML = '<p>没有匹配的条目。</p>'; return; }
    box.innerHTML = '<ul class="search-results">' + hits.map(function (x) {
      return '<li><a href="' + x.url + '">' + x.title + '</a> <small>' + x.date +
        (x.author ? ' · ' + x.author : '') + (x.tags ? ' · ' + x.tags : '') + '</small></li>';
    }).join('') + '</ul>';
  }
  input.addEventListener('focus', load);
  input.addEventListener('input', function () { load(); run(); });
})();
