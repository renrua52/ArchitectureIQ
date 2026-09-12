/* ============================================================
   engine.js —【引擎】幻灯片导航 + 演示体验
   ------------------------------------------------------------
   自动扫描所有 .slide, 生成圆点/目录, 接管翻页, 并提供:
     · 顶部进度条        (#progress, 可选)
     · 分步出现 fragment (幕内 .fragment, 翻页逐个点出)
     · 全屏切换          (#fsBtn 或按 F)
     · 演讲者备注        (幕内 .notes, 按 S 切换 #notesPanel)
     · 快捷键帮助        (按 ? 切换 #helpOverlay)
     · 主题循环          (按 T, 配合 themes.css)
   带 (可选) 的 DOM 不存在时自动跳过, 不报错。

   暴露到全局供动画脚本使用:
     window.slides  — NodeList   window.go(i) — 跳到第 i 幕(从 0 数)
   ============================================================ */
(function () {
  const slides = document.querySelectorAll('.slide');
  const $ = id => document.getElementById(id);
  const dotsContainer = $('dots'), tocContainer = $('toc'), counter = $('counter');
  const progress = $('progress'), notesPanel = $('notesPanel'), helpOverlay = $('helpOverlay');
  let current = 0, step = 0;   // step = 当前幕已点出的 fragment 数

  const frags = s => s.querySelectorAll('.fragment');

  slides.forEach((s, i) => {
    if (dotsContainer) {
      const dot = document.createElement('div');
      dot.className = 'dot'; dot.onclick = () => go(i);
      dotsContainer.appendChild(dot);
    }
    if (tocContainer) {
      const tocItem = document.createElement('a');
      tocItem.className = 'toc-item';
      tocItem.innerHTML = `<span class="num">${String(i).padStart(2, '0')}</span>${s.getAttribute('data-title') || 'Slide'}`;
      tocItem.onclick = () => go(i);
      tocContainer.appendChild(tocItem);
    }
  });

  // 跳到第 i 幕。revealFrags=true 时(通常是往回翻)直接展开本幕全部 fragment
  function go(i, revealFrags = false) {
    if (i < 0 || i >= slides.length) return;
    slides[current].classList.remove('active');
    slides[i].classList.add('active');
    slides[i].scrollTop = 0;
    current = i;
    const fs = frags(slides[i]);
    step = revealFrags ? fs.length : 0;
    fs.forEach((f, k) => f.classList.toggle('revealed', k < step));
    updateNav();
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([slides[i]]);
  }

  // 下一步: 先点 fragment, 点完才翻下一幕
  function next() {
    const fs = frags(slides[current]);
    if (step < fs.length) { fs[step].classList.add('revealed'); step++; updateNav(); }
    else go(current + 1);
  }
  // 上一步: 先收 fragment, 收完才回上一幕(并展开上一幕全部)
  function prev() {
    if (step > 0) { step--; frags(slides[current])[step].classList.remove('revealed'); updateNav(); }
    else go(current - 1, true);
  }

  function updateNav() {
    document.querySelectorAll('.dot').forEach((d, i) => d.classList.toggle('active', i === current));
    document.querySelectorAll('.toc-item').forEach((d, i) => d.classList.toggle('current', i === current));
    if (counter) counter.textContent = `${current + 1} / ${slides.length}`;
    if ($('prev')) $('prev').disabled = current === 0 && step === 0;
    if ($('next')) $('next').disabled = current === slides.length - 1 && step === frags(slides[current]).length;
    if (progress) progress.style.width = `${(current + 1) / slides.length * 100}%`;
    syncNotes();
  }

  // 演讲者备注: 把当前幕的 .notes 内容塞进面板
  function syncNotes() {
    if (!notesPanel) return;
    const note = slides[current].querySelector('.notes');
    notesPanel.innerHTML = '<div class="notes-head">演讲者备注 · Speaker notes</div>' +
      (note ? note.innerHTML : '<em style="color:var(--dim)">(本幕无备注)</em>');
  }

  /* —— 按钮与键盘 —— */
  if ($('prev')) $('prev').onclick = prev;
  if ($('next')) $('next').onclick = next;

  function toggleFullscreen() {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
    else document.exitFullscreen?.();
  }
  if ($('fsBtn')) $('fsBtn').onclick = toggleFullscreen;

  function isScrollableSlide(slide) {
    return slide.classList.contains('scrollable') || slide.dataset.scroll === 'true';
  }

  function canScroll(slide, direction) {
    if (!isScrollableSlide(slide)) return false;
    const top = slide.scrollTop;
    const max = slide.scrollHeight - slide.clientHeight;
    if (max <= 2) return false;
    return direction > 0 ? top < max - 2 : top > 2;
  }

  function scrollSlide(direction, paging = false) {
    const slide = slides[current];
    if (!canScroll(slide, direction)) return false;
    const distance = paging ? slide.clientHeight * 0.82 : Math.min(96, slide.clientHeight * 0.18);
    slide.scrollBy({ top: direction * distance, behavior: 'smooth' });
    return true;
  }

  const THEMES = ['', 'light', 'academic', 'warm', 'projector'];
  let themeIdx = 0;
  function cycleTheme() {
    themeIdx = (themeIdx + 1) % THEMES.length;
    const t = THEMES[themeIdx];
    if (t) document.documentElement.setAttribute('data-theme', t);
    else document.documentElement.removeAttribute('data-theme');
    window.refreshCanvasColors && window.refreshCanvasColors();
  }

  document.addEventListener('keydown', e => {
    if (e.key === '?' || (e.shiftKey && e.key === '/')) { helpOverlay && helpOverlay.classList.toggle('show'); return; }
    if (helpOverlay && helpOverlay.classList.contains('show') && e.key === 'Escape') { helpOverlay.classList.remove('show'); return; }
    if (['ArrowRight'].includes(e.key)) { e.preventDefault(); next(); }
    else if ([' ', 'PageDown'].includes(e.key)) {
      e.preventDefault();
      if (!scrollSlide(1, true)) next();
    }
    else if (e.key === 'ArrowDown') {
      if (scrollSlide(1)) e.preventDefault();
    }
    else if (e.key === 'ArrowUp') {
      if (scrollSlide(-1)) e.preventDefault();
    }
    else if (['ArrowLeft'].includes(e.key)) { e.preventDefault(); prev(); }
    else if (e.key === 'PageUp') {
      e.preventDefault();
      if (!scrollSlide(-1, true)) prev();
    }
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(slides.length - 1, true);
    else if (e.key === 'f' || e.key === 'F') toggleFullscreen();
    else if (e.key === 's' || e.key === 'S') notesPanel && notesPanel.classList.toggle('show');
    else if (e.key === 't' || e.key === 'T') cycleTheme();
    else if (e.key >= '0' && e.key <= '9') { const n = parseInt(e.key); if (n < slides.length) go(n); }
  });
  if (helpOverlay) helpOverlay.onclick = e => { if (e.target === helpOverlay) helpOverlay.classList.remove('show'); };

  go(0);

  // 暴露给动画脚本
  window.slides = slides;
  window.go = go;
})();
