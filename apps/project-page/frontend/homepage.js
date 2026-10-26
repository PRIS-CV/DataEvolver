var zh = document.documentElement.lang === 'zh-Hans';
var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
var lightbox = document.getElementById('lightbox');
function openLightbox(img) {
  var preview = document.getElementById('lightbox-img');
  preview.src = img.src;
  preview.alt = img.alt;
  lightbox.showModal();
}
function closeLightbox() { lightbox.close(); }
lightbox.addEventListener('click', function (event) {
  if (event.target === lightbox) closeLightbox();
});
document.querySelectorAll('img[onclick]').forEach(function (img) {
  img.tabIndex = 0;
  img.setAttribute('role', 'button');
  img.setAttribute('aria-label', (zh ? '放大：' : 'Enlarge: ') + img.alt);
  img.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      openLightbox(img);
    }
  });
});

var menuToggle = document.querySelector('.nav-toggle');
var navLinks = document.getElementById('navLinks');
function closeMenu() {
  menuToggle.setAttribute('aria-expanded', 'false');
  navLinks.classList.remove('is-open');
}
menuToggle.addEventListener('click', function () {
  var open = menuToggle.getAttribute('aria-expanded') !== 'true';
  menuToggle.setAttribute('aria-expanded', String(open));
  navLinks.classList.toggle('is-open', open);
});
navLinks.addEventListener('click', function (event) {
  if (event.target.closest('a')) closeMenu();
});
document.addEventListener('keydown', function (event) {
  if (event.key === 'Escape' && menuToggle.getAttribute('aria-expanded') === 'true') {
    closeMenu();
    menuToggle.focus();
  }
});
window.matchMedia('(min-width: 1281px)').addEventListener('change', closeMenu);

function scrollGallery(dir) { scrollGalleryById('galleryScroll', dir); }
function scrollResearchGallery(dir) { scrollGalleryById('researchGalleryScroll', dir); }
function scrollGalleryById(id, dir) {
  var gallery = document.getElementById(id);
  gallery.scrollBy({ left: dir * gallery.clientWidth, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
}
[['galleryScroll', 'galleryIndicator'], ['researchGalleryScroll', 'researchGalleryIndicator']].forEach(function (ids) {
  var gallery = document.getElementById(ids[0]);
  var indicator = document.getElementById(ids[1]);
  function updateDots() {
    var max = gallery.scrollWidth - gallery.clientWidth;
    var page = Math.round(gallery.scrollLeft / gallery.clientWidth);
    if (gallery.scrollLeft >= max - 2) page = indicator.children.length - 1;
    Array.from(indicator.children).forEach(function (dot, i) {
      dot.classList.toggle('active', i === page);
      dot.setAttribute('aria-current', i === page ? 'true' : 'false');
    });
  }
  function buildDots() {
    var pages = Math.ceil((gallery.scrollWidth - gallery.clientWidth) / gallery.clientWidth) + 1;
    indicator.replaceChildren();
    for (var i = 0; i < pages; i++) {
      var dot = document.createElement('button');
      dot.type = 'button';
      dot.className = 'dot';
      dot.dataset.page = i;
      dot.setAttribute('aria-label', (zh ? '图库页 ' : 'Gallery page ') + (i + 1));
      dot.addEventListener('click', function (event) {
        gallery.scrollTo({ left: Number(event.currentTarget.dataset.page) * gallery.clientWidth, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
      });
      indicator.appendChild(dot);
    }
    updateDots();
  }
  gallery.tabIndex = 0;
  gallery.setAttribute('role', 'region');
  gallery.setAttribute('aria-label', zh ? '渲染图库，可用左右方向键浏览。' : 'Rendering gallery. Use left and right arrow keys to browse.');
  gallery.addEventListener('keydown', function (event) {
    if (event.target === gallery && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
      event.preventDefault();
      scrollGalleryById(ids[0], event.key === 'ArrowLeft' ? -1 : 1);
    }
  });
  gallery.addEventListener('scroll', updateDots, { passive: true });
  window.addEventListener('resize', buildDots);
  buildDots();
});

// Subpage links stay ordinary links; only local sections participate in reading position.
var sectionLinks = Array.from(navLinks.querySelectorAll('a[href^="#"]'));
var readingSections = sectionLinks.map(function (link) { return document.querySelector(link.getAttribute('href')); });
if ('IntersectionObserver' in window) {
  var locationObserver = new IntersectionObserver(function () {
    var current = readingSections.filter(function (section) { return section.getBoundingClientRect().top <= window.innerHeight * 0.4; }).pop();
    sectionLinks.forEach(function (link) {
      if (current && link.getAttribute('href') === '#' + current.id) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }, { rootMargin: '-15% 0px -60% 0px', threshold: 0 });
  readingSections.forEach(function (section) { locationObserver.observe(section); });
}

var progress = document.querySelector('.nav-progress');
function updateProgress() {
  var max = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.transform = 'scaleX(' + (max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0) + ')';
}
window.addEventListener('scroll', updateProgress, { passive: true });
window.addEventListener('resize', updateProgress);
updateProgress();

var selectCitation = document.querySelector('.select-citation');
var citationStatus = document.getElementById('citation-status');
var selectionReset;
selectCitation.addEventListener('click', function () {
  clearTimeout(selectionReset);
  var selection = window.getSelection();
  var range = document.createRange();
  range.selectNodeContents(document.querySelector('#citation pre'));
  selection.removeAllRanges();
  selection.addRange(range);
  selectCitation.classList.add('is-selected');
  selectCitation.querySelector('.citation-label').textContent = zh ? '已选中' : 'Selected';
  citationStatus.textContent = zh ? '已选中 BibTeX。按 Ctrl/Cmd + C 复制。' : 'BibTeX selected. Press Ctrl/Cmd + C to copy.';
  selectionReset = setTimeout(function () {
    selectCitation.classList.remove('is-selected');
    selectCitation.querySelector('.citation-label').textContent = zh ? '选中 BibTeX' : 'Select BibTeX';
    citationStatus.textContent = '';
  }, 4000);
});

var themeToggle = document.querySelector('.theme-toggle');
function syncThemeToggle() {
  themeToggle.setAttribute('aria-checked', String(document.documentElement.getAttribute('data-theme') === 'light'));
}
syncThemeToggle();
themeToggle.addEventListener('click', function () {
  var next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('de-theme', next); } catch (_) { /* Storage is optional. */ }
  document.cookie = 'de-theme=' + next + ';path=/;max-age=31536000;samesite=lax';
  syncThemeToggle();
});
