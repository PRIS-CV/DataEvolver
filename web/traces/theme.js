(() => {
  let theme;
  try { theme = localStorage.getItem('de-theme'); } catch (_) { /* Storage optional. */ }
  if (!['light', 'dark'].includes(theme)) theme = matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
})();
