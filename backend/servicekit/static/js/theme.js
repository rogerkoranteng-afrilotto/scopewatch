// Dark / light with a remembered override. System preference is the default.

const KEY = 'opencv26-theme';
const ORDER = ['', 'light', 'dark'];        // '' means: follow the system
const LABEL = { '': 'Theme: auto', light: 'Theme: light', dark: 'Theme: dark' };

function read() {
  try { return localStorage.getItem(KEY) ?? ''; } catch { return ''; }
}

function write(value) {
  try { value ? localStorage.setItem(KEY, value) : localStorage.removeItem(KEY); }
  catch { /* private window: the theme just will not persist */ }
}

export function initTheme(button) {
  let current = read();
  const apply = () => {
    document.documentElement.dataset.theme = current;
    if (button) button.textContent = LABEL[current];
  };
  apply();
  button?.addEventListener('click', () => {
    current = ORDER[(ORDER.indexOf(current) + 1) % ORDER.length];
    write(current);
    apply();
  });
}
