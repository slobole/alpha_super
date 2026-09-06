// Rebuild from the repository root:
// pnpm dlx tailwindcss@3.4.17 -c scripts/review/dashboard_tailwind.config.cjs -i scripts/review/dashboard_tailwind.css -o alpha/live/dashboard_v3/static/utilities.css --minify
module.exports = {
  content: ['./alpha/live/dashboard_v3/templates/**/*.html', './alpha/live/dashboard_v3/static/*.js', './alpha/live/dashboard_v3/app.py'],
  theme: { extend: {
    colors: {
      ink: { 900: '#16181d', 700: '#334155', 500: '#64748b', 300: '#cbd5e1', 100: '#e2e8f0', 50: '#f8fafc' },
      ok: { DEFAULT: '#15803d', soft: '#f0fdf4' },
      warn: { DEFAULT: '#d97706', soft: '#fffbeb' },
      bad: { DEFAULT: '#dc2626', soft: '#fef2f2' },
      mute: { DEFAULT: '#64748b', soft: '#f8fafc' },
      accent: { DEFAULT: '#2563eb', soft: '#eff6ff' },
    },
    fontFamily: {
      sans: ['IBM Plex Sans', 'Segoe UI', 'system-ui', 'sans-serif'],
      mono: ['IBM Plex Mono', 'Cascadia Mono', 'Consolas', 'monospace'],
    },
  } },
};
