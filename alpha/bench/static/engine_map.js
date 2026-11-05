/* Strategy map, Bench's /map page. Ported from the Engine Map's template.html: same data, same arithmetic, same views.
   What changed for Bench: the data URL comes from the page (data-src on #emap), colours are Bench's (stage colours on the
   dots and badges, green up and red down on the heat, lines in ink), light theme only, a Charts switch on the tiles
   (growth of $1 or a rolling 3-year Sharpe), and the header is Bench's catalog head and stat band. window.EM stays for tests. */
(async () => {
const ROOT = document.getElementById('emap');
let D = window.__EM_DATA__;
if (!D) {
  const answer = await fetch(ROOT.dataset.src, { cache: 'no-store', signal: AbortSignal.timeout(30000) });
  if (!answer.ok) throw new Error(`data.json answered ${answer.status}`);
  D = await answer.json();
}
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const minus = (v, d) => { const s = Math.abs(v).toFixed(d); return (v < 0 && +s !== 0 ? '−' : '') + s; };
const pct = (v, d = 1) => v == null || Number.isNaN(v) ? '' : minus(v, d) + '%';
const num = (v) => v == null ? '' : minus(v, 2);
const signed = (v, d = 1) => (v > 0 ? '+' : '') + minus(v, d);
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* page works without it */ } } };

/* decode the daily data */
function bytes(s) { const b = atob(s); const u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; }
function decodeR(s) { const v = new Int16Array(bytes(s).buffer); const o = new Float64Array(v.length); for (let i = 0; i < v.length; i++) o[i] = v[i] === -32768 ? NaN : v[i] / 1e5; return o; }
const OFF = new Uint16Array(bytes(D.days).buffer);
const T0 = Date.parse(D.start + 'T00:00:00Z');
const DATES = Array.from(OFF, (d) => new Date(T0 + d * 864e5));
const ISO = DATES.map((d) => d.toISOString().slice(0, 10));
const YEAR = DATES.map((d) => d.getUTCFullYear());
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const shortLabel = (iso) => { const [y, m, d] = iso.split('-'); return `${MONTHS[+m - 1]} ${+d}, ${y}`; };
const dayLabel = (i) => shortLabel(ISO[i]);
const monthDay = (iso) => { const [, m, d] = iso.split('-'); return `${MONTHS[+m - 1]} ${+d}`; };
const spanLabel = (i, j) => (YEAR[i] === YEAR[j] ? monthDay(ISO[i]) : dayLabel(i)) + ` to ${dayLabel(j)}`;
const N = OFF.length, LAST = N - 1;
const idxOf = (iso) => { let lo = 0, hi = LAST; while (lo < hi) { const m = (lo + hi) >> 1; if (ISO[m] < iso) lo = m + 1; else hi = m; } return lo; };
const TB = decodeR(D.tbill), QQQ = decodeR(D.qqq);
const R = Object.fromEntries(Object.entries(D.r).map(([k, s]) => [k, decodeR(s)]));
const E = D.engines;
const byId = Object.fromEntries(E.map((e) => [e.id, e]));
const lastValid = (r) => { let i = LAST; while (i > 0 && Number.isNaN(r[i])) i--; return i; };
E.forEach((e) => { e.lastI = lastValid(R[e.id]); });
// an engine of a family the data does not list goes in an Other group, so it stays in every view
const listedFamily = (id) => D.families.some((f) => f.id === id);
E.forEach((e) => { if (!listedFamily(e.fam)) e.fam = 'other'; });
const FAM = D.families.concat(E.some((e) => e.fam === 'other') && !listedFamily('other') ? { id: 'other', name: 'Other' } : []);
// a stage the page does not know reads like PM READY: outlined, hollow on the maps, and it shows and counts
const chipClass = (s) => s === 'LIVE' || s === 'WIRED' ? s : 'PM';
const stageTip = (s) => (D.stageTips && D.stageTips[s]) || '';
// Bench's maturity colours: LIVE solid, WIRED green, PM READY an amber ring, CANDIDATE a dashed grey ring
const stageKey = (s) => s === 'LIVE' ? 'live' : s === 'WIRED' ? 'wired' : s === 'CANDIDATE' ? 'cand' : 'pm';
const BADGE = { live: 'live', wired: 'wired', pm: 'pm', cand: 'candidate' };
const STAGE_WORD = { LIVE: 'Live', WIRED: 'Wired', 'PM READY': 'PM ready', CANDIDATE: 'Candidate' };
const stageWord = (s) => STAGE_WORD[s] || s;
const chip = (s) => `<span class="maturity maturity-${BADGE[stageKey(s)]}"${tipAttr(stageTip(s))}>${esc(s)}</span>`;
// LIVE says for how long; the stage itself still counts and filters as LIVE
function chipWord(e) {
  if (!e.live) return e.stage;
  const days = (Date.parse(D.asOf) - Date.parse(e.live)) / 864e5, weeks = Math.max(1, Math.ceil(days / 7));
  return days < 0 ? `First trade ${monthDay(e.live)}` : days < 56 ? `LIVE ${weeks} week${weeks === 1 ? '' : 's'}` : `LIVE ${Math.round(days / 30.44)} months`;
}
function chipFor(e) {
  if (!e.live) return chip(e.stage);
  return `<span class="maturity maturity-${BADGE[stageKey(e.stage)]}"${tipAttr(e.liveTip || stageTip(e.stage))}>${esc(chipWord(e))}</span>`;
}
const stages = [...new Set(['LIVE', 'WIRED', 'PM READY', 'CANDIDATE', ...E.map((e) => e.stage)])];
const eName = (e) => `${esc(e.name)}`;
const PARKED_NOTE = 'In T-bills since June 2022';
const standInLabel = (e) => { const [y, m] = e.standIn.to.split('-'); return `${e.standIn.word} to ${+m === 1 ? y : `${MONTHS[+m - 1]} ${y}`}`; };

/* statistics */
const SHARPE_MIN = 20;
function statsOf(r, a, b, base = 0) {
  let n = 0, eq = 1, peak = 1, worst = 0, s = 0, s2 = 0, x = 0, x2 = 0;
  for (let i = a; i <= b; i++) {
    const v = r[i - base]; if (v === undefined || Number.isNaN(v)) continue;
    n++; eq *= 1 + v; if (eq > peak) peak = eq; const dd = eq / peak - 1; if (dd < worst) worst = dd;
    s += v; s2 += v * v; const e = v - (Number.isNaN(TB[i]) ? 0 : TB[i]); x += e; x2 += e * e;
  }
  if (n < 1) return null;
  const m = s / n, vol = n < 2 ? null : Math.sqrt(Math.max(0, (s2 - n * m * m) / (n - 1))) * Math.sqrt(252);
  const xm = x / n, xs = n < 2 ? 0 : Math.sqrt(Math.max(0, (x2 - n * xm * xm) / (n - 1)));
  const ann = n > 260;
  return { n, ann, ret: (ann ? Math.pow(eq, 252 / n) - 1 : eq - 1) * 100, total: (eq - 1) * 100, vol: vol == null ? null : vol * 100, sharpe: n >= SHARPE_MIN && xs > 1e-7 && vol > 0.015 ? xm / xs * Math.sqrt(252) : null, dd: worst * 100, end: eq };
}
// NaN when no session in the span has a value: no data is not the same as flat
function compound(r, a, b) { let eq = 1, any = false; for (let i = a; i <= b; i++) { const v = r[i]; if (!Number.isNaN(v)) { eq *= 1 + v; any = true; } } return any ? (eq - 1) * 100 : NaN; }
function ownWorst(r) { let eq = 1, pk = null, pkI = 0, low = 0, lowI = 0, lowPk = 0;
  for (let i = 0; i < r.length; i++) { const v = r[i]; if (Number.isNaN(v)) continue;
    eq *= 1 + v; if (pk === null || eq > pk) { pk = eq; pkI = i; }
    const d = eq / pk - 1; if (d < low) { low = d; lowI = i; lowPk = pkI; } }
  return { v: low * 100, from: lowPk, to: lowI, qqq: compound(QQQ, lowPk + 1, lowI) }; }
function monthStart(lastI) { let i = lastI; while (i > 0 && DATES[i - 1].getUTCMonth() === DATES[lastI].getUTCMonth()) i--; return i; }
const ordinal = (n) => { const t = n % 100; return n + (t >= 10 && t <= 20 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] || 'th'); };
// every stretch of this many sessions in its history, oldest first: the last one is today's
function stretches(r, lastI, len) {
  const days = []; for (let i = 0; i <= lastI; i++) if (!Number.isNaN(r[i])) days.push(r[i]);
  if (len === 1) return days;
  const eq = [1]; days.forEach((v) => eq.push(eq[eq.length - 1] * (1 + v)));
  const w = []; for (let k = len; k < eq.length; k++) w.push(eq[k] / eq[k - len] - 1);
  return w;
}
function rankTip(word, w, unit, who) {
  if (!w.length) return '';
  const today = w[w.length - 1], worst = 1 + w.filter((x) => x < today).length, best = 1 + w.filter((x) => x > today).length, rank = Math.min(worst, best);
  const head = `${word} ${Math.abs(today * 100) < 0.05 ? '0.0' : signed(today * 100)}%.`;
  if (rank > 0.1 * w.length) return `${head} Ordinary for ${who}.`;
  const side = worst <= best ? 'worst' : 'best';
  return `${head} ${rank === 1 ? side[0].toUpperCase() + side.slice(1) : `${ordinal(rank)} ${side}`} of ${w.length.toLocaleString('en-US')} ${unit} since 2008.`;
}
// the heat's tooltips and the tile's tag: how rare today is, on the whole history
function rarityOf(r, lastI, own, who = 'this engine') {
  const mlen = lastI - monthStart(lastI) + 1, falls = [];
  let eq = 1, pk = null;
  for (let i = 0; i <= lastI; i++) { const v = r[i]; if (Number.isNaN(v)) continue; eq *= 1 + v; if (pk === null || eq > pk) pk = eq; falls.push(eq / pk - 1); }
  const today = falls[falls.length - 1], share = falls.filter((x) => x > today).length / falls.length * 100;
  return {
    share,
    day: rankTip('Day', stretches(r, lastI, 1), 'days', who),
    week: rankTip('5 days', stretches(r, lastI, 5), 'five-day stretches', who),
    month: rankTip('This month', stretches(r, lastI, mlen), `stretches of ${mlen} session${mlen === 1 ? '' : 's'}`, who),
    peak: own.to === lastI ? 'Deepest since 2008.' : today >= 0 ? 'At its peak.' : `Deeper than ${Math.min(99.9, share).toFixed(1)}% of its days since 2008.`,
  };
}
E.forEach((e) => { e.own = ownWorst(R[e.id]); e.rare = rarityOf(R[e.id], e.lastI, e.own); });
const QQQ_OWN = ownWorst(QQQ), QQQ_RARE = rarityOf(QQQ, lastValid(QQQ), QQQ_OWN, 'QQQ');
function sampled(r, a, b, maxPts = 240, base = 0) {
  const step = Math.max(1, Math.ceil((b - a + 1) / maxPts));
  let eq = 1, peak = 1, low = 0, seen = false, lastGood = -1; const out = [];
  for (let i = a; i <= b; i++) {
    const v = r[i - base];
    if (v !== undefined && !Number.isNaN(v)) { eq *= 1 + v; if (eq > peak) peak = eq; const dd = eq / peak - 1; if (dd < low) low = dd; seen = true; lastGood = i; }
    if (((i - a) % step === step - 1 || i === b) && seen && lastGood > (out.length ? out[out.length - 1].i : -1)) { out.push({ i: lastGood, eq, dd: low * 100 }); low = 0; }
  }
  return out;
}

/* period */
// the first session after the same date a year earlier; Feb 29 looks back to Feb 28
function yearBack(i) {
  const iso = ISO[i], from = `${+iso.slice(0, 4) - 1}${iso.slice(4)}`.replace('-02-29', '-02-28'), k = idxOf(from);
  return ISO[k] > from ? k : k + 1;
}
const ytdStart = () => { let i = LAST; while (i > 0 && YEAR[i - 1] === YEAR[LAST]) i--; return i; };
const YTD_MIN = 5, ytdDays = LAST - ytdStart() + 1;
const PERIODS = [
  { id: 'all', name: 'Since 2008', a: () => 0 },
  { id: '2012', name: 'Since Oct 2012', a: () => idxOf('2012-10-02') },
  { id: '2020', name: 'Since 2020', a: () => idxOf('2020-01-02') },
  { id: '2023', name: 'Since 2023', a: () => idxOf('2023-01-03') },
  { id: '1y', name: '1 year', a: () => yearBack(LAST) },
  { id: 'ytd', name: 'This year', a: ytdStart, off: ytdDays < YTD_MIN ? `${plural(ytdDays, 'session')} so far. This year opens at ${YTD_MIN}.` : '' },
];
const storedPeriod = PERIODS.find((p) => p.id === store.get('em-period'));
let period = storedPeriod && storedPeriod.off ? PERIODS.find((p) => p.id === '1y') : storedPeriod || PERIODS[0];
let stageFilter = 'all';
const winA = () => period.a();
const periodWords = () => period.name[0].toLowerCase() + period.name.slice(1);
function cvarOf(r, a, b) {
  const days = []; for (let i = a; i <= b; i++) if (!Number.isNaN(r[i])) days.push(r[i]);
  if (days.length < 20) return null;
  days.sort((p, q) => p - q);
  const k = Math.max(1, Math.floor(days.length * 0.05));
  return days.slice(0, k).reduce((s, v) => s + v, 0) / k * 100;
}
const withCvar = (s, r, a) => { if (s) s.cvar = cvarOf(r, a, LAST); return s; };
function periodStats() { const a = winA(); E.forEach((e) => { e.s = withCvar(statsOf(R[e.id], a, LAST), R[e.id], a); }); return { qqq: withCvar(statsOf(QQQ, a, LAST), QQQ, a), tbill: withCvar(statsOf(TB, a, LAST), TB, a) }; }
let BENCH = periodStats();
const wordOf = (s) => (s && s.ann ? 'CAGR' : 'Return');
const retWord = () => wordOf(BENCH.qqq);
// an engine whose own days are too few for the window's word says so beside its number
const retCell = (s) => pct(s.ret) + (wordOf(s) === retWord() ? '' : ` <small>${wordOf(s)}</small>`);

/* tooltip */
const tip = $('tip');
function showTip(html, x, y) {
  tip.innerHTML = html; tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let left = x + 16, top = y + 16;
  if (left + r.width > innerWidth - 8) left = x - r.width - 16;
  if (top + r.height > innerHeight - 8) top = y - r.height - 16;
  tip.style.left = Math.max(4, left) + 'px'; tip.style.top = Math.max(4, top) + 'px';
}
const hideTip = () => { tip.hidden = true; };

/* heat colours: green up, red down */
const heatShare = (v, cap) => (Math.min(1, Math.abs(v) / cap) * 100).toFixed(0);
function heat(v, cap = 1) { return `color-mix(in oklab, var(${v >= 0 ? '--pos' : '--neg'}) ${heatShare(v, cap)}%, var(--mid))`; }
// a cell's text: white or the theme's ink, whichever has more contrast on the fill it really paints. Read once per fill; a theme change clears this
const fillProbe = document.createElement('canvas').getContext('2d', { willReadFrequently: true }), inkPicked = new Map();
function luminance(css) {
  fillProbe.fillStyle = css; fillProbe.fillRect(0, 0, 1, 1);
  const [r, g, b] = Array.from(fillProbe.getImageData(0, 0, 1, 1).data.slice(0, 3), (c) => (c / 255 <= 0.04045 ? c / 255 / 12.92 : ((c / 255 + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function inkOn(v, cap = 1) {
  const share = heatShare(v, cap), side = v >= 0 ? '--pos' : '--neg', key = side + share;
  if (!inkPicked.has(key)) {
    const token = (name) => getComputedStyle(ROOT).getPropertyValue(name).trim();
    const fill = luminance(`color-mix(in oklab, ${token(side)} ${share}%, ${token('--mid')})`);
    const contrast = (name) => { const x = luminance(token(name)); return (Math.max(x, fill) + 0.05) / (Math.min(x, fill) + 0.05); };
    inkPicked.set(key, contrast('--on-fill') > contrast('--ink') ? 'var(--on-fill)' : 'var(--ink)');
  }
  return inkPicked.get(key);
}
function heatCell(v, cap, text, tipText) {
  if (v == null || Number.isNaN(v)) return `<td class="plain"${tipText ? ` data-tip="${esc(tipText)}"` : ''}></td>`;
  return `<td style="background:${heat(v, cap)};color:${inkOn(v, cap)}"${tipText ? ` data-tip="${esc(tipText)}"` : ''}>${text}</td>`;
}
const tipAttr = (text) => text ? ` data-tip="${esc(text)}"` : '';
const plainCell = (text, tipText) => `<td class="plain"${tipAttr(tipText)}>${esc(text)}</td>`;
const nameCell = (e) => `<td class="n${e.stage === 'LIVE' ? ' live' : ''}"${tipAttr(e.sub)}><span class="dot s s-${stageKey(e.stage)}"></span>${eName(e)}</td>`;
const sharpeText = (s, e) => s.sharpe != null ? num(s.sharpe) : s.n < SHARPE_MIN ? 'Too few days' : e && e.parked ? 'In cash' : 'Too flat';
const sharpeTip = (e, s) => {
  if (s.sharpe != null) return '';
  if (s.n < SHARPE_MIN) return ` class="word"${tipAttr(`${plural(s.n, 'session')} in this period. A Sharpe needs ${SHARPE_MIN}.`)}`;
  return `${e && e.parked ? '' : ' class="word"'}${tipAttr(`Too little movement for a Sharpe.${e && e.parked ? ` ${PARKED_NOTE}.` : ''}`)}`;
};

/* sortable heat: the first click by the table's own direction, the second reverses, the third restores the default order */
const heatSort = { now: null, years: null, factors: null }, FIRST_DIR = { now: 1, years: 1, factors: -1 };
function stepSort(table, col) {
  const s = heatSort[table], first = FIRST_DIR[table];
  heatSort[table] = !s || s.col !== col ? { col, dir: first } : s.dir === first ? { col, dir: -first } : null;
}
function sortTh(table, col, label, extra = '') {
  const s = heatSort[table], on = s && s.col === col;
  return `<th data-hsort="${table}:${col}" tabindex="0"${on ? ` aria-sort="${s.dir > 0 ? 'ascending' : 'descending'}"` : ''}${extra}>${label}${on ? `<span class="harrow">${s.dir > 0 ? '↑' : '↓'}</span>` : ''}</th>`;
}
const TH_NAME = '<th aria-label="Engine"></th>', TH_GAP = '<th aria-hidden="true"></th>';
function sortRows(table, rows, key) { // a missing value goes last either way; ties keep the default order
  const s = heatSort[table];
  if (!s) return rows;
  return rows.map((x) => { const v = key(x, s.col); return [x, v == null || Number.isNaN(v) ? null : v]; })
    .sort((p, q) => p[1] === null || q[1] === null ? (p[1] === null) - (q[1] === null) : (p[1] - q[1]) * s.dir).map((p) => p[0]);
}

/* header */
// weekdays after the last close and before today
function sessionsBehind() {
  const now = new Date(), today = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
  let n = 0;
  for (let t = Date.parse(D.asOf) + 864e5; t < today; t += 864e5) { const day = new Date(t).getUTCDay(); if (day > 0 && day < 6) n++; }
  return n;
}
function renderHeader() {
  const behind = sessionsBehind();
  $('asof').innerHTML = esc(`Last close ${shortLabel(D.asOf)}, updated ${D.built}`) + (behind >= 2 ? ` <span class="ttag" style="position:static">${esc(`Data ${behind} sessions old`)}</span>` : '');
  $('periods').innerHTML = PERIODS.map((p) => `<button class="filter-chip" data-period="${p.id}" aria-pressed="${p === period}"${p.off ? ` aria-disabled="true"${tipAttr(p.off)}` : ''}>${esc(p.name)}</button>`).join('');
  const filters = [['all', 'All'], ...stages.map((s) => [s, stageWord(s)])];
  $('stagefilter').innerHTML = filters.map(([k, n]) => `<button class="filter-chip" data-stage="${esc(k)}" aria-pressed="${stageFilter === k}"${tipAttr(stageTip(k))}>${esc(n)}</button>`).join('');
  $('mapy').innerHTML = MAP_Y.map(([k, n]) => `<button class="filter-chip" data-mapy="${k}" aria-pressed="${mapY === k}">${esc(n)}</button>`).join('');
  $('mapx').innerHTML = MAP_X.map(([k, n, tip]) => `<button class="filter-chip" data-mapx="${k}" aria-pressed="${mapX === k}"${tipAttr(tip)}>${esc(n)}</button>`).join('');
  $('chartmode').innerHTML = CHART_MODES.map(([k, n]) => `<button class="filter-chip" data-chart="${k}" aria-pressed="${chartMode === k}">${esc(n)}</button>`).join('');
  // the stat band: every strategy, then each stage; an unknown stage counts as PM ready, as it draws
  $('n-all').textContent = E.length;
  if (D.scopeNote) $('t-all').dataset.tip = D.scopeNote;
  [['live', 'LIVE'], ['wired', 'WIRED'], ['pm', 'PM READY'], ['cand', 'CANDIDATE']].forEach(([k, s]) => { $(`n-${k}`).textContent = E.filter((e) => stageKey(e.stage) === k).length; if (stageTip(s)) $(`t-${k}`).dataset.tip = stageTip(s); });
  $('mapkey').innerHTML = [['live', 'LIVE', '<circle cx="6" cy="6" r="5.5" fill="var(--c)"/>'], ['wired', 'WIRED', '<circle cx="6" cy="6" r="5" fill="var(--c)"/>'], ['pm', 'PM READY', '<circle cx="6" cy="6" r="4.5" fill="var(--surface)" stroke="var(--c)" stroke-width="2"/>']]
    .map(([k, s, mark]) => `<span class="s-${k}"${tipAttr(stageTip(s))}><svg width="12" height="12" aria-hidden="true">${mark}</svg>${esc(stageWord(s))}</span>`).join('') +
    `<span class="s-cand"${tipAttr(stageTip('CANDIDATE'))}><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4.5" fill="var(--surface)" stroke="var(--c)" stroke-width="1.5" stroke-dasharray="2 2"/></svg>${esc(stageWord('CANDIDATE'))}</span>`;
  $('foot').innerHTML = esc(`Backtests, ${periodWords()} to ${shortLabel(D.asOf)}. After trading costs, before fund fees. Idle cash earns T-bills less 0.5%; that rate is held at its Aug 19 value. Sharpe is over T-bills. Through Aug 19, 2026 the history is the fund study's, then fresh runs of the same rules. Flow Vote and the SPMO versions: their own research runs. TAA before Oct 2012 is a proxy run, with stand-in funds until they listed in 2010-11. Factors: ${factorSpan()}. Shown where t is at least 2.`) + (D.scopeNote ? `<div>${esc(D.scopeNote)}</div>` : '') + (D.leftOut ? `<div>${esc(D.leftOut)}</div>` : '') + (D.offMap.length ? `<div>${esc(`Not on this map yet: ${D.offMap.map((s) => (D.offMapWhy && D.offMapWhy[s] ? `${D.offMapWhy[s].name} (${D.offMapWhy[s].why})` : s.replace(/^strategy_/, '').replace(/_/g, ' '))).join('; ')}.`)}</div>` : '') +
    ((D.candidatesOff || []).length ? `<div>${esc(`Scout candidates not on this map: ${D.candidatesOff.map((c) => `${c.name} (${c.why})`).join('; ')}.`)}</div>` : '');
  $('yearstitle').innerHTML = esc('Every engine, every year, every QQQ fall');
  $('yearsnote').innerHTML = esc(`Return in each calendar year${yearSpan(YEAR[0]) === String(YEAR[0]) ? '' : ` (${YEAR[0]} starts ${monthDay(ISO[0])})`}, then what each engine did while QQQ fell 15% or more, then its own worst fall. Its worst fall may have nothing to do with QQQ.`);
  $('factnote').innerHTML = esc(`How much each engine leaned on each driver: ${D.regYears}-year averages, not its current lean; a rotation's lean changes month to month. Five funds: QQQ, IEF, GLD, DBC, UUP, over T-bills. An empty cell means t under 2 on that average; the lean can still flip inside the window. Momentum is a separate monthly regression on Market, Size, Value and Momentum, to ${D.regWindowFF}. Alpha is what remains after the five funds, per year, before and after fees of 2% and 20%. Bold is t of at least 2 on this window, with no allowance for the search behind each rule. Click a column to sort.`);
  $('buildtag').textContent = `Backtest, before fund fees. ${D.menuNote}`;
}
const shows = (e) => stageFilter === 'all' || e.stage === stageFilter;

/* label placement shared by both scatter maps */
// does a segment [x1, y1, x2, y2] touch a box? Liang-Barsky clipping
function segTouches(s, b) {
  let t0 = 0, t1 = 1;
  const dx = s[2] - s[0], dy = s[3] - s[1];
  return [[-dx, s[0] - b.x0], [dx, b.x1 - s[0]], [-dy, s[1] - b.y0], [dy, b.y1 - s[1]]].every(([p, q]) => { if (p === 0) return q >= 0; const r = q / p; if (p < 0) t0 = Math.max(t0, r); else t1 = Math.min(t1, r); return t0 <= t1; });
}
function labelPlacer(W, H, m, boxes, segs = []) {
  const ctx = document.createElement('canvas').getContext('2d');
  ctx.font = "600 13px 'IBM Plex Sans', system-ui, sans-serif";
  const hits = (b) => boxes.some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0);
  return (px, py, r, text) => {
    const w = ctx.measureText(text).width, h = 16;
    const ring = (gap) => { const d = gap * 0.74; return [[px + gap, py - h / 2], [px - gap - w, py - h / 2], [px - w / 2, py - gap - h], [px - w / 2, py + gap], [px + d, py - d - h], [px + d, py + d], [px - d - w, py - d - h], [px - d - w, py + d]]; };
    const gaps = [r + 4, r + 18, r + 34, r + 52, r + 72, r + 96, r + 124];
    // first a spot no line runs through and whose own leader crosses no label or dot, then any free spot
    for (const clear of [true, false]) for (let k = 0; k < gaps.length; k++) {
      for (const [lx, ly] of ring(gaps[k])) {
        const b = { x0: lx - 3, x1: lx + w + 3, y0: ly - 1, y1: ly + h + 1 };
        if (b.x0 < m.l + 2 || b.x1 > W - m.r || b.y0 < m.t || b.y1 > H - m.b || hits(b)) continue;
        const end = [Math.max(lx, Math.min(px, lx + w)), Math.max(ly + 1, Math.min(py, ly + h - 1))], leader = [px, py, end[0], end[1]];
        if (clear && (segs.some((s) => segTouches(s, b)) || (k > 0 && boxes.some((o) => !(o.x0 <= px && px <= o.x1 && o.y0 <= py && py <= o.y1) && segTouches(leader, o))))) continue;
        boxes.push(b);
        if (k > 0) segs.push(leader);
        return { x: lx, y: ly + 12, leader: k > 0 ? end : null };
      }
    }
    return { x: Math.max(m.l + 2, Math.min(px + r + 4, W - m.r - w)), y: py + 5, leader: null };
  };
}
const niceStep = (range, target) => { const raw = Math.max(range, 0.5) / target; const p = Math.pow(10, Math.floor(Math.log10(raw))); const f = raw / p; return (f < 1.5 ? 1 : f < 3.5 ? 2.5 : f < 7.5 ? 5 : 10) * p; };
// a tick's decimals come from the step: 5 and 2.5 and 0.25 all print in full, whole values with none
const stepDigits = (step) => { let d = 0; while (d < 3 && Math.abs(step * 10 ** d - Math.round(step * 10 ** d)) > 1e-9) d++; return d; };
const tickText = (v, step) => (v < -1e-9 ? '−' : '') + String(+Math.abs(v).toFixed(stepDigits(step)));
// both maps: the pointer belongs to the nearest dot centre within 16 px, whichever dot's circle it touches
const dotSpots = { mapbox: [], simbox: [] };
function dotAt(ev) {
  const svg = ev.target.closest && ev.target.closest('#mapbox svg, #simbox svg');
  if (!svg) return null;
  const r = svg.getBoundingClientRect(), k = svg.viewBox.baseVal.width / r.width, x = (ev.clientX - r.left) * k, y = (ev.clientY - r.top) * k;
  let near = 16, id = null;
  dotSpots[svg.parentNode.id].forEach((p) => { const d = Math.hypot(p.x - x, p.y - y); if (d <= near) { near = d; id = p.id; } });
  return id ? svg.querySelector(`.pt[data-hover="${id}"]`) : null;
}

/* return against worst fall */
let mapNodes = null;
const MAP_Y = [['ret', 'Return'], ['sharpe', 'Sharpe']];
const MAP_X = [['dd', 'Max DD', 'Its deepest fall from a peak in the period.'], ['cvar', 'CVaR 5%', 'The average of its worst 5% of days in the period.']];
const pick = (list, key, dflt) => (list.some((c) => c[0] === store.get(key)) ? store.get(key) : dflt);
let mapY = pick(MAP_Y, 'em-mapy', 'ret'), mapX = pick(MAP_X, 'em-mapx', 'dd');
function renderMap() {
  const box = $('mapbox');
  const W = Math.max(700, box.clientWidth), H = Math.max(560, Math.min(640, W * 0.46));
  const m = { l: 48, r: 20, t: 10, b: 30 };
  // up: return or Sharpe; across: the fall, deepest from peak or the average of the worst 5% of days. T-bills sit at Sharpe 0
  const yOf = (s) => (mapY === 'sharpe' ? s.sharpe : s.ret), xOf = (s) => (mapX === 'cvar' ? s.cvar : s.dd);
  const ok = (s) => s && yOf(s) != null && xOf(s) != null;
  const live = E.filter((e) => ok(e.s));
  const qq = ok(BENCH.qqq) ? BENCH.qqq : null, tbs = BENCH.tbill && xOf(BENCH.tbill) != null ? BENCH.tbill : null;
  const tbY = tbs ? (mapY === 'sharpe' ? 0 : tbs.ret) : null;
  const worst = Math.max(...live.map((e) => Math.abs(xOf(e.s))), mapX === 'cvar' ? 0.5 : 4);
  const qqIn = qq && Math.abs(xOf(qq)) <= worst * 1.3;
  const xStep = mapX === 'cvar' ? niceStep(worst, 6) : 5;
  const XMAX = Math.ceil((qqIn ? Math.max(worst, Math.abs(xOf(qq))) : worst) * 1.12 / xStep) * xStep;
  const rets = live.map((e) => yOf(e.s)).concat(qq ? [yOf(qq)] : []);
  const yStep = niceStep(Math.max(...rets) - Math.min(0, ...rets), 6);
  const YMAX = Math.ceil((Math.max(...rets) + yStep * 0.15) / yStep) * yStep, YMIN = Math.min(0, Math.floor((Math.min(...rets) - yStep * 0.15) / yStep) * yStep);
  const yUnit = mapY === 'sharpe' ? '' : '%', yName = mapY === 'sharpe' ? 'Sharpe' : retWord(), xName = mapX === 'cvar' ? 'CVaR 5%' : 'Max DD';
  const x = (dd) => m.l + (Math.abs(dd) / XMAX) * (W - m.l - m.r);
  const y = (c) => H - m.b - ((c - YMIN) / (YMAX - YMIN)) * (H - m.t - m.b);
  let g = '';
  for (let v = YMIN; v <= YMAX + 1e-9; v += yStep) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--${Math.abs(v) < 1e-9 ? 'axis' : 'grid'})"/><text class="axis-t" x="${m.l - 9}" y="${y(v) + 4}" text-anchor="end">${tickText(v, yStep)}${yUnit}</text>`;
  for (let v = 0; v <= XMAX + 1e-9; v += xStep) g += `<line x1="${x(v)}" x2="${x(v)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--grid)"/><text class="axis-t" x="${x(v)}" y="${H - m.b + 19}" text-anchor="middle">${v ? '−' + tickText(v, xStep) : 0}%</text>`;
  g += `<text class="axis-n" x="${W - m.r - 6}" y="${H - m.b - 8}" text-anchor="end">${xName}</text><text class="axis-n" x="${m.l + 8}" y="${m.t + 16}">${yName}</text>`;
  // the return = worst fall line only means something on those two axes
  const ray = mapY === 'ret' && mapX === 'dd' ? Math.min(YMAX, XMAX) : 0;
  let rayBox = null;
  if (ray > 0) {
    // the label sits at the upper end, on the free side of the line
    const text = `${retWord()} = Max DD`, tw = text.length * 6.7, slope = (y(0) - y(ray)) / (x(ray) - x(0));
    const tx = Math.max(m.l + 4 + tw, x(ray) - 18 / slope), ty = y(ray) + 13;
    g += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(ray)}" y2="${y(ray)}" stroke="var(--axis)"/><text class="axis-t" x="${tx}" y="${ty}" text-anchor="end">${esc(text)}</text>`;
    rayBox = { x0: tx - tw - 3, x1: tx + 3, y0: ty - 13, y1: ty + 4 };
  }

  const rOf = (e) => e.stage === 'LIVE' ? 10 : 8;
  const points = E.map((e) => ok(e.s) ? { e, px: x(xOf(e.s)), py: y(yOf(e.s)), r: rOf(e) } : null).filter(Boolean);
  const tb = tbs ? { px: x(xOf(tbs)), py: y(tbY) } : null;
  const qText = qq ? (qqIn ? 'QQQ' : `QQQ ${pct(xOf(qq), mapX === 'cvar' ? 2 : 1)}`) : '';
  const q = qq ? { px: qqIn ? x(xOf(qq)) : W - m.r - 3, py: y(yOf(qq)) } : null;
  const boxes = [{ x0: m.l + 4, x1: m.l + 70, y0: m.t + 2, y1: m.t + 22 }, { x0: W - m.r - 70, x1: W - m.r, y0: H - m.b - 22, y1: H - m.b }];
  points.forEach((p) => boxes.push({ x0: p.px - p.r - 2, x1: p.px + p.r + 2, y0: p.py - p.r - 2, y1: p.py + p.r + 2 }));
  if (tb) boxes.push({ x0: tb.px - 6, x1: tb.px + 6, y0: tb.py - 6, y1: tb.py + 6 });
  if (q) boxes.push({ x0: q.px - (qqIn ? 8 : 120), x1: q.px + 8, y0: q.py - 9, y1: q.py + 9 });
  if (rayBox) boxes.push(rayBox);
  const place = labelPlacer(W, H, m, boxes, ray > 0 ? [[x(0), y(0), x(ray), y(ray)]] : []);
  let refs = '';
  if (tb) { const at = place(tb.px, tb.py, 6, 'T-bills'); refs += `${at.leader ? `<line x1="${tb.px}" y1="${tb.py}" x2="${at.leader[0]}" y2="${at.leader[1]}" stroke="var(--axis)"/>` : ''}<circle cx="${tb.px}" cy="${tb.py}" r="5.5" fill="var(--ref)" stroke="var(--surface)" stroke-width="2"/><text class="lbl ref" x="${at.x}" y="${at.y}">T-bills</text>`; }
  if (q) {
    if (qqIn) { const at = place(q.px, q.py, 6, qText); refs += `${at.leader ? `<line x1="${q.px}" y1="${q.py}" x2="${at.leader[0]}" y2="${at.leader[1]}" stroke="var(--axis)"/>` : ''}<circle cx="${q.px}" cy="${q.py}" r="5.5" fill="var(--ref)" stroke="var(--surface)" stroke-width="2"/><text class="lbl ref" x="${at.x}" y="${at.y}">${esc(qText)}</text>`; }
    else refs += `<path d="M${q.px + 3} ${q.py}L${q.px - 7} ${q.py - 6}V${q.py + 6}Z" fill="var(--ref)"/><text class="lbl ref" x="${q.px - 12}" y="${q.py + 5}" text-anchor="end">${esc(qText)}</text>`;
  }
  points.slice().sort((a, b) => (b.e.stage === 'LIVE') - (a.e.stage === 'LIVE')).forEach((p) => { p.label = place(p.px, p.py, p.r, p.e.name); });
  const leaders = points.map((p) => p.label.leader ? `<line${shows(p.e) ? '' : ' class="dim"'} x1="${p.px}" y1="${p.py}" x2="${p.label.leader[0]}" y2="${p.label.leader[1]}" stroke="var(--axis)"/>` : '').join('');

  let fresh = false;
  if (!mapNodes || mapNodes.W !== W || mapNodes.H !== H) {
    const nodes = E.map((e) => {
      const hollow = chipClass(e.stage) === 'PM';
      return `<g class="mover pt s-${stageKey(e.stage)}" data-open="${e.id}" data-hover="${e.id}" tabindex="0" role="button" aria-label="${esc(e.name)}"><circle class="hit" r="16"/><circle class="mk" r="${rOf(e)}" style="fill:${hollow ? 'var(--surface)' : 'var(--c)'};stroke:${hollow ? 'var(--c)' : 'var(--surface)'}" stroke-width="2.5"/></g><text class="mover lbl${e.stage === 'LIVE' ? ' live' : ''}" data-lbl="${e.id}" data-open="${e.id}" data-hover="${e.id}">${esc(e.name)}</text>`;
    }).join('');
    box.innerHTML = `<svg class="noanim" viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="Return against worst fall for each engine"><g id="m-grid"></g><g id="m-leaders"></g><g id="m-refs"></g><g id="m-pts">${nodes}</g></svg>`;
    mapNodes = { W, H }; fresh = true;
  }
  $('m-grid').innerHTML = g; $('m-leaders').innerHTML = leaders; $('m-refs').innerHTML = refs;
  // an engine with no data in this period has no dot and no label
  E.forEach((e) => { box.querySelector(`.pt[data-hover="${e.id}"]`).toggleAttribute('hidden', !ok(e.s)); box.querySelector(`[data-lbl="${e.id}"]`).toggleAttribute('hidden', !ok(e.s)); });
  dotSpots.mapbox = points.map((p) => ({ id: p.e.id, x: p.px, y: p.py }));
  points.forEach((p) => {
    const dot = box.querySelector(`.pt[data-hover="${p.e.id}"]`), text = box.querySelector(`[data-lbl="${p.e.id}"]`);
    dot.style.transform = `translate(${p.px}px, ${p.py}px)`; text.style.transform = `translate(${p.label.x}px, ${p.label.y}px)`;
    const on = shows(p.e); dot.classList.toggle('dim', !on); text.classList.toggle('dim', !on);
  });
  if (fresh) requestAnimationFrame(() => requestAnimationFrame(() => { const s = box.querySelector('svg'); if (s) s.classList.remove('noanim'); }));
  $('maptitle').innerHTML = esc(`Backtested ${mapY === 'sharpe' ? 'Sharpe' : 'return'} against ${mapX === 'cvar' ? 'CVaR 5%' : 'worst fall'}, ${periodWords()}`);
}

/* latest returns */
function nowRow(r, lastI) {
  const yStart = (() => { let i = lastI; while (i > 0 && YEAR[i - 1] === YEAR[lastI]) i--; return i; })();
  const mStart = monthStart(lastI);
  let eq = 1, peak = 1; for (let i = 0; i <= lastI; i++) { const v = r[i]; if (!Number.isNaN(v)) { eq *= 1 + v; if (eq > peak) peak = eq; } }
  return { day: r[lastI] * 100, week: compound(r, Math.max(0, lastI - 4), lastI), month: compound(r, mStart, lastI), ytd: compound(r, yStart, lastI), y1: compound(r, yearBack(LAST), lastI), peak: (eq / peak - 1) * 100 };
}
const NOWCOLS = [['day', 'Day', 5], ['week', '5 days', 8], ['month', 'This month', 15], ['ytd', 'This year', 30], ['y1', '12 months', 40], ['peak', 'From peak', 25]];
const nowText = (v) => (Number.isNaN(v) ? '' : minus(v, 1));
// its deepest fall ends at this close: a word says so, the fill stays as it is
const nowWord = (col, v, own, lastI) => (col === 'peak' && own.to === lastI ? `${nowText(v)} worst` : nowText(v));
// Year stops counting once the calendar year has moved on, 12 months once the last close's own 12 months hold none of its data
const nowStale = (e, col) => e.lastI < LAST && (col === 'ytd' ? YEAR[e.lastI] !== YEAR[LAST] : col === 'y1' ? e.lastI < yearBack(LAST) : true);
const toEnd = (e) => `to ${YEAR[e.lastI] === YEAR[LAST] ? monthDay(e.end) : shortLabel(e.end)}`;
const staleTip = (e, col) => nowStale(e, col) ? `${e.name}: data to ${shortLabel(e.end)}.${e.parked ? ` ${PARKED_NOTE}.` : ''}` : toEnd(e);
function renderNow() {
  $('nowtag').textContent = `To ${shortLabel(D.asOf)}`;
  $('nowtitle').innerHTML = esc(`To the close of ${shortLabel(D.asOf)}, in percent. Green is up, red is down, darker is bigger. Worst: the deepest fall since 2008 ends at this close. Hover a cell to see how rare it is for that engine. Click a column to sort.`);
  const head = `<thead><tr>${TH_NAME}${NOWCOLS.map((c) => sortTh('now', c[0], c[1])).join('')}</tr></thead>`;
  E.forEach((e) => { e.now = e.now || nowRow(R[e.id], e.lastI); });
  const cell = (e, c) => {
    const v = e.now[c[0]];
    if (e.lastI === LAST) return heatCell(v, c[2], nowWord(c[0], v, e.own, e.lastI), e.rare[c[0]]);
    return plainCell(nowStale(e, c[0]) ? 'no data' : nowText(v), staleTip(e, c[0]));
  };
  const row = (e) => `<tr class="clk${shows(e) ? '' : ' dim'}" data-open="${e.id}" data-hover="${e.id}">${nameCell(e)}` + NOWCOLS.map((c) => cell(e, c)).join('') + '</tr>';
  const qLast = lastValid(QQQ), q = nowRow(QQQ, qLast);
  const qCell = (c) => qLast === LAST ? heatCell(q[c[0]], c[2], nowWord(c[0], q[c[0]], QQQ_OWN, qLast), QQQ_RARE[c[0]]) : plainCell(nowText(q[c[0]]), `QQQ: data to ${dayLabel(qLast)}.`);
  const qRow = `<tr class="ref"><td class="n">${esc(qLast === LAST ? 'QQQ' : `QQQ, to ${monthDay(ISO[qLast])}`)}</td>${NOWCOLS.map(qCell).join('')}</tr>`;
  const rowsOf = (f) => sortRows('now', E.filter((e) => e.fam === f.id), (e, col) => nowStale(e, col) ? null : e.now[col]);
  $('nowbox').innerHTML = FAM.map((f) => `<div><h3 class="famh c-${f.id}"><span class="dot"></span>${esc(f.name)}</h3><table class="heat fixed">${head}<tbody>${rowsOf(f).map(row).join('')}${qRow}</tbody></table></div>`).join('');
}

/* tiles */
let sparkData = {};
function sparkHTML(e, scale) {
  const W = 300, EQ = 112, TOP = 118, DDH = 32, H = TOP + DDH, a = winA();
  const X = (i) => ((i - a) / Math.max(1, LAST - a)) * W;
  const Y = (v) => EQ - ((Math.log(v) - scale.lo) / (scale.hi - scale.lo)) * EQ;
  const DY = (v) => TOP + (Math.min(scale.dd, Math.abs(v)) / scale.dd) * DDH;
  const pts = sampled(R[e.id], a, LAST);
  sparkData[e.id] = pts;
  let eq = '', dd = `M0 ${TOP}`, tb = '';
  pts.forEach((p, k) => { eq += `${k ? 'L' : 'M'}${X(p.i).toFixed(1)} ${Y(p.eq).toFixed(1)}`; dd += `L${X(p.i).toFixed(1)} ${DY(p.dd).toFixed(1)}`; });
  const lastX = pts.length ? X(pts[pts.length - 1].i) : W;
  dd += `L${lastX.toFixed(1)} ${TOP}Z`;
  sampled(TB, a, LAST).forEach((p, k) => { tb += `${k ? 'L' : 'M'}${X(p.i).toFixed(1)} ${Y(p.eq).toFixed(1)}`; });
  const bands = D.shaded.map((c) => { const i0 = idxOf(c.from), i1 = idxOf(c.to); if (i1 < a) return ''; const x0 = X(Math.max(i0, a)); return `<rect x="${x0}" width="${Math.max(2, X(i1) - x0)}" y="0" height="${H}" fill="var(--band)"/>`; }).join('');
  const area = `${eq}L${lastX.toFixed(1)} ${EQ}L0 ${EQ}Z`;
  const ten = Math.log(10) > scale.lo && Math.log(10) < scale.hi ? `<line x1="0" x2="${W}" y1="${Y(10)}" y2="${Y(10)}" stroke="var(--grid)"/>` : '';
  const one = `<line x1="0" x2="${W}" y1="${Y(1)}" y2="${Y(1)}" stroke="var(--grid)"/>`;
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" data-spark="${e.id}" data-mode="growth" aria-hidden="true">${bands}${one}${ten}<path d="${area}" fill="var(--c)" fill-opacity=".10"/><path d="${tb}" fill="none" stroke="var(--ref)" stroke-width="1"/><path d="${eq}" fill="none" stroke="var(--c)" stroke-width="2.25" stroke-linejoin="round"/><path d="${dd}" fill="var(--c)" fill-opacity=".42"/><line class="xl" x1="0" x2="0" y1="0" y2="${H}" stroke="var(--ink-3)" stroke-width="1" visibility="hidden"/></svg>`;
}
// the Charts switch: Sharpe over T-bills on the last 756 sessions, once 600 of them have data, on one scale of -1 to 3
const ROLL = 756, ROLL_MIN = 600, ROLL_LO = -1, ROLL_HI = 3;
function rollSharpeAt(r, end) {
  let n = 0, x = 0, x2 = 0;
  for (let i = Math.max(0, end - (ROLL - 1)); i <= end; i++) { const v = r[i]; if (Number.isNaN(v)) continue; const ex = v - (Number.isNaN(TB[i]) ? 0 : TB[i]); n++; x += ex; x2 += ex * ex; }
  if (n < ROLL_MIN) return NaN;
  const m = x / n, sd = Math.sqrt(Math.max(0, (x2 - n * m * m) / (n - 1)));
  return sd * Math.sqrt(252) > 0.015 ? m / sd * Math.sqrt(252) : NaN;
}
function sharpeHTML(e) {
  const W = 300, H = 150, a = winA();
  const X = (i) => ((i - a) / Math.max(1, LAST - a)) * W;
  const Y = (v) => H - 4 - ((Math.min(ROLL_HI, Math.max(ROLL_LO, v)) - ROLL_LO) / (ROLL_HI - ROLL_LO)) * (H - 8);
  const pts = []; let d = '', gap = true;
  for (let i = a; i <= LAST; i += Math.max(1, Math.ceil((LAST - a) / 160))) {
    const v = rollSharpeAt(R[e.id], i);
    if (Number.isNaN(v)) { gap = true; continue; }
    d += `${gap ? 'M' : 'L'}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`; gap = false; pts.push({ i, v });
  }
  sparkData[e.id] = pts;
  // dashed: its own Sharpe since 2008, whatever the period
  const whole = statsOf(R[e.id], 0, LAST), ref = whole && whole.sharpe != null ? `<line x1="0" x2="${W}" y1="${Y(whole.sharpe).toFixed(1)}" y2="${Y(whole.sharpe).toFixed(1)}" stroke="var(--ink-3)" stroke-dasharray="4 3"/>` : '';
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" data-spark="${e.id}" data-mode="sharpe" aria-hidden="true"><line x1="0" x2="${W}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--edge)"/><line x1="0" x2="${W}" y1="${Y(1)}" y2="${Y(1)}" stroke="var(--grid)"/>${ref}<path d="${d}" fill="none" stroke="var(--c)" stroke-width="1.6" stroke-linejoin="round"/><line class="xl" x1="0" x2="0" y1="0" y2="${H}" stroke="var(--ink-3)" stroke-width="1" visibility="hidden"/></svg>`;
}
const chartHTML = (e, scale) => (chartMode === 'sharpe' ? sharpeHTML(e) : sparkHTML(e, scale));
// at most one tag, on the whole history whatever the period; the first rule that matches wins
function tileTag(e) {
  const peak = (e.now || (e.now = nowRow(R[e.id], e.lastI))).peak;
  if (e.parked) return `${PARKED_NOTE}, data ${toEnd(e)}`;
  if (e.lastI < LAST) return `Data to ${shortLabel(e.end)}`;
  if (e.card && e.card.verdict === 'REJECTED') return 'Scout: Rejected';
  if (e.own.to === e.lastI) return `Deepest since 2008: ${pct(peak, 0)} from peak`;
  if (e.rare.share >= 95) return `Worst 5% of its days: ${pct(peak, 0)} from peak`;
  return '';
}
function tileHTML(e, scale) {
  const s = e.s;
  if (!s) return `<button class="panel tile nodata" data-open="${e.id}" data-hover="${e.id}" aria-label="${esc(e.name)}, ${esc(chipWord(e))}: no data in this period"><div class="thead"><span class="tname">${eName(e)}</span>${chipFor(e)}</div><div class="tsub">${esc(e.sub)}</div><div class="tempty">No data in this period</div></button>`;
  const words = tileTag(e), tag = words ? `<div class="ttag"${tipAttr(words)}>${esc(words)}</div>` : '';
  return `<button class="panel tile" data-open="${e.id}" data-hover="${e.id}" aria-label="${esc(e.name)}, ${esc(chipWord(e))}: ${pct(s.ret)} ${wordOf(s)}, Sharpe ${sharpeText(s, e)}, ${pct(s.dd)} Max DD${words ? `. ${esc(words)}` : ''}">` +
    `<div class="thead"><span class="tname">${eName(e)}</span>${chipFor(e)}</div><div class="tsub">${esc(e.sub)}</div><div class="sparkwrap">${chartHTML(e, scale)}${tag}</div>` +
    `<div class="nums"><div><b${s.ret < 0 ? ' class="neg"' : ''}>${pct(s.ret)}</b><span>${wordOf(s)}</span></div><div><b${sharpeTip(e, s)}>${sharpeText(s, e)}</b><span>Sharpe</span></div><div><b${s.dd < 0 ? ' class="neg"' : ''}>${pct(s.dd)}</b><span>Max DD</span></div></div></button>`;
}
const GROUP_MODES = [['fam', 'Type'], ['stage', 'Stage'], ['cluster', 'Cluster']];
const CHART_MODES = [['growth', 'Growth of $1'], ['sharpe', 'Sharpe, 3-year rolling']];
let chartMode = CHART_MODES.some((c) => c[0] === store.get('em-chart')) ? store.get('em-chart') : 'growth';
let groupMode = GROUP_MODES.some((g) => g[0] === store.get('em-group')) ? store.get('em-group') : 'fam';
function groups() {
  const pool = E.filter(shows);
  if (groupMode === 'stage') return stages.map((s) => ({ key: s, name: s, items: pool.filter((e) => e.stage === s) })).filter((g) => g.items.length);
  if (groupMode === 'cluster') {
    const found = betGroups(shortWindow() ? TREE_ALL : TREE), alone = found.filter((g) => g.length === 1).flat();
    return found.filter((g) => g.length > 1).map((g, k) => ({ key: k, name: majorityFamily(g), items: g.map((i) => E[i]) })).concat({ key: 'alone', name: 'On their own', items: alone.map((i) => E[i]) }, { key: 'parked', name: 'Parked', items: E.filter((e) => e.parked) })
      .map((g) => ({ ...g, items: g.items.filter(shows) })).filter((g) => g.items.length);
  }
  return FAM.map((f) => ({ key: f.id, name: f.name, fam: f.id, items: pool.filter((e) => e.fam === f.id) })).filter((g) => g.items.length);
}
function sparkScale() {
  const a = winA(); let lo = 1, hi = 1, dd = 5;
  E.forEach((e) => { if (e.s) { sampled(R[e.id], a, LAST).forEach((p) => { if (p.eq < lo) lo = p.eq; if (p.eq > hi) hi = p.eq; }); dd = Math.max(dd, Math.abs(e.s.dd)); } });
  return { lo: Math.log(lo * 0.97), hi: Math.log(hi * 1.03), dd };
}
function renderTiles() {
  const scale = sparkScale(); sparkData = {};
  const bandNote = groupMode === 'cluster' && shortWindow() ? `<div class="bandnote">${esc(`Groups use the whole history: ${plural(windowWeeks(), 'week')} is too short.`)}</div>` : '';
  const none = groups().length ? '' : `<div class="bandnote">${esc(`No strategy is ${stageWord(stageFilter)} on this map.`)}</div>`;
  $('bands').innerHTML = bandNote + none + groups().map((g) => `<div><h3 class="famh${g.fam ? ` c-${g.fam}` : ''}" style="margin-bottom:14px">${g.fam ? '<span class="dot"></span>' : ''}${esc(g.name)}<small>${g.items.length}</small></h3><div class="tiles">${g.items.map((e) => tileHTML(e, scale)).join('')}</div></div>`).join('');
  const shaded = D.shaded.filter((c) => idxOf(c.to) >= winA()).map((c) => c.name);
  $('tiletag').textContent = chartMode === 'sharpe' ? 'Dashed: its Sharpe since 2008' : '';
  if (chartMode === 'sharpe') $('sparknote').innerHTML = esc(`Sharpe over T-bills on the last 3 years (756 sessions) at each date, drawn once 600 of them have data, ${periodWords()}. One scale for all charts, −1 to 3; a value past it sits on the edge. The dashed line is its own Sharpe since 2008, the thin line 1.0, the darker one zero.`);
  else $('sparknote').innerHTML = esc(`Growth of $1 from ${winA() ? `the close of ${dayLabel(winA() - 1)}` : dayLabel(0)}, on one scale across all charts so you can compare by eye. Below the line: the fall from peak. The thin line is T-bills.${shaded.length ? ` Grey bands: ${shaded.join(', ')}.` : ''}`);
  $('groupby').innerHTML = GROUP_MODES.map(([k, n]) => `<button class="filter-chip" data-group="${k}" aria-pressed="${groupMode === k}">${n}</button>`).join('');
}

/* table */
const VERDICTS = ['CANDIDATE', 'WATCHLIST', 'REJECTED'];
const verdictWord = (v) => v[0] + v.slice(1).toLowerCase();
const cardOf = (e) => e.card && e.card.date ? `Scout card of ${monthDay(e.card.date)}.` : '';
// Plan Sharpe is a statement about the whole history, so it shows under Since 2008 only
const planOn = () => period.id === 'all';
const PLAN_NONE = 'Its Scout card figures sit above this Sharpe: nothing lower to plan on.';
const ALPHA_GROSS = 'Over the five funds, per year, before fund fees.', ALPHA_NET = 'After 2% a year and 20% over a high-water mark, no hurdle.';
// sitting in T-bills, with no Sharpe in this window: its fall is too small to divide by
const parkedFlat = (e) => e.parked && e.s.sharpe == null;
const nowCol = (name, key) => [name, (e) => nowStale(e, key) ? null : e.now[key], (e) => nowStale(e, key) ? '<small>no data</small>' : pct(e.now[key]), (e) => e.lastI < LAST ? tipAttr(staleTip(e, key)) : ''];
// each column: name, sort key (null sorts last either way), cell, cell attributes, tip on its head
const COLS = [
  ['Strategy', (e) => e.name, (e) => `<span class="dot s s-${stageKey(e.stage)}"></span>${esc(e.name)}`, (e) => tipAttr(e.sub)],
  ['Stage', (e) => stages.indexOf(e.stage), (e) => chipFor(e), null, D.stageHeadTip],
  ['Scout', (e) => e.card && e.card.verdict ? VERDICTS.indexOf(e.card.verdict) : e.gradeOnly ? VERDICTS.indexOf(e.gradeOnly.verdict) : null, (e) => !e.card ? `<small>${esc(e.gradeOnly ? e.gradeOnly.word : 'No Scout card')}</small>` : e.card.verdict ? verdictWord(e.card.verdict) : '<small>Not tested</small>', (e) => tipAttr(e.card ? cardOf(e) : e.cardNote), D.scoutTip],
  ['CAGR', (e) => e.s ? e.s.ret : null, (e) => e.s ? retCell(e.s) : '<small>No data</small>', (e) => e.s ? '' : tipAttr('No data in this period.')],
  ['Sharpe', (e) => e.s && e.s.sharpe != null ? e.s.sharpe : null, (e) => e.s ? sharpeText(e.s, e) : '', (e) => e.s ? sharpeTip(e, e.s) : ''],
  ['Plan Sharpe', (e) => e.plan ? e.plan.v : null, (e) => e.plan ? `${num(e.plan.v)} <small>${esc(e.plan.kind)}</small>` : '<small>None lower</small>', (e) => tipAttr(e.plan ? e.plan.src : PLAN_NONE)],
  ['Max DD', (e) => e.s ? e.s.dd : null, (e) => e.s ? pct(e.s.dd) : ''],
  ['Vol', (e) => e.s ? e.s.vol : null, (e) => e.s ? pct(e.s.vol) : ''],
  ['Return / fall', (e) => e.s && e.s.dd !== 0 && !parkedFlat(e) ? e.s.ret / Math.abs(e.s.dd) : null, (e) => e.s && e.s.dd !== 0 && !parkedFlat(e) ? minus(e.s.ret / Math.abs(e.s.dd), 2) : ''],
  nowCol('Day', 'day'),
  nowCol('This year', 'ytd'),
  nowCol('From peak', 'peak'),
  ['Alpha', (e) => e.alpha, (e) => signed(e.alpha) + '%', (e) => alphaAttrs(e, e.alphaT, ALPHA_GROSS, e.alphaTip), `${ALPHA_GROSS} ${D.regWindow}, whatever the period. Bold at t of 2 or more.`],
  ['Alpha net', (e) => e.alphaNet, (e) => signed(e.alphaNet) + '%', (e) => alphaAttrs(e, e.alphaNetT, ALPHA_NET, e.alphaNetTip), `${ALPHA_NET} ${D.regWindow}, whatever the period. Bold at t of 2 or more.`],
];
const SORT_DEFAULT = 4;
let sortCol = SORT_DEFAULT, sortDir = -1;
function renderTable() {
  E.forEach((e) => { e.now = e.now || nowRow(R[e.id], e.lastI); });
  // Plan Sharpe leaves under other periods; a table sorted by it falls back to Sharpe until it returns
  const shown = COLS.map((c, i) => [c, i]).filter(([c]) => c[0] !== 'Plan Sharpe' || planOn()), by = shown.some(([, i]) => i === sortCol) ? sortCol : SORT_DEFAULT;
  const rows = E.filter(shows).map((e) => [e, COLS[by][1](e)]).sort((p, q) => p[1] == null || q[1] == null ? (p[1] == null) - (q[1] == null) : (p[1] > q[1] ? 1 : p[1] < q[1] ? -1 : 0) * sortDir).map((p) => p[0]);
  // the window each column reads: the period, the last close, the regression
  const windows = `<tr><th aria-hidden="true"></th><th colspan="2" aria-hidden="true"></th><th class="group" colspan="${shown.length - 8}"><span>${esc(period.name)}</span></th><th class="group" colspan="3"><span>${esc(`To ${monthDay(D.asOf)}`)}</span></th><th class="group" colspan="2"><span>${esc(D.regWindow)}</span></th></tr>`;
  $('tableview').innerHTML = `<table class="etable${planOn() ? ' plan' : ''}"><thead>${windows}<tr>${shown.map(([c, i]) => `<th data-sort="${i}" tabindex="0"${i === by ? ` aria-sort="${sortDir > 0 ? 'ascending' : 'descending'}"` : ''}${tipAttr(c[4])}>${i === 3 ? retWord() : c[0]}${i === by ? `<span class="harrow">${sortDir > 0 ? '↑' : '↓'}</span>` : ''}</th>`).join('')}</tr></thead><tbody>${rows.map((e) => `<tr data-open="${e.id}">${shown.map(([c]) => `<td${c[3] ? c[3](e) : ''}>${c[2](e)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

/* weekly returns, correlation, clusters */
const WEEKS = D.weeks;
function weeklyOf(r) { const o = new Float64Array(WEEKS.length); let start = 0; WEEKS.forEach((end, k) => { let eq = 1, any = false; for (let i = start; i <= end; i++) { const v = r[i]; if (!Number.isNaN(v)) { eq *= 1 + v; any = true; } } o[k] = any ? eq - 1 : NaN; start = end + 1; }); return o; }
const WK = E.map((e) => weeklyOf(R[e.id]));
const firstWeek = (a) => { let k = 0; while (k < WEEKS.length && WEEKS[k] < a) k++; return k; };
function corrPair(x, y, k0) {
  let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let k = k0; k < x.length; k++) { const a = x[k], b = y[k]; if (Number.isNaN(a) || Number.isNaN(b)) continue; n++; sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; }
  const dx = sxx - sx * sx / n, dy = syy - sy * sy / n;
  return n > 5 && dx > 0 && dy > 0 ? (sxy - sx * sy / n) / Math.sqrt(dx * dy) : 0;
}
let CORR = null, TREE = null, cutValue = 0.6;
function corrFor(a) {
  const k0 = firstWeek(a), n = E.length;
  const C = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) { C[i][i] = 1; for (let j = i + 1; j < n; j++) { C[i][j] = C[j][i] = corrPair(WK[i], WK[j], k0); } }
  // average linkage on 1 - correlation
  const active = new Map(); for (let i = 0; i < n; i++) active.set(i, { id: i, leaves: [i], h: 0, left: null, right: null });
  const dist = (p, q) => { let s = 0; for (const i of p.leaves) for (const j of q.leaves) s += 1 - C[i][j]; return s / (p.leaves.length * q.leaves.length); };
  let next = n;
  while (active.size > 1) {
    let best = Infinity, ka = null, kb = null; const nodes = [...active.values()];
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) { const d = dist(nodes[i], nodes[j]); if (d < best) { best = d; ka = nodes[i]; kb = nodes[j]; } }
    const [left, right] = Math.min(...ka.leaves) <= Math.min(...kb.leaves) ? [ka, kb] : [kb, ka];
    active.delete(ka.id); active.delete(kb.id);
    active.set(next, { id: next, leaves: [...left.leaves, ...right.leaves], h: best, left, right }); next++;
  }
  return { CORR: C, TREE: [...active.values()][0] };
}
// under a quarter of weeks a correlation is noise: that window reads the whole history, and says so
const corrSeen = new Map();
const corrOf = (a) => { if (!corrSeen.has(a)) corrSeen.set(a, corrFor(a)); return corrSeen.get(a); };
const WEEKS_MIN = 13;
function computeCorr() { const c = corrOf(fewWeeks() ? 0 : winA()); CORR = c.CORR; TREE = c.TREE; }
function clustersOf(tree) {
  const cut = 1 - cutValue, out = [];
  (function walk(node) { if (node.h <= cut || !node.left) out.push(node.leaves.slice()); else { walk(node.left); walk(node.right); } })(tree);
  return out;
}
// what the page calls groups: the clusters with the parked engine taken out
const betGroups = (tree = TREE) => clustersOf(tree).map((g) => g.filter((i) => !E[i].parked)).filter((g) => g.length);
const groupsOnMap = () => betGroups().map((g) => g.filter((i) => simOn[i])).filter((g) => g.length);
// under 104 weeks a correlation tree is too noisy to call groups
const windowWeeks = () => WEEKS.length - firstWeek(winA());
const shortWindow = () => windowWeeks() < 104;
const fewWeeks = () => windowWeeks() < WEEKS_MIN;
// the family at least two thirds of the group belongs to, else Mixed
function majorityFamily(g) {
  let top = FAM[0].name, most = 0;
  FAM.forEach((f) => { const c = g.filter((i) => E[i].fam === f.id).length; if (c > most) { most = c; top = f.name; } });
  return most * 3 >= g.length * 2 ? top : 'Mixed';
}

/* similarity map: SMACOF on 1 - correlation, warm-started so a day or a click cannot throw it into another solution */
function classicalStart(C) {
  const n = E.length, D2 = C.map((row) => Array.from(row, (c) => 2 * (1 - c)));
  const rowMean = D2.map((r) => r.reduce((a, b) => a + b, 0) / n), all = rowMean.reduce((a, b) => a + b, 0) / n;
  const B = D2.map((r, i) => r.map((v, j) => -0.5 * (v - rowMean[i] - rowMean[j] + all)));
  const eig = (M, prev) => {
    let v = Array.from({ length: n }, (_, i) => Math.sin(i * 1.7 + 1 + (prev ? 2 : 0))), lambda = 0;
    for (let it = 0; it < 300; it++) {
      let w = M.map((r) => r.reduce((s, x, j) => s + x * v[j], 0));
      if (prev) { const d = w.reduce((s, x, i) => s + x * prev[i], 0); w = w.map((x, i) => x - d * prev[i]); }
      const norm = Math.sqrt(w.reduce((s, x) => s + x * x, 0)) || 1; v = w.map((x) => x / norm); lambda = norm;
    }
    return { v, lambda };
  };
  const p = eig(B), q = eig(B, p.v);
  const xs = p.v.map((x) => x * Math.sqrt(Math.max(p.lambda, 0))), ys = q.v.map((x) => x * Math.sqrt(Math.max(q.lambda, 0)));
  return xs.map((x, i) => [x, ys[i]]);
}
// only the engines marked on take part; the rest keep their start. At 200 steps the stress has stopped moving long before the end
function fitCoords(C, start, on, steps = 200) {
  const idx = E.map((e, i) => i).filter((i) => on[i]), n = idx.length;
  let XY = start.map((p) => p.slice());
  for (let it = 0; it < steps; it++) {
    const Y = XY.map((p) => p.slice());
    for (const i of idx) {
      let sx = 0, sy = 0;
      for (const j of idx) { if (i === j) continue;
        const dx = XY[i][0] - XY[j][0], dy = XY[i][1] - XY[j][1], d = Math.hypot(dx, dy) || 1e-9, r = (1 - C[i][j]) / d;
        sx += r * dx; sy += r * dy; }
      Y[i] = [sx / n, sy / n];
    }
    XY = Y;
  }
  return XY;
}
// rotation, reflection and translation of X onto T, over the engines marked on
function alignTo(T, X, on) {
  const idx = E.map((e, i) => i).filter((i) => on[i]), mid = (P, k) => idx.reduce((s, i) => s + P[i][k], 0) / idx.length;
  const tx = mid(T, 0), ty = mid(T, 1), mx = mid(X, 0), my = mid(X, 1), Q = X.map((p) => [p[0] - mx, p[1] - my]);
  const fit = (P) => { let a = 0, b = 0; idx.forEach((i) => { const u = T[i][0] - tx, v = T[i][1] - ty; a += P[i][0] * u + P[i][1] * v; b += P[i][0] * v - P[i][1] * u; });
    const t = Math.atan2(b, a), c = Math.cos(t), s = Math.sin(t); return { q: Math.hypot(a, b), P: P.map((p) => [p[0] * c - p[1] * s + tx, p[0] * s + p[1] * c + ty]) }; };
  const A = fit(Q), B = fit(Q.map((p) => [p[0], -p[1]]));
  return A.q >= B.q ? A.P : B.P;
}
// map_base.json is a fixed input. An engine it does not hold starts at the weighted mean of its three most correlated engines
function startFromFile(C) {
  const known = E.map((e) => D.mapBase[e.id] || null);
  return known.map((p, i) => {
    if (p) return p.slice();
    const near = known.map((q, j) => [q, Math.max(C[i][j], 0) + 1e-6]).filter((u) => u[0]).sort((u, v) => v[1] - u[1]).slice(0, 3), sum = near.reduce((s, u) => s + u[1], 0);
    return [0, 1].map((k) => near.reduce((s, u) => s + u[0][k] * u[1], 0) / sum);
  });
}
// one frame for all periods: the fixed layout refitted to since 2008 and held to it, so a new close moves dots a little and never flips the map
const SINCE_START = corrOf(0), TREE_ALL = SINCE_START.TREE;
const BASE = (() => {
  const everyone = E.map(() => true), C = SINCE_START.CORR;
  if (D.mapBase && E.filter((e) => D.mapBase[e.id]).length >= 3) { const start = startFromFile(C); return alignTo(start, fitCoords(C, start, everyone), E.map((e) => !!D.mapBase[e.id])); }
  // no layout on file: since 2008 on its principal axes, rotation to the left, momentum to the top
  const XY = fitCoords(C, classicalStart(C), everyone);
  let sxx = 0, syy = 0, sxy = 0; XY.forEach((p) => { sxx += p[0] * p[0]; syy += p[1] * p[1]; sxy += p[0] * p[1]; });
  const th = 0.5 * Math.atan2(2 * sxy, sxx - syy), c = Math.cos(th), s = Math.sin(th);
  const xs = XY.map((p) => p[0] * c + p[1] * s), ys = XY.map((p) => -p[0] * s + p[1] * c);
  const avg = (arr, fam) => { const idx = E.map((e, i) => e.fam === fam ? i : -1).filter((i) => i >= 0); return idx.reduce((t, i) => t + arr[i], 0) / idx.length; };
  const fx = avg(xs, 'rotation') > avg(xs, 'dips') ? -1 : 1, fy = avg(ys, 'momentum') > avg(ys, 'macro') ? -1 : 1;
  return xs.map((x, i) => [x * fx, ys[i] * fy]);
})();
// who can sit on the map in a window: data in it, and enough weeks of its own for correlations
function placedIn(a) {
  const k0 = firstWeek(a), few = WEEKS.length - k0 < WEEKS_MIN;
  return E.map((e, i) => { if (e.lastI < a) return false; if (few) return true; let n = 0; for (let k = k0; k < WEEKS.length; k++) if (!Number.isNaN(WK[i][k])) n++; return n >= WEEKS_MIN; });
}
// every other window starts from BASE and is held to it, so a click moves a dot as far as its correlations moved.
// 30 steps: enough for the shapes, and a dot with little to go on stays put instead of hopping from one close to the next
const WINDOW_STEPS = 30;
const COORDS = new Map(PERIODS.map((p) => {
  const a = p.a(), on = placedIn(a), weeks = WEEKS.length - firstWeek(a);
  return [p.id, { on, weeks, XY: a === 0 || weeks < WEEKS_MIN ? BASE : alignTo(BASE, fitCoords(corrOf(a).CORR, BASE, on, WINDOW_STEPS), on) }];
}));
// one frame, so a click moves a dot only as far as its coordinates moved: its width from windows of half a year or more, its height from the long ones
const FRAME = (() => {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  COORDS.forEach(({ XY, on, weeks }) => XY.forEach((p, i) => { if (on[i] && weeks >= 26) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); if (weeks >= 104) { y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); } } }));
  return { x0, x1, y0, y1 };
})();
let simCoords = null, simOn = null;
function computeCoords() { ({ XY: simCoords, on: simOn } = COORDS.get(period.id)); }
function hullOf(pts) {
  if (pts.length < 3) return pts;
  const s = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = []; s.forEach((p) => { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); });
  const up = []; s.slice().reverse().forEach((p) => { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); });
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
let simPts = [], simHot = null, simOpened = false;
function renderSim() {
  const box = $('simbox'), n = E.length;
  const W = Math.max(700, box.clientWidth), side = W < 1000 ? 120 : 240;
  // equal axes: a distance reads the same in every direction. One scale for all windows; a short window that reaches past the frame draws taller
  const placed = simCoords.filter((c, i) => simOn[i]), xs = placed.map((c) => c[0]), ys = placed.map((c) => c[1]);
  const x0 = Math.min(FRAME.x0, ...xs), rx = Math.max(FRAME.x1, ...xs) - x0 || 1, y0 = Math.min(FRAME.y0, ...ys), ry = Math.max(FRAME.y1, ...ys) - y0 || 1;
  const k = Math.min((860 - 110) / (FRAME.y1 - FRAME.y0 || 1), (W - side) / rx);
  const H = Math.round(Math.max(520, 110 + k * ry));
  const ox = (W - k * rx) / 2, oy = (H - k * ry) / 2;
  const pts = simPts = simCoords.map((c) => [ox + k * (c[0] - x0), oy + k * (c[1] - y0)]);
  simHot = null;
  dotSpots.simbox = E.map((e, i) => ({ id: e.id, x: pts[i][0], y: pts[i][1] })).filter((p, i) => simOn[i]);
  const short = shortWindow(), found = groupsOnMap(), several = short ? [] : found.filter((g) => g.length > 1);
  const hulls = several.map((g) => {
    const hull = hullOf(g.map((i) => pts[i]));
    return `<path class="hull${g.some((i) => shows(E[i])) ? '' : ' dim'}" d="${hull.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join('')}${hull.length > 2 ? 'Z' : ''}" stroke-width="46"/>`;
  }).join('');
  // lines: pairs a shape does not already show, never the parked engine
  const groupOf = new Array(n).fill(-1); found.forEach((g, q) => g.forEach((i) => { groupOf[i] = q; }));
  let pairs = '';
  const segs = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    if (!simOn[i] || !simOn[j] || E[i].parked || E[j].parked || CORR[i][j] < cutValue || (!short && groupOf[i] === groupOf[j])) continue;
    segs.push([pts[i][0], pts[i][1], pts[j][0], pts[j][1]]);
    pairs += `<line class="pair${shows(E[i]) && shows(E[j]) ? '' : ' dim'}" x1="${pts[i][0].toFixed(1)}" y1="${pts[i][1].toFixed(1)}" x2="${pts[j][0].toFixed(1)}" y2="${pts[j][1].toFixed(1)}" stroke="var(--axis)"/>`;
  }
  const caption = `${fewWeeks() ? `Flat fit of the whole history: ${plural(windowWeeks(), 'week')} is too short.` : 'Flat fit.'} Hover a dot for its exact pairs.`;
  const boxes = pts.filter((p, i) => simOn[i]).map((p) => ({ x0: p[0] - 10, x1: p[0] + 10, y0: p[1] - 10, y1: p[1] + 10 }));
  // the caption starts where the scroller opens
  const capX = Math.max(4, (W - box.clientWidth) / 2 + 4);
  boxes.push({ x0: capX - 4, x1: capX + caption.length * 6.6 + 8, y0: H - 24, y1: H });
  // group labels sit above their shape, clear of dots and of each other
  const hits = (b) => boxes.some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0);
  const glabels = several.map((g) => {
    const text = `${majorityFamily(g)} ${g.length}`, half = text.length * 3.7 + 4;
    const inside = (x) => Math.max(half + 2, Math.min(W - half - 2, x));
    const peak = g.reduce((t, i) => pts[i][1] < pts[t][1] ? i : t, g[0]);
    const mid = inside(g.reduce((s, i) => s + pts[i][0], 0) / g.length), over = inside(pts[peak][0]), top = pts[peak][1] - 30;
    const rect = ([x, y]) => ({ x0: x - half, x1: x + half, y0: y - 13, y1: y + 4 });
    // over the middle of the group, else over its top dot, else a little higher
    const spots = [0, 14, 28].flatMap((up) => [[mid, Math.max(16, top - up)], [over, Math.max(16, top - up)]]);
    const spot = spots.find((p) => !hits(rect(p))) || spots[0];
    boxes.push(rect(spot));
    return `<text class="glabel${g.some((i) => shows(E[i])) ? '' : ' dim'}" x="${spot[0].toFixed(1)}" y="${spot[1].toFixed(1)}" text-anchor="middle">${esc(text)}</text>`;
  }).join('');
  const place = labelPlacer(W, H, { l: 2, r: 2, t: 2, b: 2 }, boxes, segs);
  const order = E.map((e, i) => i).filter((i) => simOn[i]).sort((p, q) => (E[q].stage === 'LIVE') - (E[p].stage === 'LIVE'));
  const rOf = (e) => e.stage === 'LIVE' ? 10 : 8;
  const labels = [];
  order.forEach((i) => { labels[i] = place(pts[i][0], pts[i][1], rOf(E[i]), E[i].name); });
  const lines = labels.map((l, i) => l && l.leader ? `<line${shows(E[i]) ? '' : ' class="dim"'} x1="${pts[i][0]}" y1="${pts[i][1]}" x2="${l.leader[0]}" y2="${l.leader[1]}" stroke="var(--axis)"/>` : '').join('');
  const dots = E.map((e, i) => {
    if (!simOn[i]) return '';
    const hollow = chipClass(e.stage) === 'PM', state = (shows(e) ? '' : ' dim') + (e.parked ? ' parked' : '');
    return `<g class="pt s-${stageKey(e.stage)}${state}" data-open="${e.id}" data-hover="${e.id}" tabindex="0" role="button" aria-label="${esc(e.name)}"><circle class="hit" cx="${pts[i][0]}" cy="${pts[i][1]}" r="16"/><circle class="mk" cx="${pts[i][0]}" cy="${pts[i][1]}" r="${rOf(e)}" style="fill:${hollow ? 'var(--surface)' : 'var(--c)'};stroke:${hollow ? 'var(--c)' : 'var(--surface)'}" stroke-width="2.5"/></g><text class="lbl${e.stage === 'LIVE' ? ' live' : ''}${state}" data-open="${e.id}" data-hover="${e.id}" x="${labels[i].x}" y="${labels[i].y}">${esc(e.name)}</text>`;
  }).join('');
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="Engines by how much they move together">${hulls}${pairs}${lines}${glabels}<g id="s-hover"></g>${dots}<text class="axis-t" x="${capX}" y="${H - 8}">${esc(caption)}</text></svg>`;
  // wider than its box, it opens on its middle, where the groups are; after that the place is the reader's
  if (!simOpened) { simOpened = true; box.scrollLeft = (W - box.clientWidth) / 2; }
}
// everything at or above the cut, closest first
function partnersOf(i) { return E.map((e, j) => j).filter((j) => j !== i && simOn[j] && CORR[i][j] >= cutValue).sort((p, q) => CORR[i][q] - CORR[i][p]); }
function simHover(id) {
  const g = $('s-hover');
  if (!g || id === simHot) return;
  simHot = id;
  const i = id ? E.indexOf(byId[id]) : -1;
  g.innerHTML = i < 0 || E[i].parked ? '' : partnersOf(i).filter((j) => !E[j].parked).map((j) => `<line x1="${simPts[i][0].toFixed(1)}" y1="${simPts[i][1].toFixed(1)}" x2="${simPts[j][0].toFixed(1)}" y2="${simPts[j][1].toFixed(1)}" stroke="var(--ink-3)"/>`).join('');
}
function simTip(e) {
  const i = E.indexOf(e), found = partnersOf(i);
  return `<b>${eName(e)}</b>` + (found.length ? found.slice(0, 5).map((j) => `<div class="r"><i>${eName(E[j])}</i><b>${num(CORR[i][j])}</b></div>`).join('') : `<div style="opacity:.8">${esc(`Nothing above ${cutValue.toFixed(2)}`)}</div>`);
}
function renderBetStats() {
  const short = shortWindow(), found = groupsOnMap(), several = found.filter((g) => g.length > 1).length, cut = cutValue.toFixed(2), weeks = plural(windowWeeks(), 'week'), parked = E.filter((e) => e.parked).map((e) => e.name);
  $('simnote').innerHTML = esc(`Engines sit by how much they move together. A grey shape wraps one group. A thin line joins two engines from different groups that still move together. Two dimensions cannot hold all ${E.length}, so read shapes and lines, not small gaps.`);
  $('b-count').hidden = short;
  $('b-groups').textContent = found.length;
  $('b-split').textContent = `${several} with several engines, ${found.length - several} alone${parked.length ? `, ${parked.join(' and ')} parked` : ''}`;
  $('b-cutlabel').innerHTML = esc(fewWeeks() ? `Lines join pairs above ${cut} over the whole history: ${weeks} is too short` : short ? `Lines join pairs above ${cut}. ${weeks} is too short to group` : `Groups join when their average correlation is above ${cut}. ${weeks}`);
}

/* correlation heat with its tree */
function renderCorr() {
  const n = E.length, LABEL = 118, TOP = 112, room = $('corrbox').parentNode.clientWidth - LABEL - 52, TREEW = room >= 110 + 24 * n ? 110 : 56;
  // cells take the room there is, 36 px at most; at 24 px the tree narrows, then the matrix scrolls: the tree goes first and the names stay
  const CELL = Math.max(24, Math.min(36, Math.floor((room - TREEW) / n)));
  const order = TREE.leaves, pos = new Array(n); order.forEach((id, k) => { pos[id] = k; });
  const size = n * CELL;
  const maxH = TREE.h || 1;
  const hx = (h) => Math.max(2, TREEW - 2 - (h / maxH) * (TREEW - 6));
  let tree = '';
  const walk = (node) => {
    if (!node.left) return { y: TOP + pos[node.id] * CELL + CELL / 2, x: hx(0) };
    const a = walk(node.left), b = walk(node.right), x = hx(node.h);
    tree += `M${a.x} ${a.y}H${x}V${b.y}H${b.x}`;
    return { y: (a.y + b.y) / 2, x };
  };
  walk(TREE);
  const cutX = hx(1 - cutValue);
  const treePart = `<path class="tree" d="${tree}"/><line class="cut" x1="${cutX}" x2="${cutX}" y1="${TOP - 4}" y2="${TOP + size + 2}"/>`;
  let names = '', s = '';
  order.forEach((id, k) => {
    const a = E[id], y = TOP + k * CELL, on = (a.stage === 'LIVE' ? ' live' : '') + (shows(a) && simOn[id] ? '' : ' dim');
    names += `<g class="s-${stageKey(a.stage)}${shows(a) && simOn[id] ? '' : ' dim'}"><circle cx="8" cy="${y + CELL / 2 - 1}" r="4.5" style="fill:${chipClass(a.stage) === 'PM' ? 'var(--surface)' : 'var(--c)'};stroke:var(--c);stroke-width:1.5"/></g><text class="rowl${on}" data-r="${id}" data-open="${a.id}"${tipAttr(a.sub)} x="${LABEL - 8}" y="${y + CELL / 2 + 4}" text-anchor="end">${esc(a.name)}</text>`;
    s += `<text class="rowl${on}" data-c="${id}" data-open="${a.id}"${tipAttr(a.sub)} transform="translate(${k * CELL + CELL / 2 + 3},${TOP - 8}) rotate(-52)">${esc(a.name)}</text>`;
    order.forEach((jd, j) => {
      if (!simOn[id] || !simOn[jd]) return;
      const v = CORR[id][jd];
      s += `<rect class="cell${shows(a) && shows(E[jd]) ? '' : ' dim'}" data-i="${id}" data-j="${jd}" x="${j * CELL}" y="${y}" width="${CELL - 3}" height="${CELL - 3}" rx="3" style="fill:${id === jd ? 'var(--axis)' : heat(v)}"/>`;
    });
  });
  if (!shortWindow()) betGroups().forEach((g) => {
    if (g.length < 2) return;
    const ks = g.map((i) => pos[i]), k0 = Math.min(...ks), len = Math.max(...ks) - k0 + 1;
    s += `<rect class="gbox${g.some((i) => shows(E[i])) ? '' : ' dim'}" x="${k0 * CELL - 2}" y="${TOP + k0 * CELL - 2}" width="${len * CELL + 1}" height="${len * CELL + 1}" rx="4"/>`;
  });
  // the last column's slanted label reaches past the cells
  const SW = size + 46 + (36 - CELL) / 2, SH = TOP + size + 6;
  const part = (w, inner, cls = '') => `<svg${cls ? ` class="${cls}"` : ''} width="${w}" height="${SH}" viewBox="0 0 ${w} ${SH}">${inner}</svg>`;
  $('corrnote').textContent = fewWeeks() ? `Whole history: ${plural(windowWeeks(), 'week')} is too short.` : '';
  $('corrbox').innerHTML = `<div class="corrparts" role="img" aria-label="Weekly correlation between engines, by cluster">${part(TREEW, treePart)}${part(LABEL, names, 'corrnames')}${part(SW, s)}</div>`;
}
function setCutFill() { $('b-cut').style.setProperty('--fill', ((cutValue - 0.2) / 0.7 * 100) + '%'); }
function renderBets() { computeCorr(); computeCoords(); renderBetStats(); renderSim(); renderCorr(); }
function renderCut() { renderBetStats(); renderSim(); renderCorr(); if (groupMode === 'cluster') renderTiles(); }

/* calm and stress: all history, weekly. Stress is a week that ends with QQQ 10%+ below its peak, or one of QQQ's worst 10% of weeks */
const QW = weeklyOf(QQQ);
const QDD = (() => { const o = new Float64Array(WEEKS.length); let eq = 1, pk = 1, start = 0; WEEKS.forEach((end, k) => { for (let i = start; i <= end; i++) if (!Number.isNaN(QQQ[i])) { eq *= 1 + QQQ[i]; pk = Math.max(pk, eq); } start = end + 1; o[k] = eq / pk - 1; }); return o; })();
const STRESS_DEFS = [['dd', 'QQQ 10%+ below its peak', 'Weeks that end with QQQ at least 10% below its highest close so far.'], ['tail', "QQQ's worst 10% of weeks", 'The tenth of all weeks in which QQQ fell most.']];
let stressDef = pick(STRESS_DEFS, 'em-stress', 'dd');
const STRESS_MIN = 13, JUMP_MIN = 52;
function corrOn(x, y, keep) {
  let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let k = 0; k < x.length; k++) { const a = x[k], b = y[k]; if (!keep[k] || Number.isNaN(a) || Number.isNaN(b)) continue; n++; sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; }
  const dx = sxx - sx * sx / n, dy = syy - sy * sy / n;
  return { v: n >= STRESS_MIN && dx > 0 && dy > 0 ? (sxy - sx * sy / n) / Math.sqrt(dx * dy) : NaN, n };
}
// a strategy's average stress week, and QQQ's in the same weeks: a short history sees only the stress it lived through
function meanOn(x, keep) { let n = 0, s = 0, q = 0; for (let k = 0; k < x.length; k++) if (keep[k] && !Number.isNaN(x[k])) { n++; s += x[k]; q += QW[k]; } return { v: n ? s / n * 100 : NaN, q: n ? q / n * 100 : NaN, n }; }
const stressSeen = new Map();
function stressOf(def) {
  if (stressSeen.has(def)) return stressSeen.get(def);
  const ok = Array.from(QW, (v) => !Number.isNaN(v));
  const sorted = Array.from(QW).filter((v) => !Number.isNaN(v)).sort((p, q) => p - q), cut = sorted[Math.max(0, Math.floor(sorted.length * 0.1) - 1)];
  const stress = ok.map((y, k) => y && (def === 'dd' ? QDD[k] <= -0.10 : QW[k] <= cut)), calm = ok.map((y, k) => y && !stress[k]);
  const n = E.length, pair = (keep) => { const M = Array.from({ length: n }, () => new Array(n)); for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) M[i][j] = M[j][i] = corrOn(WK[i], WK[j], keep); return M; };
  const qMean = (keep) => meanOn(QW, keep).v;
  const out = { stress, calm, nS: stress.filter(Boolean).length, nC: calm.filter(Boolean).length, qS: qMean(stress), qC: qMean(calm),
    rows: E.map((e, i) => ({ i, c: corrOn(WK[i], QW, calm), s: corrOn(WK[i], QW, stress), m: meanOn(WK[i], stress) })), CC: pair(calm), CS: pair(stress) };
  stressSeen.set(def, out);
  return out;
}
function tailOf(i) {
  const ks = []; for (let k = 0; k < WEEKS.length; k++) if (!Number.isNaN(WK[i][k]) && !Number.isNaN(QW[k])) ks.push(k);
  const cutOf = (x) => { const v = ks.map((k) => x[k]).sort((p, q) => p - q); return v[Math.max(0, Math.floor(v.length * 0.1) - 1)]; };
  const qc = cutOf(QW), sc = cutOf(WK[i]);
  let n = 0, hit = 0; ks.forEach((k) => { if (QW[k] <= qc) { n++; if (WK[i][k] <= sc) hit++; } });
  return { i, n, hit, share: n ? hit / n : NaN };
}
const corrWord = (c) => (Number.isNaN(c.v) ? `too few weeks (${c.n})` : num(c.v));
function renderStress() {
  const S = stressOf(stressDef), n = E.length;
  $('stressdef').innerHTML = STRESS_DEFS.map(([k, name, tipText]) => `<button class="filter-chip" data-stressdef="${k}" aria-pressed="${stressDef === k}"${tipAttr(tipText)}>${esc(name)}</button>`).join('');
  const avg = (M) => { let s = 0, k = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (!Number.isNaN(M[i][j].v)) { s += M[i][j].v; k++; } return k ? s / k : NaN; };
  const rated = S.rows.filter((r) => !Number.isNaN(r.s.v)), gains = rated.filter((r) => r.m.v > 0).length;
  $('stressband').innerHTML =
    `<div${tipAttr('A correlation measured on this many weeks moves by about this much from noise alone.')}><span>Stress weeks</span><b>${S.nS}</b><small>of ${S.nS + S.nC}, noise about ±${(2 / Math.sqrt(Math.max(1, S.nS))).toFixed(2)}</small></div>` +
    `<div><span>QQQ, average week</span><b>${pct(S.qS, 2)}</b><small>in stress, ${pct(S.qC, 2)} in calm</small></div>` +
    `<div><span>Gain in stress</span><b>${gains} of ${rated.length}</b><small>strategies up in an average stress week</small></div>` +
    `<div${tipAttr('Average of every pair of strategies.')}><span>Strategies together</span><b>${num(avg(S.CC))} → ${num(avg(S.CS))}</b><small>average correlation, calm to stress</small></div>`;

  // one row per strategy: correlation with QQQ, calm and stress; then its average stress week against QQQ in the same weeks
  const list = S.rows.slice().sort((p, q) => (Number.isNaN(p.s.v) ? 9 : -p.s.v) - (Number.isNaN(q.s.v) ? 9 : -q.s.v));
  const box = $('stressrows'), W = Math.max(720, box.clientWidth), NAME = 150, GAP = 34, RH = 23, TOP = 34, H = TOP + list.length * RH + 26;
  const c0 = NAME, c1 = c0 + Math.round((W - NAME - GAP) * 0.56), b0 = c1 + GAP, b1 = W - 62;
  const cs = list.flatMap((r) => [r.c.v, r.s.v]).filter((v) => !Number.isNaN(v));
  const CLO = Math.min(-0.5, Math.floor((Math.min(...cs) - 0.05) * 10) / 10);
  const cx = (v) => c0 + 8 + (v - CLO) / (1 - CLO) * (c1 - c0 - 16);
  const ms = list.flatMap((r) => [r.m.v, r.m.q]).filter((v) => !Number.isNaN(v));
  const step = niceStep(Math.max(...ms, 0) - Math.min(...ms, 0), 5), BLO = Math.floor(Math.min(0, ...ms) / step) * step, BHI = Math.ceil(Math.max(0, ...ms) / step) * step;
  const bx = (v) => b0 + (v - BLO) / (BHI - BLO || 1) * (b1 - b0);
  let s = `<text class="axis-n" x="${c0 + 8}" y="14">Correlation with QQQ</text><text class="axis-n" x="${b0}" y="14">Average week in stress</text>`;
  for (let v = Math.ceil(CLO * 2) / 2; v <= 1 + 1e-9; v += 0.5) s += `<line x1="${cx(v)}" x2="${cx(v)}" y1="${TOP - 8}" y2="${H - 22}" stroke="${Math.abs(v) < 1e-9 ? 'var(--axis)' : 'var(--grid)'}"/><text class="axis-t" x="${cx(v)}" y="${H - 6}" text-anchor="middle">${tickText(v, 0.5)}</text>`;
  for (let v = BLO; v <= BHI + 1e-9; v += step) s += `<line x1="${bx(v)}" x2="${bx(v)}" y1="${TOP - 8}" y2="${H - 22}" stroke="${Math.abs(v) < 1e-9 ? 'var(--axis)' : 'var(--grid)'}"/><text class="axis-t" x="${bx(v)}" y="${H - 6}" text-anchor="middle">${tickText(v, step)}%</text>`;
  list.forEach((r, k) => {
    const e = E[r.i], y = TOP + k * RH + RH / 2, dim = shows(e) ? '' : ' dim';
    const cTip = `${e.name}, correlation with QQQ: calm ${corrWord(r.c)}, stress ${corrWord(r.s)} on ${plural(r.s.n, 'week')}, noise about ±${(2 / Math.sqrt(Math.max(1, r.s.n))).toFixed(2)}`;
    s += `<g class="s-${stageKey(e.stage)}${dim}"><circle cx="8" cy="${y}" r="4.5" style="fill:${chipClass(e.stage) === 'PM' ? 'var(--surface)' : 'var(--c)'};stroke:var(--c);stroke-width:1.5"/></g>`;
    s += `<text class="rowl${e.stage === 'LIVE' ? ' live' : ''}${dim}" data-open="${e.id}"${tipAttr(e.sub)} x="${NAME - 10}" y="${y + 4}" text-anchor="end">${esc(e.name)}</text>`;
    s += `<g class="${dim.trim()}">`;
    if (!Number.isNaN(r.s.v)) { const nz = 2 / Math.sqrt(r.s.n), lo = Math.max(CLO, r.s.v - nz), hiv = Math.min(1, r.s.v + nz); s += `<rect class="k-noise" x="${cx(lo)}" y="${y - 7}" width="${cx(hiv) - cx(lo)}" height="14" rx="3"/>`; }
    if (!Number.isNaN(r.c.v) && !Number.isNaN(r.s.v)) s += `<line x1="${cx(r.c.v)}" x2="${cx(r.s.v)}" y1="${y}" y2="${y}" stroke="var(--edge)" stroke-width="3"/>`;
    if (!Number.isNaN(r.c.v)) s += `<circle class="k-calm" cx="${cx(r.c.v)}" cy="${y}" r="4.5"${tipAttr(cTip)}/>`;
    if (!Number.isNaN(r.s.v)) s += `<circle class="k-stress" cx="${cx(r.s.v)}" cy="${y}" r="5"${tipAttr(cTip)}/>`;
    else s += `<text class="axis-t" x="${cx(0)}" y="${y + 4}" text-anchor="middle">Too few stress weeks</text>`;
    if (!Number.isNaN(r.m.v)) {
      const a = bx(Math.min(0, r.m.v)), w = Math.max(1, Math.abs(bx(r.m.v) - bx(0)));
      const mTip = `${e.name}: ${pct(r.m.v, 2)} in an average stress week, QQQ ${pct(r.m.q, 2)} in the same ${plural(r.m.n, 'week')}`;
      s += `<rect x="${a}" y="${y - 6}" width="${w}" height="12" rx="2" fill="var(${r.m.v >= 0 ? '--pos' : '--neg'})" opacity=".85"${tipAttr(mTip)}/>`;
      s += `<line class="k-tick" x1="${bx(r.m.q)}" x2="${bx(r.m.q)}" y1="${y - 9}" y2="${y + 9}"${tipAttr(mTip)}/>`;
      // the numbers stand in one column at the right, clear of bars and ticks
      s += `<text class="axis-t${r.m.v < 0 ? ' neg' : ''}" x="${W - 4}" y="${y + 4}" text-anchor="end">${pct(r.m.v, 2)}</text>`;
    }
    s += '</g>';
  });
  box.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Correlation with QQQ in calm and stress, and the average stress week">${s}</svg>`;

  // tail dependence: of QQQ's worst 10% of weeks in the strategy's history, how many were also among the strategy's own worst 10%
  const tails = E.map((e, i) => tailOf(i)).filter((r) => r.n >= STRESS_MIN).sort((p, q) => q.share - p.share);
  const tbox = $('stresstail'), TW = Math.max(420, tbox.clientWidth), TNAME = 150, TTOP = 46, TH = TTOP + tails.length * RH + 26, t0 = TNAME, t1 = TW - 96;
  const hi = Math.max(0.3, ...tails.map((r) => r.share)), tstep = hi > 0.5 ? 0.2 : 0.1, THI = Math.ceil(hi / tstep) * tstep;
  const tx = (v) => t0 + v / THI * (t1 - t0);
  let g = `<text class="axis-n" x="${t0}" y="14">Share of QQQ's worst weeks</text>`;
  for (let v = 0; v <= THI + 1e-9; v += tstep) g += `<line x1="${tx(v)}" x2="${tx(v)}" y1="${TTOP - 8}" y2="${TH - 22}" stroke="var(--grid)"/><text class="axis-t" x="${tx(v)}" y="${TH - 6}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
  g += `<line x1="${tx(0.1)}" x2="${tx(0.1)}" y1="${TTOP - 8}" y2="${TH - 22}" stroke="var(--ink)" stroke-dasharray="3 3"/><text class="axis-t" x="${tx(0.1)}" y="${TTOP - 12}" text-anchor="middle">Unrelated 10%</text>`;
  tails.forEach((r, k) => {
    const e = E[r.i], y = TTOP + k * RH + RH / 2, dim = shows(e) ? '' : ' dim';
    const noise = 2 * Math.sqrt(0.1 * 0.9 / r.n);
    const tTip = `${e.name}: in ${r.hit} of QQQ's ${plural(r.n, 'worst week')} it had one of its own. Unrelated moves would give about 10%, ±${Math.round(noise * 100)}% from noise.`;
    g += `<g class="s-${stageKey(e.stage)}${dim}"><circle cx="8" cy="${y}" r="4.5" style="fill:${chipClass(e.stage) === 'PM' ? 'var(--surface)' : 'var(--c)'};stroke:var(--c);stroke-width:1.5"/></g>`;
    g += `<text class="rowl${e.stage === 'LIVE' ? ' live' : ''}${dim}" data-open="${e.id}"${tipAttr(e.sub)} x="${TNAME - 10}" y="${y + 4}" text-anchor="end">${esc(e.name)}</text>`;
    g += `<g class="${dim.trim()}"><rect x="${t0}" y="${y - 6}" width="${Math.max(1, tx(r.share) - t0)}" height="12" rx="2" fill="var(${r.share > 0.1 + noise ? '--neg' : '--ink-3'})" opacity="${r.share > 0.1 + noise ? '.85' : '.45'}"${tipAttr(tTip)}/>`;
    g += `<text class="axis-t" x="${TW - 4}" y="${y + 4}" text-anchor="end">${Math.round(r.share * 100)}% of ${r.n}</text></g>`;
  });
  tbox.innerHTML = `<svg width="${TW}" height="${TH}" viewBox="0 0 ${TW} ${TH}" role="img" aria-label="Share of QQQ's worst weeks in which each strategy also had one of its own worst">${g}</svg>`;

  // the pairs whose correlation rises most, on enough stress weeks to mean something
  const pairs = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const c = S.CC[i][j], st = S.CS[i][j];
    if (Number.isNaN(c.v) || Number.isNaN(st.v) || st.n < JUMP_MIN || !shows(E[i]) || !shows(E[j])) continue;
    pairs.push({ i, j, c: c.v, s: st.v, n: st.n, d: st.v - c.v });
  }
  pairs.sort((p, q) => q.d - p.d);
  // at most two rows for one pair of types, so one family's siblings do not fill the table
  const perKind = new Map(), top = [];
  for (const p of pairs) { if (p.d <= 0 || top.length >= 10) break; const kind = [E[p.i].fam, E[p.j].fam].sort().join('|'), k = perKind.get(kind) || 0; if (k < 2) { perKind.set(kind, k + 1); top.push(p); } }
  $('stressjumps').innerHTML = '<thead><tr><th>Pair</th><th class="r">Calm</th><th class="r">Stress</th><th class="r">Change</th><th class="r">Weeks</th></tr></thead><tbody>' +
    (top.length ? top.map((p) => `<tr><td><span data-open="${E[p.i].id}">${esc(E[p.i].name)}</span> · <span data-open="${E[p.j].id}">${esc(E[p.j].name)}</span></td><td class="r">${num(p.c)}</td><td class="r">${num(p.s)}</td><td class="r neg">${signed(p.d, 2)}</td><td class="r">${p.n}</td></tr>`).join('') : '<tr><td colspan="5">No pair rises in stress.</td></tr>') + '</tbody>';
  $('stressjumpnote').textContent = `Pairs with at least ${JUMP_MIN} stress weeks together; at most two for one pair of types.`;
}

/* years and falls */
const YEARS = [...new Set(YEAR)];
function yearReturns(r) { const o = {}; YEARS.forEach((y) => { const a = YEAR.indexOf(y), b = YEAR.lastIndexOf(y); let eq = 1, any = false; for (let i = a; i <= b; i++) { const v = r[i]; if (!Number.isNaN(v)) { eq *= 1 + v; any = true; } } o[y] = any ? (eq - 1) * 100 : NaN; }); return o; }
const FALLS = D.falls.map((f) => ({ ...f, a: idxOf(f.from) + 1, b: idxOf(f.to) }));
// a first year that opens late says so
function yearSpan(y) { return y === YEAR[0] && ISO[0].slice(5) > '01-07' ? spanLabel(0, YEAR.lastIndexOf(y)) : String(y); }
function ownCell(own, e) {
  const tip = `${spanLabel(own.from, own.to)}: ${pct(own.v)}${Number.isNaN(own.qqq) ? '' : `, QQQ ${signed(own.qqq)}%`}`;
  if (!e || !e.standIn || ISO[own.to] >= e.standIn.to) return heatCell(own.v, 40, minus(own.v, 0), tip);
  // the fall sits in years the traded rule did not exist: no fill, and the tooltip says why
  return `<td class="plain dotted" data-tip="${esc(`${tip}. ${standInLabel(e)}. Since then ${pct(statsOf(R[e.id], idxOf(e.standIn.to), LAST).dd)}.`)}">${minus(own.v, 0)}</td>`;
}
function renderYears() {
  const yearCols = YEARS.map((y) => sortTh('years', `y${y}`, String(y).slice(2), yearSpan(y) === String(y) ? '' : tipAttr(yearSpan(y)))).join('');
  const fallCols = FALLS.map((f, k) => sortTh('years', `f${k}`, `${f.to.slice(2, 4)}<br>${minus(f.depth, 0)}%`, ` data-tip="${esc(shortLabel(f.from))} to ${esc(shortLabel(f.to))}"`)).join('');
  const ownCol = sortTh('years', 'own', 'Own<br>worst', tipAttr('Its own deepest fall since 2008, whatever QQQ did.'));
  const row = (name, r, attrs, own, e) => {
    const y = yearReturns(r), part = e && e.lastI < LAST ? YEAR[e.lastI] : null;
    return `<tr ${attrs}>${name}${YEARS.map((yr) => yr === part ? plainCell(Number.isNaN(y[yr]) ? '' : minus(y[yr], 0), `to ${monthDay(e.end)}`) : heatCell(y[yr], 40, Number.isNaN(y[yr]) ? '' : minus(y[yr], 0), Number.isNaN(y[yr]) ? '' : `${yearSpan(yr)}: ${pct(y[yr])}`)).join('')}<td class="gap"></td>${FALLS.map((f) => { const v = compound(r, f.a, f.b), none = Number.isNaN(v), span = `${shortLabel(f.from)} to ${shortLabel(f.to)}`; return heatCell(v, 40, none ? '' : minus(v, 0), none ? `${span}: no data` : `${span}: ${pct(v)}, QQQ ${pct(f.depth)}`); }).join('')}<td class="gap"></td>${ownCell(own, e)}</tr>`;
  };
  const key = (e, col) => col === 'own' ? e.own.v : col[0] === 'y' ? yearReturns(R[e.id])[col.slice(1)] : compound(R[e.id], FALLS[+col.slice(1)].a, FALLS[+col.slice(1)].b);
  // equal columns per group: the name and the other fixed columns, 28 px a figure at the least, and the 3 px between them
  const figures = YEARS.length + FALLS.length, narrowest = 226 + 28 * figures + 3 * (figures + 5);
  let h = `<table class="heat fixed" style="min-width:${narrowest}px"><colgroup><col style="width:124px"><col span="${YEARS.length}"><col style="width:22px"><col span="${FALLS.length}" style="width:3.45%"><col style="width:22px"><col style="width:58px"></colgroup><thead><tr>${TH_GAP}<th class="group" colspan="${YEARS.length}">Calendar year, %</th>${TH_GAP}<th class="group" colspan="${FALLS.length}">${esc('While QQQ fell, % (year of the low, depth)')}</th>${TH_GAP}${TH_GAP}</tr><tr>${TH_NAME}${yearCols}${TH_GAP}${fallCols}${TH_GAP}${ownCol}</tr></thead><tbody>`;
  sortRows('years', E, key).forEach((e) => { h += row(nameCell(e), R[e.id], `class="clk${shows(e) ? '' : ' dim'}" data-open="${e.id}" data-hover="${e.id}"`, e.own, e); });
  h += row('<td class="n">QQQ</td>', QQQ, 'class="ref"', QQQ_OWN);
  $('yearbox').innerHTML = h + '</tbody></table>';
}

/* rolling Sharpe */
function rollingSharpe(r, win, endI) {
  let n = 0, x = 0, x2 = 0;
  for (let i = Math.max(0, endI - win + 1); i <= endI; i++) { const v = r[i]; if (Number.isNaN(v)) continue; const e = v - (Number.isNaN(TB[i]) ? 0 : TB[i]); n++; x += e; x2 += e * e; }
  if (n < win * 0.8) return NaN;
  const m = x / n, s = Math.sqrt(Math.max(0, (x2 - n * m * m) / (n - 1)));
  return s * Math.sqrt(252) > 0.015 ? m / s * Math.sqrt(252) : NaN;
}
/* factors */
const FCOLS = [['etf', 'QQQ', 'QQQ'], ['etf', 'IEF', 'Bonds'], ['etf', 'GLD', 'Gold'], ['etf', 'DBC', 'Commodities'], ['etf', 'UUP', 'Dollar'], ['ff', 'Mkt-RF', 'Market'], ['ff', 'SMB', 'Size'], ['ff', 'HML', 'Value'], ['ff', 'MOM', 'Momentum']];
// in a narrow column the long name may break, with its hyphen
const fHead = (c) => c[2].replace('Commodities', 'Commod&shy;ities');
const factorSpan = () => `${D.regWindow}; Momentum, monthly, to ${D.regWindowFF}`;
const QQQ_TIP = "QQQ is the model's only stock fund, so it also carries S&P 500 exposure.";
// one rule for Factors and the Table: bold at a t of 2 or more, on the t itself and not its rounding, never for a parked engine
function alphaAttrs(e, t, words, extra) {
  const bold = !e.parked && Math.abs(t) >= 2, span = (e.factorWindow || D.regWindow).split(';')[0];
  return `${tipAttr(`${words} ${span}, whatever the period. t ${minus(t, 2)}${extra ? `. ${extra}` : ''}`)} style="color:${bold ? 'var(--ink)' : 'var(--ink-3)'};font-weight:${bold ? 600 : 400}"`;
}
function factorCell(e, c) {
  const f = e[c[0]][c[1]];
  const on = Math.abs(f.t) >= 2;
  const title = `${e.name}: ${c[2]} ${signed(f.b, 2)}, t ${minus(f.t, 2)}`;
  return `<td data-tip="${esc(title)}" style="background:${on ? heat(f.b) : 'transparent'};color:${on ? inkOn(f.b) : 'transparent'}">${on ? minus(f.b, 2) : ''}</td>`;
}
function renderFactors() {
  const leaning = E.filter((e) => Math.abs(e.etf.QQQ.b) >= 0.3 && Math.abs(e.etf.QQQ.t) >= 2).length;
  $('facttitle').innerHTML = esc(`What do they move with? Stock beta of 0.3 or more: ${leaning} of ${E.length}`);
  $('facttag').textContent = factorSpan();
  const funds = FCOLS.slice(0, 5), mom = FCOLS[8];
  // an empty cell is no significant tie, so it sorts as zero
  const key = (e, col) => { if (col === 'r2') return e.etfR2; if (col === 'alpha') return e.alpha; if (col === 'net') return e.alphaNet; const f = e.etf[col] || e.ff[col]; return Math.abs(f.t) >= 2 ? f.b : 0; };
  // 722 px: Momentum and Net of 2/20 stay whole; Commodities may break
  let h = `<table class="heat fixed" style="min-width:722px"><colgroup><col style="width:124px"><col span="9"></colgroup><thead><tr>${TH_GAP}<th class="group" colspan="6">Five funds</th><th class="group">Stock factor</th><th class="group" colspan="2">Alpha a year, bold at t of 2 or more</th></tr><tr>${TH_NAME}` +
    funds.map((c) => sortTh('factors', c[1], fHead(c), c[1] === 'QQQ' ? tipAttr(QQQ_TIP) : '')).join('') + sortTh('factors', 'r2', 'R²') + sortTh('factors', mom[1], mom[2]) + sortTh('factors', 'alpha', 'Gross') + sortTh('factors', 'net', 'Net of 2/20') + '</tr></thead><tbody>';
  sortRows('factors', E, key).forEach((e) => {
    h += `<tr class="clk${shows(e) ? '' : ' dim'}" data-open="${e.id}" data-hover="${e.id}">${nameCell(e)}` + funds.map((c) => factorCell(e, c)).join('') + `<td class="plain">${e.etfR2.toFixed(2)}</td>` + factorCell(e, mom) +
      `<td${alphaAttrs(e, e.alphaT, ALPHA_GROSS, e.alphaTip)}><span class="ar">${signed(e.alpha)}%</span></td><td${alphaAttrs(e, e.alphaNetT, ALPHA_NET, e.alphaNetTip)}><span class="ar">${signed(e.alphaNet)}%</span></td></tr>`;
  });
  $('factbox').innerHTML = h + '</tbody></table>';
}

/* generic line and bar charts */
function bindCross(el, X, inner, a, b, html, m) {
  const svg = el.querySelector('svg'), line = el.querySelector('.xl');
  svg.addEventListener('pointermove', (ev) => {
    const at = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const i = Math.max(a, Math.min(b, Math.round(a + ((at.x - m.l) / inner) * (b - a))));
    line.setAttribute('x1', X(i)); line.setAttribute('x2', X(i)); line.setAttribute('visibility', 'visible');
    showTip(html(i), ev.clientX, ev.clientY);
  });
  svg.addEventListener('pointerleave', () => { line.setAttribute('visibility', 'hidden'); hideTip(); });
}
function eqAt(r, a, i, base = 0) { let eq = 1; for (let k = a; k <= i; k++) { const v = r[k - base]; if (v !== undefined && !Number.isNaN(v)) eq *= 1 + v; } return eq; }
const dashedIn = (c, a, b) => c.dashTo != null && c.dashTo > a && c.dashTo <= b;
// dashed up to dashTo, solid after it, and both parts share the point at the split
function eqPaths(c, a, b, X, Y) {
  const pts = sampled(c.r, a, b, 400, c.base || 0), line = (list) => list.map((p, k) => `${k ? 'L' : 'M'}${X(p.i).toFixed(1)} ${Y(p.eq).toFixed(1)}`).join('');
  const look = `fill="none" stroke="${c.color}" stroke-width="${c.width || 1.5}" stroke-linejoin="round"${c.opacity ? ` stroke-opacity="${c.opacity}"` : ''}`;
  if (!dashedIn(c, a, b)) return `<path d="${line(pts)}" ${look}/>`;
  const joint = { i: c.dashTo, eq: eqAt(c.r, a, c.dashTo, c.base || 0) };
  return `<path d="${line(pts.filter((p) => p.i < joint.i).concat(joint))}" stroke-dasharray="4 3" ${look}/><path d="${line([joint].concat(pts.filter((p) => p.i > joint.i)))}" ${look}/>`;
}
const axisInk = document.createElement('canvas').getContext('2d');
const axisWidth = (text) => { axisInk.font = "11px 'IBM Plex Mono', Consolas, monospace"; return axisInk.measureText(text).width; };
// the first spot where the text stays inside the plot and no line runs through it; spots are [x, baseline, anchor]
function freeSpot(spots, text, lines, taken, plot) {
  const tw = axisWidth(text);
  const boxOf = ([x, y, anchor]) => ({ x0: anchor === 'end' ? x - tw : x, x1: anchor === 'end' ? x : x + tw, y0: y - 11, y1: y + 3 });
  const fits = (b) => b.x0 >= plot.x0 && b.x1 <= plot.x1 && b.y0 >= plot.y0 && b.y1 <= plot.y1 && !taken.some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0);
  const clear = (b) => !lines.some((pts) => pts.some((p) => p.x > b.x0 - 3 && p.x < b.x1 + 3 && p.y > b.y0 - 3 && p.y < b.y1 + 3));
  const spot = spots.find((p) => fits(boxOf(p)) && clear(boxOf(p))) || spots.find((p) => fits(boxOf(p))) || spots[0];
  taken.push(boxOf(spot));
  return spot;
}
// a label at the end of a line wraps inside the right margin, so a long portfolio name stays on the panel
function wrapEnd(text, strong, room) {
  const scale = strong ? 1.06 : 1, lines = [];
  text.split(' ').forEach((word) => {
    const last = lines.length - 1, joined = last < 0 ? word : `${lines[last]} ${word}`;
    if (last >= 0 && axisWidth(joined) * scale > room) lines.push(word); else lines[Math.max(last, 0)] = joined;
  });
  return lines;
}
const endSpans = (lines, x) => lines.map((t, k) => `<tspan x="${x}"${k ? ' dy="15"' : ''}>${esc(t)}</tspan>`).join('');
// a main line's label takes its colour, deep enough to read as text in both themes
const strongInk = (color) => `color-mix(in oklab, ${color} 70%, var(--ink))`;
function growthChart(el, list, a, b, right = 116) {
  // list: [{name, r, base, color, width, strong, opacity, label, dashTo, dashLabel, mark, markLabel}]
  const marked = list.find((c) => c.mark != null && c.mark >= a && c.mark <= b);
  const W = el.clientWidth, H = 340, m = { l: 42, r: right, t: marked ? 26 : 10, b: 26 }; if (W <= m.l + m.r) return; // no room yet, as in a hidden frame: it draws on the resize when it shows
  let lo = 1, hi = 1;
  list.forEach((s) => sampled(s.r, a, b, 400, s.base || 0).forEach((p) => { if (p.eq < lo) lo = p.eq; if (p.eq > hi) hi = p.eq; }));
  const L0 = Math.log(lo * 0.94), L1 = Math.log(hi * 1.06);
  const X = (i) => m.l + ((i - a) / Math.max(1, b - a)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((Math.log(v) - L0) / (L1 - L0)) * (H - m.t - m.b);
  let s = '';
  D.shaded.forEach((c) => { const i0 = idxOf(c.from), i1 = idxOf(c.to); if (i1 >= a) { const x0 = X(Math.max(i0, a)); s += `<rect x="${x0}" width="${Math.max(2, X(i1) - x0)}" y="${m.t}" height="${H - m.t - m.b}" fill="var(--band)"/>`; } });
  const candidates = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.5, 2, 3, 5, 10, 20, 50, 100, 200];
  const useful = candidates.filter((v) => Math.log(v) >= L0 && Math.log(v) <= L1);
  const ticks = useful.length > 6 ? useful.filter((v) => [0.5, 1, 2, 5, 10, 20, 50, 100, 200].includes(v)) : useful;
  ticks.forEach((v) => { s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text class="axis-t" x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">$${v}</text>`; });
  const span = YEAR[b] - YEAR[a];
  if (span < 2) {
    // a year or less: months, and January names the year when the window crosses it
    const starts = []; for (let i = a; i <= b; i++) if (!i || DATES[i].getUTCMonth() !== DATES[i - 1].getUTCMonth()) starts.push(i);
    const gap = (W - m.l - m.r) / Math.max(1, starts.length), every = b - a > 150 || gap < 34 ? [2, 3, 4, 6].find((k) => k * gap >= 28) || 12 : 1;
    starts.forEach((i) => { const mon = DATES[i].getUTCMonth(); if (mon % every === 0) s += `<text class="axis-t" x="${X(i)}" y="${H - 6}" text-anchor="middle">${mon || !span ? MONTHS[mon] : YEAR[i]}</text>`; });
  } else {
    const perYear = (W - m.l - m.r) * 252 / Math.max(252, b - a);
    let every = span > 12 ? 4 : span > 6 ? 2 : 1;
    while (every < 64 && every * perYear < 32) every *= 2;
    for (let y = YEAR[a] + 1; y <= YEAR[b]; y++) { if ((y - 2000) % every === 0) { const i = YEAR.indexOf(y); s += `<text class="axis-t" x="${X(i)}" y="${H - 6}" text-anchor="middle">${y}</text>`; } }
  }
  list.slice().reverse().forEach((c) => { s += eqPaths(c, a, b, X, Y); });
  // the label sits by the split, where no line runs through it; two lines that split at the same date get one label
  const dashed = list.filter((c) => c.dashLabel && dashedIn(c, a, b));
  if (dashed.length) {
    const plot = { x0: m.l + 2, x1: W - m.r, y0: m.t + 2, y1: H - m.b - 2 }, taken = [], said = new Set();
    const inPx = list.map((c) => sampled(c.r, a, b, 400, c.base || 0).map((p) => ({ x: X(p.i), y: Y(p.eq) })));
    dashed.forEach((c) => {
      if (said.has(c.dashTo)) return;
      said.add(c.dashTo);
      const jx = X(c.dashTo), jy = Y(eqAt(c.r, a, c.dashTo, c.base || 0));
      const near = [[jx - 6, jy - 8, 'end'], [jx + 6, jy + 16, 'start'], [jx + 6, jy - 8, 'start'], [jx - 6, jy + 16, 'end'], [jx - 6, jy - 24, 'end'], [jx + 6, jy + 32, 'start']];
      const higher = [24, 40, 56, 72, 88, 104].flatMap((up) => [[jx + 6, jy - up, 'start'], [jx - 6, jy - up, 'end'], [plot.x0, jy - up, 'start']]);
      const [tx, ty, anchor] = freeSpot(near.concat(higher, [[plot.x0, plot.y0 + 12, 'start']]), c.dashLabel, inPx, taken, plot);
      s += `<text class="axis-t" x="${tx.toFixed(1)}" y="${ty.toFixed(1)}" text-anchor="${anchor}">${esc(c.dashLabel)}</text>`;
    });
  }
  if (marked) {
    const mx = X(marked.mark), fits = mx - axisWidth(marked.markLabel) >= m.l;
    s += `<line x1="${mx.toFixed(1)}" x2="${mx.toFixed(1)}" y1="${m.t}" y2="${m.t + 10}" stroke="var(--ink-2)" stroke-width="1.5"/><text class="axis-t" x="${mx.toFixed(1)}" y="${m.t - 5}" text-anchor="${fits ? 'end' : 'start'}">${esc(marked.markLabel)}</text>`;
  }
  const ends = list.filter((c) => c.label !== false).map((c) => ({ c, v: eqAt(c.r, a, b, c.base || 0) })).sort((p, q) => Y(p.v) - Y(q.v));
  let prev = -99;
  ends.forEach(({ c, v }) => {
    const lines = wrapEnd(`${c.name} $${v >= 10 ? v.toFixed(0) : v.toFixed(2)}`, c.strong, m.r - 8), yy = Math.max(Y(v) + 4, prev + 16);
    prev = yy + (lines.length - 1) * 15;
    s += `<text class="axis-t" y="${yy}" style="fill:${c.strong ? strongInk(c.color) : 'var(--ink-3)'};font-weight:${c.strong ? 600 : 400}">${endSpans(lines, W - m.r + 8)}</text>`;
  });
  s += `<line class="xl" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="var(--ink-3)" visibility="hidden"/><rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="Growth of $1">${s}</svg>`;
  bindCross(el, X, W - m.l - m.r, a, b, (i) => `<b>${dayLabel(i)}</b>` + list.filter((c) => c.label !== false).map((c) => `<div class="r"><i>${esc(c.name)}</i>${c.strong ? '<b>' : ''}$${eqAt(c.r, a, i, c.base || 0).toFixed(2)}${c.strong ? '</b>' : ''}</div>`).join(''), m);
}
function ddChart(el, r, a, b, base, name, ref = null, right = 116) {
  // ref: {name, r, base}, a dashed line to compare with
  const W = el.clientWidth, H = 182, m = { l: 42, r: right, t: 6, b: 6 }; if (W <= m.l + m.r) return;
  const mine = sampled(r, a, b, 400, base), qq = sampled(QQQ, a, b, 400), back = ref ? sampled(ref.r, a, b, 400, ref.base) : [];
  const floor = Math.min(-5, ...mine.map((p) => p.dd), ...qq.map((p) => p.dd), ...back.map((p) => p.dd)) * 1.04;
  const X = (i) => m.l + ((i - a) / Math.max(1, b - a)) * (W - m.l - m.r);
  const Y = (v) => m.t + (v / floor) * (H - m.t - m.b);
  let area = `M${X(a)} ${Y(0)}`, qp = '';
  mine.forEach((p) => { area += `L${X(p.i).toFixed(1)} ${Y(p.dd).toFixed(1)}`; });
  area += `L${X(mine.length ? mine[mine.length - 1].i : b)} ${Y(0)}Z`;
  qq.forEach((p, k) => { qp += `${k ? 'L' : 'M'}${X(p.i).toFixed(1)} ${Y(p.dd).toFixed(1)}`; });
  let s = '';
  D.shaded.forEach((c) => { const i0 = idxOf(c.from), i1 = idxOf(c.to); if (i1 >= a) { const x0 = X(Math.max(i0, a)); s += `<rect x="${x0}" width="${Math.max(2, X(i1) - x0)}" y="${m.t}" height="${H - m.t - m.b}" fill="var(--band)"/>`; } });
  const tick = floor < -30 ? 20 : floor < -12 ? 10 : 5;
  for (let v = 0; v >= floor; v -= tick) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text class="axis-t" x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${v ? '−' + Math.abs(v) : 0}%</text>`;
  s += `<path d="${qp}" fill="none" stroke="var(--ink-3)" stroke-width="1.25"/><path d="${area}" fill="var(--c)" fill-opacity=".55"/>`;
  if (back.length) s += `<path d="${back.map((p, k) => `${k ? 'L' : 'M'}${X(p.i).toFixed(1)} ${Y(p.dd).toFixed(1)}`).join('')}" fill="none" stroke="var(--ink-2)" stroke-width="1.5" stroke-dasharray="6 3" stroke-linejoin="round"/>`;
  const running = (rr, i, bs) => { let eq = 1, peak = 1; for (let k = a; k <= i; k++) { const v = rr[k - bs]; if (v !== undefined && !Number.isNaN(v)) { eq *= 1 + v; if (eq > peak) peak = eq; } } return (eq / peak - 1) * 100; };
  // each line named at its own end, with where it stands now; labels stack as on the growth chart, and a stack past the bottom moves up
  const ends = [{ name, pts: mine, rr: r, bs: base, strong: true }, { name: ref && ref.name, pts: back, rr: ref && ref.r, bs: ref && ref.base }, { name: 'QQQ', pts: qq, rr: QQQ, bs: 0 }].filter((c) => c.pts.length).map((c) => ({ ...c, at: Y(c.pts[c.pts.length - 1].dd) })).sort((p, q) => p.at - q.at);
  let prev = -99;
  const placed = ends.map((c) => { const lines = wrapEnd(`${c.name} ${pct(running(c.rr, b, c.bs), 0)}`, c.strong, m.r - 8), yy = Math.max(c.at + 4, prev + 16); prev = yy + (lines.length - 1) * 15; return { c, lines, yy }; });
  let floorY = H - 2;
  placed.slice().reverse().forEach((p) => { p.yy = Math.min(p.yy, floorY - (p.lines.length - 1) * 15); floorY = p.yy - 16; });
  placed.forEach(({ c, lines, yy }) => { s += `<text class="axis-t" y="${yy.toFixed(1)}"${c.strong ? ` style="fill:${strongInk('var(--c)')};font-weight:600"` : ''}>${endSpans(lines, W - m.r + 8)}</text>`; });
  s += `<line class="xl" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="var(--ink-3)" visibility="hidden"/><rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="Drawdown">${s}</svg>`;
  bindCross(el, X, W - m.l - m.r, a, b, (i) => `<b>${dayLabel(i)}</b>${mine.length ? `<div class="r"><i>${esc(name)}</i><b>${pct(running(r, i, base))}</b></div>` : ''}${ref ? `<div class="r"><i>${esc(ref.name)}</i>${pct(running(ref.r, i, ref.base))}</div>` : ''}<div class="r"><i>QQQ</i>${pct(running(QQQ, i, 0))}</div>`, m);
}
function yearChart(el, values, a = 0, b = LAST, right = 116) {
  // values: [{y, v, to}], all history; to: the last date of a year the data stops in before b. Each bar sits under its stretch of a Growth chart of the whole history; a year before the window a starts fades
  const W = el.clientWidth, H = 170, m = { l: 42, r: right, t: 18, b: 22 }; if (W <= m.l + m.r) return;
  const vals = values.map((o) => o.v);
  const hi = Math.max(10, ...vals), lo = Math.min(0, ...vals);
  const room = lo < 0 ? 16 : 0;
  const Y = (v) => m.t + ((hi - v) / (hi - lo)) * (H - m.t - m.b - room);
  const X = (i) => m.l + (i / Math.max(1, b)) * (W - m.l - m.r);
  const full = (W - m.l - m.r) * 252 / Math.max(252, b), fits = full >= 30, every = full < 8 ? 4 : 2, best = Math.max(...vals), worst = Math.min(...vals);
  let s = `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)"/>`;
  values.forEach(({ y, v, to }) => {
    const left = X(YEAR.indexOf(y)), slot = X(Math.min(b, YEAR.lastIndexOf(y)) + 1) - left, cx = left + slot / 2, bw = Math.max(3, Math.min(26, slot - 4)), x0 = cx - bw / 2, y0 = Y(Math.max(v, 0)), hh = Math.max(1, Math.abs(Y(v) - Y(0))), rr = Math.min(4, hh, bw / 2), hit = Math.max(slot, bw);
    const d = v >= 0 ? `M${x0} ${y0 + hh}V${y0 + rr}Q${x0} ${y0} ${x0 + rr} ${y0}H${x0 + bw - rr}Q${x0 + bw} ${y0} ${x0 + bw} ${y0 + rr}V${y0 + hh}Z` : `M${x0} ${y0}V${y0 + hh - rr}Q${x0} ${y0 + hh} ${x0 + rr} ${y0 + hh}H${x0 + bw - rr}Q${x0 + bw} ${y0 + hh} ${x0 + bw} ${y0 + hh - rr}V${y0}Z`;
    s += `<g${y < YEAR[a] ? ' opacity=".3"' : ''}><path d="${d}" fill="var(--c)"/><rect x="${cx - hit / 2}" y="0" width="${hit}" height="${H}" fill="transparent" data-tip="${yearSpan(y)}${to || y === YEAR[b] ? `, to ${monthDay(to || ISO[b])}` : ''}: ${pct(v)}"/>`;
    // a value on every bar when a year has the room; a year just begun writes its own to the right, clear of last year's
    const spot = fits && slot < 20 ? `x="${left - 1}" text-anchor="start"` : `x="${cx}" text-anchor="middle"`;
    if (fits || v === best || (v === worst && v <= -0.5)) s += `<text class="axis-t" ${spot} y="${v >= 0 ? Y(v) - 5 : Y(v) + 14}" style="fill:var(--ink-2)">${pct(v, 0)}</text>`;
    if (fits || slot >= 20 || y % every === 0) s += `<text class="axis-t" ${spot} y="${H - 5}">${String(y).slice(2)}</text>`;
    s += '</g>';
  });
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="Return by year">${s}</svg>`;
}
function rollingChart(el, r, S, parked, right = 116) {
  const W = el.clientWidth, H = 130, m = { l: 42, r: right, t: 8, b: 22 }, lastI = lastValid(r); if (W <= m.l + m.r) return;
  const all = []; for (let i = 756; i <= lastI; i += 21) all.push({ i, v: rollingSharpe(r, 756, i) });
  // the steps stop up to 20 sessions short; the last close is always a point
  if (all.length && all[all.length - 1].i < lastI) all.push({ i: lastI, v: rollingSharpe(r, 756, lastI) });
  while (all.length && Number.isNaN(all[0].v)) all.shift();
  const pts = all.filter((p) => !Number.isNaN(p.v));
  if (!pts.length) { el.innerHTML = ''; return; }
  // the Growth chart's whole-history scale: the first three years stay empty
  const a = 0, b = LAST;
  // two standard errors around the since-2008 Sharpe, over 3 years
  const half = 2 * Math.sqrt((1 + S * S / 2) / 3), top = S + half, low = S - half;
  const hi = Math.max(1, top, ...pts.map((p) => p.v)), lo = Math.min(0, low, ...pts.map((p) => p.v));
  const X = (i) => m.l + ((i - a) / Math.max(1, b - a)) * (W - m.l - m.r);
  const Y = (v) => m.t + ((hi - v) / (hi - lo)) * (H - m.t - m.b);
  // the line breaks where there is no Sharpe, and a gap of 3 or more points says why
  let d = '', gap = 0, flat = ''; const taken = [], FLAT_WORDS = 'Too little movement', flatWidth = axisWidth(FLAT_WORDS);
  all.forEach((p, k) => {
    if (Number.isNaN(p.v)) { gap++; if (k < all.length - 1) return; }
    if (gap >= 3 && parked) {
      const i1 = Number.isNaN(p.v) ? p.i : all[k - 1].i, i0 = all[k - gap + (Number.isNaN(p.v) ? 1 : 0)].i, cx = X((i0 + i1) / 2);
      if (X(i1) - X(i0) >= flatWidth) { flat += `<text class="axis-t" x="${cx}" y="${Y(0) - 8}" text-anchor="middle">${FLAT_WORDS}</text>`; taken.push({ x0: cx - flatWidth / 2 - 2, x1: cx + flatWidth / 2 + 2, y0: Y(0) - 19, y1: Y(0) - 5 }); }
    }
    if (!Number.isNaN(p.v)) { d += `${gap || !k ? 'M' : 'L'}${X(p.i).toFixed(1)} ${Y(p.v).toFixed(1)}`; gap = 0; }
  });
  const ticks = [-3, -2, -1, 1, 2, 3, 4].filter((v) => v <= hi && v >= lo);
  // the note sits between two rules, where the line does not run: the top gap first, then the bottom, then the rest
  const edges = [top, ...ticks.concat(0).filter((v) => v < top && v > low).sort((p, q) => q - p), low].map(Y);
  const gaps = edges.slice(1).map((y, k) => [edges[k], y]).filter(([p, q]) => q - p >= 13);
  if (!gaps.length) gaps.push([Y(top), Y(low)]);
  const words = W - m.l - m.r >= 250 ? 'Inside the band: too noisy to tell' : 'Too noisy to tell', x0 = m.l + 8, x1 = W - m.r - 8;
  const across = [[x0, 'start'], [x1, 'end'], ...[0.25, 0.5, 0.75].map((f) => [m.l + f * (W - m.l - m.r), 'start'])];
  const spots = [gaps[0], gaps[gaps.length - 1], ...gaps.slice(1, -1)].flatMap(([p, q]) => across.map(([x, side]) => [x, (p + q) / 2 + 4, side]));
  const [tx, ty, anchor] = freeSpot(spots, words, [pts.map((p) => ({ x: X(p.i), y: Y(p.v) }))], taken, { x0: m.l, x1: W - m.r, y0: Y(top) - 2, y1: Y(low) + 2 });
  let s = `<rect x="${m.l}" y="${Y(top).toFixed(1)}" width="${W - m.l - m.r}" height="${(Y(low) - Y(top)).toFixed(1)}" fill="var(--band)"/><line x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)"/>${flat}<text class="axis-t" x="${tx}" y="${ty.toFixed(1)}" text-anchor="${anchor}">${words}</text>`;
  ticks.forEach((v) => { s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text class="axis-t" x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${minus(v, 0)}</text>`; });
  s += `<text class="axis-t" x="${m.l - 8}" y="${Y(0) + 4}" text-anchor="end">0</text>`;
  for (let y = YEAR[a] + 1; y <= YEAR[b]; y++) { if ((y - 2000) % 4 === 0) { const i = YEAR.indexOf(y); s += `<text class="axis-t" x="${X(i)}" y="${H - 4}" text-anchor="middle">${y}</text>`; } }
  // bold means the last close; an older last value says its date
  const last = pts[pts.length - 1], old = last.i < lastI;
  s += `<path d="${d}" fill="none" stroke="var(--c)" stroke-width="2.25" stroke-linejoin="round"/><circle cx="${X(last.i)}" cy="${Y(last.v)}" r="5" fill="var(--c)" stroke="var(--surface)" stroke-width="2"/><text class="axis-t" x="${X(last.i) + 10}" y="${Y(last.v) + 4}" style="fill:var(--ink);font-weight:${old ? 400 : 600}">${num(last.v)}${old ? `, ${MONTHS[DATES[last.i].getUTCMonth()]} ${YEAR[last.i]}` : ''}</text>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="Rolling Sharpe">${s}</svg>`;
}

/* drawer */
function barRow(i, c) {
  const o = E[i];
  return `<button class="brow c-${o.fam}" data-open="${o.id}"><span>${eName(o)}</span><span class="track"><span class="fill" style="width:${(Math.max(0, c) * 100).toFixed(0)}%"></span></span><span class="v">${num(c)}</span></button>`;
}
const dCell = (label, value, tipText) => `<div class="dcell"${tipAttr(tipText)}><span>${esc(label)}</span>${value}</div>`;
const CHECK_NONE = '<b class="none">Not tested</b>';
const passWord = (ok) => ok == null ? CHECK_NONE : ok ? '<b>Pass</b>' : '<b class="fail">Fail</b>';
const caveatsHTML = (e) => e.caveats.length ? `<div class="dcav">${e.caveats.map((c) => `<b>${esc(c.d)}</b><span>${esc(c.t)}</span>`).join('')}</div>` : '';
function tradesHTML(e) {
  const t = e.trades, head = `<h4>How it trades${e.fillWord ? ` <span class="stag">${esc(e.fillWord)}</span>` : ''}</h4>`, most = t && t.namesMax != null ? Math.round(t.namesMax) : null;
  // what was not measured is left out; a row with nothing to say is one line
  const cells = !t ? [] : [
    ['Trade days a year', t.days, (n) => n],
    ['Positions', t.names, (n) => !n && most ? `Flat most days, up to ${most}` : most != null && most !== n ? `${n}, up to ${most}` : n],
    ['Holds', t.hold, (n) => plural(n, 'day')],
    ['Turnover a year', t.turnover, (n) => `${n}x`],
    ['Shorts', t.shortPct > 1 ? t.shortPct : null, (n) => `${n}% of capital,<br>${Math.round(t.shortDays)}% of days`],
  ].filter((c) => c[1] != null).map(([label, v, text]) => dCell(label, `<b>${text(Math.round(v))}</b>`));
  // a note that names other engines opens them
  const note = (e.tradesOpens || []).filter((id) => byId[id]).reduce((h, id) => h.replace(esc(byId[id].name), `<button class="dlink" data-open="${id}">${esc(byId[id].name)}</button>`), esc(e.tradesNote || 'Not measured'));
  if (!cells.length) return `<div class="dsec">${head}<div class="drow"><div class="dwide">${note}</div></div></div>`;
  return `<div class="dsec">${head}<div class="drow"${tipAttr(`Last 3 years of its saved run${t.end ? `, to ${shortLabel(t.end)}` : ''}. T-bill parking left out. Holds counts trading sessions. Turnover counts buys plus sells.`)}>${cells.join('')}</div></div>`;
}
function forwardHTML(e) {
  const parts = [], f = e.forward;
  if (e.live) parts.push(e.live > ISO[LAST] ? `First trade ${monthDay(e.live)}` : `<span${tipAttr(e.liveTip)}>${e.fillsOnRecord ? 'Real money' : `${e.liveWord} ${monthDay(e.live)}, fills not read here`}: ${plural(LAST - idxOf(e.live) + 1, 'session')}</span>`);
  if (f) {
    const months = plural(Math.ceil(f.months), 'month');
    parts.push(`<span${tipAttr('The months assume the backtest Sharpe; a lower Sharpe needs longer.')}>${f.start > ISO[LAST] ? `Simulated forward test starts ${monthDay(f.start)}, at least ${months}` : `Simulated forward test: ${plural(LAST - idxOf(f.start) + 1, 'session')} of at least ${months}`}</span>`);
  }
  return `<div class="dwide">${parts.join('. ') || 'No forward test'}</div>`;
}
function checksHTML(e) {
  const c = e.card;
  if (!c) return `<div class="drow"><div class="dwide none"${tipAttr(e.cardNote)}>${esc(e.gradeOnly ? e.gradeOnly.word : 'No Scout card')}</div>${forwardHTML(e)}</div>`;
  const report = D.fdrDate ? `Report of ${monthDay(D.fdrDate)}` : 'Report';
  return '<div class="drow">' +
    dCell('Beats shuffled prices', passWord(c.pPass), c.pParts ? `Against shuffled prices, each part must pass: ${c.pParts}.` : c.p == null ? '' : `Whole search against shuffled prices: p ${c.p.toFixed(3)}.`) +
    dCell(`Passes the multiple-test cut${D.fdrTests ? `, ${D.fdrTests} tests` : ''}`, passWord(c.qPass), c.q == null ? `Not in the multiple-test report${D.fdrDate ? ` of ${monthDay(D.fdrDate)}` : ''}.` : `q ${c.q.toFixed(4)}${D.fdrTests && D.fdrRegs ? `, over ${D.fdrTests} tested of ${D.fdrRegs} registered` : ''}. ${report}; it can change as tests are added.`) +
    dCell('Beats T-bills in TAA 3x + NDX VXN', passWord(c.slotPass), c.slotP == null ? '' : `In the 60/40 of TAA 3x and NDX VXN: this engine${c.slotName ? ` in ${c.slotName}` : ''} against T-bills there. Better on ${(c.slotP * 100).toFixed(0)}% of paths${c.slotWith == null || c.slotWithout == null ? '' : `, Sharpe ${c.slotWith.toFixed(2)} vs ${c.slotWithout.toFixed(2)}`}.`) +
    dCell('Sharpe at double costs', c.double == null ? CHECK_NONE : `<b>${num(c.double)}${c.net == null ? '' : `, from ${num(c.net)}`}</b>`, `The Scout card's own window${c.window ? `, ${c.window}` : ''}, not the Sharpe above. Costs doubled plus 10 bp a side${c.net == null ? '' : ', from the same settings at normal costs'}.`) +
    dCell('Scout verdict', c.verdict ? `<b>${verdictWord(c.verdict)}</b>` : CHECK_NONE, cardOf(e)) +
    forwardHTML(e) + '</div>';
}
// a 90% band on the gap between two Sharpes, given how much the two engines move together
function compareLine(e, o) {
  const none = [e, o].filter((x) => !x.s || x.s.sharpe == null), thin = [e, o].filter((x) => !simOn[E.indexOf(x)]);
  if (none.length) return `No Sharpe gap: ${none.length > 1 ? 'neither has a Sharpe' : `${none[0].name} has no Sharpe`} in this period.`;
  if (thin.length) return `No Sharpe gap: ${thin.map((x) => x.name).join(' and ')} ${thin.length > 1 ? 'have' : 'has'} too few weeks in this period.`;
  const rho = CORR[E.indexOf(e)][E.indexOf(o)], years = Math.min(e.s.n, o.s.n) / 252, half = 1.645 * Math.sqrt((2 - 2 * rho) / years), gap = Math.abs(e.s.sharpe - o.s.sharpe);
  return `Sharpe gap ${gap.toFixed(2)}, noise ±${half.toFixed(2)}${gap <= half ? ': cannot tell apart' : ''}`;
}
let openId = null, compareId = '', opener = null, openedAt = 0, closedAt = -Infinity;
// the page behind keeps its place and its width: the scrollbar's room stays as padding
function lockPage(on) {
  document.body.style.paddingRight = on ? `${innerWidth - document.documentElement.clientWidth}px` : '';
  document.body.style.overflow = on ? 'hidden' : '';
}
function openDrawer(id) {
  const e = byId[id]; if (!e) return;
  const a = winA(), s = e.s, dr = $('drawer'), ei = E.indexOf(e), firstOpen = dr.hidden, same = openId === id, had = document.activeElement, top = same ? dr.scrollTop : 0, cmp = same ? compareId : '';
  // walking redraws the drawer: focus stays on the control in use, or goes to Close when that control is gone
  const keep = !had || !dr.contains(had) ? null : had.dataset.step ? `[data-step="${had.dataset.step}"]` : had.id ? `#${had.id}` : '.dclose';
  const others = simOn[ei] ? E.map((o, i) => i).filter((i) => i !== ei && simOn[i]).map((i) => [i, CORR[ei][i]]).sort((p, q) => q[1] - p[1]) : [], corrWord = fewWeeks() ? ', whole history' : '';
  const fallCells = FALLS.map((f) => { const v = compound(R[id], f.a, f.b), none = Number.isNaN(v), span = `${esc(shortLabel(f.from))} to ${esc(shortLabel(f.to))}`; return `<div><em style="background:${none ? 'var(--band)' : heat(v, 40)};color:${none ? 'var(--ink-3)' : inkOn(v, 40)}" data-tip="${span}: ${none ? 'no data' : `${pct(v)}, QQQ ${pct(f.depth)}`}">${none ? '' : minus(v, 0)}</em>${f.to.slice(0, 4)}<br>${minus(f.depth, 0)}%</div>`; }).join('');
  const factors = FCOLS.map((c) => { const f = e[c[0]][c[1]]; const on = Math.abs(f.t) >= 2; return `<div><em style="background:${on ? heat(f.b) : 'var(--band)'};color:${on ? inkOn(f.b) : 'var(--ink-3)'}">${on ? minus(f.b, 2) : ''}</em>${fHead(c)}</div>`; }).join('');
  const versions = e.versions.length ? `<div class="dsec"><h4>Other versions</h4><table class="mini"><thead><tr><th aria-label="Version"></th><th>${retWord()}</th><th>Sharpe</th><th>Max DD</th><th>Since</th></tr></thead><tbody>${e.versions.map((v) => { const st = statsOf(R[v.key], a, LAST); let first = 0; while (first < N && Number.isNaN(R[v.key][first])) first++; return st ? `<tr><td>${esc(v.name)}</td><td>${retCell(st)}</td><td>${sharpeText(st)}</td><td>${pct(st.dd)}</td><td>${YEAR[first]}</td></tr>` : ''; }).join('')}</tbody></table></div>` : '';
  const options = ['<option value="">Compare with</option>'].concat(E.filter((o) => o.id !== id).map((o) => `<option value="${o.id}"${cmp === o.id ? ' selected' : ''}>${esc(o.name)}</option>`)).join('');
  // the whole text first: a bad field leaves the page as it was
  const html = `<div class="dhead"><span class="dot s-${stageKey(e.stage)}"></span><h2>${eName(e)}</h2>${chipFor(e)}<div class="dnav"><button class="dstep" data-step="-1">Previous</button><button class="dstep" data-step="1">Next</button><button class="dclose" data-close="1">Close</button></div></div>` +
    `<div class="dwords">${esc(e.words)}</div>` +
    (s ? `<div><div class="dstats"><div><b>${pct(s.ret)}</b><span>${wordOf(s)}</span></div><div><b${sharpeTip(e, s)}>${sharpeText(s, e)}</b><span>Sharpe</span></div><div><b>${pct(s.dd)}</b><span>Max DD</span></div><div><b>${pct(s.vol)}</b><span>Vol</span></div></div><div class="dfine">${esc(`Backtest, ${periodWords()} to ${shortLabel(e.end)}`)}</div></div>` : `<div class="dfine">No data in this period</div>`) +
    (e.used.length ? `<div class="dsec"><h4>In portfolios</h4><div class="pills"${tipAttr(D.portfolioTip)}>${e.used.map((u) => (presetNamed(u.p) ? `<button class="pill" data-pill="${esc(u.p)}">${esc(u.p)} <b>${u.w}%</b></button>` : `<span class="pill">${esc(u.p)} <b>${u.w}%</b></span>`)).join('')}</div></div>` : '') +
    tradesHTML(e) + caveatsHTML(e) + checksHTML(e) +
    `<div class="dsec"><div class="dtitle"><h4>Growth of $1</h4><select id="cmp" aria-label="Compare with another engine">${options}</select></div><div class="chart" id="d-growth"></div><div class="dfine" id="d-cmp" hidden></div></div>` +
    `<div class="dsec"><h4>Drawdown</h4><div class="chart" id="d-dd"></div></div>` +
    `<div class="dsec"><h4>By year <span class="stag">All history</span></h4><div class="chart" id="d-years"></div></div>` +
    `<div class="dsec"><h4>${esc('Rolling 3-year Sharpe')} <span class="stag">All history</span></h4><div class="chart" id="d-roll"></div></div>` +
    `<div class="dsec"><h4>${esc('While QQQ fell, %')}</h4><div class="frow" style="grid-template-columns:repeat(${FALLS.length},1fr)">${fallCells}</div></div>` +
    (others.length ? `<div class="bars"><div class="dsec"><h4>Moves most with${corrWord}</h4>${others.slice(0, 3).map(([i, c]) => barRow(i, c)).join('')}</div><div class="dsec"><h4>Moves least with${corrWord}</h4>${others.slice(-3).map(([i, c]) => barRow(i, c)).join('')}</div></div>` : '') +
    `<div class="dsec"><h4>Factors</h4><div class="frow">${factors}</div><div class="dfine">${esc(`Blank: t under 2. R² ${e.etfR2.toFixed(2)} on the five funds, ${e.ffR2.toFixed(2)} on the stock factors. Alpha ${signed(e.alpha)}% a year (t ${minus(e.alphaT, 2)}), net of 2/20 ${signed(e.alphaNet)}% (t ${minus(e.alphaNetT, 2)}). With QQQ's trend rule added ${signed(e.alphaTrend)}% (t ${minus(e.alphaTrendT, 2)}), net ${signed(e.alphaTrendNet)}% (t ${minus(e.alphaTrendNetT, 2)}). ${e.factorWindow || `${D.regWindow}; stock factors to ${D.regWindowFF}`}.`)}</div></div>` + versions;
  hideTip();
  openId = id; compareId = cmp;
  dr.className = '';
  dr.setAttribute('aria-label', `${e.name} details`);
  dr.innerHTML = html;
  // what Tab or Shift+Tab reaches lands below the sticky head
  dr.style.scrollPaddingTop = `${dr.querySelector('.dhead').offsetHeight + 8}px`;
  if (firstOpen) { opener = had; openedAt = performance.now(); lockPage(true); }
  dr.hidden = false; $('scrim').hidden = false;
  drawDrawerCharts(e, a);
  dr.scrollTop = top;
  $('cmp').addEventListener('change', (ev) => { compareId = ev.target.value; drawDrawerGrowth(e, a); });
  // focus goes to Close only on first open
  if (firstOpen) dr.querySelector('.dclose').focus({ preventScroll: true });
  else if (keep) dr.querySelector(keep).focus({ preventScroll: true });
}
const stepDrawer = (d) => { const i = E.indexOf(byId[openId]); openDrawer(E[(i + d + E.length) % E.length].id); };
// the label gutter of the drawer's charts, one for all four so the years line up
const drawerRight = () => ($('d-growth').clientWidth < 420 ? 100 : 116);
function drawDrawerGrowth(e, a) {
  // the line stays the model: dashes where its early years are not the traded rule, a tick where real money starts
  const standIn = (o) => o.standIn ? { dashTo: idxOf(o.standIn.to), dashLabel: standInLabel(o) } : {};
  const live = e.live && e.live <= ISO[LAST] ? { mark: idxOf(e.live), markLabel: `${e.fillsOnRecord ? 'Live' : e.liveWord === 'Switched on' ? 'On' : e.liveWord} ${monthDay(e.live)}` } : {};
  // no data in the window: no line to draw and no $1.00 to print
  const list = [{ name: e.name, r: R[e.id], color: 'var(--c)', width: 2.5, strong: true, ...standIn(e), ...live, ...(e.s ? {} : { label: false }) }];
  e.versions.forEach((v) => list.push({ name: v.name, r: R[v.key], color: 'var(--c)', width: 1.25, opacity: 0.45, label: false }));
  const other = compareId ? byId[compareId] : null;
  if (other) list.push({ name: other.name, r: R[other.id], color: 'var(--accent)', width: 2, strong: true, ...standIn(other), ...(other.s ? {} : { label: false }) });
  list.push({ name: 'QQQ', r: QQQ, color: 'var(--ink-3)', width: 1.5 }, { name: 'T-bills', r: TB, color: 'var(--ref)', width: 1.5 });
  growthChart($('d-growth'), list, a, LAST, drawerRight());
  const line = other ? compareLine(e, other) : '';
  $('d-cmp').textContent = line; $('d-cmp').hidden = !line;
}
function drawDrawerCharts(e, a) {
  drawDrawerGrowth(e, a);
  ddChart($('d-dd'), R[e.id], a, LAST, 0, e.name, null, drawerRight());
  const yr = yearReturns(R[e.id]);
  yearChart($('d-years'), YEARS.filter((y) => !Number.isNaN(yr[y])).map((y) => ({ y, v: yr[y], to: e.lastI < LAST && y === YEAR[e.lastI] ? e.end : null })), a, LAST, drawerRight());
  const whole = statsOf(R[e.id], 0, LAST);
  rollingChart($('d-roll'), R[e.id], whole ? whole.sharpe : null, e.parked, drawerRight());
}
function closeDrawer() {
  hideTip();
  if ($('drawer').hidden) return;
  $('drawer').hidden = true; $('scrim').hidden = true; openId = null; lockPage(false); closedAt = performance.now();
  // focus goes back to what opened the drawer; a redraw may have replaced it
  const key = opener && opener.dataset ? opener.dataset.open : null, back = opener && opener.isConnected ? opener : key ? document.querySelector(`.tile[data-open="${key}"], .pt[data-open="${key}"]`) : null;
  opener = null;
  if (back && back !== document.body) back.focus({ preventScroll: true });
}
// Tab stays inside the open drawer
function holdTab(ev) {
  const dr = $('drawer'), stops = [...dr.querySelectorAll('button, select')], at = document.activeElement, edge = stops[ev.shiftKey ? 0 : stops.length - 1], wrap = stops[ev.shiftKey ? stops.length - 1 : 0];
  if (dr.contains(at) && at !== edge) return;
  ev.preventDefault();
  if (dr.contains(at)) { wrap.focus(); return; }
  // a click on plain text leaves focus outside: Tab goes on from what the reader sees, and the drawer keeps its place
  const headEnd = dr.querySelector('.dhead').getBoundingClientRect().bottom, end = dr.getBoundingClientRect().bottom;
  const seen = stops.filter((b) => { const r = b.getBoundingClientRect(); return r.top >= headEnd && r.bottom <= end; });
  (seen[ev.shiftKey ? seen.length - 1 : 0] || wrap).focus({ preventScroll: true });
}

/* builder */
// activePreset: the mix is that preset. basePreset: the preset it started from, kept when a slider moves.
let mix = {}, activePreset = '', basePreset = '';
const presetNamed = (name) => D.presets.find((x) => x.name === name);
// a leg is at a preset's weight exactly, or at the slider's stop for it: 33.4 rests at 33
const atPreset = (p) => [...new Set([...Object.keys(mix), ...Object.keys(p.w)])].every((k) => { const v = mix[k] || 0, x = p.w[k] || 0; return v === x || v === Math.round(x); });
// weights at the preset they started from are that preset again. Presets come with data.json, so this also tests what storage holds under a name
function settlePreset() { const p = presetNamed(basePreset); activePreset = p && atPreset(p) ? p.name : ''; if (activePreset) mix = { ...p.w }; }
// storage may hold anything: only real weights of engines with a slider stay. A name no longer on the menu reads as custom, with no base
try {
  const saved = JSON.parse(store.get('em-mix') || 'null');
  if (saved && saved.mix) {
    mix = Object.fromEntries(Object.entries(saved.mix).filter(([k, v]) => k !== 'capsule' && E.some((e) => e.id === k) && Number.isFinite(v) && v > 0).map(([k, v]) => [k, Math.min(v, 100)]));
    basePreset = [saved.base, saved.preset].find((n) => presetNamed(n)) || '';
  }
} catch (e) { /* use the default below */ }
if (!Object.keys(mix).length) { const p = presetNamed('Growth') || D.presets[0]; mix = { ...p.w }; basePreset = p.name; }
settlePreset();
// shift = {id, k}, or a list of them, takes k off that engine's daily return. Two mixes compared always pass the same end.
function mixSeries(w = mix, a = winA(), endI = Infinity, shift = null) {
  const ids = Object.keys(w).filter((k) => w[k] > 0 && R[k] && byId[k] && byId[k].lastI >= a);
  if (!ids.length) return null;
  const total = ids.reduce((s, k) => s + w[k], 0);
  const cuts = [].concat(shift || []), wts = ids.map((k) => w[k] / total), rs = ids.map((k) => R[k]), off = ids.map((k) => (cuts.find((c) => c.id === k) || { k: 0 }).k);
  const b = Math.min(endI, ...ids.map((k) => byId[k].lastI));
  const out = new Float64Array(Math.max(0, b - a + 1));
  let held = wts.slice(), value = 1;
  for (let i = a; i <= b; i++) {
    if (i === a || YEAR[i] !== YEAR[i - 1]) held = wts.map((x) => x * value);
    const prev = value; value = 0;
    for (let k = 0; k < held.length; k++) { const v = rs[k][i]; held[k] *= 1 + (Number.isNaN(v) ? 0 : v - off[k]); value += held[k]; }
    out[i - a] = value / prev - 1;
  }
  return { ids, w: wts, a, b, out, total };
}
const seriesStats = (m) => m ? statsOf(m.out, m.a, m.b, m.a) : null;
// a difference that rounds to zero carries no sign
const changeText = (v, d) => (+Math.abs(v).toFixed(d) === 0 ? minus(0, d) : signed(v, d));
const averageOf = (x) => x.reduce((p, v) => p + v, 0) / x.length;
// daily return over T-bills on [a, b]; a day with no value does not count
function excessOf(r, a, b) {
  let n = 0, x = 0, x2 = 0;
  for (let i = a; i <= b; i++) { const v = r[i]; if (Number.isNaN(v)) continue; const e = v - (Number.isNaN(TB[i]) ? 0 : TB[i]); n++; x += e; x2 += e * e; }
  const mean = n ? x / n : 0;
  return { mean, sd: n > 1 ? Math.sqrt(Math.max(0, (x2 - n * mean * mean) / (n - 1))) : 0 };
}
// one set of weeks for risk shares and bets: weekly legs with no value as 0, and the weekly portfolio
function mixWeeks(m) {
  const k0 = firstWeek(m.a), ends = WEEKS.slice(k0).filter((end) => end <= m.b);
  const legs = m.ids.map((id) => { const x = WK[E.indexOf(byId[id])]; return Float64Array.from(ends, (end, k) => Number.isNaN(x[k0 + k]) ? 0 : x[k0 + k]); });
  const port = new Float64Array(ends.length);
  let start = m.a;
  ends.forEach((end, k) => { let eq = 1; for (let i = start; i <= end; i++) eq *= 1 + m.out[i - m.a]; port[k] = eq - 1; start = end + 1; });
  return { legs, port };
}
// (sum of weight times sd) squared over the mix's variance, never more than the number of engines
function betsOf(m, { legs }) {
  const T = legs.length ? legs[0].length : 0;
  if (legs.length < 2 || T < 2) return 1;
  const devs = legs.map((x) => { const lm = averageOf(x); return x.map((v) => v - lm); });
  const cov = (p, q) => { let c = 0; for (let k = 0; k < T; k++) c += p[k] * q[k]; return c / (T - 1); };
  let spread = 0, variance = 0;
  devs.forEach((p, i) => { spread += m.w[i] * Math.sqrt(cov(p, p)); devs.forEach((q, j) => { variance += m.w[i] * m.w[j] * cov(p, q); }); });
  return variance > 0 ? Math.min(spread * spread / variance, legs.length) : 1;
}
// largest risk share first. gain is the share of the return over T-bills: it can be negative, and it is null when the mix has under 1% a year to share.
const GAIN_MIN = 0.01 / 252;
function sharesOf(m, { legs, port }) {
  const pm = averageOf(port);
  const covs = legs.map((x, j) => { const lm = averageOf(x); let c = 0; for (let k = 0; k < x.length; k++) c += (x[k] - lm) * (port[k] - pm); return m.w[j] * c; });
  const totalCov = covs.reduce((p, v) => p + v, 0) || 1;
  const gains = m.ids.map((id, j) => m.w[j] * excessOf(R[id], m.a, m.b).mean), totalGain = gains.reduce((p, v) => p + v, 0);
  return m.ids.map((id, j) => ({ e: byId[id], w: m.w[j], share: covs[j] / totalCov * 100, gain: totalGain >= GAIN_MIN ? gains[j] / totalGain * 100 : null })).sort((p, q) => q.share - p.share);
}
function mixStats(w = mix, a = winA(), endI = Infinity) {
  const m = mixSeries(w, a, endI), s = seriesStats(m);
  if (!s) return null;
  const weeks = mixWeeks(m);
  return { ...s, a, b: m.b, series: m, bets: betsOf(m, weeks), shares: sharesOf(m, weeks) };
}
// names the engine whose lost edge cuts the Sharpe most. One at or under T-bills has no edge to lose; with no Sharpe to compare, the return decides.
function lineOne(w = mix, a = winA(), st = mixStats(w, a)) {
  if (!st) return '';
  const tested = st.shares.map((x) => ({ x, k: excessOf(R[x.e.id], a, st.b).mean })).filter((c) => c.k > 0).map((c) => ({ x: c.x, s: seriesStats(mixSeries(w, a, st.b, { id: c.x.e.id, k: c.k })) }));
  if (!tested.length) {
    const names = st.shares.map((x) => x.e.name);
    return names.length > 2 ? `None of its ${names.length} engines has beaten T-bills in this period.` : `${names.join(' and ')} ${names.length > 1 ? 'have' : 'has'} not beaten T-bills in this period.`;
  }
  const rated = tested.filter((c) => c.s.sharpe != null), by = rated.length ? 'sharpe' : 'ret';
  const { x: main, s } = (rated.length ? rated : tested).reduce((p, c) => (c.s[by] < p.s[by] ? c : p));
  return `${main.e.name} is ${minus(main.share, 0)}% of risk${main.gain == null ? '' : `, ${minus(main.gain, 0)}% of return`}. If its edge stops: ${s.sharpe == null ? '' : `Sharpe ${num(s.sharpe)}, `}${s.ann ? 'CAGR' : 'Return'} ${pct(s.ret)}.`;
}
// what comes off an engine's daily return so that its own Sharpe reads its fade over [a, end]; null if it is there already
function fadeShift(e, a, end) {
  const x = excessOf(R[e.id], a, end), k = x.mean - e.fade.sharpe / Math.sqrt(252) * x.sd;
  return k > 0 ? { id: e.id, k } : null;
}
const NEXT_ERAS = [0, idxOf('2012-10-02'), idxOf('2018-01-02'), idxOf('2023-01-03'), LAST + 1].map((x, k, all) => [x, all[k + 1] - 1]).slice(0, 4);
// the mix at 90% of its weights, plus 10% of this engine
function plusTen(w, id) {
  const total = Object.keys(w).reduce((s, k) => s + (w[k] > 0 ? w[k] : 0), 0), out = {};
  Object.keys(w).forEach((k) => { if (w[k] > 0) out[k] = w[k] / total * 90; });
  out[id] = 10;
  return out;
}
function lineTwo(w = mix, a = winA()) {
  const m = mixSeries(w, a);
  if (!m) return '';
  const weeks = WEEKS.slice(firstWeek(a)).filter((end) => end <= m.b).length;
  if (weeks < 104) return `Next 10%: too short to tell (${plural(weeks, 'week')}).`;
  const sharpe = (wts, from, to, shift = null) => { const s = seriesStats(mixSeries(wts, from, to, shift)); return s ? s.sharpe : null; };
  const own = new Map(), ownSharpe = (from, to) => { const key = `${from}:${to}`; if (!own.has(key)) own.set(key, sharpe(w, from, to)); return own.get(key); };
  if (ownSharpe(a, m.b) == null) return 'Next 10%: too little movement for a Sharpe.';
  // mix and candidate over the same days
  const gap = (id, from, to, shift) => { const x = sharpe(plusTen(w, id), from, to, shift), y = ownSharpe(from, to); return x == null || y == null ? null : x - y; };
  // an era with no Sharpe does not vote: 3 of the eras that do, or 2 of 2
  const holds = (c) => { const votes = NEXT_ERAS.map(([x, y]) => gap(c.e.id, x, Math.min(y, c.end))).filter((v) => v != null); return votes.filter((v) => Math.sign(v) === Math.sign(c.d)).length >= (votes.length > 2 ? 3 : 2); };
  // one idea counts once: a mix holding either rule of it takes neither
  const ideas = new Set(Object.keys(w).filter((k) => w[k] > 0 && byId[k] && byId[k].idea).map((k) => byId[k].idea));
  const kept = E.filter((e) => !(w[e.id] > 0) && e.id !== 'capsule' && !e.parked && e.candidate !== false && !ideas.has(e.idea)).map((e) => { const end = Math.min(m.b, e.lastI); return { e, end, d: gap(e.id, a, end) }; }).filter((c) => c.d != null && Math.abs(c.d) > 0.03 && holds(c));
  const best = kept.filter((c) => c.d > 0).sort((p, q) => q.d - p.d)[0], worst = kept.filter((c) => c.d < 0).sort((p, q) => p.d - q.d)[0];
  // a lower Sharpe to say alongside the gain, whole history only; later windows claim no haircut
  const fade = (c) => {
    const cut = c.e.fade && a === 0 ? fadeShift(c.e, a, c.end) : null, v = cut ? gap(c.e.id, a, c.end, cut) : null;
    return v == null ? '' : `, ${changeText(v, 2)} ${c.e.fade.word}`;
  };
  const worstText = worst ? `Worst that holds: ${worst.e.name} ${signed(worst.d, 2)}.` : '';
  // the other rule of an idea it holds was not tried, so the words cover only what was
  const nothing = ideas.size ? 'no new idea' : 'nothing';
  if (best) return `Next 10%: ${best.e.name} ${signed(best.d, 2)} Sharpe${fade(best)}. ${worstText || `${nothing[0].toUpperCase()}${nothing.slice(1)} steady worse than −0.03.`}`;
  return worst ? `Next 10%: ${nothing} steady beyond +0.03 Sharpe. ${worstText}` : `Next 10%: ${nothing} steady beyond 0.03 Sharpe either way.`;
}
// Since 2008 only: this mix with what it holds at those lower Sharpes, against the preset it started from at its own such figures
function lineFade(w = mix, a = winA(), base = null) {
  const m = a === 0 ? mixSeries(w, a) : null;
  if (!m) return '';
  const end = base ? base.b : m.b, cutsOf = (ids) => ids.filter((id) => byId[id] && byId[id].fade).map((id) => ({ e: byId[id], cut: fadeShift(byId[id], a, end) })).filter((c) => c.cut);
  const held = cutsOf(m.ids), s = held.length ? seriesStats(mixSeries(w, a, end, held.map((c) => c.cut))) : null;
  if (!s || s.sharpe == null) return '';
  const start = base ? presetNamed(basePreset).w : null;
  const from = base ? seriesStats(mixSeries(start, a, end, cutsOf(Object.keys(start).filter((id) => start[id] > 0)).map((c) => c.cut))) : null;
  return `At ${held.map(({ e }) => `${e.name}'s ${e.fade.label} ${num(e.fade.sharpe)}`).join(' and ')}: Sharpe ${num(s.sharpe)}${from && from.sharpe != null ? `, ${changeText(s.sharpe - from.sharpe, 2)} vs ${basePreset}` : ''}.`;
}
const BETS_TIP = 'How many unrelated bets of equal risk this mix is like, given its weights. Two copies of one engine read 1.0.';
const NEXT_TIP = ['Takes 10% from the rest of the mix. A name shows only if the Sharpe moves more than 0.03 and the sign holds in 3 of the eras that vote, or 2 of 2. An era with no Sharpe does not vote. Eras: 2008 to Sep 2012, Oct 2012 to 2017, 2018 to 2022, 2023 on.',
  ...E.filter((e) => e.candidate === false).map((e) => `${e.name} is not offered: ${e.candidateWhy}.`),
  ...[...new Set(E.map((e) => e.idea).filter(Boolean))].map((idea) => `A mix holding ${E.filter((e) => e.idea === idea).map((e) => e.name).join(' or ')} takes neither.`)].join(' ');
const EDGE_TIP = "The engine whose lost edge cuts the Sharpe most. It keeps its swings but earns only T-bills on average in this period: the fund study's engine-dead test.";
const FADE_TIP = 'Takes enough off that engine every day to bring its own Sharpe to that figure over this period, then reruns the mix, and the preset at its own such figures.';
const CHANGE_TIP = 'This mix minus the preset, over the same days. On Max DD a plus is a shallower fall.';
// the label gutter of Build's charts: fixed, so nothing jumps when a name changes
const buildRight = (W) => (W < 420 ? 100 : W < 700 ? 116 : 190);
const presetTip = (p) => p.tip;
const totalText = (total) => esc(`Total ${total.toFixed(0)}%${total > 0 && total.toFixed(0) !== '100' ? ', scaled to 100%' : ''}`);
function renderWeights() {
  let h = '';
  FAM.forEach((f) => {
    h += `<div class="wfam c-${f.id}"><span class="dot s" style="margin:0"></span>${esc(f.name)}</div>`;
    E.filter((e) => e.fam === f.id && e.id !== 'capsule').forEach((e) => {
      const w = mix[e.id] || 0;
      h += `<div class="wrow c-${e.fam}${w > 0 ? ' on' : ''}"><label for="w-${e.id}"${tipAttr(e.sub)}>${eName(e)}</label><input id="w-${e.id}" type="range" min="0" max="100" step="1" value="${w}" style="--fill:${w}%" data-w="${e.id}" aria-label="Weight of ${esc(e.name)}"><input id="wn-${e.id}" type="number" min="0" max="100" step="any" value="${+w.toFixed(1)}" data-w="${e.id}" aria-label="Weight of ${esc(e.name)} in percent"></div>`;
    });
  });
  const total = Object.values(mix).reduce((s, v) => s + (+v || 0), 0);
  h += `<div class="wsum"><span id="wtotal">${totalText(total)}</span><button data-clear="1">Clear</button></div>`;
  $('weights').innerHTML = h;
  $('presets').innerHTML = D.presets.map((p) => `<button data-preset="${esc(p.name)}" aria-pressed="${activePreset === p.name}"${tipAttr(presetTip(p))}>${esc(p.name)}</button>`).join('');
}
function renderMix() {
  store.set('em-mix', JSON.stringify({ mix, preset: activePreset, base: basePreset }));
  const st = mixStats();
  document.querySelectorAll('#s-build .dsec').forEach((d) => { d.hidden = !st; });
  // a leg with no data in this window: the mix computes without it and names it
  const legsOut = Object.keys(mix).filter((k) => mix[k] > 0 && byId[k] && byId[k].lastI < winA()).map((k) => byId[k].name);
  if (!st) { $('mixstats').innerHTML = `<div style="color:var(--ink-3);grid-column:1/-1">${esc(legsOut.length ? `${legsOut.join(' and ')} ${legsOut.length > 1 ? 'have' : 'has'} no data in this period.` : 'Set a weight to see the portfolio.')}</div>`; ['m-growth', 'm-dd', 'm-years', 'm-risk', 'm-risklist', 'm-lines'].forEach((id) => { $(id).innerHTML = ''; }); return; }
  const m = st.series;
  // under a quarter of weeks a split of risk or return is noise
  const weeksHeld = WEEKS.slice(firstWeek(m.a)).filter((end) => end <= m.b).length, thin = weeksHeld < WEEKS_MIN;
  // where it started: both over the same days
  const base = basePreset && !activePreset ? mixStats(presetNamed(basePreset).w, m.a, m.b) : null;
  const now = base && base.b < m.b ? mixStats(mix, m.a, base.b) : st;
  // the row keeps its room with no base, so the charts below stay put when a slider moves
  const change = (key, d) => `<small>${!base || now[key] == null || base[key] == null ? '&nbsp;' : changeText(now[key] - base[key], d)}</small>`;
  $('mixstats').innerHTML = `<div><b>${pct(st.ret)}</b><span>${st.ann ? 'CAGR' : 'Return'}</span>${change('ret', 1)}</div><div><b${sharpeTip(null, st)}>${sharpeText(st)}</b><span>Sharpe</span>${change('sharpe', 2)}</div><div><b>${pct(st.dd)}</b><span>Max DD</span>${change('dd', 1)}</div><div><b>${pct(st.vol)}</b><span>Vol</span>${change('vol', 1)}</div><div${tipAttr(BETS_TIP)}>${thin ? '<b class="word">Too few weeks</b>' : `<b>${st.bets.toFixed(1)}</b>`}<span>Independent bets</span>${change(thin ? '' : 'bets', 1)}</div><div class="mixbase"${base ? tipAttr(CHANGE_TIP) : ''}>${base ? esc(`Change against ${basePreset}`) : '&nbsp;'}</div>`;
  $('s-build').style.setProperty('--c', 'var(--ink-2)');
  const label = activePreset || (basePreset ? 'Changed' : 'Portfolio'), right = buildRight($('m-growth').clientWidth);
  const baseLine = base ? { name: basePreset, r: base.series.out, base: m.a } : null;
  growthChart($('m-growth'), [{ name: label, r: m.out, base: m.a, color: 'var(--ink)', width: 2.5, strong: true }].concat(baseLine ? { ...baseLine, color: 'var(--ink-2)', width: 1.5 } : [], { name: 'QQQ', r: QQQ, color: 'var(--ink-3)', width: 1.5 }, { name: 'T-bills', r: TB, color: 'var(--ref)', width: 1.5 }), m.a, m.b, right);
  ddChart($('m-dd'), m.out, m.a, m.b, m.a, label, baseLine, right);
  // all history, as in the drawer. A leg that stopped ends that mix, so any later year comes from this window's mix
  const whole = mixSeries(mix, 0);
  const yearsOf = (x) => [...new Set(YEAR.slice(x.a, x.b + 1))].map((y) => { const last = Math.min(x.b, YEAR.lastIndexOf(y)); let eq = 1; for (let i = Math.max(x.a, YEAR.indexOf(y)); i <= last; i++) eq *= 1 + x.out[i - x.a]; return { y, v: (eq - 1) * 100, to: x.b < YEAR.lastIndexOf(y) ? ISO[x.b] : null }; });
  yearChart($('m-years'), yearsOf(whole).concat(yearsOf(m).filter((o) => o.y > YEAR[whole.b])), m.a, Math.max(m.b, whole.b), right);
  const rest = (x) => `${x.gain == null ? '' : `${minus(x.gain, 0)}% of return, `}${(x.w * 100).toFixed(0)}% of money`;
  $('m-risk').innerHTML = thin ? '' : st.shares.filter((x) => x.share > 0.3).map((x) => `<i class="c-${x.e.fam}" style="flex:${x.share.toFixed(1)} 0 0" data-tip="${esc(x.e.name)}: ${minus(x.share, 0)}% of risk, ${rest(x)}"></i>`).join('');
  // only money shows under a quarter of weeks, so money sets the order
  $('m-risklist').innerHTML = (thin ? st.shares.slice().sort((p, q) => q.w - p.w) : st.shares).map((x) => `<span data-open="${x.e.id}"${tipAttr(x.e.sub)}><span class="dot s s-${stageKey(x.e.stage)}"></span>${eName(x.e)} ${thin ? `${(x.w * 100).toFixed(0)}% of money` : `<b>${minus(x.share, 0)}%</b> of risk, ${rest(x)}`}</span>`).join('');
  const stops = m.b < LAST ? m.ids.filter((id) => byId[id].lastI === m.b).map((id) => byId[id].name) : [];
  const one = thin ? `Too few weeks to split the risk: ${plural(weeksHeld, 'week')}.` : lineOne(mix, m.a, st), faded = lineFade(mix, m.a, base);
  $('m-lines').innerHTML = `<div${tipAttr(one.includes('If its edge stops') ? EDGE_TIP : '')}>${esc(one)}</div><div${tipAttr(NEXT_TIP)}>${esc(lineTwo(mix, m.a))}</div>` + (faded ? `<div${tipAttr(FADE_TIP)}>${esc(faded)}</div>` : '') + (stops.length ? `<div>${esc(`This mix ends ${dayLabel(m.b)}: ${stops.join(' and ')} ${stops.length > 1 ? 'have' : 'has'} no later data.`)}</div>` : '') + (legsOut.length ? `<div>${esc(`Computed without ${legsOut.join(' and ')}: no data in this period.`)}</div>` : '');
}

/* events */
function sortTable(th) {
  const i = +th.dataset.sort;
  // Plan Sharpe off the table sorts as Sharpe: the column that shows the arrow reverses
  if (i === (planOn() || sortCol !== 5 ? sortCol : SORT_DEFAULT)) { sortCol = i; sortDir = -sortDir; } else { sortCol = i; sortDir = i < 3 ? 1 : -1; }
  redrawKeeping(th, 'data-sort', renderTable);
}
// the title lands below the bar whatever the bar's height, and focus goes along, so the next Tab goes on from there
const setBarHeight = () => { const bar = document.querySelector('.system-topbar'); document.documentElement.style.setProperty('--bar-h', `${bar ? bar.offsetHeight : 0}px`); };
function jumpTo(section) {
  setBarHeight();
  section.tabIndex = -1;
  section.focus({ preventScroll: true });
  section.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
}
function sortHeat(spec, th) {
  const [table, col] = spec.split(':'), heads = () => [...document.querySelectorAll(`[data-hsort="${spec}"]`)], at = Math.max(0, heads().indexOf(th));
  stepSort(table, col);
  ({ now: renderNow, years: renderYears, factors: renderFactors })[table]();
  heads()[at].focus({ preventScroll: true });
}
// a redraw replaces the buttons: focus stays on the one pressed
function redrawKeeping(button, attr, draw) {
  const had = document.activeElement === button, value = button.getAttribute(attr);
  draw();
  if (had) { const again = document.querySelector(`[${attr}="${CSS.escape(value)}"]`); if (again) again.focus({ preventScroll: true }); }
}
// a redraw above the reader must not move what the reader is looking at: one thing is measured before it, and the page scrolls by the difference.
// That thing is the one given, else a section title on screen, else a tile or row on screen that outlives the redraw, else the last title above the view
function holdPlace(draw, given = null) {
  const bar = document.querySelector('.system-topbar'), top = bar && /sticky|fixed/.test(getComputedStyle(bar).position) ? bar.getBoundingClientRect().bottom : 0;
  const yOf = (n) => n.getBoundingClientRect().top, onScreen = (n) => yOf(n) >= top && yOf(n) < innerHeight, titles = [...document.querySelectorAll('.shead')];
  const fixed = given || titles.find(onScreen);
  const seen = fixed ? [] : [...document.querySelectorAll('#bands .tile, #tableview tbody tr')].filter((n) => n.offsetParent && onScreen(n)).map((n) => [n.dataset.open, yOf(n)]);
  const mark = fixed || titles.filter((n) => yOf(n) < top).pop(), was = mark ? yOf(mark) : 0;
  draw();
  for (const [id, y] of seen) {
    const again = document.querySelector(`#bands .tile[data-open="${id}"], #tableview tbody tr[data-open="${id}"]`);
    if (again && again.offsetParent) { scrollBy(0, yOf(again) - y); return; }
  }
  // none of the tiles in view is left and the band now ends above the view: back to its start
  const band = seen.length ? ($('bands').hidden ? $('tableview') : $('bands')).getBoundingClientRect() : null;
  if (band && band.bottom < top) { scrollBy(0, band.top - top - 16); return; }
  if (mark) scrollBy(0, yOf(mark) - was);
}
document.addEventListener('click', (ev) => {
  const t = ev.target;
  // the map's controls and drawer all sit inside its root; the rest of Bench keeps its own clicks
  if (!ROOT.contains(t)) return;
  // the second click of the double click that closed the drawer lands on the page behind
  if (ev.detail > 1 && performance.now() - closedAt < 500) return;
  // the second click of the double click that opened the drawer lands on whatever now sits under the pointer
  if (ev.detail > 1 && (t === $('scrim') || ($('drawer').contains(t) && performance.now() - openedAt < 500))) return;
  if (t.closest('[data-close]') || t === $('scrim')) { closeDrawer(); return; }
  const step = t.closest('[data-step]');
  if (step) { stepDrawer(+step.dataset.step); return; }
  const q = t.closest('.qbtn');
  if (q) { q.setAttribute('aria-expanded', q.closest('.shead').classList.toggle('open')); return; }
  const hs = t.closest('[data-hsort]');
  if (hs) { sortHeat(hs.dataset.hsort, hs); return; }
  const per = t.closest('[data-period]');
  if (per) { if (per.getAttribute('aria-disabled') === 'true') return; period = PERIODS.find((p) => p.id === per.dataset.period); store.set('em-period', period.id); holdPlace(() => redrawKeeping(per, 'data-period', refreshAll)); return; }
  const my = t.closest('[data-mapy]'), mx = t.closest('[data-mapx]');
  if (my || mx) { if (my) { mapY = my.dataset.mapy; store.set('em-mapy', mapY); } else { mapX = mx.dataset.mapx; store.set('em-mapx', mapX); } const b = my || mx; redrawKeeping(b, my ? 'data-mapy' : 'data-mapx', () => { renderHeader(); renderMap(); }); return; }
  const cm = t.closest('[data-chart]');
  if (cm) { chartMode = cm.dataset.chart; store.set('em-chart', chartMode); holdPlace(() => redrawKeeping(cm, 'data-chart', () => { renderHeader(); showView('chart'); renderTiles(); })); return; }
  const st = t.closest('[data-stage]');
  if (st) { stageFilter = st.dataset.stage; holdPlace(() => redrawKeeping(st, 'data-stage', () => { renderHeader(); renderTiles(); renderMap(); renderNow(); renderSim(); renderCorr(); renderStress(); renderYears(); renderFactors(); if (!$('tableview').hidden) renderTable(); })); return; }
  const sd = t.closest('[data-stressdef]');
  if (sd) { stressDef = sd.dataset.stressdef; store.set('em-stress', stressDef); redrawKeeping(sd, 'data-stressdef', renderStress); return; }
  const gr = t.closest('[data-group]');
  if (gr) { groupMode = gr.dataset.group; store.set('em-group', groupMode); showView('chart'); redrawKeeping(gr, 'data-group', renderTiles); return; }
  const jump = t.closest('[data-jump]');
  if (jump) { jumpTo($(jump.dataset.jump)); return; }
  const preset = t.closest('[data-preset]');
  if (preset) { const p = D.presets.find((x) => x.name === preset.dataset.preset); mix = { ...p.w }; activePreset = basePreset = p.name; redrawKeeping(preset, 'data-preset', () => { renderWeights(); renderMix(); }); return; }
  const clear = t.closest('[data-clear]');
  if (clear) { mix = {}; activePreset = basePreset = ''; redrawKeeping(clear, 'data-clear', () => { renderWeights(); renderMix(); }); return; }
  const th = t.closest('[data-sort]');
  if (th) { sortTable(th); return; }
  const pill = t.closest('[data-pill]');
  if (pill) { const p = presetNamed(pill.dataset.pill); mix = { ...p.w }; activePreset = basePreset = p.name; closeDrawer(); renderWeights(); renderMix(); jumpTo($('s-build')); return; }
  const open = dotAt(ev) || t.closest('[data-open]');
  if (open) openDrawer(open.dataset.open);
});
document.addEventListener('input', (ev) => {
  const t = ev.target;
  if (t.id === 'b-cut') { cutValue = +t.value; setCutFill(); holdPlace(renderCut, t); return; }
  if (t.dataset && t.dataset.w) {
    const v = Math.max(0, Math.min(100, +t.value || 0)), id = t.dataset.w;
    if (v > 0) mix[id] = v; else delete mix[id];
    settlePreset();
    // every control shows the weight in use, but for the box being typed in: that one settles when focus leaves it
    document.querySelectorAll('.wrow input[type=number]').forEach((b) => { if (b !== t) b.value = mix[b.dataset.w] || 0; });
    if (t.type !== 'range') $('w-' + id).value = mix[id] || 0;
    $('w-' + id).style.setProperty('--fill', v + '%');
    t.closest('.wrow').classList.toggle('on', v > 0);
    const total = Object.values(mix).reduce((s, x) => s + x, 0);
    $('wtotal').innerHTML = totalText(total);
    document.querySelectorAll('#presets [data-preset]').forEach((b) => b.setAttribute('aria-pressed', b.dataset.preset === activePreset));
    clearTimeout(window.mixTimer); window.mixTimer = setTimeout(renderMix, 60);
  }
});
document.addEventListener('change', (ev) => { const t = ev.target; if (t.matches && t.matches('.wrow input[type=number]')) t.value = mix[t.dataset.w] || 0; });
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape') closeDrawer();
  if (ev.key === 'Tab' && openId) { holdTab(ev); return; }
  const inField = ev.target.matches && ev.target.matches('input, select, textarea');
  if (openId && (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') && !inField && !ev.altKey && !ev.ctrlKey && !ev.metaKey && !ev.shiftKey) { ev.preventDefault(); stepDrawer(ev.key === 'ArrowLeft' ? -1 : 1); return; }
  if ((ev.key === 'Enter' || ev.key === ' ') && ev.target.matches && ev.target.matches('[data-hsort]')) { ev.preventDefault(); sortHeat(ev.target.dataset.hsort, ev.target); return; }
  if ((ev.key === 'Enter' || ev.key === ' ') && ev.target.matches && ev.target.matches('[data-sort]')) { ev.preventDefault(); sortTable(ev.target); return; }
  if ((ev.key === 'Enter' || ev.key === ' ') && ev.target.matches && ev.target.matches('g[data-open]')) { ev.preventDefault(); openDrawer(ev.target.dataset.open); }
});
function sparkCross(ev, svg) {
  const r = svg.getBoundingClientRect(), a = winA();
  const i = Math.max(a, Math.min(LAST, Math.round(a + ((ev.clientX - r.left) / r.width) * (LAST - a))));
  const x = ((i - a) / Math.max(1, LAST - a)) * 300;
  document.querySelectorAll('.spark .xl').forEach((l) => { l.setAttribute('x1', x); l.setAttribute('x2', x); l.setAttribute('visibility', 'visible'); });
  const e = byId[svg.dataset.spark], pts = sparkData[e.id] || [];
  let lo = 0, hi = pts.length - 1; while (lo < hi) { const k = (lo + hi) >> 1; if (pts[k].i < i) lo = k + 1; else hi = k; }
  const p = pts[lo];
  if (p) showTip(`<b>${dayLabel(p.i)}</b><div class="r"><i>${eName(e)}${p.eq == null ? ', 3-year Sharpe' : ''}</i><b>${p.eq == null ? num(p.v) : `$${p.eq.toFixed(2)}`}</b></div>`, ev.clientX, ev.clientY);
}
let hotId = null;
function setHot(id) {
  if (id === hotId) return;
  if (hotId) document.querySelectorAll(`[data-hover="${hotId}"]`).forEach((n) => n.classList.remove('hot'));
  hotId = id;
  if (id) document.querySelectorAll(`.pt[data-hover="${id}"], .tile[data-hover="${id}"]`).forEach((n) => n.classList.add('hot'));
}
document.addEventListener('pointermove', (ev) => {
  const t = ev.target;
  if (!t.closest) return;
  // a name on a map acts as its dot
  const name = t.closest('text.lbl[data-open]'), pt = dotAt(ev) || (name && name.ownerSVGElement.querySelector(`.pt[data-hover="${name.dataset.open}"]`)) || t.closest('.pt'), hov = pt || t.closest('[data-hover]');
  setHot(hov ? hov.dataset.hover : null);
  const simDot = pt && pt.closest('#simbox') ? pt : null;
  simHover(simDot ? simDot.dataset.hover : null);
  const spark = t.closest('.spark');
  if (spark) { sparkCross(ev, spark); return; }
  document.querySelectorAll('.spark .xl[visibility="visible"]').forEach((l) => l.setAttribute('visibility', 'hidden'));
  const cell = t.closest('.cell');
  if (cell) {
    const i = +cell.dataset.i, j = +cell.dataset.j;
    document.querySelectorAll('#corrbox .rowl.on').forEach((x) => x.classList.remove('on'));
    document.querySelector(`#corrbox [data-r="${i}"]`).classList.add('on');
    document.querySelector(`#corrbox [data-c="${j}"]`).classList.add('on');
    showTip(i === j ? `<b>${eName(E[i])}</b>` : `<b>${num(CORR[i][j])}</b> <i>${eName(E[i])}, ${eName(E[j])}</i>`, ev.clientX, ev.clientY);
    return;
  }
  document.querySelectorAll('#corrbox .rowl.on').forEach((x) => x.classList.remove('on'));
  if (pt) { const e = byId[pt.dataset.hover], s = e.s; if (simDot) showTip(simTip(e), ev.clientX, ev.clientY); else if (s) showTip(`<b>${eName(e)}</b><div style="opacity:.8">${esc(e.sub)}</div><div class="r"><i>${wordOf(s)}${e.lastI < LAST ? `, ${toEnd(e)}` : ''}</i><b>${pct(s.ret)}</b></div><div class="r"><i>Sharpe</i><b>${sharpeText(s, e)}</b></div><div class="r"><i>Max DD</i><b>${pct(s.dd)}</b></div>`, ev.clientX, ev.clientY); return; }
  const tipped = t.closest('[data-tip]');
  if (tipped) { showTip(esc(tipped.dataset.tip), ev.clientX, ev.clientY); return; }
  // only a chart with a cross keeps its tip off its marks
  const chart = t.closest('.chart svg');
  if (!chart || !chart.querySelector('.xl')) hideTip();
});
// a tip and a crosshair describe what is under the pointer: they go when the page moves under it, or when it leaves the window
function clearPointer() {
  hideTip(); setHot(null); simHover(null);
  document.querySelectorAll('.xl[visibility="visible"]').forEach((l) => l.setAttribute('visibility', 'hidden'));
  document.querySelectorAll('#corrbox .rowl.on').forEach((x) => x.classList.remove('on'));
}
addEventListener('scroll', clearPointer, { capture: true, passive: true });
document.documentElement.addEventListener('pointerleave', clearPointer);

function setPressed(ids, on) { ids.forEach((b) => $(b).setAttribute('aria-pressed', b === on)); }
function showView(v) {
  setPressed(['v-chart', 'v-table'], v === 'chart' ? 'v-chart' : 'v-table');
  $('bands').hidden = v !== 'chart'; $('tableview').hidden = v === 'chart';
  if (v === 'table') renderTable();
}
$('v-chart').addEventListener('click', () => showView('chart'));
$('v-table').addEventListener('click', () => showView('table'));

function refreshAll() {
  BENCH = periodStats();
  renderHeader();
  renderMap();
  renderBets();
  renderTiles();
  if (!$('tableview').hidden) renderTable();
  renderMix();
  if (openId) openDrawer(openId);
}
setCutFill(); renderHeader(); renderNow(); computeCorr(); computeCoords(); renderMap(); renderTiles(); renderBetStats(); renderSim(); renderCorr(); renderStress(); renderYears(); renderFactors(); renderWeights(); renderMix();
setBarHeight();
ROOT.classList.add('ready');
window.EM = { openDrawer, rollSharpeAt, get chartMode() { return chartMode; }, statsOf, ownWorst, R, E, TB, QQQ, idxOf, LAST, D, get simCoords() { return simCoords; }, get simOn() { return simOn; }, get CORR() { return CORR; }, FRAME, COORDS, lineOne, lineTwo, lineFade,
  mixStats: (w, a, endI) => { const s = mixStats(w, a, endI); return s && { ret: s.ret, sharpe: s.sharpe, dd: s.dd, vol: s.vol, n: s.n, end: ISO[s.b], bets: s.bets, shares: s.shares.map((x) => ({ name: x.e.name, risk: x.share, ret: x.gain, money: x.w * 100 })) }; },
  bets: (w, a) => { const s = mixStats(w, a); return s ? s.bets : null; } };
let resizeTimer, lastWidth = innerWidth;
// no drawing depends on the window's height
addEventListener('resize', () => { if (innerWidth === lastWidth) return; lastWidth = innerWidth; clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { setBarHeight(); mapNodes = null; simOpened = false; renderMap(); renderSim(); renderCorr(); renderStress(); renderMix(); if (openId) drawDrawerCharts(byId[openId], winA()); }, 120); });
if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { mapNodes = null; renderMap(); renderSim(); });
})().catch((error) => {
  // anything on the way up, the fetch or the first draw: one plain message and no half-drawn page
  console.error(error);
  const root = document.getElementById('emap'), body = root && root.querySelector('.emap-body');
  if (root) root.classList.add('ready');
  if (body) body.innerHTML = '<div class="notice notice-block"><div class="notice-title">The map did not load.</div><div class="notice-detail">Reload the page. If it stays, start Refresh map, or run python -m alpha.bench.engine_map build.</div></div>';
});
