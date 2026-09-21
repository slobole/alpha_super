const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source_str = fs.readFileSync(path.join(__dirname, '../alpha/live/dashboard_v4/static/performance.js'), 'utf8');

function environment_obj() {
  const handlers_dict = {};
  const document_obj = {activeElement: null, addEventListener(name_str, callback_fn) { (handlers_dict[name_str] ||= []).push(callback_fn); }};
  function emit(name_str, target_obj, options_dict = {}) {
    const event_obj = {target: target_obj, relatedTarget: null, prevented_bool: false,
      preventDefault() { this.prevented_bool = true; }, ...options_dict};
    for (const callback_fn of handlers_dict[name_str] || []) callback_fn(event_obj);
    return event_obj;
  }
  function element_obj(attributes_dict = {}, parent_obj = null) {
    const classes_set = new Set();
    const result_obj = {parent_obj, children_list: [], hidden: false, textContent: '', tabIndex: -1,
      getAttribute(key_str) { return attributes_dict[key_str] ?? null; },
      setAttribute(key_str, value_str) { attributes_dict[key_str] = value_str; },
      matches(selector_str) { return selector_str.startsWith('.') ? classes_set.has(selector_str.slice(1)) : Object.hasOwn(attributes_dict, selector_str.slice(1, -1)); },
      closest(selector_str) { return this.matches(selector_str) ? this : parent_obj?.closest(selector_str) || null; },
      contains(node_obj) { return node_obj === this || this.children_list.some(child_obj => child_obj.contains(node_obj)); },
      querySelectorAll(selector_str) { return this.children_list.flatMap(child_obj => [
        ...(selector_str.split(', ').some(item_str => child_obj.matches(item_str)) ? [child_obj] : []), ...child_obj.querySelectorAll(selector_str)]); },
      querySelector(selector_str) { return this.querySelectorAll(selector_str)[0] || null; },
      classList: {toggle(name_str, enabled_bool) { enabled_bool ? classes_set.add(name_str) : classes_set.delete(name_str); }, contains(name_str) { return classes_set.has(name_str); }},
      focus() { document_obj.activeElement = this; emit('focusin', this); }};
    if (parent_obj) parent_obj.children_list.push(result_obj);
    return result_obj;
  }
  const panel_obj = element_obj({'data-daily-panel': '', 'data-date-label': 'Fri 2026-09-11', 'data-pnl': '+$2.00', 'data-return': '+0.02%', 'data-tone': 'pos', 'data-last-readout': 'Last Fri 2026-09-11 · +$2.00 · +0.02%'});
  const readout_obj = element_obj({'data-daily-readout': ''}, panel_obj);
  const last_obj = element_obj({'data-daily-last': ''}, readout_obj);
  const date_obj = element_obj({'data-daily-date': ''}, readout_obj);
  const value_obj = element_obj({'data-daily-value': ''}, readout_obj);
  const return_obj = element_obj({'data-daily-return': ''}, readout_obj);
  last_obj.textContent = 'Last ';
  date_obj.textContent = panel_obj.getAttribute('data-date-label');
  value_obj.textContent = panel_obj.getAttribute('data-pnl');
  return_obj.textContent = panel_obj.getAttribute('data-return');
  value_obj.classList.toggle('pos', true);
  Object.defineProperty(readout_obj, 'textContent', {get() { return `${last_obj.textContent}${date_obj.textContent} · ${value_obj.textContent} · ${return_obj.textContent}`; }});
  const mode_list = ['bars', 'numbers'].map(mode_str => element_obj({'data-daily-mode': mode_str, 'aria-pressed': String(mode_str === 'bars')}, panel_obj));
  const view_list = ['bars', 'numbers'].map(mode_str => element_obj({'data-daily-view': mode_str}, panel_obj));
  view_list[1].hidden = true;
  const chart_obj = element_obj({}, view_list[0]);
  chart_obj.classList.toggle('daily-chart', true);
  const readout_list = ['Tue 2026-09-08 · $0.00 · 0.00%', 'Wed 2026-09-09 · — · —', 'Fri 2026-09-11 · +$2.00 · +0.02%'];
  const day_lists = view_list.map(view_obj => readout_list.map((readout_str, index_int) => {
    const [date_str, pnl_str, return_str] = readout_str.split(' · ');
    const day_obj = element_obj({'data-daily-day': String(index_int), 'data-readout': readout_str,
      'data-date-label': date_str, 'data-pnl': pnl_str, 'data-return': return_str, 'data-tone': index_int === 2 ? 'pos' : ''}, view_obj);
    day_obj.tabIndex = index_int === 2 ? 0 : -1;
    return day_obj;
  }));
  const paint_list = [0, 2].map(index_int => element_obj({'data-daily-paint': String(index_int)}, chart_obj));
  const header_obj = element_obj({}, panel_obj);
  const outside_obj = element_obj();
  vm.runInNewContext(source_str, {document: document_obj, URL, window: {location: {}}});
  return {emit, panel_obj, readout_obj, value_obj, return_obj, mode_list, view_list, chart_obj, day_lists, paint_list, document_obj, header_obj, outside_obj};
}

test('Bars and Numbers switch locally and expose the selected state', () => {
  const env_obj = environment_obj();
  env_obj.emit('click', env_obj.mode_list[1]);
  assert.deepEqual(env_obj.view_list.map(view_obj => view_obj.hidden), [true, false]);
  assert.deepEqual(env_obj.mode_list.map(mode_obj => mode_obj.getAttribute('aria-pressed')), ['false', 'true']);
  env_obj.emit('click', env_obj.mode_list[0]);
  assert.deepEqual(env_obj.view_list.map(view_obj => view_obj.hidden), [false, true]);
});

for (const event_str of ['pointerover', 'focusin', 'click']) test(`${event_str} shows exact saved readout and dims other bars`, () => {
  const env_obj = environment_obj();
  const day_obj = env_obj.day_lists[0][0];
  env_obj.emit(event_str, day_obj);
  assert.equal(env_obj.readout_obj.textContent, day_obj.getAttribute('data-readout'));
  assert.equal(env_obj.chart_obj.classList.contains('is-active'), true);
  assert.deepEqual(env_obj.paint_list.map(paint_obj => paint_obj.classList.contains('is-active')), [true, false]);
  assert.deepEqual(env_obj.day_lists[0].map(item_obj => item_obj.tabIndex), [0, -1, -1]);
});

test('missing day remains selectable and never fabricates a number', () => {
  const env_obj = environment_obj();
  env_obj.emit('click', env_obj.day_lists[1][1]);
  assert.equal(env_obj.readout_obj.textContent, 'Wed 2026-09-09 · — · —');
  assert.equal(env_obj.paint_list.some(paint_obj => paint_obj.classList.contains('is-active')), false);
});

for (const view_int of [0, 1]) test(`readout colour follows selected gain/loss and clears for zero/missing in view ${view_int}`, () => {
  const env_obj = environment_obj();
  const day_list = env_obj.day_lists[view_int];
  const loss_obj = day_list[2];
  loss_obj.setAttribute('data-pnl', '-$1,320.00');
  loss_obj.setAttribute('data-tone', 'neg');
  loss_obj.setAttribute('data-return', '-1.06%');
  env_obj.emit('click', loss_obj);
  assert.equal(env_obj.value_obj.textContent, '-$1,320.00');
  assert.equal(env_obj.value_obj.classList.contains('neg'), true);
  assert.equal(env_obj.value_obj.classList.contains('pos'), false);
  assert.equal(env_obj.return_obj.textContent, '-1.06%');
  assert.equal(env_obj.return_obj.classList.contains('neg'), false);
  for (const day_obj of day_list.slice(0, 2)) {
    env_obj.emit('focusin', day_obj);
    assert.equal(env_obj.value_obj.classList.contains('neg'), false);
    assert.equal(env_obj.value_obj.classList.contains('pos'), false);
  }
  env_obj.emit('pointerout', day_list[1], {relatedTarget: env_obj.header_obj});
  assert.equal(env_obj.value_obj.textContent, '+$2.00');
  assert.equal(env_obj.value_obj.classList.contains('pos'), true);
  env_obj.emit('click', loss_obj);
  env_obj.emit('click', env_obj.mode_list[1 - view_int]);
  assert.equal(env_obj.value_obj.classList.contains('neg'), false);
  assert.equal(env_obj.value_obj.classList.contains('pos'), true);
});

test('leaving a hovered day for the panel header restores last session', () => {
  const env_obj = environment_obj();
  env_obj.emit('pointerover', env_obj.day_lists[0][0]);
  env_obj.emit('pointerout', env_obj.day_lists[0][0], {relatedTarget: env_obj.header_obj});
  assert.equal(env_obj.readout_obj.textContent, env_obj.panel_obj.getAttribute('data-last-readout'));
  assert.equal(env_obj.chart_obj.classList.contains('is-active'), false);
});

test('pointer leaves but keyboard-focused day remains selected; blur restores last', () => {
  const env_obj = environment_obj();
  const day_obj = env_obj.day_lists[0][0];
  day_obj.focus();
  env_obj.emit('pointerout', day_obj, {relatedTarget: env_obj.outside_obj});
  assert.equal(env_obj.readout_obj.textContent, day_obj.getAttribute('data-readout'));
  env_obj.emit('focusout', day_obj, {relatedTarget: env_obj.outside_obj});
  assert.equal(env_obj.readout_obj.textContent, env_obj.panel_obj.getAttribute('data-last-readout'));
});

for (const view_int of [0, 1]) test(`keyboard arrows/Home/End include zero and missing days and clamp in view ${view_int}`, () => {
  const env_obj = environment_obj();
  const day_list = env_obj.day_lists[view_int];
  const home_obj = env_obj.emit('keydown', day_list[2], {key: 'Home'});
  assert.equal(home_obj.prevented_bool, true);
  assert.equal(env_obj.document_obj.activeElement, day_list[0]);
  env_obj.emit('keydown', day_list[0], {key: 'ArrowLeft'});
  assert.equal(env_obj.document_obj.activeElement, day_list[0]);
  env_obj.emit('keydown', day_list[0], {key: 'ArrowRight'});
  assert.equal(env_obj.document_obj.activeElement, day_list[1]);
  assert.match(env_obj.readout_obj.textContent, /— · —/);
  env_obj.emit('keydown', day_list[1], {key: 'End'});
  assert.equal(env_obj.document_obj.activeElement, day_list[2]);
  env_obj.emit('keydown', day_list[2], {key: 'ArrowRight'});
  assert.equal(env_obj.document_obj.activeElement, day_list[2]);
  assert.deepEqual(day_list.map(day_obj => day_obj.tabIndex), [-1, -1, 0]);
});
