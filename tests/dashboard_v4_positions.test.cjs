const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source_str = fs.readFileSync(path.join(__dirname, '../alpha/live/dashboard_v4/static/positions.js'), 'utf8');

function load_api(storage_dict = {}) {
  let api_obj = null;
  const listener_dict = {};
  const document_obj = {
    addEventListener(name_str, callback_fn) { (listener_dict[name_str] ||= []).push(callback_fn); },
    querySelector: () => null, querySelectorAll: () => [], getElementById: () => null,
  };
  const window_obj = {
    sessionStorage: {getItem: (key_str) => storage_dict[key_str] ?? null, setItem: (key_str, value_str) => { storage_dict[key_str] = value_str; }},
    addEventListener() {}, requestAnimationFrame: (callback_fn) => { callback_fn(); return 1; },
  };
  vm.runInNewContext(source_str, {document: document_obj, window: window_obj, positions_test_hook: (hook_obj) => { api_obj = hook_obj; }});
  return {api_obj, listener_dict, document_obj, storage_dict};
}

// Values made inside the vm context have another realm's prototypes.
const plain = (value_obj) => JSON.parse(JSON.stringify(value_obj));

function row(attribute_dict) {
  return {getAttribute: (name_str) => (Object.hasOwn(attribute_dict, name_str) ? attribute_dict[name_str] : null), attribute_dict};
}

function table(key_str, row_list, head_key_list) {
  const body_obj = {row_list: row_list.slice(),
    querySelectorAll() { return this.row_list.slice(); },
    appendChild(row_obj) { this.row_list = this.row_list.filter((item_obj) => item_obj !== row_obj).concat([row_obj]); }};
  const head_list = head_key_list.map((head_key_str) => {
    const head_dict = {'aria-sort': 'none'};
    const button_obj = {getAttribute: (name_str) => (name_str === 'data-sort-key' ? head_key_str : name_str === 'data-sort-type' ? (head_key_str === 'symbol' ? 'text' : 'number') : null),
      closest: (selector_str) => (selector_str === '[data-sortable-table]' ? table_obj : null)};
    return {getAttribute: (name_str) => head_dict[name_str], setAttribute: (name_str, value_str) => { head_dict[name_str] = value_str; },
      querySelector: () => button_obj, button_obj, head_dict};
  });
  const table_obj = {getAttribute: (name_str) => (name_str === 'data-sortable-table' ? key_str : null),
    querySelectorAll: () => head_list, querySelector: () => body_obj, body_obj, head_list};
  return table_obj;
}

test('squarify fills the box exactly with areas proportional to size', () => {
  const {api_obj} = load_api();
  const item_list = [6, 6, 4, 3, 2, 2, 1].map((size_float, index_int) => ({size_float, index_int}));
  const rect_list = api_obj.squarify_list(item_list, 10, 20, 600, 400);
  assert.equal(rect_list.length, item_list.length);
  const total_float = rect_list.reduce((sum_float, rect_obj) => sum_float + rect_obj.w * rect_obj.h, 0);
  assert.ok(Math.abs(total_float - 600 * 400) < 1e-6);
  for (const rect_obj of rect_list) {
    assert.ok(rect_obj.x >= 10 - 1e-9 && rect_obj.y >= 20 - 1e-9 && rect_obj.x + rect_obj.w <= 610 + 1e-6 && rect_obj.y + rect_obj.h <= 420 + 1e-6);
    assert.ok(Math.abs(rect_obj.w * rect_obj.h / (600 * 400) - rect_obj.item_obj.size_float / 24) < 1e-9);
  }
  assert.deepEqual(plain(api_obj.squarify_list([], 0, 0, 10, 10)), []);
  assert.deepEqual(plain(api_obj.squarify_list([{size_float: 0}], 0, 0, 10, 10)), []);
});

test('colour buckets use 0.5%, 3% and 10% edges and treat unknown as neutral', () => {
  const {api_obj} = load_api();
  assert.deepEqual(plain([null, 0.004, 0.005, -0.029, 0.03, -0.0999, 0.1, -0.5].map(api_obj.bucket_str)), ['0', '0', 'p1', 'n1', 'p2', 'n2', 'p3', 'n3']);
  assert.equal(api_obj.compact_money_str(-1234.4, true), '−$1,234');
  assert.equal(api_obj.compact_money_str(12.6, true), '+$13');
});

test('blocks keep the largest tiles, fold the rest into one, and show cash only by value', () => {
  const {api_obj} = load_api();
  const pod_obj = {name_str: 'Pod', nav_float: 1000, nav_str: '$1,000', cash_float: 100, cash_str: '$100', pnl_str: '+$5', return_str: '+0.5%',
    item_list: Array.from({length: 6}, (_, index_int) => ({key_str: 'S' + index_int + '|p', symbol_str: 'S' + index_int, value_float: 100 + index_int,
      multiplier_float: index_int === 0 ? 3 : 1, pnl_float: index_int, return_float: 0.01, pnl_str: '', return_str: '', status_str: 'open', tip_list: []}))};
  const value_list = api_obj.block_item_list(pod_obj, 'value', 4);
  assert.deepEqual(plain(value_list.map((item_obj) => item_obj.symbol_str)), ['S5', 'S4', 'S3', '+3 more', 'Cash']);
  assert.equal(value_list[3].value_float, 100 + 101 + 102);
  assert.equal(value_list[3].pnl_float, 0 + 1 + 2);
  const exposure_list = api_obj.block_item_list(pod_obj, 'exposure', 10);
  assert.equal(exposure_list[0].symbol_str, 'S0', 'TQQQ-like 3x tile is largest by exposure');
  assert.ok(!exposure_list.some((item_obj) => item_obj.cash_bool));
});

test('block headers drop detail until they fit', () => {
  const {api_obj} = load_api();
  const pod_obj = {name_str: 'TAA BTAL', nav_float: 18000, nav_str: '$18,000', pnl_str: '+$645', return_str: '+4.0%'};
  assert.deepEqual(plain(api_obj.header_text_list(pod_obj, 'value', 18000, 1000)), ['TAA BTAL  ·  $18,000  ·  ', '+$645 (+4.0%)']);
  assert.deepEqual(plain(api_obj.header_text_list(pod_obj, 'value', 18000, 140)), ['TAA BTAL  ·  ', '+$645']);
  assert.deepEqual(plain(api_obj.header_text_list(pod_obj, 'value', 18000, 40)), ['TAA BTAL', '']);
  assert.match(api_obj.header_text_list(pod_obj, 'exposure', 30000, 1000)[0], /exp \$30,000 · 1\.67x NAV/);
});

test('folded leveraged holdings label gross exposure separately from value', () => {
  const {api_obj} = load_api();
  const pod_obj = {cash_float: null, item_list: Array.from({length: 15}, (_, index_int) => ({
    key_str: 'S' + index_int, symbol_str: 'S' + index_int, value_float: 100, multiplier_float: 3, pnl_float: 0}))};
  const value_obj = api_obj.block_item_list(pod_obj, 'value', 14).at(-1);
  const exposure_obj = api_obj.block_item_list(pod_obj, 'exposure', 14).at(-1);
  assert.deepEqual(plain(value_obj.tip_list[1]), ['Value', '$200']);
  assert.deepEqual(plain(exposure_obj.tip_list[1]), ['Exposure', '$600']);
  assert.equal(exposure_obj.value_float, 600, 'folded area retains the summed exposure');
  assert.ok(!api_obj.block_item_list(pod_obj, 'value', 14).some((item_obj) => item_obj.cash_bool));
});

test('row comparison: numbers, natural text, missing values last, ties in server order', () => {
  const {api_obj} = load_api();
  const a = row({'data-sort-index': '0', 'data-sort-pnl': '5', 'data-sort-symbol': 'B10'});
  const b = row({'data-sort-index': '1', 'data-sort-pnl': '', 'data-sort-symbol': 'B9'});
  const c = row({'data-sort-index': '2', 'data-sort-pnl': '-3', 'data-sort-symbol': 'a1'});
  const d = row({'data-sort-index': '3', 'data-sort-pnl': '5', 'data-sort-symbol': 'b10'});
  const by = (key_str, type_str, direction_int) => [a, b, c, d].sort((x, y) => api_obj.compare_rows(x, y, key_str, type_str, direction_int))
    .map((item_obj) => item_obj.attribute_dict['data-sort-index']);
  assert.deepEqual(by('pnl', 'number', -1), ['0', '3', '2', '1']);
  assert.deepEqual(by('pnl', 'number', 1), ['2', '0', '3', '1']);
  assert.deepEqual(by('symbol', 'text', 1), ['2', '1', '0', '3']);
});

test('header clicks sort, flip and survive a refreshed table', () => {
  const {api_obj} = load_api();
  const make_rows = () => [row({'data-sort-index': '0', 'data-sort-value': '10', 'data-sort-symbol': 'C'}),
    row({'data-sort-index': '1', 'data-sort-value': '30', 'data-sort-symbol': 'A'}),
    row({'data-sort-index': '2', 'data-sort-value': '20', 'data-sort-symbol': 'B'})];
  const table_obj = table('positions-now', make_rows(), ['symbol', 'value']);
  const value_button_obj = table_obj.head_list[1].button_obj;
  api_obj.sort_click(value_button_obj);
  assert.deepEqual(table_obj.body_obj.row_list.map((row_obj) => row_obj.attribute_dict['data-sort-value']), ['30', '20', '10']);
  assert.equal(table_obj.head_list[1].head_dict['aria-sort'], 'descending');
  assert.equal(table_obj.head_list[0].head_dict['aria-sort'], 'none');
  api_obj.sort_click(value_button_obj);
  assert.deepEqual(table_obj.body_obj.row_list.map((row_obj) => row_obj.attribute_dict['data-sort-value']), ['10', '20', '30']);
  api_obj.sort_click(table_obj.head_list[0].button_obj);
  assert.deepEqual(table_obj.body_obj.row_list.map((row_obj) => row_obj.attribute_dict['data-sort-symbol']), ['A', 'B', 'C']);
  // The 15-second refresh brings a new table in server order; the choice is re-applied.
  const refreshed_obj = table('positions-now', make_rows(), ['symbol', 'value']);
  api_obj.apply_sort(refreshed_obj);
  assert.deepEqual(refreshed_obj.body_obj.row_list.map((row_obj) => row_obj.attribute_dict['data-sort-symbol']), ['A', 'B', 'C']);
  assert.equal(refreshed_obj.head_list[0].head_dict['aria-sort'], 'ascending');
  // Another table keeps its own server order.
  const other_obj = table('positions-month', make_rows(), ['symbol', 'value']);
  api_obj.apply_sort(other_obj);
  assert.deepEqual(other_obj.body_obj.row_list.map((row_obj) => row_obj.attribute_dict['data-sort-index']), ['0', '1', '2']);
});

test('the tile size choice is restored from session storage and blocked storage is harmless', () => {
  assert.equal(load_api({'v4-positions-map-size': 'exposure'}).api_obj.get_state().size_mode_str, 'exposure');
  assert.equal(load_api({'v4-positions-map-size': 'bogus'}).api_obj.get_state().size_mode_str, 'value');
  let api_obj = null;
  vm.runInNewContext(source_str, {document: {addEventListener() {}, querySelector: () => null, querySelectorAll: () => []},
    window: {get sessionStorage() { throw new Error('blocked'); }, addEventListener() {}}, positions_test_hook: (hook_obj) => { api_obj = hook_obj; }});
  assert.equal(api_obj.get_state().size_mode_str, 'value');
});

// ---------- a minimal DOM for drawing, selection and refresh ----------
function dom_env(map_obj, {scope_str = 'month:2026-08:all', storage_dict = {}, json_str = null} = {}) {
  const listener_dict = {};
  const document_obj = {activeElement: null};
  function matches_simple(node_obj, selector_str) {
    const match_obj = selector_str.match(/^([a-zA-Z][\w-]*)?((?:\[[^\]]+\]|\.[\w-]+)*)$/);
    if (!match_obj) throw new Error('unsupported selector ' + selector_str);
    if (match_obj[1] && node_obj.tag_str.toLowerCase() !== match_obj[1].toLowerCase()) return false;
    return (match_obj[2].match(/\[[^\]]+\]|\.[\w-]+/g) || []).every((part_str) => {
      if (part_str[0] === '.') return node_obj.class_set.has(part_str.slice(1));
      const inner_str = part_str.slice(1, -1);
      const index_int = inner_str.indexOf('=');
      if (index_int < 0) return Object.hasOwn(node_obj.attribute_dict, inner_str);
      return node_obj.attribute_dict[inner_str.slice(0, index_int)] === inner_str.slice(index_int + 1).replace(/^"|"$/g, '');
    });
  }
  class Node {
    constructor(tag_str) {
      Object.assign(this, {tag_str, attribute_dict: {}, children: [], parent: null, listener_dict: {}, own_text: '',
        class_set: new Set(), style: {}, hidden: false, clientWidth: 0});
    }
    setAttribute(key_str, value_obj) {
      this.attribute_dict[key_str] = String(value_obj);
      if (key_str === 'class') this.class_set = new Set(String(value_obj).split(/\s+/).filter(Boolean));
    }
    getAttribute(key_str) {
      if (key_str === 'class') return [...this.class_set].join(' ');
      return Object.hasOwn(this.attribute_dict, key_str) ? this.attribute_dict[key_str] : null;
    }
    get classList() {
      const self_obj = this;
      return {toggle(name_str, on_bool) { if (on_bool) self_obj.class_set.add(name_str); else self_obj.class_set.delete(name_str); },
        contains: (name_str) => self_obj.class_set.has(name_str)};
    }
    appendChild(child_obj) {
      if (child_obj.parent) child_obj.parent.children = child_obj.parent.children.filter((item_obj) => item_obj !== child_obj);
      child_obj.parent = this;
      this.children.push(child_obj);
      return child_obj;
    }
    append(...child_list) { child_list.forEach((child_obj) => this.appendChild(child_obj)); }
    get textContent() { return this.own_text + this.children.map((child_obj) => child_obj.textContent).join(''); }
    set textContent(value_str) { this.own_text = String(value_str); this.children = []; }
    addEventListener(name_str, callback_fn) { (this.listener_dict[name_str] ||= []).push(callback_fn); }
    fire(name_str) {
      (this.listener_dict[name_str] || []).forEach((callback_fn) => callback_fn({type: name_str, target: this, clientX: 10, clientY: 10,
        key: '', preventDefault() {}}));
    }
    contains(node_obj) {
      let current_obj = node_obj;
      while (current_obj) { if (current_obj === this) return true; current_obj = current_obj.parent; }
      return false;
    }
    matches(selector_str) { return selector_str.split(',').some((part_str) => matches_simple(this, part_str.trim())); }
    closest(selector_str) {
      let current_obj = this;
      while (current_obj) { if (current_obj.matches(selector_str)) return current_obj; current_obj = current_obj.parent; }
      return null;
    }
    querySelectorAll(selector_str) {
      const out_list = [];
      const walk = (node_obj) => node_obj.children.forEach((child_obj) => { if (child_obj.matches(selector_str)) out_list.push(child_obj); walk(child_obj); });
      walk(this);
      return out_list;
    }
    querySelector(selector_str) { return this.querySelectorAll(selector_str)[0] || null; }
    focus() { document_obj.activeElement = this; this.fire('focus'); }
    getBoundingClientRect() { return {left: 0, top: 0, width: 900, height: 470}; }
    get offsetWidth() { return 200; }
    get offsetHeight() { return 80; }
  }
  const body_obj = new Node('body');
  function element(tag_str, attribute_dict = {}, parent_obj = body_obj) {
    const node_obj = new Node(tag_str);
    for (const [key_str, value_str] of Object.entries(attribute_dict)) node_obj.setAttribute(key_str, value_str);
    parent_obj.appendChild(node_obj);
    return node_obj;
  }
  const panel_obj = element('section', {'data-positions-map': '', 'data-map-scope': scope_str});
  const wrap_obj = element('div', {'data-map-wrap': ''}, panel_obj);
  const canvas_obj = element('div', {'data-map-canvas': '', 'data-map-json': json_str === null ? JSON.stringify(map_obj) : json_str}, wrap_obj);
  canvas_obj.clientWidth = 900;
  const tip_obj = element('div', {'data-map-tip': ''}, wrap_obj);
  tip_obj.hidden = true;
  const legend_obj = element('span', {'data-map-size-legend': '', 'data-value-legend': 'Size = value', 'data-exposure-legend': 'Size = exposure'}, panel_obj);
  const value_button_obj = element('button', {'data-map-size': 'value', 'aria-pressed': 'true'}, panel_obj);
  const exposure_button_obj = element('button', {'data-map-size': 'exposure', 'aria-pressed': 'false'}, panel_obj);
  const table_obj = element('table', {'data-sortable-table': 'positions-now'});
  const head_obj = element('th', {'aria-sort': 'none'}, element('tr', {}, element('thead', {}, table_obj)));
  element('button', {'data-sort-key': 'value', 'data-sort-type': 'number'}, head_obj);
  const body_obj_table = element('tbody', {}, table_obj);
  const row_list = ['AAA|p1', 'BBB|p1'].map((key_str, index_int) => element('tr', {'data-position-row': '', 'data-position-key': key_str,
    'data-position-symbol': key_str.split('|')[0], 'data-sort-row': '', 'data-sort-index': String(index_int),
    'data-sort-value': String(10 + 20 * index_int)}, body_obj_table));
  Object.assign(document_obj, {body: body_obj,
    createElementNS: (_, tag_str) => new Node(tag_str), createElement: (tag_str) => new Node(tag_str),
    querySelector: (selector_str) => body_obj.querySelector(selector_str), querySelectorAll: (selector_str) => body_obj.querySelectorAll(selector_str),
    getElementById: () => null, addEventListener: (name_str, callback_fn) => { (listener_dict[name_str] ||= []).push(callback_fn); }});
  const window_obj = {sessionStorage: {getItem: (key_str) => storage_dict[key_str] ?? null, setItem: (key_str, value_str) => { storage_dict[key_str] = value_str; }},
    addEventListener() {}, requestAnimationFrame: (callback_fn) => { callback_fn(); return 1; }};
  let api_obj = null;
  vm.runInNewContext(source_str, {document: document_obj, window: window_obj, positions_test_hook: (hook_obj) => { api_obj = hook_obj; }});
  const click = (target_obj) => (listener_dict.click || []).forEach((callback_fn) => callback_fn({target: target_obj}));
  const tile = (key_str) => body_obj.querySelectorAll('[data-map-key]').find((tile_obj) => tile_obj.getAttribute('data-map-key') === key_str);
  return {api_obj, document_obj, panel_obj, canvas_obj, tip_obj, legend_obj, value_button_obj, exposure_button_obj, row_list, storage_dict, click, tile,
    head_obj, body_obj_table};
}

const MINUS = '−';
const MAP_OBJ = {available_bool: true, mode_str: 'month', book_value_float: 3000, pod_list: [{pod_id_str: 'p1', name_str: 'Pod One', color_str: '#123456',
  nav_float: 3000, nav_str: '$3,000 avg', cash_float: 500, cash_str: '$500', pnl_float: 40, pnl_str: '+$40', return_str: '+1.3%',
  item_list: [
    {key_str: 'AAA|p1', symbol_str: 'AAA', pod_name_str: 'Pod One', value_float: 1500, multiplier_float: 3, pnl_float: 60, return_float: 0.04,
      pnl_str: '+$60', return_str: '+4.0%', status_str: 'kept', status_note_str: '', tip_list: [['Pod', 'Pod One']]},
    {key_str: 'BBB|p1', symbol_str: 'BBB', pod_name_str: 'Pod One', value_float: 1000, multiplier_float: 1, pnl_float: -20, return_float: -0.02,
      pnl_str: MINUS + '$20', return_str: MINUS + '2.0%', status_str: 'sold', status_note_str: 'sold 08-12', tip_list: [['Pod', 'Pod One']]}]}]};

test('the map draws tiles, cash and hatched sold tiles from the data attribute', () => {
  const env_obj = dom_env(MAP_OBJ);
  assert.ok(env_obj.tile('AAA|p1') && env_obj.tile('BBB|p1'));
  assert.ok(env_obj.tile('BBB|p1').querySelector('.pm-hatch'), 'sold tile is hatched');
  assert.equal(env_obj.tile('AAA|p1').querySelector('.pm-hatch'), null);
  assert.ok(env_obj.canvas_obj.textContent.includes('Cash'));
  assert.ok(env_obj.tile('AAA|p1').class_set.has('tm-p2') && env_obj.tile('BBB|p1').class_set.has('tm-n1'));
  assert.equal(env_obj.tile('AAA|p1').getAttribute('tabindex'), '0');
});

test('bad map data says so instead of leaving a blank map', () => {
  const env_obj = dom_env(MAP_OBJ, {json_str: '{"available_bool": NaN'});
  assert.equal(env_obj.canvas_obj.textContent, 'The map could not be drawn.');
});

test('selection highlights tile and row, toggles off, and survives a same-scope refresh only', () => {
  const env_obj = dom_env(MAP_OBJ);
  env_obj.api_obj.select_key('AAA|p1', false);
  assert.ok(env_obj.tile('AAA|p1').class_set.has('sel'));
  assert.deepEqual(env_obj.row_list.map((row_obj) => row_obj.class_set.has('hl')), [true, false]);
  env_obj.api_obj.refresh_page();
  assert.ok(env_obj.tile('AAA|p1').class_set.has('sel'), 'same scope keeps the selection');
  env_obj.panel_obj.setAttribute('data-map-scope', 'month:2026-07:all');
  env_obj.api_obj.refresh_page();
  assert.equal(env_obj.api_obj.get_state().selected_key_str, '');
  env_obj.api_obj.select_key('BBB|p1', false);
  env_obj.api_obj.select_key('BBB|p1', false);
  assert.equal(env_obj.api_obj.get_state().selected_key_str, '', 'a second click clears it');
  assert.deepEqual(env_obj.row_list.map((row_obj) => row_obj.class_set.has('hl')), [false, false]);
});

test('the size buttons switch to exposure, remember it and drop cash', () => {
  const env_obj = dom_env(MAP_OBJ);
  env_obj.click(env_obj.exposure_button_obj);
  assert.equal(env_obj.storage_dict['v4-positions-map-size'], 'exposure');
  assert.equal(env_obj.exposure_button_obj.getAttribute('aria-pressed'), 'true');
  assert.equal(env_obj.value_button_obj.getAttribute('aria-pressed'), 'false');
  assert.equal(env_obj.legend_obj.textContent, 'Size = exposure');
  assert.ok(!env_obj.canvas_obj.textContent.includes('Cash'));
  assert.match(env_obj.canvas_obj.textContent, /exp \$5,500 · 1\.83x NAV/);
});

test('a click anywhere in a header cell sorts, and a scrolling tile click closes the facts', () => {
  const env_obj = dom_env(MAP_OBJ);
  env_obj.click(env_obj.head_obj);
  assert.equal(env_obj.head_obj.getAttribute('aria-sort'), 'descending');
  assert.deepEqual(env_obj.body_obj_table.children.map((row_obj) => row_obj.getAttribute('data-sort-value')), ['30', '10']);
  env_obj.tip_obj.hidden = false;
  env_obj.api_obj.select_key('AAA|p1', true);
  assert.equal(env_obj.tip_obj.hidden, true);
});

test('keyboard focus stays on the same holding across a redraw', () => {
  const env_obj = dom_env(MAP_OBJ);
  env_obj.tile('BBB|p1').focus();
  env_obj.api_obj.draw_map(env_obj.panel_obj);
  assert.equal(env_obj.document_obj.activeElement.getAttribute('data-map-key'), 'BBB|p1');
  assert.equal(env_obj.tip_obj.hidden, false, 'focus shows the facts');
});
