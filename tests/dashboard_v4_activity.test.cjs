/* Behavior checks for the browser-only Activity visit and refresh state. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source_str = fs.readFileSync(path.join(__dirname, '../alpha/live/dashboard_v4/static/activity.js'), 'utf8');
const KEY_STR = 'dashboard_v4.activity.demo.live';
const BASELINE_STR = '2026-09-21T13:00:00Z';
const ASOF_STR = '2026-09-21T14:00:00Z';
const DEFAULT_ROWS = [
  {id: 'fail', time: '2026-09-21T13:50:00Z', type: 'alerts', state: 'fail', pod: 'a', text: 'Broker ACK missing'},
  {id: 'late', time: '2026-09-21T13:40:00Z', type: 'cycles', state: 'late', pod: 'b', text: 'Fills late'},
  {id: 'cycle', time: '2026-09-21T13:30:00Z', type: 'cycles', state: 'done', pod: 'a', text: 'Open cycle complete'},
  {id: 'step', parent: 'cycle', time: '2026-09-21T13:29:00Z', type: 'cycles', state: 'done', pod: 'a', text: 'Reconciled'},
  {id: 'old', time: '2026-09-20T12:00:00Z', type: 'system', state: 'fail', pod: '', text: 'Old system failure'},
];

function environment_obj(options_dict = {}) {
  const handlers_dict = {};
  const store_map = new Map(options_dict.saved === undefined ? [[KEY_STR, BASELINE_STR]] : options_dict.saved === null ? [] : [[KEY_STR, options_dict.saved]]);
  const write_list = [];
  const document_obj = {hidden: Boolean(options_dict.hidden), activeElement: null,
    addEventListener(name_str, callback_fn) { (handlers_dict[name_str] ||= []).push(callback_fn); }};
  function element_obj(tag_str, attribute_dict = {}, text_str = '') {
    const class_set = new Set();
    let content_str = text_str;
    const result_obj = {tagName: tag_str.toUpperCase(), id: attribute_dict.id || '', parentElement: null, children_list: [], hidden: false, value: '',
      getAttribute(key_str) { return attribute_dict[key_str] ?? null; }, hasAttribute(key_str) { return Object.hasOwn(attribute_dict, key_str); },
      setAttribute(key_str, value_str) { attribute_dict[key_str] = value_str; },
      matches(selector_str) { return selector_str.startsWith('#') ? this.id === selector_str.slice(1) : selector_str.startsWith('[') ? this.hasAttribute(selector_str.slice(1, -1)) : selector_str.toUpperCase() === this.tagName; },
      closest(selector_str) { return this.matches(selector_str) ? this : this.parentElement?.closest(selector_str) || null; },
      contains(node_obj) { return node_obj === this || this.children_list.some(child_obj => child_obj.contains(node_obj)); },
      querySelectorAll(selector_str) { return this.children_list.flatMap(child_obj => [...(child_obj.matches(selector_str) ? [child_obj] : []), ...child_obj.querySelectorAll(selector_str)]); },
      querySelector(selector_str) { return this.querySelectorAll(selector_str)[0] || null; },
      append(child_obj) { child_obj.parentElement = this; this.children_list.push(child_obj); return child_obj; },
      insertBefore(child_obj, before_obj) {
        if (child_obj.parentElement) child_obj.parentElement.children_list = child_obj.parentElement.children_list.filter(item_obj => item_obj !== child_obj);
        child_obj.parentElement = this;
        const index_int = before_obj ? this.children_list.indexOf(before_obj) : this.children_list.length;
        this.children_list.splice(index_int, 0, child_obj);
      },
      focus() { document_obj.activeElement = this; },
      setSelectionRange(start_int, end_int) { this.selectionStart = start_int; this.selectionEnd = end_int; },
      classList: {toggle(name_str, selected_bool) { selected_bool ? class_set.add(name_str) : class_set.delete(name_str); }, contains(name_str) { return class_set.has(name_str); }},
    };
    Object.defineProperty(result_obj, 'textContent', {get() { return content_str + this.children_list.map(child_obj => child_obj.textContent).join(' '); },
      set(value_str) { content_str = value_str; this.children_list = []; }});
    return result_obj;
  }
  document_obj.body = element_obj('body');
  document_obj.activeElement = document_obj.body;
  document_obj.querySelector = selector_str => document_obj.body.querySelector(selector_str);
  let shell_obj, page_obj, input_obj, pod_obj, verdict_obj, body_obj, marker_obj;
  let row_map, proof_map, cycle_button_map, evidence_button_map, type_map;
  function build(rows_list = DEFAULT_ROWS, asof_str = ASOF_STR, complete_bool = true, key_str = KEY_STR, feed_available_bool = complete_bool) {
    document_obj.body.children_list = [];
    shell_obj = document_obj.body.append(element_obj('div', {id: 'overview-shell'}));
    page_obj = shell_obj.append(element_obj('section', {'data-activity-page': '', 'data-as-of': asof_str, 'data-storage-key': key_str, 'data-complete': String(complete_bool), 'data-feed-available': String(feed_available_bool)}));
    verdict_obj = page_obj.append(element_obj('b', {'data-activity-verdict': ''}));
    page_obj.append(element_obj('span', {'data-activity-verdict-detail': ''}));
    pod_obj = page_obj.append(element_obj('select', {'data-activity-pod': ''}));
    input_obj = page_obj.append(element_obj('input', {'data-activity-search': ''}));
    type_map = new Map(['all', 'cycles', 'alerts', 'operator', 'system'].map(type_str => [type_str, page_obj.append(element_obj('button', {'data-activity-type': type_str}))]));
    page_obj.append(element_obj('button', {'data-activity-late': ''}));
    page_obj.append(element_obj('button', {'data-activity-codes': ''}));
    body_obj = page_obj.append(element_obj('tbody', {'data-activity-body': ''}));
    row_map = new Map(); proof_map = new Map(); cycle_button_map = new Map(); evidence_button_map = new Map();
    let day_str = '';
    for (const row_dict of rows_list) {
      const next_day_str = row_dict.time.slice(0, 10);
      if (!row_dict.parent && day_str !== next_day_str) {
        body_obj.append(element_obj('tr', {'data-activity-day': next_day_str})); day_str = next_day_str;
      }
      const row_obj = body_obj.append(element_obj('tr', {'data-activity-row': row_dict.id, 'data-activity-parent': row_dict.parent || '',
        'data-timestamp': row_dict.time, 'data-day': next_day_str, 'data-pod': row_dict.pod, 'data-related-pods': JSON.stringify(row_dict.related || []), 'data-type': row_dict.type, 'data-state': row_dict.state}, row_dict.text));
      const code_obj = row_obj.append(element_obj('code', {'data-activity-code': ''}, 'RAW_' + row_dict.id)); code_obj.hidden = true;
      row_map.set(row_dict.id, row_obj);
      if (rows_list.some(child_dict => child_dict.parent === row_dict.id)) cycle_button_map.set(row_dict.id, row_obj.append(element_obj('button', {'data-activity-cycle': row_dict.id})));
      evidence_button_map.set(row_dict.id, row_obj.append(element_obj('button', {'data-activity-evidence': row_dict.id})));
      const proof_obj = body_obj.append(element_obj('tr', {'data-activity-proof': row_dict.id}, 'Evidence ' + row_dict.id));
      proof_obj.hidden = true; proof_map.set(row_dict.id, proof_obj);
    }
    marker_obj = body_obj.append(element_obj('tr', {'data-activity-marker': ''}));
    marker_obj.append(element_obj('span', {'data-activity-marker-label': ''}));
    const empty_obj = body_obj.append(element_obj('tr', {'data-activity-empty': ''})); empty_obj.append(element_obj('td'));
  }
  function emit(name_str, target_obj = page_obj, detail_dict = {}) {
    for (const callback_fn of handlers_dict[name_str] || []) callback_fn({target: target_obj, detail: {target: shell_obj, ...detail_dict}});
  }
  build(options_dict.rows || DEFAULT_ROWS, options_dict.asof || ASOF_STR, options_dict.complete !== false, KEY_STR, options_dict.feed_available ?? options_dict.complete !== false);
  vm.runInNewContext(source_str, {document: document_obj, window: {localStorage: {
    getItem(key_str) { if (options_dict.blocked) throw new Error('storage blocked'); return store_map.get(key_str) || null; },
    setItem(key_str, value_str) { if (options_dict.blocked) throw new Error('storage blocked'); store_map.set(key_str, value_str); write_list.push([key_str, value_str]); },
  }}});
  return {emit, document_obj, store_map, write_list,
    get page_obj() { return page_obj; }, get shell_obj() { return shell_obj; }, get input_obj() { return input_obj; }, get pod_obj() { return pod_obj; }, get verdict_str() { return verdict_obj.textContent; },
    get marker_obj() { return marker_obj; }, get body_obj() { return body_obj; }, get row_map() { return row_map; }, get proof_map() { return proof_map; },
    click_type(type_str) { emit('click', type_map.get(type_str)); },
    click(attribute_str) { emit('click', page_obj.querySelector(`[${attribute_str}]`)); },
    expand(id_str) { emit('click', cycle_button_map.get(id_str)); }, evidence(id_str) { emit('click', evidence_button_map.get(id_str)); },
    search(value_str) { input_obj.value = value_str; emit('input', input_obj); },
    filter_pod(value_str) { pod_obj.value = value_str; emit('change', pod_obj); },
    swap(rows_list = DEFAULT_ROWS, asof_str = ASOF_STR, complete_bool = true, key_str = KEY_STR) {
      emit('htmx:beforeSwap'); build(rows_list, asof_str, complete_bool, key_str); document_obj.activeElement = document_obj.body;
      emit('htmx:afterSwap', page_obj, {xhr: {status: 200}});
    },
  };
}

test('counts only new top-level failures and late events and places the ET boundary', () => {
  const env_obj = environment_obj();
  assert.equal(env_obj.verdict_str, '1 failed, 1 late');
  assert.equal(env_obj.marker_obj.hidden, false);
  assert.match(env_obj.marker_obj.textContent, /You last looked here.*09:00:00 ET/);
  assert.equal(env_obj.body_obj.children_list.indexOf(env_obj.marker_obj) + 1, env_obj.body_obj.children_list.indexOf(env_obj.row_map.get('old')));
  assert.equal(env_obj.store_map.get(KEY_STR), '2026-09-21T14:00:00.000Z');
});

test('polling and Load older retain the original baseline while accumulating new events', () => {
  const env_obj = environment_obj();
  const incoming_dict = {id: 'new', time: '2026-09-21T14:10:00Z', type: 'alerts', state: 'fail', pod: 'a', text: 'New failure'};
  env_obj.swap([incoming_dict, ...DEFAULT_ROWS], '2026-09-21T14:20:00Z');
  assert.equal(env_obj.verdict_str, '2 failed, 1 late');
  env_obj.click_type('system'); assert.equal(env_obj.verdict_str, 'No new events.');
  env_obj.click_type('all'); assert.equal(env_obj.verdict_str, '2 failed, 1 late');
  env_obj.swap([incoming_dict, ...DEFAULT_ROWS, {id: 'older', time: '2026-09-10T12:00:00Z', type: 'alerts', state: 'fail', pod: 'a', text: 'Older failure'}], '2026-09-21T14:21:00Z');
  assert.equal(env_obj.verdict_str, '2 failed, 1 late');
  assert.match(env_obj.marker_obj.textContent, /09:00:00 ET/);
});

test('search text, focus and selection survive a shell refresh', () => {
  const env_obj = environment_obj();
  env_obj.search('ACK'); env_obj.input_obj.focus(); env_obj.input_obj.setSelectionRange(1, 3);
  env_obj.swap();
  assert.equal(env_obj.input_obj.value, 'ACK');
  assert.equal(env_obj.document_obj.activeElement, env_obj.input_obj);
  assert.deepEqual([env_obj.input_obj.selectionStart, env_obj.input_obj.selectionEnd], [1, 3]);
  assert.equal(env_obj.row_map.get('fail').hidden, false);
  assert.equal(env_obj.row_map.get('late').hidden, true);
});

test('pod/type/late/code filters persist across swaps and hide unused date groups', () => {
  const env_obj = environment_obj();
  env_obj.filter_pod('a'); env_obj.click_type('alerts'); env_obj.click('data-activity-late'); env_obj.click('data-activity-codes');
  env_obj.swap();
  assert.equal(env_obj.pod_obj.value, 'a');
  assert.equal(env_obj.verdict_str, '1 failed');
  assert.equal(env_obj.row_map.get('fail').hidden, false);
  assert.equal(env_obj.row_map.get('cycle').hidden, true);
  assert.equal(env_obj.row_map.get('fail').querySelector('[data-activity-code]').hidden, false);
  assert.equal(env_obj.page_obj.querySelectorAll('[data-activity-day]').find(item_obj => item_obj.getAttribute('data-activity-day') === '2026-09-20').hidden, true);
});

test('cycle steps and inline evidence retain their open state through refresh and filtering', () => {
  const env_obj = environment_obj();
  assert.equal(env_obj.row_map.get('step').hidden, true);
  env_obj.expand('cycle'); env_obj.evidence('step'); env_obj.swap();
  assert.equal(env_obj.row_map.get('step').hidden, false);
  assert.equal(env_obj.proof_map.get('step').hidden, false);
  env_obj.click_type('alerts'); assert.equal(env_obj.proof_map.get('step').hidden, true);
  env_obj.click_type('all'); assert.equal(env_obj.proof_map.get('step').hidden, false);
  env_obj.expand('cycle'); assert.equal(env_obj.proof_map.get('step').hidden, true);
});

test('search finds evidence and nested steps without inventing extra top-level counts', () => {
  const env_obj = environment_obj();
  env_obj.search('Evidence step');
  assert.equal(env_obj.row_map.get('cycle').hidden, false);
  assert.equal(env_obj.row_map.get('fail').hidden, true);
  assert.equal(env_obj.verdict_str, '1 new event.');
});

for (const saved_str of [null, 'broken timestamp', '2026-09-22T14:00:00Z']) test(`first/invalid/future stored visit is not an invented historical alert count: ${saved_str}`, () => {
  const env_obj = environment_obj({saved: saved_str});
  assert.equal(env_obj.verdict_str, 'Recent activity.');
  assert.equal(env_obj.marker_obj.hidden, true);
  assert.equal(env_obj.store_map.get(KEY_STR), '2026-09-21T14:00:00.000Z');
  env_obj.swap([{id: 'incoming', time: '2026-09-21T14:10:00Z', type: 'alerts', state: 'fail', pod: 'a', text: 'Incoming'}, ...DEFAULT_ROWS], '2026-09-21T14:20:00Z');
  assert.equal(env_obj.verdict_str, '1 failed');
});

test('storage failures do not break filtering or the current visit baseline', () => {
  const env_obj = environment_obj({blocked: true});
  env_obj.search('ACK'); env_obj.swap();
  assert.equal(env_obj.input_obj.value, 'ACK');
  assert.equal(env_obj.row_map.get('fail').hidden, false);
  assert.equal(env_obj.write_list.length, 0);
});

test('hidden pages, incomplete responses and failed transport never advance last looked', () => {
  const hidden_obj = environment_obj({hidden: true});
  assert.equal(hidden_obj.write_list.length, 0);
  hidden_obj.document_obj.hidden = false; hidden_obj.emit('visibilitychange');
  assert.equal(hidden_obj.write_list.length, 1);
  const incomplete_obj = environment_obj({complete: false});
  assert.equal(incomplete_obj.write_list.length, 0);
  incomplete_obj.click_type('operator'); assert.equal(incomplete_obj.verdict_str, 'Activity may be incomplete.');
  const failed_obj = environment_obj();
  const write_count_int = failed_obj.write_list.length;
  failed_obj.emit('htmx:responseError'); failed_obj.emit('htmx:timeout');
  assert.equal(failed_obj.write_list.length, write_count_int);
  assert.equal(failed_obj.verdict_str, '1 failed, 1 late');
});

test('a previously hidden page cannot save its observation after shell freshness fails', () => {
  const env_obj = environment_obj({hidden: true});
  env_obj.shell_obj.setAttribute('data-source-stale', 'true');
  env_obj.document_obj.hidden = false;
  env_obj.emit('visibilitychange');
  assert.equal(env_obj.write_list.length, 0);
});

test('a bounded but available current feed can advance the marker without claiming completeness', () => {
  const env_obj = environment_obj({complete: false, feed_available: true});
  assert.equal(env_obj.write_list.length, 1);
  env_obj.click_type('operator');
  assert.equal(env_obj.verdict_str, 'Activity may be incomplete.');
});

test('a scoped system event remains reachable through each related pod filter', () => {
  const env_obj = environment_obj({rows: [{id: 'shared', time: '2026-09-21T13:50:00Z', type: 'system', state: 'fail', pod: '', related: ['a', 'b'], text: 'Shared scheduler error'}]});
  env_obj.filter_pod('a'); assert.equal(env_obj.row_map.get('shared').hidden, false);
  env_obj.filter_pod('b'); assert.equal(env_obj.row_map.get('shared').hidden, false);
  env_obj.filter_pod('c'); assert.equal(env_obj.row_map.get('shared').hidden, true);
});

test('a slower tab cannot regress the saved observation and another client gets its own baseline', () => {
  const env_obj = environment_obj();
  env_obj.store_map.set(KEY_STR, '2026-09-21T16:00:00.000Z');
  env_obj.swap(DEFAULT_ROWS, '2026-09-21T14:10:00Z');
  assert.equal(env_obj.store_map.get(KEY_STR), '2026-09-21T16:00:00.000Z');
  env_obj.search('ACK'); env_obj.swap(DEFAULT_ROWS, ASOF_STR, true, 'different-client.live');
  assert.equal(env_obj.verdict_str, 'Recent activity.');
  assert.equal(env_obj.input_obj.value, '');
  assert.equal(env_obj.store_map.get('different-client.live'), '2026-09-21T14:00:00.000Z');
});
