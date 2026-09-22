const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source_str = fs.readFileSync(path.join(__dirname, '../alpha/live/dashboard_v4/static/donut_interaction.js'), 'utf8');

function environment_obj() {
  const handlers_dict = {};
  const document_obj = {activeElement: null, getElementById(id_str) { return shell_obj.id === id_str ? shell_obj : null; },
    addEventListener(name_str, callback_fn) { (handlers_dict[name_str] ||= []).push(callback_fn); }};
  function emit(name_str, target_obj, relatedTarget = null, detail = {}) {
    for (const callback_fn of handlers_dict[name_str] || []) callback_fn({type: name_str, target: target_obj, relatedTarget, detail});
  }
  function element_obj(attributes_dict = {}, parent_obj = null) {
    const classes_set = new Set();
    const result_obj = {children_list: [], textContent: '', id: attributes_dict.id || '', focused_int: 0,
      getAttribute(key_str) { return attributes_dict[key_str] ?? null; },
      setAttribute(key_str, value_str) { attributes_dict[key_str] = value_str; },
      matches(selector_str) { return Object.hasOwn(attributes_dict, selector_str.slice(1, -1)); },
      closest(selector_str) { return this.matches(selector_str) ? this : parent_obj?.closest(selector_str) || null; },
      contains(node_obj) { return node_obj === this || this.children_list.some(child_obj => child_obj.contains(node_obj)); },
      querySelectorAll(selector_str) { return this.children_list.flatMap(child_obj => [
        ...(child_obj.matches(selector_str) ? [child_obj] : []), ...child_obj.querySelectorAll(selector_str)]); },
      querySelector(selector_str) { return this.querySelectorAll(selector_str)[0] || null; },
      classList: {toggle(name_str, enabled_bool) { enabled_bool ? classes_set.add(name_str) : classes_set.delete(name_str); }, contains(name_str) { return classes_set.has(name_str); }},
      focus(options_obj) { const previous_obj = document_obj.activeElement; document_obj.activeElement = this;
        this.focused_int += 1; this.focus_options_obj = options_obj;
        if (previous_obj && previous_obj !== this) emit('focusout', previous_obj, this);
        emit('focusin', this, previous_obj); }};
    if (parent_obj) parent_obj.children_list.push(result_obj);
    return result_obj;
  }
  document_obj.body = element_obj();
  let shell_obj;
  function replace_shell() {
    shell_obj = element_obj({id: 'overview-shell', 'data-selection-scope': 'overview:All'}, document_obj.body);
    document_obj.activeElement = document_obj.body;
    return shell_obj;
  }
  replace_shell();
  function donut_obj(pod_bool) {
    const panel_obj = element_obj(pod_bool ? {'data-pod-allocation': ''} : {}, shell_obj);
    const inspector_obj = element_obj({'data-donut-inspector': '',
      ...(!pod_bool ? {'data-donut-id': 'overview-allocation', 'data-donut-date': '2026-09-18'} : {})}, panel_obj);
    const readout_obj = element_obj({'data-donut-value': ''}, inspector_obj);
    const slice_list = ['position:ABC', 'cash'].map((key_str, index_int) => element_obj({
      'data-donut-slice': '', 'data-donut-readout': ['ABC · 80.0% · $1,000.00', 'Cash · 20.0% · $250.00'][index_int],
      ...(pod_bool ? {'data-allocation-key': key_str} : {'data-donut-key': key_str})}, inspector_obj));
    const row_list = pod_bool ? ['position:ABC', 'cash', 'position:ZERO'].map(key_str => element_obj({'data-allocation-key': key_str}, panel_obj)) : [];
    return {panel_obj, inspector_obj, readout_obj, slice_list, row_list};
  }
  vm.runInNewContext(source_str, {document: document_obj});
  return {emit, element_obj, donut_obj, document_obj, replace_shell, get shell_obj() { return shell_obj; }, outside_obj: element_obj()};
}

for (const event_str of ['pointerover', 'focusin', 'click']) test(`${event_str} shows exact slice text including cash`, () => {
  const env_obj = environment_obj();
  const donut_obj = env_obj.donut_obj(false);
  for (const slice_obj of donut_obj.slice_list) {
    env_obj.emit(event_str, slice_obj);
    assert.equal(donut_obj.readout_obj.textContent, slice_obj.getAttribute('data-donut-readout'));
    assert.equal(slice_obj.classList.contains('is-donut-active'), true);
    assert.equal(donut_obj.slice_list.filter(item_obj => item_obj.classList.contains('is-donut-active')).length, 1);
  }
});

test('pointer leaving without focused selection clears the readout and stroke', () => {
  const env_obj = environment_obj();
  const donut_obj = env_obj.donut_obj(false);
  env_obj.emit('pointerover', donut_obj.slice_list[0]);
  env_obj.emit('pointerout', donut_obj.slice_list[0], env_obj.outside_obj);
  assert.equal(donut_obj.readout_obj.textContent, '');
  assert.equal(donut_obj.slice_list.some(item_obj => item_obj.classList.contains('is-donut-active')), false);
});

test('tap retains the readout through pointer exit until focus leaves', () => {
  const env_obj = environment_obj();
  const donut_obj = env_obj.donut_obj(false);
  env_obj.emit('click', donut_obj.slice_list[1]);
  assert.equal(env_obj.document_obj.activeElement, donut_obj.slice_list[1]);
  env_obj.emit('pointerout', donut_obj.slice_list[1], env_obj.outside_obj);
  assert.equal(donut_obj.readout_obj.textContent, 'Cash · 20.0% · $250.00');
  env_obj.outside_obj.focus();
  assert.equal(donut_obj.readout_obj.textContent, '');
});

test('hover temporarily overrides keyboard selection and restores it on leave', () => {
  const env_obj = environment_obj();
  const donut_obj = env_obj.donut_obj(true);
  donut_obj.row_list[0].focus();
  env_obj.emit('pointerover', donut_obj.slice_list[1]);
  assert.equal(donut_obj.readout_obj.textContent, 'Cash · 20.0% · $250.00');
  env_obj.emit('pointerout', donut_obj.slice_list[1], env_obj.outside_obj);
  assert.equal(donut_obj.readout_obj.textContent, 'ABC · 80.0% · $1,000.00');
});

for (const event_str of ['pointerover', 'focusin', 'click']) test(`Pod row ${event_str} links only its own matching slice and preserves existing highlight class`, () => {
  const env_obj = environment_obj();
  const first_obj = env_obj.donut_obj(true), second_obj = env_obj.donut_obj(true);
  first_obj.row_list[1].classList.toggle('is-highlighted', true);
  env_obj.emit(event_str, first_obj.row_list[1]);
  assert.equal(first_obj.readout_obj.textContent, 'Cash · 20.0% · $250.00');
  assert.equal(second_obj.readout_obj.textContent, '');
  assert.equal(first_obj.row_list[1].classList.contains('is-highlighted'), true);
});

test('missing or ambiguous row-to-slice matches do not borrow another value', () => {
  const env_obj = environment_obj();
  const donut_obj = env_obj.donut_obj(true);
  env_obj.emit('pointerover', donut_obj.row_list[2]);
  assert.equal(donut_obj.readout_obj.textContent, '');
  donut_obj.slice_list[0].setAttribute('data-allocation-key', 'cash');
  env_obj.emit('pointerover', donut_obj.row_list[1]);
  assert.equal(donut_obj.readout_obj.textContent, '');
});

test('crossing children or another slice uses the correct current value', () => {
  const env_obj = environment_obj();
  const donut_obj = env_obj.donut_obj(false);
  const child_obj = env_obj.element_obj({}, donut_obj.slice_list[0]);
  env_obj.emit('pointerover', child_obj);
  env_obj.emit('pointerout', donut_obj.slice_list[0], child_obj);
  assert.equal(donut_obj.readout_obj.textContent, 'ABC · 80.0% · $1,000.00');
  env_obj.emit('pointerout', child_obj, donut_obj.slice_list[1]);
  assert.equal(donut_obj.readout_obj.textContent, 'Cash · 20.0% · $250.00');
});

test('delegated events use refreshed slice values and cannot carry values to another panel', () => {
  const env_obj = environment_obj();
  const old_obj = env_obj.donut_obj(true);
  old_obj.row_list[0].focus();
  const refreshed_obj = env_obj.donut_obj(true);
  refreshed_obj.slice_list[0].setAttribute('data-donut-readout', 'ABC · 75.0% · $900.00');
  refreshed_obj.row_list[0].focus();
  assert.equal(refreshed_obj.readout_obj.textContent, 'ABC · 75.0% · $900.00');
  assert.equal(old_obj.readout_obj.textContent, '');
});

test('labels are assigned as text and empty metadata never invents a value', () => {
  const env_obj = environment_obj();
  const donut_obj = env_obj.donut_obj(false);
  donut_obj.slice_list[0].setAttribute('data-donut-readout', '<img src=x> · 0.0%');
  env_obj.emit('pointerover', donut_obj.slice_list[0]);
  assert.equal(donut_obj.readout_obj.textContent, '<img src=x> · 0.0%');
  donut_obj.slice_list[0].setAttribute('data-donut-readout', '');
  env_obj.emit('focusin', donut_obj.slice_list[0]);
  assert.equal(donut_obj.readout_obj.textContent, '');
});

test('Overview tap and keyboard selection survive a matching full refresh without scrolling', () => {
  const env_obj = environment_obj();
  const old_obj = env_obj.donut_obj(false);
  env_obj.emit('click', old_obj.slice_list[1]);
  env_obj.emit('htmx:beforeSwap', env_obj.shell_obj, null, {target: env_obj.shell_obj});
  env_obj.replace_shell();
  const refreshed_obj = env_obj.donut_obj(false);
  env_obj.emit('htmx:afterSwap', env_obj.shell_obj, null, {target: env_obj.shell_obj});
  assert.equal(env_obj.document_obj.activeElement, refreshed_obj.slice_list[1]);
  assert.equal(refreshed_obj.slice_list[1].focus_options_obj.preventScroll, true);
  assert.equal(refreshed_obj.readout_obj.textContent, 'Cash · 20.0% · $250.00');
  assert.equal(refreshed_obj.slice_list[1].classList.contains('is-donut-active'), true);
  env_obj.emit('htmx:afterSwap', env_obj.shell_obj, null, {target: env_obj.shell_obj});
  assert.equal(refreshed_obj.slice_list[1].focused_int, 1);
});

for (const change_str of ['scope', 'date', 'missing date', 'key', 'duplicate slice', 'duplicate inspector',
  'initial duplicate', 'initial missing date', 'stale', 'error', 'HTTP failure', 'cancelled', 'transport failure',
  'moved focus', 'cleared focus', 'hover only', 'Pod']) {
  test(`donut refresh does not restore a selection after ${change_str}`, () => {
    const env_obj = environment_obj();
    const old_obj = env_obj.donut_obj(change_str === 'Pod');
    if (change_str === 'hover only') env_obj.emit('pointerover', old_obj.slice_list[1]);
    else old_obj.slice_list[1].focus();
    if (change_str === 'initial duplicate') old_obj.slice_list[0].setAttribute('data-donut-key', 'cash');
    if (change_str === 'initial missing date') old_obj.inspector_obj.setAttribute('data-donut-date', '');
    if (change_str === 'cleared focus') env_obj.outside_obj.focus();
    env_obj.emit('htmx:beforeSwap', env_obj.shell_obj, null, {target: env_obj.shell_obj, shouldSwap: change_str !== 'cancelled'});
    if (change_str === 'transport failure') env_obj.emit('htmx:sendError', env_obj.shell_obj);
    env_obj.replace_shell();
    const refreshed_obj = env_obj.donut_obj(change_str === 'Pod');
    if (change_str === 'scope') env_obj.shell_obj.setAttribute('data-selection-scope', 'overview:3M');
    if (change_str === 'date') refreshed_obj.inspector_obj.setAttribute('data-donut-date', '2026-09-21');
    if (change_str === 'missing date') refreshed_obj.inspector_obj.setAttribute('data-donut-date', '');
    if (change_str === 'key') refreshed_obj.slice_list[1].setAttribute('data-donut-key', 'another slice');
    if (change_str === 'duplicate slice') refreshed_obj.slice_list[0].setAttribute('data-donut-key', 'cash');
    if (change_str === 'duplicate inspector') env_obj.donut_obj(false);
    if (change_str === 'stale') env_obj.shell_obj.setAttribute('data-source-stale', 'true');
    if (change_str === 'moved focus') env_obj.outside_obj.focus();
    env_obj.emit('htmx:afterSwap', env_obj.shell_obj, null, {target: env_obj.shell_obj,
      isError: change_str === 'error', xhr: {status: change_str === 'HTTP failure' ? 503 : 200}});
    assert.equal(refreshed_obj.readout_obj.textContent, '');
    assert.equal(refreshed_obj.slice_list[1].focused_int, 0);
    assert.equal(env_obj.document_obj.activeElement, change_str === 'moved focus' ? env_obj.outside_obj : env_obj.document_obj.body);
  });
}

test('status-only refresh cannot replace or steal a focused Overview slice', () => {
  const env_obj = environment_obj();
  const donut_obj = env_obj.donut_obj(false);
  donut_obj.slice_list[0].focus();
  const status_obj = env_obj.element_obj({id: 'performance-status'});
  env_obj.emit('htmx:beforeSwap', status_obj, null, {target: status_obj, shouldSwap: false});
  env_obj.emit('htmx:afterSwap', status_obj, null, {target: status_obj, shouldSwap: false});
  assert.equal(env_obj.document_obj.activeElement, donut_obj.slice_list[0]);
  assert.equal(donut_obj.slice_list[0].focused_int, 1);
});
