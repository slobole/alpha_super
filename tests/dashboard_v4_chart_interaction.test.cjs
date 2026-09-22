const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source_str = fs.readFileSync(path.join(__dirname, '../alpha/live/dashboard_v4/static/chart_interaction.js'), 'utf8');

function environment_obj() {
  const handlers_dict = {};
  const document_obj = {activeElement: null, addEventListener(name_str, callback_fn) { (handlers_dict[name_str] ||= []).push(callback_fn); }};
  function emit(name_str, target_obj, options_dict = {}) {
    const event_obj = {type: name_str, target: target_obj, relatedTarget: null, prevented_bool: false,
      preventDefault() { this.prevented_bool = true; }, ...options_dict};
    for (const callback_fn of handlers_dict[name_str] || []) callback_fn(event_obj);
    return event_obj;
  }
  function element_obj(attributes_dict = {}, parent_obj = null) {
    let tab_index_int = -1;
    const result_obj = {parent_obj, children_list: [], hidden: false, textContent: '', style: {}, focus_count_int: 0, write_count_int: 0,
      get tabIndex() { return tab_index_int; },
      set tabIndex(value_int) { this.write_count_int += 1; tab_index_int = value_int; },
      get id() { return attributes_dict.id || ''; },
      getAttribute(key_str) { return attributes_dict[key_str] ?? null; },
      setAttribute(key_str, value_str) { this.write_count_int += 1; attributes_dict[key_str] = String(value_str); },
      matches(selector_str) { return selector_str.startsWith('#') ? this.id === selector_str.slice(1) : Object.hasOwn(attributes_dict, selector_str.slice(1, -1)); },
      closest(selector_str) { return this.matches(selector_str) ? this : this.parent_obj?.closest(selector_str) || null; },
      contains(node_obj) { return node_obj === this || this.children_list.some(child_obj => child_obj.contains(node_obj)); },
      querySelectorAll(selector_str) { return this.children_list.flatMap(child_obj => [
        ...(selector_str.split(', ').some(item_str => child_obj.matches(item_str)) ? [child_obj] : []), ...child_obj.querySelectorAll(selector_str)]); },
      querySelector(selector_str) { return this.querySelectorAll(selector_str)[0] || null; },
      focus(options_dict) {
        this.focus_count_int += 1;
        this.focus_options_dict = options_dict;
        const previous_obj = document_obj.activeElement;
        if (previous_obj === this) return;
        if (previous_obj) emit('focusout', previous_obj, {relatedTarget: this});
        document_obj.activeElement = this;
        emit('focusin', this, {relatedTarget: previous_obj});
      }};
    if (parent_obj) parent_obj.children_list.push(result_obj);
    return result_obj;
  }
  document_obj.body = element_obj();
  document_obj.activeElement = document_obj.body;
  document_obj.getElementById = id_str => document_obj.body.querySelector('#' + id_str);
  function shell_obj(scope_str = 'overview:All') {
    return element_obj({id: 'overview-shell', 'data-selection-scope': scope_str}, document_obj.body);
  }
  function chart_obj(parent_obj, options_dict = {}) {
    const date_list = options_dict.date_list || ['2026-08-31', '2026-09-01', '2026-09-02'];
    const series_int = options_dict.series_int || 1;
    const root_obj = element_obj({'data-history-chart': '', 'data-chart-id': options_dict.id_str || 'Portfolio return', 'data-chart-default': date_list.at(-1)}, parent_obj);
    const readout_obj = element_obj({'data-chart-readout': ''}, root_obj);
    const date_obj = element_obj({'data-chart-date': ''}, readout_obj);
    const readout_list = Array.from({length: series_int}, (_, index_int) => {
      const row_obj = element_obj({'data-chart-value': '', 'data-series-index': String(index_int)}, readout_obj);
      const name_obj = element_obj({'data-chart-name': ''}, row_obj);
      const number_obj = element_obj({'data-chart-number': ''}, row_obj);
      const key_obj = element_obj({'data-chart-key': ''}, row_obj);
      return {row_obj, name_obj, number_obj, key_obj};
    });
    const plot_obj = element_obj({'data-chart-plot': ''}, root_obj);
    const day_list = date_list.map((date_str, index_int) => {
      const x_float = date_list.length > 1 ? index_int * 100 / (date_list.length - 1) : 50;
      const values_list = Array.from({length: series_int}, (_, series_index_int) => ({
        name_str: ['DVO2', 'QPI'][series_index_int], color_str: ['var(--blue)', '#b06a34'][series_index_int],
        label_str: ['0.00%', '-1.23%', '+2.00%'][index_int], x_percent_float: x_float,
        y_percent_float: [100, 75, 0][index_int], available_bool: true,
      }));
      const day_obj = element_obj({'data-chart-day': date_str, 'data-date-label': 'Date ' + date_str,
        'data-chart-x': String(x_float), 'data-chart-values': JSON.stringify(values_list)}, plot_obj);
      day_obj.tabIndex = index_int === date_list.length - 1 ? 0 : -1;
      return day_obj;
    });
    const marker_list = Array.from({length: series_int}, (_, index_int) => {
      const marker_obj = element_obj({'data-chart-marker': '', 'data-series-index': String(index_int)}, plot_obj);
      marker_obj.hidden = true;
      return marker_obj;
    });
    const crosshair_obj = element_obj({'data-chart-crosshair': ''}, plot_obj);
    crosshair_obj.hidden = true;
    const last_list = JSON.parse(day_list.at(-1).getAttribute('data-chart-values'));
    date_obj.textContent = day_list.at(-1).getAttribute('data-date-label');
    for (const [index_int, row_dict] of readout_list.entries()) {
      row_dict.name_obj.textContent = last_list[index_int].name_str;
      row_dict.number_obj.textContent = last_list[index_int].label_str;
      row_dict.key_obj.style.backgroundColor = last_list[index_int].color_str;
    }
    return {root_obj, readout_obj, date_obj, readout_list, plot_obj, day_list, marker_list, crosshair_obj};
  }
  function replace_shell(old_obj, options_dict = {}) {
    document_obj.body.children_list = document_obj.body.children_list.filter(item_obj => item_obj !== old_obj);
    const next_obj = shell_obj(options_dict.scope_str);
    if (options_dict.stale_bool) next_obj.setAttribute('data-source-stale', 'true');
    const next_chart_obj = chart_obj(next_obj, options_dict);
    if (!options_dict.keep_focus_bool) document_obj.activeElement = document_obj.body;
    return {shell_obj: next_obj, ...next_chart_obj};
  }
  vm.runInNewContext(source_str, {document: document_obj});
  return {emit, element_obj, document_obj, handlers_dict, shell_obj, chart_obj, replace_shell};
}

function fixture_obj(options_dict = {}) {
  const env_obj = environment_obj();
  const shell_obj = env_obj.shell_obj();
  return {env_obj, shell_obj, ...env_obj.chart_obj(shell_obj, options_dict)};
}

function set_value(day_obj, index_int, patch_dict) {
  const value_list = JSON.parse(day_obj.getAttribute('data-chart-values'));
  Object.assign(value_list[index_int], patch_dict);
  day_obj.setAttribute('data-chart-values', JSON.stringify(value_list));
}

test('initial load leaves the server default untouched with no selection marker', () => {
  const fixture_dict = fixture_obj();
  assert.equal(fixture_dict.date_obj.textContent, 'Date 2026-09-02');
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '+2.00%');
  assert.equal(fixture_dict.crosshair_obj.hidden, true);
  assert.equal(fixture_dict.marker_list[0].hidden, true);
  assert.deepEqual(fixture_dict.day_list.map(day_obj => day_obj.tabIndex), [-1, -1, 0]);
});

for (const event_str of ['pointerover', 'focusin', 'click']) test(`${event_str} shows exact zero value, date and boundary marker`, () => {
  const fixture_dict = fixture_obj();
  fixture_dict.env_obj.emit(event_str, fixture_dict.day_list[0]);
  assert.equal(fixture_dict.date_obj.textContent, 'Date 2026-08-31');
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '0.00%');
  assert.equal(fixture_dict.marker_list[0].hidden, false);
  assert.equal(fixture_dict.marker_list[0].style.left, '0%');
  assert.equal(fixture_dict.marker_list[0].style.top, '100%');
  assert.equal(fixture_dict.crosshair_obj.style.left, '0%');
  if (event_str === 'click') assert.equal(fixture_dict.env_obj.document_obj.activeElement, fixture_dict.day_list[0]);
});

test('whole-column child targets work and pointer exit restores the latest date without focus changes', () => {
  const fixture_dict = fixture_obj();
  const child_obj = fixture_dict.env_obj.element_obj({}, fixture_dict.day_list[1]);
  fixture_dict.env_obj.emit('pointerover', child_obj);
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '-1.23%');
  assert.equal(fixture_dict.marker_list[0].style.backgroundColor, 'var(--blue)');
  fixture_dict.env_obj.emit('pointerout', child_obj, {relatedTarget: fixture_dict.readout_obj});
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '+2.00%');
  assert.equal(fixture_dict.crosshair_obj.hidden, true);
  assert.equal(fixture_dict.marker_list[0].hidden, true);
  assert.equal(fixture_dict.env_obj.document_obj.activeElement, fixture_dict.env_obj.document_obj.body);
});

test('hovering another day does not replace the keyboard selection on pointer exit; blur restores default', () => {
  const fixture_dict = fixture_obj();
  fixture_dict.day_list[0].focus();
  fixture_dict.env_obj.emit('pointerover', fixture_dict.day_list[1]);
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '-1.23%');
  fixture_dict.env_obj.emit('pointerout', fixture_dict.day_list[1], {relatedTarget: fixture_dict.readout_obj});
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '0.00%');
  assert.equal(fixture_dict.marker_list[0].hidden, false);
  fixture_dict.env_obj.element_obj({}, fixture_dict.shell_obj).focus();
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '+2.00%');
  assert.equal(fixture_dict.marker_list[0].hidden, true);
});

test('gap dates remain selectable without interpolation while other Pod values retain their own colors', () => {
  const fixture_dict = fixture_obj({series_int: 2});
  set_value(fixture_dict.day_list[1], 0, {available_bool: false, label_str: '+999%', y_percent_float: null});
  set_value(fixture_dict.day_list[1], 1, {name_str: 'Long Pod name', label_str: '$1,234,567.89', y_percent_float: 12.5});
  fixture_dict.env_obj.emit('click', fixture_dict.day_list[1]);
  assert.deepEqual(fixture_dict.readout_list.map(row_dict => row_dict.number_obj.textContent), ['—', '$1,234,567.89']);
  assert.deepEqual(fixture_dict.marker_list.map(marker_obj => marker_obj.hidden), [true, false]);
  assert.equal(fixture_dict.readout_list[1].name_obj.textContent, 'Long Pod name');
  assert.equal(fixture_dict.readout_list[1].key_obj.style.backgroundColor, '#b06a34');
  assert.equal(fixture_dict.marker_list[1].style.backgroundColor, '#b06a34');
  assert.equal(fixture_dict.marker_list[1].style.top, '12.5%');
  assert.equal(fixture_dict.crosshair_obj.hidden, false);
  assert.equal(fixture_dict.crosshair_obj.style.left, '50%');
});

for (const payload_str of ['{broken', '{}', 'null', '[null]']) test(`malformed optional payload ${payload_str} clears old numbers and markers`, () => {
  const fixture_dict = fixture_obj();
  fixture_dict.env_obj.emit('click', fixture_dict.day_list[0]);
  fixture_dict.day_list[1].setAttribute('data-chart-values', payload_str);
  fixture_dict.env_obj.emit('click', fixture_dict.day_list[1]);
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '—');
  assert.equal(fixture_dict.marker_list[0].hidden, true);
});

test('invalid coordinates never become fabricated edge markers and text is not HTML', () => {
  const fixture_dict = fixture_obj();
  set_value(fixture_dict.day_list[1], 0, {label_str: '<b>-1.23%</b>', x_percent_float: null, y_percent_float: 101});
  fixture_dict.day_list[1].setAttribute('data-chart-x', '');
  fixture_dict.env_obj.emit('click', fixture_dict.day_list[1]);
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '<b>-1.23%</b>');
  assert.equal(fixture_dict.marker_list[0].hidden, true);
  assert.equal(fixture_dict.crosshair_obj.hidden, true);
});

test('arrows/Home/End move one roving tab stop across every date and clamp at the ends', () => {
  const fixture_dict = fixture_obj();
  const emit_fn = (index_int, key_str) => fixture_dict.env_obj.emit('keydown', fixture_dict.day_list[index_int], {key: key_str});
  assert.equal(emit_fn(2, 'Home').prevented_bool, true);
  assert.equal(fixture_dict.env_obj.document_obj.activeElement, fixture_dict.day_list[0]);
  emit_fn(0, 'ArrowLeft');
  assert.equal(fixture_dict.env_obj.document_obj.activeElement, fixture_dict.day_list[0]);
  emit_fn(0, 'ArrowRight');
  assert.equal(fixture_dict.env_obj.document_obj.activeElement, fixture_dict.day_list[1]);
  emit_fn(1, 'End');
  emit_fn(2, 'ArrowRight');
  assert.equal(fixture_dict.env_obj.document_obj.activeElement, fixture_dict.day_list[2]);
  assert.deepEqual(fixture_dict.day_list.map(day_obj => day_obj.tabIndex), [-1, -1, 0]);
  assert.equal(emit_fn(2, 'Tab').prevented_bool, false);
  assert.equal(emit_fn(2, 'ArrowDown').prevented_bool, false);
});

test('large All history updates only the previous/current date buttons', () => {
  const fixture_dict = fixture_obj({date_list: Array.from({length: 3000}, (_, index_int) => 'day-' + index_int)});
  for (const day_obj of fixture_dict.day_list) day_obj.write_count_int = 0;
  fixture_dict.env_obj.emit('pointerover', fixture_dict.day_list[20]);
  fixture_dict.env_obj.emit('pointerover', fixture_dict.day_list[21]);
  assert.equal(fixture_dict.day_list.reduce((count_int, day_obj) => count_int + day_obj.write_count_int, 0), 3);
  for (const day_obj of fixture_dict.day_list) day_obj.write_count_int = 0;
  fixture_dict.env_obj.emit('click', fixture_dict.day_list[22]);
  assert.equal(fixture_dict.day_list.reduce((count_int, day_obj) => count_int + day_obj.write_count_int, 0), 4);
  assert.equal(fixture_dict.day_list.filter(day_obj => day_obj.tabIndex === 0).length, 1);
});

test('Escape resets the latest readout and markers while leaving keyboard focus usable', () => {
  const fixture_dict = fixture_obj();
  fixture_dict.env_obj.emit('click', fixture_dict.day_list[0]);
  fixture_dict.env_obj.emit('keydown', fixture_dict.day_list[0], {key: 'Escape'});
  assert.equal(fixture_dict.env_obj.document_obj.activeElement, fixture_dict.day_list[2]);
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '+2.00%');
  assert.equal(fixture_dict.marker_list[0].hidden, true);
  fixture_dict.env_obj.emit('pointerout', fixture_dict.day_list[2], {relatedTarget: fixture_dict.readout_obj});
  assert.equal(fixture_dict.marker_list[0].hidden, true);
  fixture_dict.env_obj.emit('keydown', fixture_dict.day_list[2], {key: 'ArrowLeft'});
  assert.equal(fixture_dict.marker_list[0].hidden, false);
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '-1.23%');
});

test('single-date chart supports keyboard, tap and reset at the server coordinate', () => {
  const fixture_dict = fixture_obj({date_list: ['2026-09-02']});
  fixture_dict.env_obj.emit('keydown', fixture_dict.day_list[0], {key: 'ArrowRight'});
  assert.equal(fixture_dict.crosshair_obj.style.left, '50%');
  fixture_dict.env_obj.emit('keydown', fixture_dict.day_list[0], {key: 'Escape'});
  assert.equal(fixture_dict.crosshair_obj.hidden, true);
  fixture_dict.env_obj.emit('click', fixture_dict.day_list[0]);
  assert.equal(fixture_dict.crosshair_obj.hidden, false);
});

test('successful same-chart refresh restores the exact focused date using new server values without adding listeners', () => {
  const fixture_dict = fixture_obj();
  const env_obj = fixture_dict.env_obj;
  const count_list = Object.values(env_obj.handlers_dict).map(handler_list => handler_list.length);
  fixture_dict.day_list[1].focus();
  env_obj.emit('htmx:beforeSwap', fixture_dict.shell_obj, {detail: {target: fixture_dict.shell_obj}});
  const next_dict = env_obj.replace_shell(fixture_dict.shell_obj);
  set_value(next_dict.day_list[1], 0, {label_str: '-1.24%'});
  env_obj.emit('htmx:afterSwap', next_dict.shell_obj, {detail: {target: next_dict.shell_obj}});
  assert.equal(env_obj.document_obj.activeElement, next_dict.day_list[1]);
  assert.equal(next_dict.readout_list[0].number_obj.textContent, '-1.24%');
  assert.equal(next_dict.marker_list[0].hidden, false);
  assert.equal(next_dict.day_list[1].focus_options_dict.preventScroll, true);
  assert.deepEqual(Object.values(env_obj.handlers_dict).map(handler_list => handler_list.length), count_list);
});

for (const case_str of ['scope', 'chart id', 'removed date', 'duplicate chart', 'duplicate date', 'new focus', 'stale', 'error', 'cancelled', 'transport failure']) {
  test(`refresh never restores chart focus after ${case_str}`, () => {
    const fixture_dict = fixture_obj();
    const env_obj = fixture_dict.env_obj;
    fixture_dict.day_list[1].focus();
    env_obj.emit('htmx:beforeSwap', fixture_dict.shell_obj, {detail: {target: fixture_dict.shell_obj, shouldSwap: case_str !== 'cancelled'}});
    if (case_str === 'transport failure') env_obj.emit('htmx:sendError', fixture_dict.shell_obj, {detail: {target: fixture_dict.shell_obj}});
    const next_dict = env_obj.replace_shell(fixture_dict.shell_obj, {
      scope_str: case_str === 'scope' ? 'pod:other:All' : undefined,
      id_str: case_str === 'chart id' ? 'Pod return' : undefined,
      date_list: case_str === 'removed date' ? ['2026-08-31', '2026-09-02'] : undefined,
      stale_bool: case_str === 'stale',
    });
    if (case_str === 'duplicate chart') env_obj.chart_obj(next_dict.shell_obj);
    if (case_str === 'duplicate date') next_dict.day_list[0].setAttribute('data-chart-day', '2026-09-01');
    const outside_obj = env_obj.element_obj({}, next_dict.shell_obj);
    if (case_str === 'new focus') outside_obj.focus();
    env_obj.emit('htmx:afterSwap', next_dict.shell_obj, {detail: {target: next_dict.shell_obj, isError: case_str === 'error'}});
    assert.equal(env_obj.document_obj.activeElement, case_str === 'new focus' ? outside_obj : env_obj.document_obj.body);
    assert.equal(next_dict.marker_list[0].hidden, true);
  });
}

test('capture occurs at replacement, so cleared focus is not revived', () => {
  const fixture_dict = fixture_obj();
  const env_obj = fixture_dict.env_obj;
  fixture_dict.day_list[1].focus();
  env_obj.emit('htmx:beforeRequest', fixture_dict.shell_obj, {detail: {target: fixture_dict.shell_obj}});
  env_obj.element_obj({}, fixture_dict.shell_obj).focus();
  env_obj.emit('htmx:beforeSwap', fixture_dict.shell_obj, {detail: {target: fixture_dict.shell_obj}});
  const next_dict = env_obj.replace_shell(fixture_dict.shell_obj);
  env_obj.emit('htmx:afterSwap', next_dict.shell_obj, {detail: {target: next_dict.shell_obj}});
  assert.equal(env_obj.document_obj.activeElement, env_obj.document_obj.body);
});

test('Performance status-only refresh and its failure leave dated chart selection intact', () => {
  const fixture_dict = fixture_obj();
  fixture_dict.day_list[1].focus();
  const status_obj = fixture_dict.env_obj.element_obj({id: 'performance-status'}, fixture_dict.shell_obj);
  for (const event_str of ['htmx:beforeSwap', 'htmx:afterSwap', 'htmx:responseError']) {
    fixture_dict.env_obj.emit(event_str, status_obj, {detail: {target: status_obj, shouldSwap: false}});
  }
  assert.equal(fixture_dict.env_obj.document_obj.activeElement, fixture_dict.day_list[1]);
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '-1.23%');
  assert.equal(fixture_dict.marker_list[0].hidden, false);
});

test('other charts and Daily P&L controls are unaffected', () => {
  const fixture_dict = fixture_obj();
  const other_dict = fixture_dict.env_obj.chart_obj(fixture_dict.shell_obj, {id_str: 'Another chart'});
  const daily_obj = fixture_dict.env_obj.element_obj({'data-daily-panel': ''}, fixture_dict.shell_obj);
  const day_obj = fixture_dict.env_obj.element_obj({'data-daily-day': '2026-09-01'}, daily_obj);
  fixture_dict.env_obj.emit('click', fixture_dict.day_list[0]);
  for (const event_str of ['pointerover', 'pointerout', 'focusin', 'focusout', 'click', 'keydown']) {
    assert.equal(fixture_dict.env_obj.emit(event_str, day_obj, {key: 'Home'}).prevented_bool, false);
  }
  assert.equal(other_dict.readout_list[0].number_obj.textContent, '+2.00%');
  assert.equal(other_dict.marker_list[0].hidden, true);
  assert.equal(fixture_dict.readout_list[0].number_obj.textContent, '0.00%');
  assert.equal(day_obj.focus_count_int, 0);
});
