const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source_str = fs.readFileSync(path.join(__dirname, '../alpha/live/dashboard_v4/static/tools.js'), 'utf8');

async function flush() { for (let index_int = 0; index_int < 12; index_int += 1) await Promise.resolve(); }
function environment_obj({demo_bool = true, clipboard_fail_bool = false, selected_tool_str = '', vplan_id_str = '42',
  ready_state_str = 'complete', auto_frame_bool = true} = {}) {
  const handler_dict = {};
  const timer_map = new Map();
  const request_list = [];
  const copied_list = [];
  const navigation_list = [];
  const popup_list = [];
  const scroll_list = [];
  const frame_list = [];
  let popup_fn = () => true;
  let timer_int = 0;
  let now_int = 1000000;
  function emit(name_str, target_obj, extra_dict = {}) {
    const event_obj = {target: target_obj, preventDefault() {}, ...extra_dict};
    for (const callback_fn of handler_dict[name_str] || []) callback_fn(event_obj);
  }
  const document_obj = {activeElement: null, readyState: ready_state_str,
    addEventListener(name_str, callback_fn) { (handler_dict[name_str] ||= []).push(callback_fn); }};
  function element_obj(tag_str, attribute_dict = {}, parent_obj = null) {
    const node_obj = {tagName: tag_str.toUpperCase(), attribute_dict, parent_obj, children_list: [], hidden: false, disabled: false,
      required: false, value: '', textContent: '', selected_bool: false,
      get name() { return attribute_dict.name; },
      hasAttribute(key_str) { return Object.hasOwn(attribute_dict, key_str); },
      getAttribute(key_str) { return attribute_dict[key_str] ?? null; },
      setAttribute(key_str, value_str) { attribute_dict[key_str] = value_str; },
      matches(selector_str) { return selector_str.startsWith('[') ? Object.hasOwn(attribute_dict, selector_str.slice(1, -1)) : tag_str === selector_str; },
      closest(selector_str) { return this.matches(selector_str) ? this : parent_obj?.closest(selector_str) || null; },
      contains(other_obj) { return this === other_obj || this.children_list.some(child_obj => child_obj.contains(other_obj)); },
      querySelectorAll(selector_str) { return this.children_list.flatMap(child_obj => [
        ...(child_obj.matches(selector_str) ? [child_obj] : []), ...child_obj.querySelectorAll(selector_str)]); },
      querySelector(selector_str) { return this.querySelectorAll(selector_str)[0] || null; },
      appendChild(child_obj) { this.children_list.push(child_obj); }, replaceChildren() { this.children_list = []; },
      scrollIntoView(options_dict) { scroll_list.push({element_obj: this, block_str: options_dict.block}); },
      focus() { document_obj.activeElement = this; }, select() { this.selected_bool = true; },
      checkValidity() {
        if (this.valid_bool === false || (this.required && !this.value.trim())) return false;
        if (attribute_dict.type !== 'number' || !this.value) return true;
        const value_float = Number(this.value);
        return Number.isFinite(value_float) && Number.isInteger(value_float)
          && value_float >= Number(attribute_dict.min) && value_float <= Number(attribute_dict.max);
      },
      reportValidity() { return this.querySelectorAll('[data-manual-field]').every(input_obj => input_obj.disabled || !input_obj.required || input_obj.value.trim())
        && this.querySelectorAll('[data-tool-parameter]').every(input_obj => input_obj.checkValidity()); }};
    if (parent_obj) parent_obj.children_list.push(node_obj);
    return node_obj;
  }
  const page_obj = element_obj('div', {'data-tools-page': '', 'data-pod-id': 'demo_1_0', 'data-pod-label': 'Demo Pod', 'data-pod-account': '···771', 'data-demo': String(demo_bool),
    'data-actions-enabled': 'true', 'data-action-token': 'demo-token', 'data-selected-tool': selected_tool_str});
  const pod_form_obj = element_obj('form', {'data-tools-pod-form': ''}, page_obj);
  const pod_select_obj = element_obj('select', {'data-tools-pod': ''}, pod_form_obj);
  function row_obj(key_str, manual_bool = false) {
    const root_obj = element_obj('div', {'data-tool': key_str, 'data-tool-action': key_str, 'data-tool-copy': 'true',
      'data-tool-argv': JSON.stringify(['uv', 'run', 'python', '-m', 'alpha.live.runner', key_str, '--pod-id', "Pod's $(private)"])}, page_obj);
    const open_obj = element_obj('button', {'data-tool-open': '', 'aria-expanded': 'false'}, root_obj);
    const body_obj = element_obj('div', {'data-tool-body': ''}, root_obj); body_obj.hidden = true;
    const form_obj = element_obj('form', {'data-tool-form': ''}, body_obj);
    const command_obj = element_obj('textarea', {'data-tool-command': ''}, form_obj);
    const copy_obj = element_obj('button', {'data-tool-copy-button': ''}, form_obj);
    const copy_status_obj = element_obj('p', {'data-tool-copy-status': ''}, form_obj);
    const preview_obj = element_obj('button', {'data-tool-preview': ''}, form_obj);
    const box_obj = element_obj('div', {'data-tool-preview-box': ''}, body_obj); box_obj.hidden = true;
    const expires_obj = element_obj('span', {'data-tool-expires': ''}, box_obj);
    const lines_obj = element_obj('ul', {'data-tool-preview-lines': ''}, box_obj);
    const confirm_obj = element_obj('button', {'data-tool-confirm': ''}, box_obj);
    const cancel_obj = element_obj('button', {'data-tool-cancel': ''}, box_obj);
    const result_obj = element_obj('div', {'data-tool-result': ''}, body_obj); result_obj.hidden = true;
    const label_obj = element_obj('b', {'data-tool-result-label': ''}, result_obj);
    const message_obj = element_obj('p', {'data-tool-result-message': ''}, result_obj);
    const field_dict = {};
    if (key_str === 'submit_vplan') {
      const field_obj = element_obj('input', {name: 'vplan_id_int', type: 'number', min: '1', max: '2147483647',
        'data-tool-parameter': '', 'data-flag': '--vplan-id'}, form_obj);
      field_obj.value = vplan_id_str; field_obj.required = true; field_dict.vplan_id_int = field_obj;
    }
    if (manual_bool) for (const [name_str, value_str] of Object.entries({asset_str: 'AAPL', side_str: 'BUY', broker_order_type_str: 'MKT',
      quantity_int: '4', limit_price_float: '', operator_id_str: 'operator', reason_str: 'demo ticket', confirmation_text_str: 'SUBMIT MANUAL ORDER'})) {
      const field_obj = element_obj('input', {name: name_str, 'data-manual-field': ''}, form_obj); field_obj.value = value_str;
      field_obj.required = name_str !== 'limit_price_float'; field_dict[name_str] = field_obj;
    }
    return {root_obj, open_obj, body_obj, form_obj, command_obj, copy_obj, copy_status_obj, preview_obj, box_obj, expires_obj, lines_obj,
      confirm_obj, cancel_obj, result_obj, label_obj, message_obj, field_dict};
  }
  const tick_obj = row_obj('tick');
  const eod_obj = row_obj('eod_snapshot');
  const submit_obj = row_obj('submit_vplan');
  const manual_obj = row_obj('manual_order', true);
  let current_page_obj = page_obj;
  document_obj.querySelector = () => current_page_obj;
  document_obj.createElement = tag_str => element_obj(tag_str);
  const window_obj = {location: {origin: 'http://127.0.0.1:8114', assign(value_str) { navigation_list.push(value_str); }},
    confirm(message_str) { popup_list.push(message_str); return popup_fn(message_str); },
    requestAnimationFrame(callback_fn) { frame_list.push(callback_fn); return frame_list.length; },
    addEventListener: document_obj.addEventListener,
    setTimeout(callback_fn, delay_int) { const key_int = ++timer_int; timer_map.set(key_int, {callback_fn, delay_int}); return key_int; },
    clearTimeout(key_int) { timer_map.delete(key_int); },
    fetch(url_str, options_dict) { return new Promise((resolve_fn, reject_fn) => request_list.push({url_str, options_dict, resolve_fn, reject_fn})); }};
  class Clock extends Date { static now() { return now_int; } }
  vm.runInNewContext(source_str, {document: document_obj, window: window_obj, Date: Clock, URL, URLSearchParams, AbortController,
    navigator: {clipboard: {async writeText(value_str) { if (clipboard_fail_bool) throw new Error('denied'); copied_list.push(value_str); }}}});
  function flush_frame() { for (const callback_fn of frame_list.splice(0)) callback_fn(); }
  if (auto_frame_bool) flush_frame();
  function resolve(index_int, payload_dict, ok_bool = true) {
    request_list[index_int].resolve_fn({ok: ok_bool, async json() { return payload_dict; }});
  }
  function preview_response(action_str = 'tick', overrides_dict = {}) {
    return {demo_bool: true, pod_id_str: 'demo_1_0', action_name_str: action_str, confirmation_nonce_str: 'demo-nonce',
      expires_in_seconds_int: 120, preview_line_list: ['Only synthetic data.', '<img src=x onerror=bad()>'], ...overrides_dict};
  }
  function job_response(status_str = 'succeeded', overrides_dict = {}) {
    return {demo_bool: true, pod_id_str: 'demo_1_0', action_name_str: 'tick', job_id_str: 'job_1', status_str, message_str: 'Synthetic response',
      poll_url_str: '/api/demo-tools/demo_1_0/jobs/job_1', ...overrides_dict};
  }
  return {emit, page_obj, tick_obj, eod_obj, submit_obj, manual_obj, document_obj, request_list, copied_list, navigation_list, popup_list, scroll_list, flush_frame,
    set_popup(callback_fn) { popup_fn = callback_fn; }, pod_form_obj, pod_select_obj, timer_map, resolve, preview_response,
    job_response, element_obj, set_page(next_obj) { current_page_obj = next_obj; }, advance(milliseconds_int) { now_int += milliseconds_int; },
    run_timer(delay_int) { const entry_obj = [...timer_map.entries()].find(([, value_dict]) => value_dict.delay_int === delay_int);
      if (entry_obj) { timer_map.delete(entry_obj[0]); entry_obj[1].callback_fn(); } }};
}

async function open_preview(env_obj, row_obj = env_obj.tick_obj) {
  env_obj.emit('click', row_obj.open_obj); env_obj.emit('submit', row_obj.form_obj); await flush();
  const index_int = env_obj.request_list.length - 1;
  env_obj.resolve(index_int, env_obj.preview_response(row_obj.root_obj.getAttribute('data-tool-action'))); await flush();
}

test('production Copy quotes PowerShell literals, never dispatches and cannot preview even with actions flag', async () => {
  const env_obj = environment_obj({demo_bool: false});
  env_obj.emit('click', env_obj.tick_obj.open_obj); env_obj.emit('click', env_obj.tick_obj.copy_obj); await flush();
  assert.match(env_obj.copied_list[0], /'Pod''s \$\(private\)'/);
  assert.ok(env_obj.copied_list[0].startsWith("& 'uv'"));
  assert.match(env_obj.tick_obj.copy_status_obj.textContent, /Nothing was run/);
  env_obj.emit('submit', env_obj.tick_obj.form_obj); env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush();
  assert.equal(env_obj.request_list.length, 0); assert.equal(env_obj.tick_obj.preview_obj.disabled, true);
});

test('denied clipboard selects the complete command for manual copying', async () => {
  const env_obj = environment_obj({demo_bool: false, clipboard_fail_bool: true});
  env_obj.emit('click', env_obj.tick_obj.copy_obj); await flush();
  assert.equal(env_obj.tick_obj.command_obj.selected_bool, true);
  assert.equal(env_obj.document_obj.activeElement, env_obj.tick_obj.command_obj);
  assert.match(env_obj.tick_obj.copy_status_obj.textContent, /Ctrl\+C/); assert.equal(env_obj.request_list.length, 0);
});

test('required parameters block copy; filled values are quoted and invalid numbers stay blocked', async () => {
  const env_obj = environment_obj();
  const parameter_obj = env_obj.element_obj('input', {'data-tool-parameter': '', 'data-flag': '--broker-client-id'}, env_obj.tick_obj.form_obj);
  parameter_obj.required = true;
  env_obj.emit('input', parameter_obj); assert.equal(env_obj.tick_obj.copy_obj.disabled, true); assert.equal(env_obj.tick_obj.command_obj.value, '');
  parameter_obj.value = '55'; env_obj.emit('input', parameter_obj);
  assert.match(env_obj.tick_obj.command_obj.value, /'--broker-client-id' '55'$/);
  parameter_obj.valid_bool = false; env_obj.emit('input', parameter_obj); assert.equal(env_obj.tick_obj.copy_obj.disabled, true);
});

test('preview stays synthetic, renders text safely and never confirms until clicked', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj);
  assert.equal(env_obj.request_list.length, 1);
  assert.equal(env_obj.request_list[0].url_str, '/api/demo-tools/demo_1_0/tick/preview');
  assert.deepEqual(JSON.parse(env_obj.request_list[0].options_dict.body), {confirmed_bool: true});
  assert.equal(env_obj.tick_obj.lines_obj.children_list[1].textContent, '<img src=x onerror=bad()>');
  assert.equal(env_obj.tick_obj.box_obj.hidden, false); assert.equal(env_obj.tick_obj.confirm_obj.disabled, false);
  assert.match(env_obj.tick_obj.expires_obj.textContent, /2:00/);
});

test('duplicate previews and confirms dispatch once, then poll the synthetic job', async () => {
  const env_obj = environment_obj(); env_obj.emit('click', env_obj.tick_obj.open_obj);
  env_obj.emit('submit', env_obj.tick_obj.form_obj); env_obj.emit('submit', env_obj.tick_obj.form_obj); await flush();
  assert.equal(env_obj.request_list.length, 1); env_obj.resolve(0, env_obj.preview_response()); await flush();
  env_obj.emit('click', env_obj.tick_obj.confirm_obj); env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush();
  assert.equal(env_obj.request_list.length, 2); assert.match(env_obj.request_list[1].url_str, /\/confirm$/);
  env_obj.resolve(1, env_obj.job_response('running')); await flush(); assert.equal(env_obj.request_list.length, 3);
  assert.equal(env_obj.request_list[2].options_dict.method, 'GET'); env_obj.resolve(2, env_obj.job_response()); await flush();
  assert.match(env_obj.tick_obj.label_obj.textContent, /Simulated · Command completed/);
  assert.match(env_obj.tick_obj.message_obj.textContent, /No broker action/);
});

test('expired preview cannot confirm and offers a fresh preview', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj); env_obj.advance(121000); env_obj.run_timer(1000);
  assert.equal(env_obj.tick_obj.confirm_obj.disabled, true); assert.equal(env_obj.tick_obj.preview_obj.disabled, false);
  env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush(); assert.equal(env_obj.request_list.length, 1);
  assert.match(env_obj.tick_obj.expires_obj.textContent, /Expired/);
});

test('cancel revokes preview without sending a confirm', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj); env_obj.emit('click', env_obj.tick_obj.cancel_obj); await flush();
  assert.equal(env_obj.tick_obj.box_obj.hidden, true); assert.equal(env_obj.tick_obj.confirm_obj.disabled, true);
  assert.match(env_obj.request_list[1].url_str, /\/cancel$/);
  env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush(); assert.equal(env_obj.request_list.length, 2);
});

test('changing tools discards a late preview and aborts its request', async () => {
  const env_obj = environment_obj(); env_obj.emit('click', env_obj.tick_obj.open_obj); env_obj.emit('submit', env_obj.tick_obj.form_obj); await flush();
  env_obj.emit('click', env_obj.eod_obj.open_obj); assert.equal(env_obj.request_list[0].options_dict.signal.aborted, true);
  env_obj.resolve(0, env_obj.preview_response()); await flush();
  assert.equal(env_obj.tick_obj.box_obj.hidden, true); assert.equal(env_obj.tick_obj.body_obj.hidden, true);
  assert.equal(env_obj.eod_obj.body_obj.hidden, false); assert.equal(env_obj.tick_obj.confirm_obj.disabled, true);
});

test('header-only refresh preserves fields, open tool, preview and focus', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj); const focus_obj = env_obj.document_obj.activeElement;
  env_obj.emit('htmx:afterSwap', env_obj.page_obj);
  assert.equal(env_obj.tick_obj.box_obj.hidden, false); assert.equal(env_obj.tick_obj.body_obj.hidden, false);
  assert.equal(env_obj.document_obj.activeElement, focus_obj); assert.equal(env_obj.request_list.length, 1);
});

test('a replaced page cannot receive an old preview response', async () => {
  const env_obj = environment_obj(); env_obj.emit('click', env_obj.tick_obj.open_obj); env_obj.emit('submit', env_obj.tick_obj.form_obj); await flush();
  env_obj.set_page(null); env_obj.emit('htmx:afterSwap', env_obj.page_obj); env_obj.resolve(0, env_obj.preview_response()); await flush();
  assert.equal(env_obj.tick_obj.box_obj.hidden, true); assert.equal(env_obj.request_list[0].options_dict.signal.aborted, true);
});

test('wrong Pod or non-demo preview is rejected without enabling confirm', async () => {
  for (const override_dict of [{pod_id_str: 'another'}, {demo_bool: false}]) {
    const env_obj = environment_obj(); env_obj.emit('click', env_obj.tick_obj.open_obj); env_obj.emit('submit', env_obj.tick_obj.form_obj); await flush();
    env_obj.resolve(0, env_obj.preview_response('tick', override_dict)); await flush();
    assert.equal(env_obj.tick_obj.confirm_obj.disabled, true); assert.match(env_obj.tick_obj.label_obj.textContent, /unavailable/);
  }
});

test('confirm timeout never retries or claims the command did not run', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj); env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush();
  env_obj.request_list[1].reject_fn(new Error('network lost')); await flush();
  assert.match(env_obj.tick_obj.label_obj.textContent, /unconfirmed/); assert.equal(env_obj.tick_obj.preview_obj.disabled, true);
  env_obj.emit('click', env_obj.tick_obj.confirm_obj); env_obj.emit('submit', env_obj.tick_obj.form_obj); await flush();
  assert.equal(env_obj.request_list.length, 2);
});

test('poll links cannot leave this origin or the selected demo Pod', async () => {
  for (const url_str of ['https://outside.example/job', '/api/tools/demo_1_0/jobs/job_1', '/api/demo-tools/another/jobs/job_1', '/api/demo-tools/demo_1_0/jobs/job_1?run=yes']) {
    const env_obj = environment_obj(); await open_preview(env_obj); env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush();
    env_obj.resolve(1, env_obj.job_response('running', {poll_url_str: url_str})); await flush();
    assert.equal(env_obj.request_list.length, 2); assert.match(env_obj.tick_obj.label_obj.textContent, /unconfirmed/);
  }
});

test('manual demo ticket requires exact phrase, omits MARKET limit and preserves typed reason as text', async () => {
  const env_obj = environment_obj(); const row_obj = env_obj.manual_obj;
  env_obj.emit('click', row_obj.open_obj); row_obj.field_dict.confirmation_text_str.value = 'submit manual order';
  env_obj.emit('submit', row_obj.form_obj); await flush(); assert.equal(env_obj.request_list.length, 0);
  row_obj.field_dict.confirmation_text_str.value = 'SUBMIT MANUAL ORDER'; row_obj.field_dict.reason_str.value = '<script>bad()</script>';
  env_obj.emit('submit', row_obj.form_obj); await flush();
  const payload_dict = JSON.parse(env_obj.request_list[0].options_dict.body);
  assert.equal(payload_dict.manual_order_dict.quantity_int, 4); assert.equal(payload_dict.manual_order_dict.time_in_force_str, 'DAY');
  assert.equal(payload_dict.manual_order_dict.reason_str, '<script>bad()</script>'); assert.equal('limit_price_float' in payload_dict.manual_order_dict, false);
});

test('LIMIT ticket requires a price and editing a preview invalidates confirmation', async () => {
  const env_obj = environment_obj(); const row_obj = env_obj.manual_obj;
  env_obj.emit('click', row_obj.open_obj); row_obj.field_dict.broker_order_type_str.value = 'LMT'; env_obj.emit('change', row_obj.field_dict.broker_order_type_str);
  assert.equal(row_obj.field_dict.limit_price_float.required, true); assert.equal(row_obj.field_dict.limit_price_float.disabled, false);
  env_obj.emit('submit', row_obj.form_obj); await flush(); assert.equal(env_obj.request_list.length, 0);
  row_obj.field_dict.limit_price_float.value = '120.25'; env_obj.emit('submit', row_obj.form_obj); await flush();
  env_obj.resolve(0, env_obj.preview_response('manual_order')); await flush();
  row_obj.field_dict.quantity_int.value = '8'; env_obj.emit('input', row_obj.field_dict.quantity_int); await flush();
  assert.equal(row_obj.box_obj.hidden, true); assert.equal(row_obj.confirm_obj.disabled, true);
  assert.equal(JSON.parse(env_obj.request_list[0].options_dict.body).manual_order_dict.limit_price_float, 120.25);
});

test('Pod form navigates with encoded identifiers without native form submission or execution', async () => {
  const env_obj = environment_obj({demo_bool: false}); let prevented_bool = false;
  env_obj.pod_select_obj.value = "long Pod & user's name";
  env_obj.emit('submit', env_obj.pod_form_obj, {preventDefault() { prevented_bool = true; }}); await flush();
  assert.equal(prevented_bool, true);
  assert.equal(env_obj.navigation_list[0], '/tools?pod=long+Pod+%26+user%27s+name');
  env_obj.pod_select_obj.value = ''; env_obj.emit('submit', env_obj.pod_form_obj);
  assert.equal(env_obj.navigation_list[1], '/tools'); assert.equal(env_obj.request_list.length, 0);
});

test('selected tool deep link opens its row without requesting a preview and survives Pod change', () => {
  const env_obj = environment_obj({selected_tool_str: 'eod_snapshot'});
  assert.equal(env_obj.eod_obj.body_obj.hidden, false); assert.equal(env_obj.tick_obj.body_obj.hidden, true);
  assert.equal(env_obj.eod_obj.open_obj.getAttribute('aria-expanded'), 'true'); assert.equal(env_obj.request_list.length, 0);
  assert.equal(env_obj.scroll_list.length, 1);
  assert.equal(env_obj.scroll_list[0].element_obj, env_obj.eod_obj.root_obj);
  assert.equal(env_obj.scroll_list[0].block_str, 'start');
  env_obj.pod_select_obj.value = 'demo_1_1'; env_obj.emit('submit', env_obj.pod_form_obj);
  assert.equal(env_obj.navigation_list[0], '/tools?pod=demo_1_1&tool=eod_snapshot');
});

test('status refresh and page resume never repeat the deep-link scroll', () => {
  const env_obj = environment_obj({selected_tool_str: 'eod_snapshot'});
  env_obj.emit('htmx:afterSwap', env_obj.page_obj);
  env_obj.emit('htmx:afterSwap', env_obj.page_obj);
  env_obj.emit('pageshow', env_obj.page_obj, {persisted: true});
  env_obj.emit('visibilitychange', env_obj.page_obj);
  env_obj.flush_frame();
  assert.equal(env_obj.scroll_list.length, 1);
  assert.equal(env_obj.eod_obj.body_obj.hidden, false);
  assert.equal(env_obj.request_list.length, 0);
});

test('manually opening another tool does not scroll the page', () => {
  const env_obj = environment_obj({selected_tool_str: 'eod_snapshot'});
  env_obj.emit('click', env_obj.tick_obj.open_obj);
  env_obj.emit('click', env_obj.tick_obj.open_obj);
  env_obj.emit('click', env_obj.eod_obj.open_obj);
  assert.equal(env_obj.scroll_list.length, 1);
});

test('missing or invalid tool selection never opens or scrolls a row', () => {
  for (const selected_tool_str of ['', 'unknown_tool']) {
    const env_obj = environment_obj({selected_tool_str});
    assert.equal(env_obj.scroll_list.length, 0);
    assert.equal(env_obj.tick_obj.body_obj.hidden, true);
    assert.equal(env_obj.eod_obj.body_obj.hidden, true);
    assert.equal(env_obj.manual_obj.body_obj.hidden, true);
    assert.equal(env_obj.request_list.length, 0);
  }
});

test('wrong job identity never becomes a completed result', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj); env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush();
  env_obj.resolve(1, env_obj.job_response('running')); await flush();
  env_obj.resolve(2, env_obj.job_response('succeeded', {job_id_str: 'other_job', message_str: 'Wrong job completed'})); await flush();
  assert.match(env_obj.tick_obj.label_obj.textContent, /unconfirmed/);
  assert.doesNotMatch(env_obj.tick_obj.message_obj.textContent, /Wrong job completed/);
  assert.equal(env_obj.tick_obj.preview_obj.disabled, true);
});

test('native confirmation shows exact tool, Pod, account and preview consequences before dispatch', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj);
  assert.equal(env_obj.popup_list.length, 0);
  env_obj.set_popup(() => { assert.equal(env_obj.request_list.length, 1); return true; });
  env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush();
  assert.equal(env_obj.popup_list.length, 1);
  assert.match(env_obj.popup_list[0], /^Are you sure\?/);
  assert.match(env_obj.popup_list[0], /SIMULATION · tick/);
  assert.match(env_obj.popup_list[0], /Pod: Demo Pod\ndemo_1_0\nAccount: ···771/);
  assert.match(env_obj.popup_list[0], /Only synthetic data\./);
  assert.deepEqual(JSON.parse(env_obj.request_list[1].options_dict.body),
    {confirmed_bool: true, browser_confirmed_bool: true, confirmation_nonce_str: 'demo-nonce'});
});

test('declining the popup sends no request and preserves the valid preview for a later choice', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj); env_obj.set_popup(() => false);
  env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush();
  assert.equal(env_obj.request_list.length, 1); assert.equal(env_obj.tick_obj.box_obj.hidden, false);
  assert.equal(env_obj.tick_obj.confirm_obj.disabled, false);
  env_obj.set_popup(() => true); env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush();
  assert.equal(env_obj.request_list.length, 2);
  assert.equal(JSON.parse(env_obj.request_list[1].options_dict.body).confirmation_nonce_str, 'demo-nonce');
});

test('a preview that expires while the popup is open never dispatches', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj);
  env_obj.set_popup(() => { env_obj.advance(121000); return true; });
  env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush();
  assert.equal(env_obj.request_list.length, 1); assert.equal(env_obj.tick_obj.confirm_obj.disabled, true);
  assert.match(env_obj.tick_obj.expires_obj.textContent, /Expired/); assert.equal(env_obj.tick_obj.preview_obj.disabled, false);
});

test('duplicate click during a popup cannot create a second popup or request', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj);
  env_obj.set_popup(() => { env_obj.emit('click', env_obj.tick_obj.confirm_obj); return true; });
  env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush();
  assert.equal(env_obj.popup_list.length, 1); assert.equal(env_obj.request_list.length, 2);
});

test('changing the selected Pod while a popup is open invalidates the preview', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj);
  env_obj.set_popup(() => { env_obj.page_obj.setAttribute('data-pod-id', 'demo_1_1'); return true; });
  env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush();
  assert.equal(env_obj.request_list.length, 1); assert.match(env_obj.tick_obj.label_obj.textContent, /Preview changed/);
  assert.equal(env_obj.tick_obj.confirm_obj.disabled, true);
});

test('blocked native popup cannot silently confirm', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj); env_obj.set_popup(() => { throw new Error('Dialog suppressed'); });
  env_obj.emit('click', env_obj.tick_obj.confirm_obj); await flush();
  assert.equal(env_obj.request_list.length, 1); assert.equal(env_obj.tick_obj.box_obj.hidden, false);
  assert.equal(env_obj.tick_obj.confirm_obj.disabled, false);
});

test('new tool preview waits for an earlier tool cancellation for the same Pod', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj);
  env_obj.emit('click', env_obj.eod_obj.open_obj); env_obj.emit('submit', env_obj.eod_obj.form_obj); await flush();
  assert.equal(env_obj.request_list.length, 2);
  assert.match(env_obj.request_list[1].url_str, /\/tick\/cancel$/);
  assert.equal(env_obj.eod_obj.confirm_obj.disabled, true);
  env_obj.resolve(1, {cancelled_bool: true}); await flush();
  assert.equal(env_obj.request_list.length, 3);
  assert.match(env_obj.request_list[2].url_str, /\/eod_snapshot\/preview$/);
  env_obj.resolve(2, env_obj.preview_response('eod_snapshot')); await flush();
  assert.equal(env_obj.eod_obj.box_obj.hidden, false); assert.equal(env_obj.eod_obj.confirm_obj.disabled, false);
});

test('navigation during a pending cancellation cannot dispatch the waiting preview', async () => {
  const env_obj = environment_obj(); await open_preview(env_obj);
  env_obj.emit('click', env_obj.eod_obj.open_obj); env_obj.emit('submit', env_obj.eod_obj.form_obj); await flush();
  env_obj.set_page(null); env_obj.emit('htmx:afterSwap', env_obj.page_obj);
  env_obj.resolve(1, {cancelled_bool: true}); await flush();
  assert.equal(env_obj.request_list.some(request_obj => /\/eod_snapshot\/preview$/.test(request_obj.url_str)), false);
  assert.equal(env_obj.eod_obj.box_obj.hidden, true);
});

test('submit Copy pins the prefilled VPlan once and editing replaces its ID without production requests', async () => {
  const env_obj = environment_obj({demo_bool: false}); const row_obj = env_obj.submit_obj;
  env_obj.emit('click', row_obj.copy_obj); await flush();
  assert.match(env_obj.copied_list[0], /'--vplan-id' '42'$/);
  assert.equal(env_obj.copied_list[0].match(/'--vplan-id'/g).length, 1);
  row_obj.field_dict.vplan_id_int.value = '77'; env_obj.emit('input', row_obj.field_dict.vplan_id_int);
  env_obj.emit('click', row_obj.copy_obj); await flush();
  assert.match(env_obj.copied_list[1], /'--vplan-id' '77'$/);
  assert.doesNotMatch(env_obj.copied_list[1], /'42'/);
  assert.equal(env_obj.copied_list[1].match(/'--vplan-id'/g).length, 1);
  assert.equal(env_obj.request_list.length, 0); assert.equal(env_obj.popup_list.length, 0);
});

test('cleared and invalid VPlan IDs block Copy and never send production requests', async () => {
  const env_obj = environment_obj({demo_bool: false}); const row_obj = env_obj.submit_obj;
  for (const value_str of ['', '0', '-1', '1.5', '2147483648', 'NaN', '1; bad']) {
    row_obj.field_dict.vplan_id_int.value = value_str; env_obj.emit('input', row_obj.field_dict.vplan_id_int);
    env_obj.emit('click', row_obj.copy_obj); await flush();
    assert.equal(row_obj.copy_obj.disabled, true, value_str); assert.equal(row_obj.command_obj.value, '', value_str);
  }
  assert.equal(env_obj.copied_list.length, 0); assert.equal(env_obj.request_list.length, 0);
});

test('submit preview sends the exact numeric VPlan ID and editing it cancels the old preview', async () => {
  const env_obj = environment_obj(); const row_obj = env_obj.submit_obj;
  env_obj.emit('click', row_obj.open_obj); env_obj.emit('submit', row_obj.form_obj); await flush();
  assert.deepEqual(JSON.parse(env_obj.request_list[0].options_dict.body), {confirmed_bool: true, vplan_id_int: 42});
  assert.match(env_obj.request_list[0].url_str, /\/submit_vplan\/preview$/);
  env_obj.resolve(0, env_obj.preview_response('submit_vplan', {preview_line_list: ['VPlan: 42']})); await flush();
  row_obj.field_dict.vplan_id_int.value = '77'; env_obj.emit('input', row_obj.field_dict.vplan_id_int); await flush();
  assert.equal(row_obj.box_obj.hidden, true); assert.equal(row_obj.confirm_obj.disabled, true);
  assert.match(env_obj.request_list[1].url_str, /\/submit_vplan\/cancel$/);
  env_obj.emit('click', row_obj.confirm_obj); await flush(); assert.equal(env_obj.request_list.length, 2);
  env_obj.emit('submit', row_obj.form_obj); env_obj.resolve(1, {cancelled_bool: true}); await flush();
  assert.deepEqual(JSON.parse(env_obj.request_list[2].options_dict.body), {confirmed_bool: true, vplan_id_int: 77});
  assert.equal(env_obj.popup_list.length, 0);
});

test('invalid submit IDs never request even a synthetic preview', async () => {
  for (const vplan_id_str of ['', '0', '-1', '2.5', '2147483648']) {
    const env_obj = environment_obj({vplan_id_str});
    env_obj.emit('click', env_obj.submit_obj.open_obj); env_obj.emit('submit', env_obj.submit_obj.form_obj); await flush();
    assert.equal(env_obj.request_list.length, 0, vplan_id_str);
  }
});

test('integral decimal and exponent IDs copy the same canonical integer sent to preview', async () => {
  for (const [vplan_id_str, expected_int] of [['42.0', 42], ['4e1', 40], ['2147483647', 2147483647]]) {
    const env_obj = environment_obj({vplan_id_str}); const row_obj = env_obj.submit_obj;
    env_obj.emit('click', row_obj.copy_obj); await flush();
    assert.ok(env_obj.copied_list[0].endsWith("'--vplan-id' '" + expected_int + "'"));
    assert.equal(env_obj.copied_list[0].match(/'--vplan-id'/g).length, 1);
    env_obj.emit('click', row_obj.open_obj); env_obj.emit('submit', row_obj.form_obj); await flush();
    assert.equal(JSON.parse(env_obj.request_list[0].options_dict.body).vplan_id_int, expected_int);
  }
});

test('VPlan range validation blocks Copy even if native input validity is bypassed', async () => {
  const env_obj = environment_obj({demo_bool: false}); const row_obj = env_obj.submit_obj;
  row_obj.field_dict.vplan_id_int.checkValidity = () => true;
  for (const value_str of ['0', '2147483648', '4.2e-1', 'NaN']) {
    row_obj.field_dict.vplan_id_int.value = value_str; env_obj.emit('input', row_obj.field_dict.vplan_id_int);
    env_obj.emit('click', row_obj.copy_obj); await flush();
    assert.equal(row_obj.copy_obj.disabled, true); assert.equal(row_obj.command_obj.value, '');
  }
  assert.equal(env_obj.copied_list.length, 0); assert.equal(env_obj.request_list.length, 0);
});

test('initial deep link waits until the frame after pageshow and history restoration', () => {
  const env_obj = environment_obj({selected_tool_str: 'eod_snapshot', ready_state_str: 'interactive'});
  assert.equal(env_obj.eod_obj.body_obj.hidden, false); assert.equal(env_obj.scroll_list.length, 0);
  env_obj.document_obj.readyState = 'complete'; env_obj.emit('load', env_obj.page_obj); env_obj.flush_frame();
  assert.equal(env_obj.scroll_list.length, 0);
  env_obj.emit('pageshow', env_obj.page_obj, {persisted: false});
  assert.equal(env_obj.scroll_list.length, 0);
  env_obj.flush_frame();
  assert.equal(env_obj.scroll_list.length, 1); assert.equal(env_obj.scroll_list[0].element_obj, env_obj.eod_obj.root_obj);
  env_obj.emit('pageshow', env_obj.page_obj, {persisted: false}); env_obj.flush_frame();
  assert.equal(env_obj.scroll_list.length, 1);
});

test('manual selection before page display cancels the delayed deep-link scroll', () => {
  const env_obj = environment_obj({selected_tool_str: 'eod_snapshot', ready_state_str: 'interactive'});
  env_obj.emit('click', env_obj.tick_obj.open_obj);
  env_obj.document_obj.readyState = 'complete'; env_obj.emit('pageshow', env_obj.page_obj, {persisted: false}); env_obj.flush_frame();
  assert.equal(env_obj.scroll_list.length, 0); assert.equal(env_obj.tick_obj.body_obj.hidden, false);
});

test('a queued deep-link frame cannot scroll after a manual choice, page replacement or resume', () => {
  for (const change_str of ['manual', 'page', 'resume']) {
    const env_obj = environment_obj({selected_tool_str: 'eod_snapshot', auto_frame_bool: false});
    assert.equal(env_obj.scroll_list.length, 0);
    if (change_str === 'manual') env_obj.emit('click', env_obj.tick_obj.open_obj);
    else if (change_str === 'resume') { env_obj.emit('pagehide', env_obj.page_obj); env_obj.emit('pageshow', env_obj.page_obj, {persisted: true}); }
    else { env_obj.set_page(null); env_obj.emit('htmx:afterSwap', env_obj.page_obj); }
    env_obj.flush_frame();
    assert.equal(env_obj.scroll_list.length, 0);
  }
});
