/* Behavioral tests for expiry/transport recovery, using a tiny DOM double. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source_str = fs.readFileSync(path.join(__dirname, '../alpha/live/dashboard_v4/static/overview.js'), 'utf8');

function element_obj(text_str = '', class_str = '') {
  const result_obj = {textContent: text_str, className: class_str, hidden: true, title: ''};
  const attribute_dict = {};
  result_obj.getAttribute = (key_str) => attribute_dict[key_str] ?? null;
  result_obj.setAttribute = (key_str, value_str) => { attribute_dict[key_str] = value_str; };
  result_obj.classList = {remove: (...name_list) => {
    result_obj.className = result_obj.className.split(' ').filter((name_str) => !name_list.includes(name_str)).join(' ');
  }};
  return result_obj;
}

function snapshot_obj(valid_ms = 120000) {
  const selector_dict = {};
  for (const selector_str of ['.refresh-error', '[data-refresh-reason]', '[data-observed-state]', '.tk',
    '[data-pod-pill]', '[data-pill-label]', '[data-pod-now]', '[data-status-label]', '[data-now-detail]',
    '[data-next-detail]', '[data-pod-next]', '[data-status-detail]', '[data-verdict-detail]', '[data-verdict]']) {
    selector_dict[selector_str] = element_obj('Current', 'st st-done');
  }
  selector_dict['[data-observed-state]'].setAttribute('aria-label', 'Fill · Done');
  selector_dict['.tk'].querySelector = () => element_obj('Fill');
  const shell_obj = element_obj();
  shell_obj.id = 'overview-shell';
  shell_obj.setAttribute('data-source-valid-ms', String(valid_ms));
  shell_obj.setAttribute('data-last-update', '09:41:07');
  shell_obj.querySelector = (selector_str) => selector_dict[selector_str] || null;
  shell_obj.querySelectorAll = (selector_str) => selector_str.split(', ').flatMap((item_str) => selector_dict[item_str] ? [selector_dict[item_str]] : []);
  return {shell_obj, selector_dict};
}

function environment_obj(valid_ms = 120000, initial_latency_ms = 0) {
  let wall_ms = 1000000;
  let current_obj = snapshot_obj(valid_ms);
  const handler_dict = {};
  let timer_fn;
  const document_obj = {
    activeElement: null,
    getElementById: () => current_obj.shell_obj,
    querySelectorAll: () => [],
    addEventListener: (name_str, handler_fn) => { handler_dict[name_str] = handler_fn; },
  };
  vm.runInNewContext(source_str, {
    document: document_obj,
    window: {addEventListener: document_obj.addEventListener},
    Date: {now: () => wall_ms}, performance: {now: () => initial_latency_ms},
    setInterval: (handler_fn) => { timer_fn = handler_fn; },
  });
  return {
    get current_obj() { return current_obj; },
    advance: (elapsed_ms) => { wall_ms += elapsed_ms; },
    timer: () => timer_fn(),
    fire: (name_str, extra_dict = {}) => handler_dict[name_str]({detail: {target: current_obj.shell_obj}, ...extra_dict}),
    replace: (remaining_ms = 120000) => { current_obj = snapshot_obj(remaining_ms); },
  };
}

test('source expires at remaining lifetime, before next poll; update time is retained', () => {
  const env_obj = environment_obj(1000);
  env_obj.advance(999); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['.refresh-error'].hidden, true);
  env_obj.advance(1); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-observed-state]'].className, 'st st-unk');
  assert.equal(env_obj.current_obj.selector_dict['[data-observed-state]'].getAttribute('aria-label'), 'Fill · Unknown');
  assert.equal(env_obj.current_obj.selector_dict['[data-verdict]'].textContent, 'Status unknown.');
  assert.equal(env_obj.current_obj.selector_dict['[data-refresh-reason]'].textContent, 'Saved status is out of date.');
  assert.equal(env_obj.current_obj.shell_obj.getAttribute('data-last-update'), '09:41:07');
});

test('initial network/parse latency cannot extend source lifetime', () => {
  const env_obj = environment_obj(1000, 1100);
  assert.equal(env_obj.current_obj.selector_dict['[data-pill-label]'].textContent, 'Unknown');
});

for (const event_str of ['htmx:responseError', 'htmx:sendError', 'htmx:timeout', 'htmx:swapError']) {
  test(event_str + ' marks every operating claim unknown while retaining the last plan label', () => {
    const env_obj = environment_obj();
    env_obj.fire(event_str);
    assert.equal(env_obj.current_obj.selector_dict['[data-pod-pill]'].className, 'pill pill-unk');
    assert.equal(env_obj.current_obj.selector_dict['[data-pod-now]'].textContent, 'Unknown');
    assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
    assert.equal(env_obj.current_obj.selector_dict['.tk'].title, 'Fill · Unknown');
    assert.equal(env_obj.current_obj.selector_dict['[data-refresh-reason]'].textContent, 'Update failed.');
    assert.equal(env_obj.current_obj.selector_dict['[data-next-detail]'].textContent, 'Last saved plan');
    assert.equal(env_obj.current_obj.shell_obj.getAttribute('data-source-stale'), 'true');
    assert.equal(env_obj.current_obj.shell_obj.getAttribute('data-last-update'), '09:41:07');
  });
}

test('background return and restored browser page never revive expired green state', () => {
  const env_obj = environment_obj();
  env_obj.advance(121000);
  env_obj.fire('visibilitychange');
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
  const restored_obj = environment_obj();
  restored_obj.fire('pageshow', {persisted: true});
  assert.equal(restored_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
});

test('new response replaces error state; round-trip latency is deducted from renewed TTL', () => {
  const env_obj = environment_obj();
  env_obj.fire('htmx:timeout');
  env_obj.fire('htmx:beforeRequest');
  env_obj.advance(2000);
  env_obj.replace(3000);
  env_obj.fire('htmx:afterSwap');
  assert.equal(env_obj.current_obj.selector_dict['.refresh-error'].hidden, true);
  env_obj.advance(999); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Current');
  env_obj.advance(1); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
});
