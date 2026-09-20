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

function snapshot_obj(valid_ms = 120000, clock_timestamp_str = '2026-09-18T13:41:07+00:00') {
  const selector_dict = {};
  for (const selector_str of ['.refresh-error', '[data-refresh-reason]', '[data-observed-state]', '.tk',
    '[data-pod-pill]', '[data-pill-label]', '[data-pod-now]', '[data-status-label]', '[data-now-detail]',
    '[data-next-detail]', '[data-pod-next]', '[data-status-detail]', '[data-verdict-detail]', '[data-verdict]',
    '.step', '[data-step-fact]', '[data-evidence-status]', '[data-cycle-verdict]']) {
    selector_dict[selector_str] = element_obj('Current', 'st st-done');
  }
  selector_dict['[data-observed-state]'].setAttribute('aria-label', 'Fill · Done');
  selector_dict['.tk'].querySelector = () => element_obj('Fill');
  selector_dict['.step'].className = 'step is-done is-sel';
  selector_dict['[data-step-fact]'].textContent = '9 of 9';
  selector_dict['[data-evidence-status]'].textContent = 'Acked';
  selector_dict['[data-live-clock]'] = [element_obj('09:41:07 ET'), element_obj('09:41:07 ET')];
  selector_dict['[data-broker-id]'] = element_obj('perm:123456789');
  const shell_obj = element_obj();
  shell_obj.id = 'overview-shell';
  shell_obj.setAttribute('data-source-valid-ms', String(valid_ms));
  shell_obj.setAttribute('data-last-update', '09:41:07');
  shell_obj.setAttribute('data-clock-timestamp', clock_timestamp_str);
  shell_obj.querySelector = (selector_str) => selector_dict[selector_str] || null;
  shell_obj.querySelectorAll = (selector_str) => selector_str.split(', ').flatMap((item_str) => selector_dict[item_str] || []);
  return {shell_obj, selector_dict};
}

function environment_obj(valid_ms = 120000, initial_latency_ms = 0, clock_timestamp_str) {
  let wall_ms = 1000000;
  let current_obj = snapshot_obj(valid_ms, clock_timestamp_str);
  const handler_dict = {};
  let timer_fn;
  const document_obj = {
    activeElement: null,
    getElementById: () => current_obj.shell_obj,
    querySelectorAll: () => [],
    addEventListener: (name_str, handler_fn) => { handler_dict[name_str] = handler_fn; },
  };
  const window_obj = {addEventListener: document_obj.addEventListener};
  vm.runInNewContext(source_str, {
    document: document_obj,
    window: window_obj,
    Date: {now: () => wall_ms, parse: Date.parse}, performance: {now: () => initial_latency_ms},
    setInterval: (handler_fn) => { timer_fn = handler_fn; },
  });
  return {
    get current_obj() { return current_obj; },
    advance: (elapsed_ms) => { wall_ms += elapsed_ms; },
    timer: () => timer_fn(),
    fire: (name_str, extra_dict = {}) => {
      const event_obj = {detail: {target: current_obj.shell_obj}, defaultPrevented: false,
        preventDefault() { this.defaultPrevented = true; }, ...extra_dict};
      handler_dict[name_str](event_obj);
      return event_obj;
    },
    select: (selection_obj) => { window_obj.getSelection = () => selection_obj; },
    replace: (remaining_ms = 120000, next_clock_str) => { current_obj = snapshot_obj(remaining_ms, next_clock_str); },
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
  assert.equal(env_obj.current_obj.selector_dict['[data-cycle-verdict]'].textContent, 'Status unknown.');
  assert.equal(env_obj.current_obj.selector_dict['[data-refresh-reason]'].textContent, 'Saved status is out of date.');
  assert.equal(env_obj.current_obj.selector_dict['.step'].className, 'step is-unk');
  assert.equal(env_obj.current_obj.selector_dict['[data-step-fact]'].textContent, 'Unknown');
  assert.equal(env_obj.current_obj.selector_dict['[data-evidence-status]'].textContent, 'Unknown');
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
    assert.equal(env_obj.current_obj.selector_dict['.step'].className, 'step is-unk');
    assert.equal(env_obj.current_obj.selector_dict['[data-step-fact]'].textContent, 'Unknown');
    assert.equal(env_obj.current_obj.selector_dict['[data-evidence-status]'].textContent, 'Unknown');
    assert.equal(env_obj.current_obj.shell_obj.getAttribute('data-source-stale'), 'true');
    assert.equal(env_obj.current_obj.shell_obj.getAttribute('data-last-update'), '09:41:07');
  });
}

for (const trigger_str of ['expiry', 'htmx:responseError', 'htmx:sendError', 'htmx:timeout', 'htmx:swapError']) {
  test('Pod ' + trigger_str + ' clears all seven steps and every visible evidence status', () => {
    const env_obj = environment_obj(1000);
    const selector_dict = env_obj.current_obj.selector_dict;
    const step_list = ['done', 'now', 'next', 'late', 'fail', 'skip', 'unk'].map(
      (state_str) => element_obj('', 'step is-' + state_str + ' is-sel'));
    const fact_list = ['Data ready', '10 targets', '9 orders', '9 sent · 9 ack', '9 of 9', '0 diffs', 'Captured'].map(
      (fact_str) => element_obj(fact_str));
    const evidence_list = ['Acked', 'No ack', 'Unknown'].map((status_str) => element_obj(status_str));
    selector_dict['.step'] = step_list;
    selector_dict['[data-step-fact]'] = fact_list;
    selector_dict['[data-evidence-status]'] = evidence_list;
    if (trigger_str === 'expiry') {
      env_obj.advance(999); env_obj.timer();
      assert.equal(fact_list[4].textContent, '9 of 9');
      assert.equal(evidence_list[0].textContent, 'Acked');
      env_obj.advance(1); env_obj.timer();
    } else {
      env_obj.fire(trigger_str);
    }
    assert.ok(step_list.every((step_obj) => step_obj.className === 'step is-unk'));
    assert.ok(fact_list.every((fact_obj) => fact_obj.textContent === 'Unknown'));
    assert.ok(evidence_list.every((status_obj) => status_obj.textContent === 'Unknown'));
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
  assert.equal(env_obj.current_obj.selector_dict['.step'].className, 'step is-done is-sel');
  assert.equal(env_obj.current_obj.selector_dict['[data-step-fact]'].textContent, '9 of 9');
  assert.equal(env_obj.current_obj.selector_dict['[data-evidence-status]'].textContent, 'Acked');
  env_obj.advance(999); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Current');
  env_obj.advance(1); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
  assert.equal(env_obj.current_obj.selector_dict['.step'].className, 'step is-unk');
  assert.equal(env_obj.current_obj.selector_dict['[data-step-fact]'].textContent, 'Unknown');
  assert.equal(env_obj.current_obj.selector_dict['[data-evidence-status]'].textContent, 'Unknown');
});

test('both ET clocks advance each second from server time, without changing saved update time or expiry', () => {
  const env_obj = environment_obj(2000);
  const clock_list = env_obj.current_obj.selector_dict['[data-live-clock]'];
  assert.ok(clock_list.every((clock_obj) => clock_obj.textContent === '09:41:07 ET'));
  env_obj.advance(1000); env_obj.timer();
  assert.ok(clock_list.every((clock_obj) => clock_obj.textContent === '09:41:08 ET'));
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Current');
  env_obj.advance(1000); env_obj.timer();
  assert.ok(clock_list.every((clock_obj) => clock_obj.textContent === '09:41:09 ET'));
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
  assert.equal(env_obj.current_obj.shell_obj.getAttribute('data-last-update'), '09:41:07');
  assert.equal(env_obj.current_obj.shell_obj.getAttribute('data-source-valid-ms'), '2000');
});

for (const [timestamp_str, expected_str] of [
  ['2026-03-08T06:59:59+00:00', '03:00:01 ET'],
  ['2026-11-01T05:59:59+00:00', '01:00:01 ET'],
]) {
  test('ET clock follows the daylight-saving boundary from ' + timestamp_str, () => {
    const env_obj = environment_obj(120000, 0, timestamp_str);
    env_obj.advance(2000); env_obj.timer();
    assert.equal(env_obj.current_obj.selector_dict['[data-live-clock]'][0].textContent, expected_str);
  });
}

test('clock continues after failed refresh but cannot restore operating status', () => {
  const env_obj = environment_obj();
  env_obj.fire('htmx:sendError');
  env_obj.advance(1000); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-live-clock]'][0].textContent, '09:41:08 ET');
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
  assert.equal(env_obj.current_obj.shell_obj.getAttribute('data-source-stale'), 'true');
});

test('missing or invalid clock anchor leaves saved clock text intact without disabling expiry', () => {
  for (const timestamp_str of [null, 'bad timestamp']) {
    const env_obj = environment_obj(1000, 0, timestamp_str);
    env_obj.advance(1000); env_obj.timer();
    assert.equal(env_obj.current_obj.selector_dict['[data-live-clock]'][0].textContent, '09:41:07 ET');
    assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
  }
});

test('selection-canceled swaps preserve broker IDs but cannot extend TTL; clearing selection permits next poll', () => {
  const env_obj = environment_obj(1200);
  const selected_shell_obj = env_obj.current_obj.shell_obj;
  const selected_id_obj = env_obj.current_obj.selector_dict['[data-broker-id]'];
  env_obj.select({isCollapsed: false, rangeCount: 1,
    getRangeAt: () => ({intersectsNode: (shell_obj) => shell_obj === selected_shell_obj})});
  env_obj.fire('htmx:beforeRequest');
  env_obj.advance(900);
  const canceled_obj = env_obj.fire('htmx:beforeSwap');
  assert.equal(canceled_obj.defaultPrevented, true);
  assert.equal(canceled_obj.detail.shouldSwap, false);
  assert.equal(env_obj.current_obj.shell_obj, selected_shell_obj);
  // No shell was replaced. Even another swap event on this same DOM cannot renew its evidence.
  env_obj.fire('htmx:afterSwap');
  env_obj.advance(299); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Current');
  env_obj.advance(1); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
  assert.equal(env_obj.current_obj.selector_dict['[data-broker-id]'], selected_id_obj);
  assert.equal(selected_id_obj.textContent, 'perm:123456789');
  assert.equal(env_obj.current_obj.shell_obj.getAttribute('data-last-update'), '09:41:07');
  assert.equal(env_obj.fire('htmx:beforeSwap').defaultPrevented, true);
  env_obj.select(null);
  env_obj.fire('htmx:beforeRequest');
  env_obj.advance(200);
  assert.equal(env_obj.fire('htmx:beforeSwap').defaultPrevented, false);
  env_obj.replace(5000, '2026-09-18T13:41:30+00:00');
  env_obj.fire('htmx:afterSwap');
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Current');
  assert.equal(env_obj.current_obj.selector_dict['[data-live-clock]'][0].textContent, '09:41:30 ET');
  env_obj.advance(1000); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-live-clock]'][0].textContent, '09:41:31 ET');
  env_obj.advance(3800); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
});

test('absent, collapsed, empty, and outside selections do not block a shell refresh', () => {
  const env_obj = environment_obj();
  assert.equal(env_obj.fire('htmx:beforeSwap').defaultPrevented, false);
  for (const selection_obj of [null, {isCollapsed: true}, {isCollapsed: false, rangeCount: 0},
    {isCollapsed: false, rangeCount: 1, getRangeAt: () => ({intersectsNode: () => false})}]) {
    env_obj.select(selection_obj);
    assert.equal(env_obj.fire('htmx:beforeSwap').defaultPrevented, false);
  }
});

test('a selected range crossing the shell prevents replacement even when another range is outside', () => {
  const env_obj = environment_obj();
  env_obj.select({isCollapsed: false, rangeCount: 2,
    getRangeAt: (range_int) => ({intersectsNode: () => range_int === 1})});
  assert.equal(env_obj.fire('htmx:beforeSwap').defaultPrevented, true);
});

test('selection does not cancel HTMX error handling before its responseError event', () => {
  const env_obj = environment_obj();
  env_obj.select({isCollapsed: false, rangeCount: 1, getRangeAt: () => ({intersectsNode: () => true})});
  const error_obj = env_obj.fire('htmx:beforeSwap', {
    detail: {target: env_obj.current_obj.shell_obj, shouldSwap: false, isError: true},
  });
  assert.equal(error_obj.defaultPrevented, false);
  env_obj.fire('htmx:responseError');
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
  assert.equal(env_obj.current_obj.selector_dict['[data-broker-id]'].textContent, 'perm:123456789');
});
