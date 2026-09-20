/* Behavioral tests for expiry/transport recovery, using a tiny DOM double. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source_str = fs.readFileSync(path.join(__dirname, '../alpha/live/dashboard_v4/static/overview.js'), 'utf8');

function element_obj(text_str = '', class_str = '') {
  const result_obj = {textContent: text_str, className: class_str, hidden: true, title: '',
    nodeType: 1, parentElement: null, childNodes: [], isConnected: true};
  const attribute_dict = {};
  result_obj.getAttribute = (key_str) => attribute_dict[key_str] ?? null;
  result_obj.setAttribute = (key_str, value_str) => { attribute_dict[key_str] = value_str; };
  result_obj.closest = (selector_str) => {
    if ((selector_str === '[data-selection-key]' && result_obj.getAttribute('data-selection-key'))
        || (selector_str === '#overview-shell' && result_obj.id === 'overview-shell')) return result_obj;
    return result_obj.parentElement ? result_obj.parentElement.closest(selector_str) : null;
  };
  result_obj.contains = (node_obj) => {
    for (; node_obj; node_obj = node_obj.parentElement) if (node_obj === result_obj) return true;
    return false;
  };
  result_obj.querySelector = () => result_obj.childNodes.find((node_obj) => node_obj.nodeType === 1 && node_obj.getAttribute('data-selection-key')) || null;
  result_obj.classList = {remove: (...name_list) => {
    result_obj.className = result_obj.className.split(' ').filter((name_str) => !name_list.includes(name_str)).join(' ');
  }};
  return result_obj;
}

function region_obj(shell_obj, key_str, text_list = ['perm:', '123456789']) {
  const result_obj = element_obj(text_list.join(''));
  result_obj.parentElement = shell_obj;
  result_obj.setAttribute('data-selection-key', key_str);
  result_obj.childNodes = text_list.map((text_str) => ({nodeType: 3, textContent: text_str, parentElement: result_obj}));
  return result_obj;
}

function selection_obj(region_obj, anchor_int = 5, focus_int = 14) {
  return {rangeCount: 1, isCollapsed: anchor_int === focus_int, restored_int: 0,
    anchorNode: region_obj.childNodes[0], anchorOffset: anchor_int,
    focusNode: region_obj.childNodes[0], focusOffset: focus_int,
    setBaseAndExtent(anchor_obj, anchor_offset_int, focus_obj, focus_offset_int) {
      this.anchorNode = anchor_obj; this.anchorOffset = anchor_offset_int;
      this.focusNode = focus_obj; this.focusOffset = focus_offset_int;
      this.isCollapsed = anchor_obj === focus_obj && anchor_offset_int === focus_offset_int;
      this.rangeCount = 1; this.restored_int += 1;
    }};
}

function text_node_list(root_obj) {
  return root_obj.nodeType === 3 ? [root_obj] : root_obj.childNodes.flatMap(text_node_list);
}

function scheduler_details_obj(snapshot_obj, open_bool = false, command_str = 'next_due --pod-id pod_one') {
  const details_obj = element_obj();
  const code_obj = region_obj(snapshot_obj.shell_obj, 'scheduler-check', [command_str]);
  details_obj.querySelector = () => code_obj;
  if (open_bool) details_obj.setAttribute('open', '');
  snapshot_obj.selector_dict['.scheduler-check'] = details_obj;
  return details_obj;
}

test('open diagnostic remains open across refresh of the same Pod and command', () => {
  const env_obj = environment_obj();
  scheduler_details_obj(env_obj.current_obj, true);
  env_obj.fire('htmx:beforeSwap');
  env_obj.replace();
  const details_obj = scheduler_details_obj(env_obj.current_obj);
  env_obj.fire('htmx:afterSwap');
  assert.equal(details_obj.getAttribute('open'), '');
});

for (const change_str of ['scope', 'command', 'closed', 'error']) {
  test(`diagnostic does not reopen after ${change_str} changes`, () => {
    const env_obj = environment_obj();
    scheduler_details_obj(env_obj.current_obj, change_str !== 'closed');
    env_obj.fire('htmx:beforeSwap');
    if (change_str === 'error') env_obj.fire('htmx:responseError');
    env_obj.replace();
    const details_obj = scheduler_details_obj(env_obj.current_obj, false,
      change_str === 'command' ? 'another command' : 'next_due --pod-id pod_one');
    if (change_str === 'scope') env_obj.current_obj.shell_obj.setAttribute('data-selection-scope', 'pod:other');
    env_obj.fire('htmx:afterSwap');
    assert.equal(details_obj.getAttribute('open'), null);
  });
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
  const shell_obj = element_obj();
  shell_obj.id = 'overview-shell';
  shell_obj.setAttribute('data-source-valid-ms', String(valid_ms));
  shell_obj.setAttribute('data-last-update', '09:41:07');
  shell_obj.setAttribute('data-clock-timestamp', clock_timestamp_str);
  shell_obj.setAttribute('data-selection-scope', 'pod:demo:cycle-2:orders:3M');
  selector_dict['[data-broker-id]'] = region_obj(shell_obj, 'evidence:orders', ['perm:123456789']);
  selector_dict['[data-selection-key]'] = [selector_dict['[data-broker-id]']];
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
    createRange: () => {
      let root_obj, end_obj, end_int;
      return {selectNodeContents: (node_obj) => { root_obj = node_obj; },
        setEnd: (node_obj, offset_int) => { end_obj = node_obj; end_int = offset_int; },
        toString: () => {
          const node_list = text_node_list(root_obj);
          const end_index_int = node_list.indexOf(end_obj);
          assert.notEqual(end_index_int, -1);
          assert.ok(end_int >= 0 && end_int <= end_obj.textContent.length);
          return node_list.slice(0, end_index_int).map((node_obj) => node_obj.textContent).join('') + end_obj.textContent.slice(0, end_int);
        }};
    },
    createTreeWalker: (root_obj) => {
      const node_list = text_node_list(root_obj);
      let node_int = 0;
      return {nextNode: () => node_list[node_int++] || null};
    },
  };
  const window_obj = {addEventListener: document_obj.addEventListener};
  vm.runInNewContext(source_str, {
    document: document_obj,
    window: window_obj,
    Date: {now: () => wall_ms, parse: Date.parse}, performance: {now: () => initial_latency_ms},
    NodeFilter: {SHOW_TEXT: 4},
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
    replace: (remaining_ms = 120000, next_clock_str) => {
      current_obj.shell_obj.isConnected = false;
      const selected_obj = window_obj.getSelection ? window_obj.getSelection() : null;
      if (selected_obj && current_obj.shell_obj.contains(selected_obj.anchorNode)) {
        selected_obj.isCollapsed = true; selected_obj.anchorNode = null; selected_obj.focusNode = null;
      }
      current_obj = snapshot_obj(remaining_ms, next_clock_str);
    },
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

test('expired next operations cannot claim either saved-plan or forecast freshness', () => {
  for (const detail_str of ['Scheduled', 'Scheduled · when data is ready', 'in 00:01:00']) {
    const env_obj = environment_obj(1000);
    env_obj.current_obj.selector_dict['[data-pod-next]'].textContent = 'EOD 16:10:00';
    env_obj.current_obj.selector_dict['[data-next-detail]'].textContent = detail_str;
    env_obj.advance(1000); env_obj.timer();
    assert.equal(env_obj.current_obj.selector_dict['[data-pod-next]'].textContent, 'EOD 16:10:00');
    assert.equal(env_obj.current_obj.selector_dict['[data-next-detail]'].textContent, 'Not current');
    assert.equal(env_obj.current_obj.selector_dict['[data-pod-now]'].textContent, 'Unknown');
  }
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
    assert.equal(env_obj.current_obj.selector_dict['[data-next-detail]'].textContent, 'Not current');
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

test('fresh atomic refreshes keep applying while the same broker ID remains selected', () => {
  const env_obj = environment_obj(120000);
  const selected_obj = selection_obj(env_obj.current_obj.selector_dict['[data-broker-id]']);
  env_obj.select(selected_obj);
  for (let poll_int = 0; poll_int < 9; poll_int += 1) {
    env_obj.advance(14000);
    env_obj.fire('htmx:beforeRequest');
    env_obj.advance(1000);
    const swap_obj = env_obj.fire('htmx:beforeSwap');
    assert.equal(swap_obj.defaultPrevented, false);
    assert.notEqual(swap_obj.detail.shouldSwap, false);
    const old_shell_obj = env_obj.current_obj.shell_obj;
    env_obj.replace();
    env_obj.fire('htmx:afterSwap');
    assert.notEqual(env_obj.current_obj.shell_obj, old_shell_obj);
    assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Current');
    assert.equal(selected_obj.anchorNode, env_obj.current_obj.selector_dict['[data-broker-id]'].childNodes[0]);
    assert.equal(selected_obj.anchorNode.textContent.slice(selected_obj.anchorOffset, selected_obj.focusOffset), '123456789');
  }
  assert.equal(selected_obj.restored_int, 9);
  env_obj.advance(118999); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Current');
  env_obj.advance(1); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
});

test('selection offsets survive unchanged text split across new markup and retain backward direction', () => {
  const env_obj = environment_obj();
  const old_region_obj = env_obj.current_obj.selector_dict['[data-broker-id]'];
  const selected_obj = selection_obj(old_region_obj, 14, 5);
  env_obj.select(selected_obj);
  env_obj.fire('htmx:beforeSwap');
  env_obj.replace();
  const new_region_obj = region_obj(env_obj.current_obj.shell_obj, 'evidence:orders');
  env_obj.current_obj.selector_dict['[data-selection-key]'] = [new_region_obj];
  env_obj.fire('htmx:afterSwap');
  assert.equal(selected_obj.restored_int, 1);
  assert.equal(selected_obj.anchorNode, new_region_obj.childNodes[1]);
  assert.equal(selected_obj.anchorOffset, 9);
  assert.equal(selected_obj.focusNode, new_region_obj.childNodes[1]);
  assert.equal(selected_obj.focusOffset, 0);
});

for (const backward_bool of [false, true]) {
  for (const last_cell_bool of [false, true]) {
    test('exact cell selection retains endpoint nodes: backward=' + backward_bool + ', last=' + last_cell_bool, () => {
      const env_obj = environment_obj();
      const broker_id_str = '1184220320';
      const cell_text_list = ['Submitted', broker_id_str, ...(last_cell_bool ? [] : ['Acked'])];
      const old_region_obj = region_obj(env_obj.current_obj.shell_obj, 'evidence:orders', [cell_text_list.join('')]);
      env_obj.current_obj.selector_dict['[data-selection-key]'] = [old_region_obj];
      const start_int = cell_text_list[0].length;
      const end_int = start_int + broker_id_str.length;
      const selected_obj = selection_obj(old_region_obj, backward_bool ? end_int : start_int, backward_bool ? start_int : end_int);
      env_obj.select(selected_obj);
      env_obj.fire('htmx:beforeSwap');
      env_obj.replace();
      const new_region_obj = region_obj(env_obj.current_obj.shell_obj, 'evidence:orders', cell_text_list);
      new_region_obj.childNodes = new_region_obj.childNodes.map((text_obj) => {
        const cell_obj = element_obj(text_obj.textContent);
        cell_obj.parentElement = new_region_obj;
        cell_obj.childNodes = [text_obj];
        text_obj.parentElement = cell_obj;
        return cell_obj;
      });
      env_obj.current_obj.selector_dict['[data-selection-key]'] = [new_region_obj];
      env_obj.fire('htmx:afterSwap');
      const broker_node_obj = new_region_obj.childNodes[1].childNodes[0];
      assert.equal(selected_obj.restored_int, 1);
      // Text alone cannot detect browser-generated tabs between table cells.
      assert.equal(selected_obj.anchorNode, broker_node_obj);
      assert.equal(selected_obj.focusNode, broker_node_obj);
      assert.equal(selected_obj.anchorOffset, backward_bool ? broker_id_str.length : 0);
      assert.equal(selected_obj.focusOffset, backward_bool ? 0 : broker_id_str.length);
    });
  }
}

for (const change_str of ['text', 'missing', 'key', 'duplicate', 'scope', 'nested']) {
  test('fresh data applies without restoring a selection after ' + change_str + ' changes', () => {
    const env_obj = environment_obj();
    const selected_obj = selection_obj(env_obj.current_obj.selector_dict['[data-broker-id]']);
    env_obj.select(selected_obj);
    env_obj.fire('htmx:beforeSwap');
    env_obj.replace();
    const new_region_obj = env_obj.current_obj.selector_dict['[data-broker-id]'];
    if (change_str === 'text') new_region_obj.textContent = 'perm:987654321';
    if (change_str === 'missing') env_obj.current_obj.selector_dict['[data-selection-key]'] = [];
    if (change_str === 'key') new_region_obj.setAttribute('data-selection-key', 'evidence:fills');
    if (change_str === 'duplicate') env_obj.current_obj.selector_dict['[data-selection-key]'].push(region_obj(env_obj.current_obj.shell_obj, 'evidence:orders'));
    if (change_str === 'scope') env_obj.current_obj.shell_obj.setAttribute('data-selection-scope', 'pod:other:cycle-3:orders:3M');
    if (change_str === 'nested') new_region_obj.childNodes.push(region_obj(new_region_obj, 'nested:field', []));
    env_obj.fire('htmx:afterSwap');
    assert.equal(selected_obj.restored_int, 0);
    assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Current');
  });
}

for (const invalid_str of ['duplicate', 'multiple_ranges', 'multiple_regions', 'missing_scope', 'nested']) {
  test('ambiguous initial ' + invalid_str + ' selection is never saved', () => {
    const env_obj = environment_obj();
    const old_region_obj = env_obj.current_obj.selector_dict['[data-broker-id]'];
    const selected_obj = selection_obj(old_region_obj);
    if (invalid_str === 'duplicate') env_obj.current_obj.selector_dict['[data-selection-key]'].push(region_obj(env_obj.current_obj.shell_obj, 'evidence:orders'));
    if (invalid_str === 'multiple_ranges') selected_obj.rangeCount = 2;
    if (invalid_str === 'multiple_regions') selected_obj.focusNode = region_obj(env_obj.current_obj.shell_obj, 'another:field').childNodes[0];
    if (invalid_str === 'missing_scope') env_obj.current_obj.shell_obj.setAttribute('data-selection-scope', '');
    if (invalid_str === 'nested') old_region_obj.childNodes.push(region_obj(old_region_obj, 'nested:field', []));
    env_obj.select(selected_obj);
    assert.equal(env_obj.fire('htmx:beforeSwap').defaultPrevented, false);
    env_obj.replace();
    env_obj.fire('htmx:afterSwap');
    assert.equal(selected_obj.restored_int, 0);
  });
}

test('clearing or changing selection while a request is in flight never revives the old selection', () => {
  for (const clear_bool of [true, false]) {
    const env_obj = environment_obj();
    const selected_obj = selection_obj(env_obj.current_obj.selector_dict['[data-broker-id]']);
    env_obj.select(selected_obj);
    env_obj.fire('htmx:beforeRequest');
    if (clear_bool) selected_obj.isCollapsed = true;
    else { selected_obj.anchorOffset = 0; selected_obj.focusOffset = 4; }
    env_obj.fire('htmx:beforeSwap');
    env_obj.replace();
    env_obj.fire('htmx:afterSwap');
    assert.equal(selected_obj.restored_int, clear_bool ? 0 : 1);
    if (!clear_bool) assert.equal(selected_obj.anchorNode.textContent.slice(selected_obj.anchorOffset, selected_obj.focusOffset), 'perm');
  }
});

test('a selection change after capture invalidates a pending restore', () => {
  const env_obj = environment_obj();
  const selected_obj = selection_obj(env_obj.current_obj.selector_dict['[data-broker-id]']);
  env_obj.select(selected_obj);
  env_obj.fire('htmx:beforeSwap');
  selected_obj.isCollapsed = true;
  env_obj.fire('selectionchange');
  env_obj.replace();
  env_obj.fire('htmx:afterSwap');
  assert.equal(selected_obj.restored_int, 0);
});

test('a newer selection outside the replaced shell is not overwritten', () => {
  const env_obj = environment_obj();
  env_obj.select(selection_obj(env_obj.current_obj.selector_dict['[data-broker-id]']));
  env_obj.fire('htmx:beforeSwap');
  env_obj.replace();
  const newer_obj = selection_obj(region_obj(null, 'outside', ['different selection']), 0, 9);
  env_obj.select(newer_obj);
  env_obj.fire('htmx:afterSwap');
  assert.equal(newer_obj.restored_int, 0);
  assert.equal(newer_obj.anchorNode.textContent.slice(0, 9), 'different');
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

test('failed refresh cannot replay a captured selection or renew evidence lifetime', () => {
  const env_obj = environment_obj(1000);
  const selected_obj = selection_obj(env_obj.current_obj.selector_dict['[data-broker-id]']);
  env_obj.select(selected_obj);
  env_obj.fire('htmx:beforeSwap');
  env_obj.fire('htmx:swapError');
  env_obj.fire('htmx:afterSwap');
  assert.equal(selected_obj.restored_int, 0);
  assert.equal(env_obj.current_obj.selector_dict['[data-status-label]'].textContent, 'Unknown');
  env_obj.advance(1000); env_obj.timer();
  assert.equal(env_obj.current_obj.selector_dict['[data-refresh-reason]'].textContent, 'Saved status is out of date.');
});
