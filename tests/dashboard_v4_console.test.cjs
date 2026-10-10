/* Behavioral tests for the Console pane with a tiny DOM double: bounded rows,
   folding, polling cadence, hidden tabs and failure backoff. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source_str = fs.readFileSync(path.join(__dirname, '../alpha/live/dashboard_v4/static/console.js'), 'utf8');

function node_obj(tag_str = 'div') {
  const attribute_dict = {};
  const class_set = new Set();
  const result_obj = {
    tagName: tag_str.toUpperCase(), children: [], hidden: false, parent_obj: null, listener_dict: {},
    scrollTop: 0, clientHeight: 400, offsetParent: {},
    get className() { return Array.from(class_set).join(' '); },
    set className(value_str) { class_set.clear(); String(value_str).split(' ').filter(Boolean).forEach((name_str) => class_set.add(name_str)); },
    classList: {
      toggle: (name_str, on_bool) => { if (on_bool === undefined ? !class_set.has(name_str) : on_bool) class_set.add(name_str); else class_set.delete(name_str); },
      contains: (name_str) => class_set.has(name_str),
    },
    _text_str: '',
    get textContent() { return this.children.length ? this.children.map((child_obj) => child_obj.textContent).join('') : this._text_str; },
    set textContent(value_str) { this.children = []; this._text_str = String(value_str); },
    get childElementCount() { return this.children.length; },
    get firstElementChild() { return this.children[0] || null; },
    get lastElementChild() { return this.children[this.children.length - 1] || null; },
    get scrollHeight() { return this.children.length * 20; },
    setAttribute: (key_str, value_str) => { attribute_dict[key_str] = String(value_str); },
    getAttribute: (key_str) => (key_str in attribute_dict ? attribute_dict[key_str] : null),
    append(...item_list) {
      for (const item_obj of item_list) {
        const added_list = item_obj.is_fragment_bool ? item_obj.children.splice(0) : [item_obj];
        for (const child_obj of added_list) { child_obj.parent_obj = this; this.children.push(child_obj); }
      }
    },
    remove() { if (this.parent_obj) this.parent_obj.children.splice(this.parent_obj.children.indexOf(this), 1); },
    querySelector(selector_str) {
      if (selector_str.startsWith('.')) return this.children.find((child_obj) => child_obj.classList.contains(selector_str.slice(1))) || null;
      return null;
    },
    addEventListener(name_str, handler_fn) { this.listener_dict[name_str] = handler_fn; },
  };
  Object.defineProperty(result_obj, 'innerHTML', {set() { throw new Error('innerHTML must not be used'); }});
  return result_obj;
}

function environment_obj() {
  let now_ms = 1_000_000;
  const timer_list = [];
  const fetch_list = [];
  const response_list = [];
  const document_listener_dict = {};
  const element_dict = {};
  for (const key_str of ['body', 'status', 'new', 'find', 'quiet', 'pause', 'copy']) element_dict[key_str] = node_obj();
  element_dict.find.value = '';
  element_dict.quiet.checked = true;
  const level_list = ['all', 'warn', 'error'].map((level_str) => {
    const button_obj = node_obj('button');
    button_obj.setAttribute('data-console-level', level_str);
    return button_obj;
  });
  const panel_obj = node_obj('section');
  panel_obj.setAttribute('data-console-tail-url', '/console/pod_a/tail');
  panel_obj.querySelector = (selector_str) => ({
    '[data-console-body]': element_dict.body, '[data-console-status]': element_dict.status,
    '[data-console-new]': element_dict.new, '[data-console-find]': element_dict.find,
    '[data-console-quiet]': element_dict.quiet, '[data-console-pause]': element_dict.pause,
    '[data-console-copy]': element_dict.copy,
  })[selector_str] || null;
  panel_obj.querySelectorAll = (selector_str) => (selector_str === '[data-console-level]' ? level_list : []);
  const document_obj = {
    hidden: false,
    getElementById: (id_str) => (id_str === 'console-panel' ? panel_obj : null),
    createElement: (tag_str) => node_obj(tag_str),
    createDocumentFragment: () => Object.assign(node_obj('fragment'), {is_fragment_bool: true}),
    addEventListener: (name_str, handler_fn) => { document_listener_dict[name_str] = handler_fn; },
  };
  const window_obj = {
    setTimeout: (handler_fn, delay_ms) => { timer_list.push({handler_fn, delay_ms, at_ms: now_ms + delay_ms, done_bool: false}); return timer_list.length; },
    clearTimeout: (id_int) => { if (id_int && timer_list[id_int - 1]) timer_list[id_int - 1].done_bool = true; },
    setInterval: () => 0,
    localStorage: {getItem: () => null, setItem: () => {}},
  };
  vm.runInNewContext(source_str, {
    document: document_obj, window: window_obj, navigator: {},
    Date: {now: () => now_ms, parse: Date.parse}, Intl, Number, Math, Array, String, Boolean, encodeURIComponent,
    AbortController: class { constructor() { this.signal = {}; } abort() { this.aborted_bool = true; } },
    fetch: (url_str) => {
      fetch_list.push(url_str);
      const next_obj = response_list.shift() || {status: 200, body_dict: {line_list: [], cursor_str: 'c'}};
      if (next_obj.error_bool) return Promise.reject(new Error('offline'));
      return Promise.resolve({status: next_obj.status, ok: next_obj.status >= 200 && next_obj.status < 300,
        headers: {get: (key_str) => (next_obj.header_dict || {})[key_str] || null},
        json: () => Promise.resolve(next_obj.body_dict)});
    },
  });
  return {
    element_dict, fetch_list, timer_list, document_obj,
    queue: (response_obj) => response_list.push(response_obj),
    pending_list: () => timer_list.filter((timer_obj) => !timer_obj.done_bool),
    async run_next() {
      const timer_obj = timer_list.filter((item_obj) => !item_obj.done_bool).sort((a_obj, b_obj) => a_obj.at_ms - b_obj.at_ms)[0];
      assert.ok(timer_obj, 'a timer is scheduled');
      timer_obj.done_bool = true;
      now_ms = Math.max(now_ms, timer_obj.at_ms);
      await timer_obj.handler_fn();
      for (let index_int = 0; index_int < 5; index_int += 1) await Promise.resolve();
      return timer_obj.delay_ms;
    },
    visibility(hidden_bool) { document_obj.hidden = hidden_bool; document_listener_dict.visibilitychange(); },
  };
}

function line_dict(index_int, extra_dict = {}) {
  return {t: '2026-10-09T13:00:00+00:00', l: 'I', a: 'fill.ok', x: 'pod=pod_a vplan=' + index_int, ...extra_dict};
}

test('first load renders rows as text and replaces the pane', async () => {
  const env_obj = environment_obj();
  env_obj.queue({status: 200, body_dict: {cursor_str: 'v1.a.10', line_list: [line_dict(1, {x: '<img src=x onerror=alert(1)>'})]}});
  await env_obj.run_next();
  const body_obj = env_obj.element_dict.body;
  assert.equal(body_obj.childElementCount, 1);
  assert.equal(body_obj.children[0].querySelector('.cx').textContent, '<img src=x onerror=alert(1)>');
  assert.equal(env_obj.fetch_list[0], '/console/pod_a/tail');
  env_obj.queue({status: 200, body_dict: {cursor_str: 'v1.a.20', line_list: [line_dict(2)]}});
  await env_obj.run_next();
  assert.equal(env_obj.fetch_list[1], '/console/pod_a/tail?cursor=v1.a.10');
  assert.equal(body_obj.childElementCount, 2);
});

test('the pane never keeps more than 1,000 rows', async () => {
  const env_obj = environment_obj();
  for (let batch_int = 0; batch_int < 3; batch_int += 1) {
    env_obj.queue({status: 200, body_dict: {cursor_str: 'v1.a.' + batch_int,
      line_list: Array.from({length: 600}, (_, index_int) => line_dict(batch_int * 600 + index_int))}});
    await env_obj.run_next();
  }
  const body_obj = env_obj.element_dict.body;
  assert.equal(body_obj.childElementCount, 1000);
  assert.equal(body_obj.lastElementChild.querySelector('.cx').textContent, 'pod=pod_a vplan=1799');
});

test('identical repeated lines fold into one row with a count', async () => {
  const env_obj = environment_obj();
  const repeat_dict = {l: 'E', a: 'cycle.fail', x: 'pod=pod_a reason=refused retry_in=60s'};
  env_obj.queue({status: 200, body_dict: {cursor_str: 'v1.a.1', line_list: [
    line_dict(0, {...repeat_dict, t: '2026-10-09T13:00:00+00:00'}),
    line_dict(0, {...repeat_dict, t: '2026-10-09T13:01:00+00:00'}),
    line_dict(0, {...repeat_dict, t: '2026-10-09T13:02:00+00:00'})]}});
  await env_obj.run_next();
  const body_obj = env_obj.element_dict.body;
  assert.equal(body_obj.childElementCount, 1);
  assert.equal(body_obj.children[0].querySelector('.cn').textContent, '×3 · to 09:02:00');
});

test('reset replaces the buffer; markers are never folded', async () => {
  const env_obj = environment_obj();
  env_obj.queue({status: 200, body_dict: {cursor_str: 'v1.a.1', line_list: [line_dict(1), line_dict(2)]}});
  await env_obj.run_next();
  env_obj.queue({status: 200, body_dict: {cursor_str: 'v1.b.1', reset_bool: true,
    line_list: [{t: null, l: 'M', a: '', x: 'Log rotated'}, line_dict(3)]}});
  await env_obj.run_next();
  const body_obj = env_obj.element_dict.body;
  assert.equal(body_obj.childElementCount, 2);
  assert.ok(body_obj.children[0].classList.contains('l-M'));
});

test('cadence: 2 s while lines arrive, slower when idle, catch-up when more is pending', async () => {
  const env_obj = environment_obj();
  env_obj.queue({status: 200, body_dict: {cursor_str: 'v1.a.1', line_list: [line_dict(1)], more_bool: true}});
  await env_obj.run_next();
  assert.equal(env_obj.pending_list()[0].delay_ms, 200);
  env_obj.queue({status: 200, body_dict: {cursor_str: 'v1.a.2', line_list: [line_dict(2)]}});
  await env_obj.run_next();
  assert.equal(env_obj.pending_list()[0].delay_ms, 2000);
  for (let index_int = 0; index_int < 3; index_int += 1) {
    env_obj.queue({status: 200, body_dict: {cursor_str: 'v1.a.2', line_list: []}});
    await env_obj.run_next();
  }
  assert.equal(env_obj.pending_list()[0].delay_ms, 5000);
});

test('a hidden tab stops polling and polls at once when visible again', async () => {
  const env_obj = environment_obj();
  await env_obj.run_next();
  env_obj.visibility(true);
  assert.equal(env_obj.pending_list().length, 0);
  const fetch_count_int = env_obj.fetch_list.length;
  env_obj.visibility(false);
  assert.equal(env_obj.pending_list()[0].delay_ms, 0);
  await env_obj.run_next();
  assert.equal(env_obj.fetch_list.length, fetch_count_int + 1);
});

test('failures back off and 429 obeys Retry-After', async () => {
  const env_obj = environment_obj();
  const delay_list = [];
  for (let index_int = 0; index_int < 5; index_int += 1) {
    env_obj.queue({error_bool: true});
    await env_obj.run_next();
    delay_list.push(env_obj.pending_list()[0].delay_ms);
  }
  assert.deepEqual(delay_list, [5000, 10000, 30000, 60000, 60000]);
  assert.ok(env_obj.element_dict.status.textContent.startsWith('Offline'));
  env_obj.queue({status: 429, header_dict: {'Retry-After': '7'}});
  await env_obj.run_next();
  assert.equal(env_obj.pending_list()[0].delay_ms, 7000);
  assert.ok(env_obj.element_dict.status.textContent.startsWith('Busy'));
});

test('pause stops polling until resumed', async () => {
  const env_obj = environment_obj();
  await env_obj.run_next();
  env_obj.element_dict.pause.listener_dict.click();
  assert.equal(env_obj.pending_list().length, 0);
  assert.equal(env_obj.element_dict.pause.textContent, 'Resume');
  env_obj.element_dict.pause.listener_dict.click();
  assert.equal(env_obj.pending_list()[0].delay_ms, 0);
});

test('pausing during a request is not reported as offline and adds no backoff', async () => {
  const env_obj = environment_obj();
  await env_obj.run_next();
  env_obj.queue({error_bool: true});
  const timer_obj = env_obj.pending_list()[0];
  timer_obj.done_bool = true;
  const poll_promise = timer_obj.handler_fn();
  env_obj.element_dict.pause.listener_dict.click();
  await poll_promise;
  for (let index_int = 0; index_int < 5; index_int += 1) await Promise.resolve();
  assert.ok(env_obj.element_dict.status.textContent.startsWith('Paused'));
  env_obj.element_dict.pause.listener_dict.click();
  env_obj.queue({error_bool: true});
  await env_obj.run_next();
  assert.equal(env_obj.pending_list()[0].delay_ms, 5000);  // First real failure: first backoff step.
});
