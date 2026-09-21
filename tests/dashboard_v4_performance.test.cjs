const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source_str = fs.readFileSync(path.join(__dirname, '../alpha/live/dashboard_v4/static/performance.js'), 'utf8');

function environment_obj() {
  const listener_dict = {};
  const document_obj = {activeElement: null, addEventListener(name_str, callback_fn) { listener_dict[name_str] = callback_fn; }};
  let panel_obj, shell_obj, navigated_str = '';
  function replace(scope_str = 'performance:portfolio:All') {
    shell_obj = {getAttribute: () => scope_str};
    const input_list = ['from', 'to'].map((name_str) => ({name: name_str, value: '2026-09-01', error_str: '', valid_bool: true,
      setCustomValidity(message_str) { this.error_str = message_str; }, reportValidity() { return this.valid_bool && !this.error_str; },
      focus() { document_obj.activeElement = this; }, closest() { return panel_obj; }, hasAttribute() { return false; }}));
    const button_obj = {closest() { return this; }, hasAttribute() { return true; }, focus() { document_obj.activeElement = this; }};
    panel_obj = {getAttribute: () => '/performance?level=pods&unit=pct', querySelectorAll: () => input_list,
      querySelector(selector_str) { return selector_str.includes('from') ? input_list[0] : selector_str.includes('to') ? input_list[1] : button_obj; }};
    document_obj.activeElement = null;
  }
  replace();
  document_obj.querySelector = () => panel_obj;
  document_obj.getElementById = () => shell_obj;
  vm.runInNewContext(source_str, {document: document_obj, URL,
    window: {location: {origin: 'http://127.0.0.1:8114', assign: (value_str) => { navigated_str = value_str; }}}});
  return {replace, document_obj, get from_obj() { return panel_obj.querySelector('[name="from"]'); },
    get to_obj() { return panel_obj.querySelector('[name="to"]'); }, get navigated_str() { return navigated_str; },
    fire(name_str, detail_dict = {}) { listener_dict[name_str]({target: panel_obj.querySelector('[data-performance-apply]'),
      detail: {target: shell_obj, ...detail_dict}}); }};
}

test('Apply navigates to exact dates while retaining the selected level and unit', () => {
  const env_obj = environment_obj();
  env_obj.from_obj.value = '2026-06-02';
  env_obj.to_obj.value = '2026-08-31';
  env_obj.fire('click');
  const url_obj = new URL(env_obj.navigated_str);
  assert.equal(url_obj.origin, 'http://127.0.0.1:8114');
  assert.deepEqual(Object.fromEntries(url_obj.searchParams), {level: 'pods', unit: 'pct', from: '2026-06-02', to: '2026-08-31'});
});

test('reversed dates stay local and editing clears their validation error', () => {
  const env_obj = environment_obj();
  env_obj.from_obj.value = '2026-09-03';
  env_obj.to_obj.value = '2026-09-01';
  env_obj.fire('click');
  assert.equal(env_obj.navigated_str, '');
  assert.match(env_obj.to_obj.error_str, /end date/);
  env_obj.fire('input');
  assert.equal(env_obj.to_obj.error_str, '');
});

test('native invalid date or future-date validation prevents navigation', () => {
  const env_obj = environment_obj();
  env_obj.to_obj.valid_bool = false;
  env_obj.fire('click');
  assert.equal(env_obj.navigated_str, '');
});

test('unfinished edits and keyboard focus survive the same-period refresh', () => {
  const env_obj = environment_obj();
  env_obj.from_obj.value = '2026-08-01';
  env_obj.to_obj.value = '2026-08-19';
  env_obj.from_obj.focus();
  env_obj.fire('htmx:beforeSwap');
  env_obj.replace();
  env_obj.fire('htmx:afterSwap');
  assert.equal(env_obj.from_obj.value, '2026-08-01');
  assert.equal(env_obj.to_obj.value, '2026-08-19');
  assert.equal(env_obj.document_obj.activeElement, env_obj.from_obj);
});

for (const case_str of ['scope', 'failed', 'cancelled']) {
  test(`date drafts do not cross ${case_str} swaps`, () => {
    const env_obj = environment_obj();
    env_obj.from_obj.value = '2026-06-01';
    env_obj.fire('htmx:beforeSwap', case_str === 'failed' ? {isError: true} : case_str === 'cancelled' ? {shouldSwap: false} : {});
    env_obj.replace(case_str === 'scope' ? 'performance:pods:All' : undefined);
    env_obj.fire('htmx:afterSwap');
    assert.equal(env_obj.from_obj.value, '2026-09-01');
  });
}
