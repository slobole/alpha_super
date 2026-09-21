const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source_str = fs.readFileSync(path.join(__dirname, '../alpha/live/dashboard_v4/static/performance.js'), 'utf8');

function environment_obj() {
  const listener_dict = {};
  const document_obj = {activeElement: null, addEventListener(name_str, callback_fn) { (listener_dict[name_str] ||= []).push(callback_fn); }};
  let panel_obj, shell_obj, navigated_str = '';
  function replace(scope_str = 'performance:portfolio:All') {
    shell_obj = {getAttribute: () => scope_str};
    const input_list = ['from', 'to'].map((name_str) => ({name: name_str, value: '2026-09-01', error_str: '', valid_bool: true,
      minimum_str: '2020-01-01', maximum_str: '2026-09-08',
      getAttribute(name_str) { return name_str === 'data-min-date' ? this.minimum_str : this.maximum_str; },
      setCustomValidity(message_str) { this.error_str = message_str; }, reportValidity() { return this.valid_bool && !this.error_str; },
      focus() { document_obj.activeElement = this; }, closest(selector_str) { return selector_str === '[data-performance-dates]' ? panel_obj : null; }, hasAttribute() { return false; }}));
    const button_obj = {closest(selector_str) { return selector_str === '[data-performance-apply]' ? this : selector_str === '[data-performance-dates]' ? panel_obj : null; }, hasAttribute() { return true; }, focus() { document_obj.activeElement = this; }};
    panel_obj = {getAttribute: () => '/performance?level=pods&unit=pct', querySelectorAll: () => input_list,
      querySelector(selector_str) { return selector_str.includes('from') ? input_list[0] : selector_str.includes('to') ? input_list[1] : button_obj; }};
    document_obj.activeElement = null;
  }
  replace();
  document_obj.querySelector = () => panel_obj;
  document_obj.getElementById = () => shell_obj;
  vm.runInNewContext(source_str, {document: document_obj, URL,
    window: {location: {origin: 'http://127.0.0.1:8114', assign: (value_str) => { navigated_str = value_str; }}}});
  return {replace, document_obj, listener_list: Object.keys(listener_dict), get from_obj() { return panel_obj.querySelector('[name="from"]'); },
    get to_obj() { return panel_obj.querySelector('[name="to"]'); }, get navigated_str() { return navigated_str; },
    fire(name_str, detail_dict = {}) { listener_dict[name_str].forEach(callback_fn => callback_fn({target: panel_obj.querySelector('[data-performance-apply]'),
      detail: {target: shell_obj, ...detail_dict}})); }};
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

test('failed input validity prevents navigation', () => {
  const env_obj = environment_obj();
  env_obj.to_obj.valid_bool = false;
  env_obj.fire('click');
  assert.equal(env_obj.navigated_str, '');
});

test('financial body has no date-draft refresh listeners', () => {
  const env_obj = environment_obj();
  assert.equal(env_obj.listener_list.some(name_str => name_str.startsWith('htmx:')), false);
});

for (const date_str of ['', '01/06/2026', '2026-6-01', '20260601', '2026-02-29', '2026-04-31', '2026-00-01', '2026-13-01', '2026-06-00', '0000-01-01', '2026-06-01T00:00:00']) {
  test(`invalid ISO date is rejected locally: ${date_str}`, () => {
    const env_obj = environment_obj();
    env_obj.from_obj.value = date_str;
    env_obj.fire('click');
    assert.equal(env_obj.navigated_str, '');
    assert.match(env_obj.from_obj.error_str, /YYYY-MM-DD/);
  });
}

test('leap day and equal dates preserve their exact ISO spelling', () => {
  const env_obj = environment_obj();
  env_obj.from_obj.value = env_obj.to_obj.value = '2024-02-29';
  env_obj.fire('click');
  const url_obj = new URL(env_obj.navigated_str);
  assert.equal(url_obj.searchParams.get('from'), '2024-02-29');
  assert.equal(url_obj.searchParams.get('to'), '2024-02-29');
});

for (const [date_str, valid_bool] of [['1900-02-29', false], ['2000-02-29', true]]) {
  test(`Gregorian century leap rule: ${date_str}`, () => {
    const env_obj = environment_obj();
    env_obj.from_obj.minimum_str = env_obj.to_obj.minimum_str = '';
    env_obj.from_obj.value = env_obj.to_obj.value = date_str;
    env_obj.fire('click');
    assert.equal(Boolean(env_obj.navigated_str), valid_bool);
  });
}

test('future and pre-history dates cannot bypass text-input bounds', () => {
  const env_obj = environment_obj();
  env_obj.to_obj.value = '2026-09-09';
  env_obj.fire('click');
  assert.equal(env_obj.navigated_str, '');
  assert.match(env_obj.to_obj.error_str, /today/);
  env_obj.to_obj.value = '2026-09-08';
  env_obj.from_obj.value = '2019-12-31';
  env_obj.fire('click');
  assert.equal(env_obj.navigated_str, '');
  assert.match(env_obj.from_obj.error_str, /available history/);
});
