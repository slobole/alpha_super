/* Activity is saved history. Filters and the last-looked marker live only in
   this browser; polling never changes the baseline of the current visit. */
(() => {
  'use strict';
  let page_obj = null;
  let storage_key_str = '';
  let baseline_ms = NaN;
  let invalid_saved_ms = NaN;
  let previous_visit_bool = false;
  let row_list = [];
  let focus_dict = null;
  const filter_dict = {pod_str: '', type_str: 'all', search_str: '', late_bool: false, codes_bool: false};
  const cycle_set = new Set();
  const evidence_set = new Set();
  const marker_formatter_obj = new Intl.DateTimeFormat('en-US', {timeZone: 'America/New_York',
    weekday: 'short', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23'});

  function saved_timestamp_ms() {
    try { return Date.parse(window.localStorage.getItem(storage_key_str)); }
    catch (error_obj) { return NaN; }
  }

  function save_observation() {
    if (!page_obj || document.hidden || page_obj.getAttribute('data-feed-available') !== 'true') return;
    const shell_obj = page_obj.closest('#overview-shell');
    if (shell_obj && shell_obj.getAttribute('data-source-stale') === 'true') return;
    const observed_ms = Date.parse(page_obj.getAttribute('data-as-of'));
    if (!Number.isFinite(observed_ms)) return;
    try {
      // A slower tab must not move a newer tab's saved observation backwards.
      const saved_ms = saved_timestamp_ms();
      if (!Number.isFinite(saved_ms) || observed_ms > saved_ms || saved_ms === invalid_saved_ms) {
        window.localStorage.setItem(storage_key_str, new Date(observed_ms).toISOString());
        invalid_saved_ms = NaN;
      }
    } catch (error_obj) { /* Blocked storage still permits this visit's filters. */ }
  }

  function render() {
    if (!page_obj) return;
    page_obj.querySelector('[data-activity-pod]').value = filter_dict.pod_str;
    page_obj.querySelector('[data-activity-search]').value = filter_dict.search_str;
    for (const button_obj of page_obj.querySelectorAll('[data-activity-type]')) {
      const selected_bool = button_obj.getAttribute('data-activity-type') === filter_dict.type_str;
      button_obj.classList.toggle('on', selected_bool);
      button_obj.setAttribute('aria-pressed', String(selected_bool));
    }
    for (const [attribute_str, selected_bool] of [['data-activity-late', filter_dict.late_bool], ['data-activity-codes', filter_dict.codes_bool]]) {
      const button_obj = page_obj.querySelector(`[${attribute_str}]`);
      button_obj.classList.toggle('on', selected_bool);
      button_obj.setAttribute('aria-pressed', String(selected_bool));
    }
    const search_str = filter_dict.search_str.trim().toLocaleLowerCase();
    const visible_list = row_list.filter(row_dict => !row_dict.parent_str
      && (!filter_dict.pod_str || row_dict.pod_str === filter_dict.pod_str || row_dict.related_pod_list.includes(filter_dict.pod_str))
      && (filter_dict.type_str === 'all' || row_dict.type_str === filter_dict.type_str)
      && (!filter_dict.late_bool || ['fail', 'late'].includes(row_dict.state_str))
      && (!search_str || row_dict.search_str.includes(search_str)));
    const visible_set = new Set(visible_list.map(row_dict => row_dict.id_str));
    const day_set = new Set(visible_list.map(row_dict => row_dict.day_str));
    for (const row_dict of row_list) {
      const visible_bool = row_dict.parent_str ? visible_set.has(row_dict.parent_str) && cycle_set.has(row_dict.parent_str) : visible_set.has(row_dict.id_str);
      row_dict.element_obj.hidden = !visible_bool;
      if (row_dict.proof_obj) row_dict.proof_obj.hidden = !visible_bool || !evidence_set.has(row_dict.id_str);
    }
    for (const group_obj of page_obj.querySelectorAll('[data-activity-day]')) group_obj.hidden = !day_set.has(group_obj.getAttribute('data-activity-day'));
    for (const code_obj of page_obj.querySelectorAll('[data-activity-code]')) code_obj.hidden = !filter_dict.codes_bool;
    for (const button_obj of page_obj.querySelectorAll('[data-activity-cycle]')) button_obj.setAttribute('aria-expanded', String(cycle_set.has(button_obj.getAttribute('data-activity-cycle'))));
    for (const button_obj of page_obj.querySelectorAll('[data-activity-evidence]')) button_obj.setAttribute('aria-expanded', String(evidence_set.has(button_obj.getAttribute('data-activity-evidence'))));
    const empty_obj = page_obj.querySelector('[data-activity-empty]');
    empty_obj.hidden = visible_list.length !== 0;
    empty_obj.querySelector('td').textContent = row_list.length ? 'No matching activity.' : 'No saved activity for this period.';

    const observed_ms = Date.parse(page_obj.getAttribute('data-as-of'));
    const new_list = visible_list.filter(row_dict => row_dict.timestamp_ms > baseline_ms && row_dict.timestamp_ms <= observed_ms);
    const count_list = [['fail', 'failed'], ['late', 'late']].map(([state_str, label_str]) => {
      const count_int = new_list.filter(row_dict => row_dict.state_str === state_str).length;
      return count_int ? `${count_int} ${label_str}` : '';
    }).filter(Boolean);
    const verdict_obj = page_obj.querySelector('[data-activity-verdict]');
    const detail_obj = page_obj.querySelector('[data-activity-verdict-detail]');
    verdict_obj.textContent = count_list.length ? count_list.join(', ') : new_list.length ? `${new_list.length} new event${new_list.length === 1 ? '' : 's'}.`
      : page_obj.getAttribute('data-complete') !== 'true' ? 'Activity may be incomplete.' : previous_visit_bool ? 'No new events.' : 'Recent activity.';
    detail_obj.textContent = count_list.length ? ' since you last looked.' : '';
    const marker_obj = page_obj.querySelector('[data-activity-marker]');
    marker_obj.hidden = !visible_list.length || !Number.isFinite(baseline_ms) || (!previous_visit_bool && !new_list.length);
    if (!marker_obj.hidden) {
      page_obj.querySelector('[data-activity-marker-label]').textContent = 'You last looked here · ' + marker_formatter_obj.format(new Date(baseline_ms)).replaceAll(',', '').replaceAll('/', '-') + ' ET';
      const older_dict = visible_list.find(row_dict => row_dict.timestamp_ms <= baseline_ms);
      page_obj.querySelector('[data-activity-body]').insertBefore(marker_obj, older_dict ? older_dict.element_obj : empty_obj);
    }
  }

  const focus_attribute_list = ['data-activity-search', 'data-activity-pod', 'data-activity-type',
    'data-activity-late', 'data-activity-codes', 'data-activity-cycle', 'data-activity-evidence'];
  function remember_focus() {
    const active_obj = document.activeElement;
    focus_dict = null;
    if (!page_obj || !active_obj || !page_obj.contains(active_obj)) return;
    const attribute_str = focus_attribute_list.find(value_str => active_obj.hasAttribute(value_str));
    if (attribute_str) focus_dict = {attribute_str, value_str: active_obj.getAttribute(attribute_str), old_obj: active_obj,
      start_int: active_obj.selectionStart, end_int: active_obj.selectionEnd};
  }

  function install_page() {
    const next_obj = document.querySelector('[data-activity-page]');
    if (!next_obj || next_obj === page_obj) return;
    page_obj = next_obj;
    const next_key_str = page_obj.getAttribute('data-storage-key');
    if (next_key_str !== storage_key_str) {
      storage_key_str = next_key_str;
      const observed_ms = Date.parse(page_obj.getAttribute('data-as-of'));
      const saved_ms = saved_timestamp_ms();
      previous_visit_bool = Number.isFinite(saved_ms) && saved_ms <= observed_ms;
      invalid_saved_ms = Number.isFinite(saved_ms) && saved_ms > observed_ms ? saved_ms : NaN;
      baseline_ms = previous_visit_bool ? saved_ms : observed_ms;
      Object.assign(filter_dict, {pod_str: '', type_str: 'all', search_str: '', late_bool: false, codes_bool: false});
      cycle_set.clear(); evidence_set.clear(); focus_dict = null;
    }
    const proof_map = new Map([...page_obj.querySelectorAll('[data-activity-proof]')].map(proof_obj => [proof_obj.getAttribute('data-activity-proof'), proof_obj]));
    row_list = [...page_obj.querySelectorAll('[data-activity-row]')].map(element_obj => {
      const id_str = element_obj.getAttribute('data-activity-row');
      const proof_obj = proof_map.get(id_str);
      return {id_str, element_obj, proof_obj, parent_str: element_obj.getAttribute('data-activity-parent'),
        timestamp_ms: Date.parse(element_obj.getAttribute('data-timestamp')), day_str: element_obj.getAttribute('data-day'),
        pod_str: element_obj.getAttribute('data-pod'), type_str: element_obj.getAttribute('data-type'), state_str: element_obj.getAttribute('data-state'),
        related_pod_list: JSON.parse(element_obj.getAttribute('data-related-pods') || '[]'),
        search_str: (element_obj.textContent + ' ' + (proof_obj ? proof_obj.textContent : '')).toLocaleLowerCase()};
    });
    const row_map = new Map(row_list.map(row_dict => [row_dict.id_str, row_dict]));
    for (const row_dict of row_list) if (row_dict.parent_str && row_map.has(row_dict.parent_str)) row_map.get(row_dict.parent_str).search_str += ' ' + row_dict.search_str;
    render();
    if (focus_dict && (!document.activeElement || document.activeElement === document.body || document.activeElement === focus_dict.old_obj)) {
      const control_obj = [...page_obj.querySelectorAll(`[${focus_dict.attribute_str}]`)].find(item_obj => item_obj.getAttribute(focus_dict.attribute_str) === focus_dict.value_str);
      if (control_obj) {
        control_obj.focus({preventScroll: true});
        if (focus_dict.attribute_str === 'data-activity-search' && Number.isInteger(focus_dict.start_int)) control_obj.setSelectionRange(focus_dict.start_int, focus_dict.end_int);
      }
    }
    focus_dict = null;
    save_observation();
  }

  document.addEventListener('input', event_obj => {
    if (page_obj && page_obj.contains(event_obj.target) && event_obj.target.hasAttribute('data-activity-search')) {
      filter_dict.search_str = event_obj.target.value; render();
    }
  });
  document.addEventListener('change', event_obj => {
    if (page_obj && page_obj.contains(event_obj.target) && event_obj.target.hasAttribute('data-activity-pod')) {
      filter_dict.pod_str = event_obj.target.value; render();
    }
  });
  document.addEventListener('click', event_obj => {
    const button_obj = event_obj.target.closest && event_obj.target.closest('button');
    if (!button_obj || !page_obj || !page_obj.contains(button_obj)) return;
    if (button_obj.hasAttribute('data-activity-type')) filter_dict.type_str = button_obj.getAttribute('data-activity-type');
    else if (button_obj.hasAttribute('data-activity-late')) filter_dict.late_bool = !filter_dict.late_bool;
    else if (button_obj.hasAttribute('data-activity-codes')) filter_dict.codes_bool = !filter_dict.codes_bool;
    else {
      const attribute_str = button_obj.hasAttribute('data-activity-cycle') ? 'data-activity-cycle' : 'data-activity-evidence';
      if (!button_obj.hasAttribute(attribute_str)) return;
      const id_str = button_obj.getAttribute(attribute_str);
      const opened_set = attribute_str === 'data-activity-cycle' ? cycle_set : evidence_set;
      opened_set.has(id_str) ? opened_set.delete(id_str) : opened_set.add(id_str);
    }
    render();
  });
  document.addEventListener('htmx:beforeSwap', event_obj => {
    if (event_obj.detail.target && event_obj.detail.target.id === 'overview-shell'
        && event_obj.detail.shouldSwap !== false && !event_obj.detail.isError) remember_focus();
  });
  document.addEventListener('htmx:afterSwap', event_obj => {
    if (event_obj.detail.target && event_obj.detail.target.id === 'overview-shell'
        && !event_obj.detail.isError && (!event_obj.detail.xhr || event_obj.detail.xhr.status < 400)) install_page();
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) save_observation(); });
  install_page();
})();
