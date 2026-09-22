/* Dated chart inspection only. Values and plot coordinates come from the server. */
(() => {
  'use strict';
  let focus_snapshot_obj = null;
  let reset_focus_obj = null;
  const pressed_map = new WeakMap();
  const tabstop_map = new WeakMap();

  function chart_day_obj(node_obj) {
    const day_obj = node_obj && node_obj.closest && node_obj.closest('[data-chart-day]');
    return day_obj && day_obj.closest('[data-history-chart]') ? day_obj : null;
  }

  function day_list(chart_obj) {
    return Array.from(chart_obj.querySelectorAll('[data-chart-day]'));
  }

  function default_day_obj(chart_obj) {
    return day_list(chart_obj).find(day_obj => day_obj.getAttribute('data-chart-day') === chart_obj.getAttribute('data-chart-default'));
  }

  function percent_bool(value_float) {
    return typeof value_float === 'number' && Number.isFinite(value_float) && value_float >= 0 && value_float <= 100;
  }

  function set_tabstop(chart_obj, day_obj) {
    if (!tabstop_map.has(chart_obj)) tabstop_map.set(chart_obj, day_list(chart_obj).find(peer_obj => peer_obj.tabIndex === 0));
    const previous_obj = tabstop_map.get(chart_obj);
    if (previous_obj === day_obj) return;
    if (previous_obj) previous_obj.tabIndex = -1;
    day_obj.tabIndex = 0;
    tabstop_map.set(chart_obj, day_obj);
  }

  function show_day(chart_obj, day_obj, selected_bool) {
    if (!day_obj) return;
    let value_list;
    try { value_list = JSON.parse(day_obj.getAttribute('data-chart-values')); } catch { value_list = []; }
    if (!Array.isArray(value_list)) value_list = [];
    const date_obj = chart_obj.querySelector('[data-chart-date]');
    if (date_obj) date_obj.textContent = day_obj.getAttribute('data-date-label') || day_obj.getAttribute('data-chart-day');
    for (const item_obj of chart_obj.querySelectorAll('[data-chart-value], [data-chart-marker]')) {
      const index_str = item_obj.getAttribute('data-series-index');
      const value_dict = /^\d+$/.test(index_str || '') ? value_list[Number(index_str)] : null;
      const available_bool = value_dict && value_dict.available_bool === true;
      const color_str = value_dict && typeof value_dict.color_str === 'string' ? value_dict.color_str : '';
      if (item_obj.getAttribute('data-chart-marker') !== null) {
        item_obj.hidden = !(selected_bool && available_bool && percent_bool(value_dict.x_percent_float) && percent_bool(value_dict.y_percent_float));
        if (!item_obj.hidden) {
          item_obj.style.left = value_dict.x_percent_float + '%';
          item_obj.style.top = value_dict.y_percent_float + '%';
          item_obj.style.backgroundColor = color_str;
        }
      } else {
        const name_obj = item_obj.querySelector('[data-chart-name]');
        const number_obj = item_obj.querySelector('[data-chart-number]');
        const key_obj = item_obj.querySelector('[data-chart-key]');
        if (name_obj && value_dict && typeof value_dict.name_str === 'string') name_obj.textContent = value_dict.name_str;
        if (number_obj) number_obj.textContent = available_bool && typeof value_dict.label_str === 'string' ? value_dict.label_str : '—';
        if (key_obj && color_str) key_obj.style.backgroundColor = color_str;
      }
    }
    const crosshair_obj = chart_obj.querySelector('[data-chart-crosshair]');
    const x_str = day_obj.getAttribute('data-chart-x');
    const x_float = x_str === null || x_str.trim() === '' ? NaN : Number(x_str);
    if (crosshair_obj) {
      crosshair_obj.hidden = !(selected_bool && percent_bool(x_float));
      if (!crosshair_obj.hidden) crosshair_obj.style.left = x_float + '%';
    }
    // All history can contain thousands of dates. Change only the two affected buttons.
    const previous_obj = pressed_map.get(chart_obj);
    const selected_obj = selected_bool ? day_obj : null;
    if (previous_obj && previous_obj !== selected_obj) previous_obj.setAttribute('aria-pressed', 'false');
    if (selected_obj && selected_obj !== previous_obj) selected_obj.setAttribute('aria-pressed', 'true');
    pressed_map.set(chart_obj, selected_obj);
  }

  function focus_day(day_obj, reset_bool = false) {
    const chart_obj = day_obj.closest('[data-history-chart]');
    reset_focus_obj = reset_bool ? day_obj : null;
    set_tabstop(chart_obj, day_obj);
    day_obj.focus({preventScroll: true});
    show_day(chart_obj, day_obj, !reset_bool);
  }

  for (const event_str of ['pointerover', 'focusin']) document.addEventListener(event_str, event_obj => {
    const day_obj = chart_day_obj(event_obj.target);
    if (!day_obj) return;
    const chart_obj = day_obj.closest('[data-history-chart]');
    if (event_str === 'pointerover') reset_focus_obj = null;
    if (event_str === 'focusin') set_tabstop(chart_obj, day_obj);
    show_day(chart_obj, day_obj, day_obj !== reset_focus_obj);
  });

  for (const event_str of ['pointerout', 'focusout']) document.addEventListener(event_str, event_obj => {
    const day_obj = chart_day_obj(event_obj.target);
    if (!day_obj || (event_obj.relatedTarget && day_obj.contains(event_obj.relatedTarget))) return;
    const chart_obj = day_obj.closest('[data-history-chart]');
    const related_day_obj = chart_day_obj(event_obj.relatedTarget);
    if (related_day_obj && chart_obj.contains(related_day_obj)) return;
    const focused_day_obj = chart_day_obj(event_str === 'focusout' ? event_obj.relatedTarget : document.activeElement);
    if (focused_day_obj && chart_obj.contains(focused_day_obj)) show_day(chart_obj, focused_day_obj, focused_day_obj !== reset_focus_obj);
    else show_day(chart_obj, default_day_obj(chart_obj), false);
  });

  document.addEventListener('click', event_obj => {
    const day_obj = chart_day_obj(event_obj.target);
    if (day_obj) focus_day(day_obj);
  });

  document.addEventListener('keydown', event_obj => {
    const day_obj = chart_day_obj(event_obj.target);
    if (!day_obj || !['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Escape'].includes(event_obj.key)) return;
    const chart_obj = day_obj.closest('[data-history-chart]');
    const peer_list = day_list(chart_obj);
    event_obj.preventDefault();
    if (event_obj.key === 'Escape') {
      const default_obj = default_day_obj(chart_obj);
      if (default_obj) focus_day(default_obj, true);
      return;
    }
    const current_int = peer_list.indexOf(day_obj);
    const next_int = event_obj.key === 'Home' ? 0 : event_obj.key === 'End' ? peer_list.length - 1
      : Math.max(0, Math.min(peer_list.length - 1, current_int + (event_obj.key === 'ArrowLeft' ? -1 : 1)));
    focus_day(peer_list[next_int]);
  });

  function shell_swap_bool(event_obj) {
    return event_obj.detail && event_obj.detail.target && event_obj.detail.target.id === 'overview-shell';
  }

  document.addEventListener('htmx:beforeSwap', event_obj => {
    if (!shell_swap_bool(event_obj)) return;
    focus_snapshot_obj = null;
    if (event_obj.detail.shouldSwap === false || event_obj.detail.isError) return;
    const shell_obj = document.getElementById('overview-shell');
    const active_obj = chart_day_obj(document.activeElement);
    if (!shell_obj || !active_obj || !shell_obj.contains(active_obj)) return;
    const chart_obj = active_obj.closest('[data-history-chart]');
    const scope_str = shell_obj.getAttribute('data-selection-scope');
    const chart_id_str = chart_obj.getAttribute('data-chart-id');
    const match_list = Array.from(shell_obj.querySelectorAll('[data-history-chart]')).filter(item_obj => item_obj.getAttribute('data-chart-id') === chart_id_str);
    if (scope_str && chart_id_str && match_list.length === 1) focus_snapshot_obj = {
      shell_obj, active_obj, scope_str, chart_id_str, date_str: active_obj.getAttribute('data-chart-day'), reset_bool: active_obj === reset_focus_obj,
    };
  });

  document.addEventListener('htmx:afterSwap', event_obj => {
    if (!shell_swap_bool(event_obj)) return;
    const snapshot_obj = focus_snapshot_obj;
    focus_snapshot_obj = null;
    if (!snapshot_obj || event_obj.detail.isError || event_obj.detail.shouldSwap === false
        || (event_obj.detail.xhr && event_obj.detail.xhr.status >= 400)) return;
    const shell_obj = document.getElementById('overview-shell');
    if (!shell_obj || shell_obj === snapshot_obj.shell_obj || shell_obj.getAttribute('data-source-stale') === 'true'
        || shell_obj.getAttribute('data-selection-scope') !== snapshot_obj.scope_str) return;
    // A removed button normally returns focus to body. Never replace new user focus.
    const active_obj = document.activeElement;
    if (active_obj && active_obj !== document.body && active_obj !== snapshot_obj.active_obj) return;
    const chart_list = Array.from(shell_obj.querySelectorAll('[data-history-chart]')).filter(item_obj => item_obj.getAttribute('data-chart-id') === snapshot_obj.chart_id_str);
    if (chart_list.length !== 1) return;
    const match_list = day_list(chart_list[0]).filter(day_obj => day_obj.getAttribute('data-chart-day') === snapshot_obj.date_str);
    if (match_list.length === 1) focus_day(match_list[0], snapshot_obj.reset_bool);
  });

  for (const event_str of ['htmx:responseError', 'htmx:sendError', 'htmx:timeout', 'htmx:swapError']) {
    document.addEventListener(event_str, () => { focus_snapshot_obj = null; });
  }
})();
