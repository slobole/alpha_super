/* ISO date selection is navigation only; the financial body does not poll. */
(() => {
  'use strict';
  function valid_date_bool(value_str) {
    if (!/^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(value_str)) return false;
    const [year_int, month_int, day_int] = value_str.split('-').map(Number);
    if (year_int < 1 || month_int < 1 || month_int > 12) return false;
    const leap_bool = year_int % 4 === 0 && (year_int % 100 !== 0 || year_int % 400 === 0);
    const month_days_list = [31, leap_bool ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    return day_int >= 1 && day_int <= month_days_list[month_int - 1];
  }
  document.addEventListener('click', (event_obj) => {
    if (!event_obj.target.closest || !event_obj.target.closest('[data-performance-apply]')) return;
    const panel_obj = document.querySelector('[data-performance-dates]');
    if (!panel_obj) return;
    const from_obj = panel_obj.querySelector('[name="from"]');
    const to_obj = panel_obj.querySelector('[name="to"]');
    for (const input_obj of [from_obj, to_obj]) {
      const minimum_str = input_obj.getAttribute('data-min-date');
      const maximum_str = input_obj.getAttribute('data-max-date');
      input_obj.setCustomValidity(!valid_date_bool(input_obj.value) ? 'Use a valid date: YYYY-MM-DD.' :
        minimum_str && input_obj.value < minimum_str ? 'Choose a date within the available history.' :
        maximum_str && input_obj.value > maximum_str ? 'Choose today or an earlier date.' : '');
      if (!input_obj.reportValidity()) return;
    }
    if (from_obj.value > to_obj.value) {
      to_obj.setCustomValidity('Choose an end date on or after the start.');
      to_obj.reportValidity();
      return;
    }
    const url_obj = new URL(panel_obj.getAttribute('data-date-url'), window.location.origin);
    url_obj.searchParams.set('from', from_obj.value);
    url_obj.searchParams.set('to', to_obj.value);
    window.location.assign(url_obj.href);
  });
  document.addEventListener('input', (event_obj) => {
    if (!event_obj.target.closest || !event_obj.target.closest('[data-performance-dates]')) return;
    const panel_obj = document.querySelector('[data-performance-dates]');
    if (panel_obj) {
      panel_obj.querySelector('[name="from"]').setCustomValidity('');
      panel_obj.querySelector('[name="to"]').setCustomValidity('');
    }
  });
})();

/* Daily values are server-rendered; interaction selects facts, never calculates. */
(() => {
  'use strict';
  function panel_for(event_obj) {
    return event_obj.target.closest && event_obj.target.closest('[data-daily-panel]');
  }
  function show_day(panel_obj, day_obj) {
    const readout_obj = panel_obj.querySelector('[data-daily-readout]');
    if (!readout_obj) return;
    const source_obj = day_obj || panel_obj;
    readout_obj.querySelector('[data-daily-last]').textContent = day_obj ? '' : 'Last ';
    readout_obj.querySelector('[data-daily-date]').textContent = source_obj.getAttribute('data-date-label');
    readout_obj.querySelector('[data-daily-return]').textContent = source_obj.getAttribute('data-return');
    const value_obj = readout_obj.querySelector('[data-daily-value]');
    value_obj.textContent = source_obj.getAttribute('data-pnl');
    for (const tone_str of ['pos', 'neg']) value_obj.classList.toggle(tone_str, source_obj.getAttribute('data-tone') === tone_str);
    const date_str = day_obj && day_obj.getAttribute('data-daily-day');
    const chart_obj = panel_obj.querySelector('.daily-chart');
    if (chart_obj) chart_obj.classList.toggle('is-active', Boolean(date_str));
    for (const mark_obj of panel_obj.querySelectorAll('[data-daily-paint], [data-daily-day]')) {
      const selected_bool = date_str && (mark_obj.getAttribute('data-daily-paint') || mark_obj.getAttribute('data-daily-day')) === date_str;
      mark_obj.classList.toggle('is-active', Boolean(selected_bool));
    }
  }
  function select_day(event_obj) {
    const panel_obj = panel_for(event_obj);
    const day_obj = panel_obj && event_obj.target.closest('[data-daily-day]');
    if (day_obj) {
      const view_obj = day_obj.closest('[data-daily-view]');
      for (const peer_obj of view_obj.querySelectorAll('[data-daily-day]')) peer_obj.tabIndex = peer_obj === day_obj ? 0 : -1;
      show_day(panel_obj, day_obj);
    }
  }
  document.addEventListener('pointerover', select_day);
  document.addEventListener('focusin', select_day);
  for (const event_str of ['pointerout', 'focusout']) document.addEventListener(event_str, (event_obj) => {
    const panel_obj = panel_for(event_obj);
    if (!panel_obj || (event_obj.relatedTarget && panel_obj.contains(event_obj.relatedTarget)
        && event_obj.relatedTarget.closest('[data-daily-day]'))) return;
    const focused_obj = event_str === 'focusout' ? event_obj.relatedTarget : document.activeElement;
    show_day(panel_obj, focused_obj && panel_obj.contains(focused_obj) && focused_obj.matches('[data-daily-day]') ? focused_obj : null);
  });
  document.addEventListener('click', (event_obj) => {
    const panel_obj = panel_for(event_obj);
    if (!panel_obj) return;
    const mode_obj = event_obj.target.closest('[data-daily-mode]');
    if (mode_obj) {
      const mode_str = mode_obj.getAttribute('data-daily-mode');
      for (const view_obj of panel_obj.querySelectorAll('[data-daily-view]')) view_obj.hidden = view_obj.getAttribute('data-daily-view') !== mode_str;
      for (const button_obj of panel_obj.querySelectorAll('[data-daily-mode]')) {
        const selected_bool = button_obj === mode_obj;
        button_obj.classList.toggle('on', selected_bool);
        button_obj.setAttribute('aria-pressed', String(selected_bool));
      }
      show_day(panel_obj, null);
    } else select_day(event_obj);
  });
  document.addEventListener('keydown', (event_obj) => {
    const panel_obj = panel_for(event_obj);
    const day_obj = panel_obj && event_obj.target.closest('[data-daily-day]');
    if (!day_obj || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event_obj.key)) return;
    const day_list = [...day_obj.closest('[data-daily-view]').querySelectorAll('[data-daily-day]')];
    const index_int = day_list.indexOf(day_obj);
    const next_int = event_obj.key === 'Home' ? 0 : event_obj.key === 'End' ? day_list.length - 1
      : Math.max(0, Math.min(day_list.length - 1, index_int + (event_obj.key === 'ArrowLeft' ? -1 : 1)));
    event_obj.preventDefault();
    day_list.forEach((item_obj, item_int) => { item_obj.tabIndex = item_int === next_int ? 0 : -1; });
    day_list[next_int].focus();
  });
})();
