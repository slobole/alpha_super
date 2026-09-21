/* Date selection is navigation only. Keep an unfinished edit across refreshes. */
(() => {
  'use strict';
  let dates_snapshot_obj = null;
  function dates_panel() {
    return document.querySelector('[data-performance-dates]');
  }
  document.addEventListener('click', (event_obj) => {
    if (!event_obj.target.closest || !event_obj.target.closest('[data-performance-apply]')) return;
    const panel_obj = dates_panel();
    if (!panel_obj) return;
    const from_obj = panel_obj.querySelector('[name="from"]');
    const to_obj = panel_obj.querySelector('[name="to"]');
    to_obj.setCustomValidity('');
    if (!from_obj.reportValidity() || !to_obj.reportValidity()) return;
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
    const panel_obj = dates_panel();
    if (panel_obj) panel_obj.querySelector('[name="to"]').setCustomValidity('');
  });
  document.addEventListener('htmx:beforeSwap', (event_obj) => {
    if (!event_obj.detail || event_obj.detail.shouldSwap === false || event_obj.detail.isError) return;
    const shell_obj = document.getElementById('overview-shell');
    if (event_obj.detail.target !== shell_obj) return;
    const panel_obj = dates_panel();
    if (!panel_obj) return;
    dates_snapshot_obj = {scope_str: shell_obj.getAttribute('data-selection-scope'), value_dict: {}, focus_str: ''};
    for (const input_obj of panel_obj.querySelectorAll('input')) {
      dates_snapshot_obj.value_dict[input_obj.name] = input_obj.value;
      if (document.activeElement === input_obj) dates_snapshot_obj.focus_str = input_obj.name;
    }
    if (document.activeElement && document.activeElement.hasAttribute('data-performance-apply')) dates_snapshot_obj.focus_str = 'apply';
  });
  document.addEventListener('htmx:afterSwap', () => {
    const snapshot_obj = dates_snapshot_obj;
    dates_snapshot_obj = null;
    const shell_obj = document.getElementById('overview-shell');
    const panel_obj = dates_panel();
    if (!snapshot_obj || !panel_obj || shell_obj.getAttribute('data-selection-scope') !== snapshot_obj.scope_str) return;
    for (const input_obj of panel_obj.querySelectorAll('input')) {
      input_obj.value = snapshot_obj.value_dict[input_obj.name];
      if (input_obj.name === snapshot_obj.focus_str) input_obj.focus({preventScroll: true});
    }
    if (snapshot_obj.focus_str === 'apply') panel_obj.querySelector('[data-performance-apply]').focus({preventScroll: true});
  });
})();
