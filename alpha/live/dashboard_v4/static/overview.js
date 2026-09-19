/* A single HTMX response replaces the whole observation atomically.
   Failed transport must never leave the previous observation looking current. */
(() => {
  'use strict';
  let focus_period_str = '';
  let observed_shell_obj = null;
  let valid_until_ms = 0;
  let request_start_ms = Date.now();

  function mark_unknown(reason_str = 'Update failed.') {
    const shell_obj = document.getElementById('overview-shell');
    if (!shell_obj) return;
    shell_obj.setAttribute('data-source-stale', 'true');
    const failure_obj = shell_obj.querySelector('.refresh-error');
    if (failure_obj) failure_obj.hidden = false;
    const reason_obj = shell_obj.querySelector('[data-refresh-reason]');
    if (reason_obj) reason_obj.textContent = reason_str;
    shell_obj.querySelectorAll('[data-observed-state]').forEach((mark_obj) => {
      const label_str = mark_obj.getAttribute('aria-label') || '';
      const step_str = label_str.includes(' · ') ? label_str.split(' · ')[0] + ' · ' : '';
      mark_obj.className = 'st st-unk';
      mark_obj.title = step_str + 'Unknown';
      mark_obj.setAttribute('aria-label', step_str + 'Unknown');
    });
    shell_obj.querySelectorAll('.tk').forEach((step_obj) => {
      step_obj.className = 'tk plan';
      const name_obj = step_obj.querySelector('.tl');
      step_obj.title = (name_obj ? name_obj.textContent : 'Step') + ' · Unknown';
    });
    shell_obj.querySelectorAll('[data-pod-pill]').forEach((pill_obj) => {
      pill_obj.className = 'pill pill-unk';
    });
    shell_obj.querySelectorAll('[data-pill-label], [data-pod-now], [data-status-label]').forEach((label_obj) => {
      label_obj.textContent = 'Unknown';
      label_obj.classList.remove('neg', 'amb');
    });
    shell_obj.querySelectorAll('[data-now-detail]').forEach((label_obj) => {
      label_obj.textContent = reason_str;
    });
    shell_obj.querySelectorAll('[data-next-detail]').forEach((label_obj) => {
      label_obj.textContent = 'Last saved plan';
    });
    shell_obj.querySelectorAll('[data-pod-next]').forEach((label_obj) => {
      label_obj.classList.remove('neg', 'amb');
    });
    shell_obj.querySelectorAll('[data-status-detail], [data-verdict-detail]').forEach((label_obj) => {
      label_obj.textContent = '';
    });
    const verdict_obj = shell_obj.querySelector('[data-verdict]');
    if (verdict_obj) verdict_obj.textContent = 'Status unknown.';
  }

  function check_expiry() {
    if (observed_shell_obj && Date.now() >= valid_until_ms) mark_unknown('Saved status is out of date.');
  }

  function observe_snapshot(elapsed_ms) {
    observed_shell_obj = document.getElementById('overview-shell');
    if (!observed_shell_obj) return;
    const remaining_ms = Number(observed_shell_obj.getAttribute('data-source-valid-ms'));
    valid_until_ms = Date.now() + Math.max(0, (Number.isFinite(remaining_ms) ? remaining_ms : 0) - elapsed_ms);
    check_expiry();
  }

  function overview_event(event_obj) {
    const target_obj = event_obj.detail && (event_obj.detail.target || event_obj.detail.elt);
    return target_obj && (target_obj.id === 'overview-shell' || target_obj.closest('#overview-shell'));
  }

  ['htmx:responseError', 'htmx:sendError', 'htmx:timeout', 'htmx:swapError'].forEach((event_str) => {
    document.addEventListener(event_str, (event_obj) => {
      if (overview_event(event_obj)) mark_unknown();
    });
  });
  document.addEventListener('htmx:beforeRequest', (event_obj) => {
    if (!overview_event(event_obj)) return;
    request_start_ms = Date.now();
    const active_obj = document.activeElement;
    focus_period_str = active_obj ? active_obj.getAttribute('data-period') || '' : '';
  });
  document.addEventListener('htmx:afterSwap', (event_obj) => {
    if (overview_event(event_obj)) observe_snapshot(Math.max(0, Date.now() - request_start_ms));
  });
  document.addEventListener('htmx:afterSettle', (event_obj) => {
    if (!overview_event(event_obj) || !focus_period_str) return;
    const period_list = document.querySelectorAll('#overview-shell [data-period]');
    for (const period_obj of period_list) {
      if (period_obj.getAttribute('data-period') === focus_period_str) {
        period_obj.focus({preventScroll: true});
        break;
      }
    }
    focus_period_str = '';
  });
  // Subtract the whole acquisition interval, conservatively, so transport or
  // a sleeping tab cannot extend the server's 120-second evidence lifetime.
  observe_snapshot(performance.now());
  setInterval(check_expiry, 1000);
  document.addEventListener('visibilitychange', check_expiry);
  window.addEventListener('pageshow', (event_obj) => {
    if (event_obj.persisted) mark_unknown('Refresh saved status.');
    else check_expiry();
  });
})();
