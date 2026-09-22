/* Show the exact saved slice label; no valuation or percentage calculation. */
(() => {
  'use strict';
  let focus_snapshot_obj = null;
  function selection_obj(target_obj) {
    if (!target_obj || !target_obj.closest) return null;
    const slice_obj = target_obj.closest('[data-donut-slice]');
    if (slice_obj) return {item_obj: slice_obj, slice_obj, inspector_obj: slice_obj.closest('[data-donut-inspector]')};
    const item_obj = target_obj.closest('[data-allocation-key]');
    const panel_obj = item_obj && item_obj.closest('[data-pod-allocation]');
    const inspector_obj = panel_obj && panel_obj.querySelector('[data-donut-inspector]');
    if (!inspector_obj) return null;
    const key_str = item_obj.getAttribute('data-allocation-key');
    const match_list = Array.from(inspector_obj.querySelectorAll('[data-donut-slice]'))
      .filter(peer_obj => peer_obj.getAttribute('data-allocation-key') === key_str);
    return {item_obj, inspector_obj, slice_obj: match_list.length === 1 ? match_list[0] : null};
  }

  function show_value(inspector_obj, slice_obj) {
    if (!inspector_obj) return;
    const readout_obj = inspector_obj.querySelector('[data-donut-value]');
    if (!readout_obj) return;
    readout_obj.textContent = slice_obj ? slice_obj.getAttribute('data-donut-readout') || '' : '';
    for (const peer_obj of inspector_obj.querySelectorAll('[data-donut-slice]')) {
      peer_obj.classList.toggle('is-donut-active', peer_obj === slice_obj);
    }
  }

  function inspect_slice(event_obj) {
    const selected_obj = selection_obj(event_obj.target);
    if (!selected_obj || !selected_obj.inspector_obj) return;
    const leaving_bool = event_obj.type === 'pointerout' || event_obj.type === 'focusout';
    if (leaving_bool && event_obj.relatedTarget && selected_obj.item_obj.contains(event_obj.relatedTarget)) return;
    let slice_obj = selected_obj.slice_obj;
    if (leaving_bool) {
      const next_obj = selection_obj(event_obj.relatedTarget);
      const active_obj = selection_obj(document.activeElement);
      slice_obj = next_obj && next_obj.inspector_obj === selected_obj.inspector_obj ? next_obj.slice_obj
        : active_obj && active_obj.inspector_obj === selected_obj.inspector_obj
          && (event_obj.type === 'pointerout' || active_obj.item_obj !== selected_obj.item_obj) ? active_obj.slice_obj : null;
    } else if (event_obj.type === 'click' && selected_obj.slice_obj && selected_obj.item_obj.focus) {
      // Touch taps keep their value readable until focus moves elsewhere.
      selected_obj.item_obj.focus({preventScroll: true});
    }
    show_value(selected_obj.inspector_obj, slice_obj);
  }

  // Delegation also handles the next HTMX refresh and restored Pod-row focus.
  for (const event_str of ['pointerover', 'pointerout', 'focusin', 'focusout', 'click']) {
    document.addEventListener(event_str, inspect_slice);
  }

  function shell_swap_bool(event_obj) {
    return event_obj.detail && event_obj.detail.target && event_obj.detail.target.id === 'overview-shell';
  }

  function matching_slice_list(shell_obj, snapshot_obj) {
    const inspector_list = Array.from(shell_obj.querySelectorAll('[data-donut-id]'))
      .filter(item_obj => item_obj.getAttribute('data-donut-id') === snapshot_obj.id_str);
    if (inspector_list.length !== 1 || inspector_list[0].getAttribute('data-donut-date') !== snapshot_obj.date_str) return [];
    return Array.from(inspector_list[0].querySelectorAll('[data-donut-key]'))
      .filter(item_obj => item_obj.getAttribute('data-donut-key') === snapshot_obj.key_str);
  }

  document.addEventListener('htmx:beforeSwap', event_obj => {
    if (!shell_swap_bool(event_obj)) return;
    focus_snapshot_obj = null;
    if (event_obj.detail.shouldSwap === false || event_obj.detail.isError) return;
    const shell_obj = document.getElementById('overview-shell');
    const selected_obj = selection_obj(document.activeElement);
    // Pod slices and rows retain their existing restoration in overview.js.
    if (!shell_obj || !selected_obj || !selected_obj.slice_obj || !shell_obj.contains(selected_obj.item_obj)
        || selected_obj.item_obj.closest('[data-pod-allocation]')) return;
    const snapshot_obj = {shell_obj, active_obj: selected_obj.item_obj, scope_str: shell_obj.getAttribute('data-selection-scope'),
      id_str: selected_obj.inspector_obj.getAttribute('data-donut-id'), date_str: selected_obj.inspector_obj.getAttribute('data-donut-date'),
      key_str: selected_obj.slice_obj.getAttribute('data-donut-key')};
    if (snapshot_obj.scope_str && snapshot_obj.id_str && snapshot_obj.date_str && snapshot_obj.key_str
        && matching_slice_list(shell_obj, snapshot_obj).length === 1) focus_snapshot_obj = snapshot_obj;
  });

  document.addEventListener('htmx:afterSwap', event_obj => {
    if (!shell_swap_bool(event_obj)) return;
    const snapshot_obj = focus_snapshot_obj;
    focus_snapshot_obj = null;
    if (!snapshot_obj || event_obj.detail.shouldSwap === false || event_obj.detail.isError
        || (event_obj.detail.xhr && event_obj.detail.xhr.status >= 400)) return;
    const shell_obj = document.getElementById('overview-shell');
    if (!shell_obj || shell_obj === snapshot_obj.shell_obj || shell_obj.getAttribute('data-source-stale') === 'true'
        || shell_obj.getAttribute('data-selection-scope') !== snapshot_obj.scope_str) return;
    const active_obj = document.activeElement;
    if (active_obj && active_obj !== document.body && active_obj !== snapshot_obj.active_obj) return;
    const match_list = matching_slice_list(shell_obj, snapshot_obj);
    if (match_list.length === 1) match_list[0].focus({preventScroll: true});
  });

  for (const event_str of ['htmx:responseError', 'htmx:sendError', 'htmx:timeout', 'htmx:swapError']) {
    document.addEventListener(event_str, () => { focus_snapshot_obj = null; });
  }
})();
