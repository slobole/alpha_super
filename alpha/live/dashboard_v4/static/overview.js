/* HTMX replaces operational observations; Performance keeps its dated report.
   Failed transport must never leave the previous observation looking current. */
(() => {
  'use strict';
  // Delegation also covers the next atomic refresh without rebinding listeners.
  function highlight_allocation(event_obj) {
    const item_obj = event_obj.target.closest && event_obj.target.closest('[data-allocation-key]');
    const panel_obj = item_obj && item_obj.closest('[data-pod-allocation]');
    if (!panel_obj) return;
    const leaving_bool = event_obj.type === 'pointerout' || event_obj.type === 'focusout';
    const related_obj = event_obj.relatedTarget;
    if (leaving_bool && related_obj && item_obj.contains(related_obj)) return;
    const active_obj = document.activeElement && document.activeElement.closest
      && document.activeElement.closest('[data-allocation-key]');
    const selected_obj = leaving_bool ? (active_obj && panel_obj.contains(active_obj)
      && (event_obj.type === 'pointerout' || active_obj !== item_obj) ? active_obj : null) : item_obj;
    const key_str = selected_obj ? selected_obj.getAttribute('data-allocation-key') : null;
    for (const peer_obj of panel_obj.querySelectorAll('[data-allocation-key]')) {
      peer_obj.classList.toggle('is-highlighted', key_str !== null && peer_obj.getAttribute('data-allocation-key') === key_str);
    }
  }
  for (const event_str of ['pointerover', 'pointerout', 'focusin', 'focusout']) {
    document.addEventListener(event_str, highlight_allocation);
  }
  let focus_period_str = '';
  let observed_shell_obj = null;
  let observed_source_obj = null;
  let valid_until_ms = 0;
  let request_start_ms = Date.now();
  let clock_anchor_ms = NaN;
  let clock_observed_ms = 0;
  let selection_snapshot_obj = null;
  let scheduler_check_snapshot_obj = null;
  let positions_search_snapshot_obj = null;
  let allocation_focus_snapshot_obj = null;
  const clock_formatter_obj = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
  });

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
    shell_obj.querySelectorAll('.step').forEach((step_obj) => {
      step_obj.className = 'step is-unk';
    });
    shell_obj.querySelectorAll('[data-step-fact], [data-evidence-status]').forEach((label_obj) => {
      label_obj.textContent = 'Unknown';
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
      label_obj.textContent = 'Not current';
    });
    shell_obj.querySelectorAll('[data-pod-next]').forEach((label_obj) => {
      label_obj.classList.remove('neg', 'amb');
    });
    shell_obj.querySelectorAll('[data-status-detail], [data-verdict-detail]').forEach((label_obj) => {
      label_obj.textContent = '';
    });
    shell_obj.querySelectorAll('[data-verdict], [data-cycle-verdict]').forEach((verdict_obj) => {
      verdict_obj.textContent = 'Status unknown.';
    });
  }

  function check_expiry() {
    if (observed_shell_obj && Date.now() >= valid_until_ms) mark_unknown('Saved status is out of date.');
  }

  function update_clock() {
    // Display time only: never renew saved operational evidence from this ticker.
    if (!observed_shell_obj || !Number.isFinite(clock_anchor_ms)) return;
    const clock_str = clock_formatter_obj.format(clock_anchor_ms + Math.max(0, Date.now() - clock_observed_ms)) + ' ET';
    observed_shell_obj.querySelectorAll('[data-live-clock]').forEach((clock_obj) => {
      if (clock_obj.textContent !== clock_str) clock_obj.textContent = clock_str;
    });
  }

  function observe_snapshot(elapsed_ms) {
    const shell_obj = document.getElementById('overview-shell');
    if (!shell_obj) return;
    const source_obj = shell_obj.querySelector('#performance-status') || shell_obj;
    if (shell_obj === observed_shell_obj && source_obj === observed_source_obj) return;
    observed_shell_obj = shell_obj;
    observed_source_obj = source_obj;
    const remaining_ms = Number(source_obj.getAttribute('data-source-valid-ms'));
    valid_until_ms = Date.now() + Math.max(0, (Number.isFinite(remaining_ms) ? remaining_ms : 0) - elapsed_ms);
    clock_anchor_ms = Date.parse(source_obj.getAttribute('data-clock-timestamp'));
    clock_observed_ms = Date.now();
    if (source_obj !== shell_obj && valid_until_ms > Date.now()) {
      shell_obj.removeAttribute('data-source-stale');
      const failure_obj = shell_obj.querySelector('.refresh-error');
      if (failure_obj) failure_obj.hidden = true;
      shell_obj.querySelectorAll('[data-update-time]').forEach((label_obj) => {
        label_obj.textContent = source_obj.getAttribute('data-last-update');
      });
    }
    update_clock();
    check_expiry();
  }

  function selected_region_obj(node_obj) {
    const element_obj = node_obj && (node_obj.nodeType === 1 ? node_obj : node_obj.parentElement);
    return element_obj ? element_obj.closest('[data-selection-key]') : null;
  }

  function matching_region_list(shell_obj, key_str) {
    return Array.from(shell_obj.querySelectorAll('[data-selection-key]'))
      .filter((region_obj) => region_obj.getAttribute('data-selection-key') === key_str);
  }

  function capture_selection(shell_obj) {
    const selection_obj = typeof window.getSelection === 'function' ? window.getSelection() : null;
    const scope_str = shell_obj.getAttribute('data-selection-scope');
    if (!scope_str || !selection_obj || selection_obj.isCollapsed || selection_obj.rangeCount !== 1) return null;
    const region_obj = selected_region_obj(selection_obj.anchorNode);
    if (!region_obj || region_obj !== selected_region_obj(selection_obj.focusNode) || !shell_obj.contains(region_obj)
        || region_obj.querySelector('[data-selection-key]')) return null;
    const key_str = region_obj.getAttribute('data-selection-key');
    if (!key_str || matching_region_list(shell_obj, key_str).length !== 1) return null;
    const prefix_range_obj = document.createRange();
    prefix_range_obj.selectNodeContents(region_obj);
    prefix_range_obj.setEnd(selection_obj.anchorNode, selection_obj.anchorOffset);
    const anchor_int = prefix_range_obj.toString().length;
    prefix_range_obj.setEnd(selection_obj.focusNode, selection_obj.focusOffset);
    const focus_int = prefix_range_obj.toString().length;
    if (anchor_int === focus_int) return null;
    return {shell_obj, scope_str, key_str, text_str: region_obj.textContent, anchor_int, focus_int,
      anchor_node_obj: selection_obj.anchorNode, anchor_offset_int: selection_obj.anchorOffset,
      focus_node_obj: selection_obj.focusNode, focus_offset_int: selection_obj.focusOffset};
  }

  function unchanged_selection_bool(selection_obj, snapshot_obj) {
    return selection_obj && !selection_obj.isCollapsed
      && selection_obj.anchorNode === snapshot_obj.anchor_node_obj && selection_obj.anchorOffset === snapshot_obj.anchor_offset_int
      && selection_obj.focusNode === snapshot_obj.focus_node_obj && selection_obj.focusOffset === snapshot_obj.focus_offset_int;
  }

  function text_position_obj(region_obj, offset_int, start_bool) {
    const walker_obj = document.createTreeWalker(region_obj, NodeFilter.SHOW_TEXT);
    let node_obj, last_node_obj;
    while ((node_obj = walker_obj.nextNode())) {
      // A start at a cell boundary belongs to the next node; an end belongs
      // to the preceding node, so copying does not add a table separator.
      if (offset_int < node_obj.textContent.length || (!start_bool && offset_int === node_obj.textContent.length)) return {node_obj, offset_int};
      offset_int -= node_obj.textContent.length;
      last_node_obj = node_obj;
    }
    return last_node_obj && offset_int === 0 ? {node_obj: last_node_obj, offset_int: last_node_obj.textContent.length} : null;
  }

  function restore_selection() {
    const snapshot_obj = selection_snapshot_obj;
    selection_snapshot_obj = null;
    const shell_obj = document.getElementById('overview-shell');
    const selection_obj = typeof window.getSelection === 'function' ? window.getSelection() : null;
    if (!snapshot_obj || !shell_obj || shell_obj === snapshot_obj.shell_obj || !selection_obj
        || typeof selection_obj.setBaseAndExtent !== 'function'
        || shell_obj.getAttribute('data-selection-scope') !== snapshot_obj.scope_str) return;
    // DOM removal normally collapses the selection. Never replace a new user selection.
    if (!selection_obj.isCollapsed && !unchanged_selection_bool(selection_obj, snapshot_obj)) return;
    const region_list = matching_region_list(shell_obj, snapshot_obj.key_str);
    if (region_list.length !== 1 || region_list[0].textContent !== snapshot_obj.text_str
        || region_list[0].querySelector('[data-selection-key]')) return;
    const anchor_obj = text_position_obj(region_list[0], snapshot_obj.anchor_int, snapshot_obj.anchor_int < snapshot_obj.focus_int);
    const focus_obj = text_position_obj(region_list[0], snapshot_obj.focus_int, snapshot_obj.focus_int < snapshot_obj.anchor_int);
    if (anchor_obj && focus_obj) selection_obj.setBaseAndExtent(anchor_obj.node_obj, anchor_obj.offset_int, focus_obj.node_obj, focus_obj.offset_int);
  }

  function overview_event(event_obj) {
    const target_obj = event_obj.detail && (event_obj.detail.target || event_obj.detail.elt);
    return target_obj && (target_obj.id === 'overview-shell' || target_obj.closest('#overview-shell'));
  }

  function filter_positions(shell_obj) {
    const input_obj = shell_obj && shell_obj.querySelector('[data-positions-search]');
    if (!input_obj) return;
    const search_str = input_obj.value.trim().toUpperCase();
    let visible_int = 0;
    shell_obj.querySelectorAll('[data-position-row]').forEach((row_obj) => {
      row_obj.hidden = !row_obj.getAttribute('data-position-symbol').toUpperCase().includes(search_str);
      if (!row_obj.hidden) visible_int += 1;
    });
    const empty_obj = shell_obj.querySelector('[data-position-search-empty]');
    if (empty_obj) empty_obj.hidden = visible_int !== 0;
  }

  function allocation_focus_match_list(shell_obj, snapshot_obj) {
    return Array.from(shell_obj.querySelectorAll('[data-allocation-key]')).filter((item_obj) => {
      const panel_obj = item_obj.closest('[data-pod-allocation]');
      return panel_obj && panel_obj.getAttribute('data-close-date') === snapshot_obj.close_date_str
        && item_obj.getAttribute('data-allocation-key') === snapshot_obj.key_str && item_obj.tagName === snapshot_obj.tag_str;
    });
  }

  function capture_allocation_focus(shell_obj) {
    const active_obj = document.activeElement;
    const panel_obj = active_obj && active_obj.closest && active_obj.closest('[data-pod-allocation]');
    if (!shell_obj || !panel_obj || !shell_obj.contains(active_obj)) return null;
    const snapshot_obj = {shell_obj, active_obj, scope_str: shell_obj.getAttribute('data-selection-scope'),
      close_date_str: panel_obj.getAttribute('data-close-date'), key_str: active_obj.getAttribute('data-allocation-key'), tag_str: active_obj.tagName};
    return snapshot_obj.scope_str && snapshot_obj.close_date_str && snapshot_obj.key_str && snapshot_obj.tag_str
      && allocation_focus_match_list(shell_obj, snapshot_obj).length === 1 ? snapshot_obj : null;
  }

  function restore_allocation_focus(shell_obj) {
    const snapshot_obj = allocation_focus_snapshot_obj;
    allocation_focus_snapshot_obj = null;
    if (!snapshot_obj || !shell_obj || shell_obj === snapshot_obj.shell_obj
        || shell_obj.getAttribute('data-selection-scope') !== snapshot_obj.scope_str) return;
    // DOM removal normally returns focus to the body. Do not replace new user focus.
    const active_obj = document.activeElement;
    if (active_obj && active_obj !== document.body && active_obj !== snapshot_obj.active_obj) return;
    const match_list = allocation_focus_match_list(shell_obj, snapshot_obj);
    if (match_list.length === 1) match_list[0].focus({preventScroll: true});
  }

  document.addEventListener('input', (event_obj) => {
    if (event_obj.target && event_obj.target.getAttribute('data-positions-search') !== null) {
      filter_positions(document.getElementById('overview-shell'));
    }
  });

  ['htmx:responseError', 'htmx:sendError', 'htmx:timeout', 'htmx:swapError'].forEach((event_str) => {
    document.addEventListener(event_str, (event_obj) => {
      if (overview_event(event_obj)) {
        selection_snapshot_obj = null;
        scheduler_check_snapshot_obj = null;
        positions_search_snapshot_obj = null;
        allocation_focus_snapshot_obj = null;
        mark_unknown();
      }
    });
  });
  document.addEventListener('htmx:beforeRequest', (event_obj) => {
    if (!overview_event(event_obj)) return;
    request_start_ms = Date.now();
    const active_obj = document.activeElement;
    focus_period_str = active_obj ? active_obj.getAttribute('data-period') || '' : '';
  });
  document.addEventListener('htmx:beforeSwap', (event_obj) => {
    if (!overview_event(event_obj) || event_obj.detail.shouldSwap === false || event_obj.detail.isError) return;
    const shell_obj = document.getElementById('overview-shell');
    // Capture at replacement time, not request start: a cleared selection stays cleared.
    selection_snapshot_obj = shell_obj ? capture_selection(shell_obj) : null;
    const check_obj = shell_obj && shell_obj.querySelector('.scheduler-check');
    const command_obj = check_obj && check_obj.querySelector('code');
    scheduler_check_snapshot_obj = check_obj && check_obj.getAttribute('open') !== null && command_obj
      ? {scope_str: shell_obj.getAttribute('data-selection-scope'), command_str: command_obj.textContent} : null;
    const input_obj = shell_obj && shell_obj.querySelector('[data-positions-search]');
    positions_search_snapshot_obj = input_obj ? {scope_str: shell_obj.getAttribute('data-selection-scope'),
      value_str: input_obj.value, focused_bool: document.activeElement === input_obj,
      start_int: input_obj.selectionStart, end_int: input_obj.selectionEnd} : null;
    allocation_focus_snapshot_obj = capture_allocation_focus(shell_obj);
    if (allocation_focus_snapshot_obj) focus_period_str = '';
  });
  document.addEventListener('htmx:afterSwap', (event_obj) => {
    if (!overview_event(event_obj)) return;
    observe_snapshot(Math.max(0, Date.now() - request_start_ms));
    const shell_obj = document.getElementById('overview-shell');
    const check_obj = shell_obj && shell_obj.querySelector('.scheduler-check');
    const command_obj = check_obj && check_obj.querySelector('code');
    if (scheduler_check_snapshot_obj && command_obj
        && shell_obj.getAttribute('data-selection-scope') === scheduler_check_snapshot_obj.scope_str
        && command_obj.textContent === scheduler_check_snapshot_obj.command_str) check_obj.setAttribute('open', '');
    scheduler_check_snapshot_obj = null;
    const input_obj = shell_obj && shell_obj.querySelector('[data-positions-search]');
    if (input_obj && positions_search_snapshot_obj
        && shell_obj.getAttribute('data-selection-scope') === positions_search_snapshot_obj.scope_str) {
      input_obj.value = positions_search_snapshot_obj.value_str;
      if (positions_search_snapshot_obj.focused_bool) {
        input_obj.focus({preventScroll: true});
        input_obj.setSelectionRange(positions_search_snapshot_obj.start_int, positions_search_snapshot_obj.end_int);
      }
    }
    positions_search_snapshot_obj = null;
    filter_positions(shell_obj);
    restore_selection();
    restore_allocation_focus(shell_obj);
  });
  document.addEventListener('htmx:afterRequest', (event_obj) => {
    // hx-swap=none still applies the header/rail out-of-band fragments. Only
    // a newly replaced stamp can renew freshness; errors cannot renew it.
    if (overview_event(event_obj) && event_obj.detail.successful) {
      observe_snapshot(Math.max(0, Date.now() - request_start_ms));
    }
  });
  document.addEventListener('selectionchange', () => {
    if (selection_snapshot_obj && selection_snapshot_obj.shell_obj.isConnected
        && !unchanged_selection_bool(window.getSelection(), selection_snapshot_obj)) selection_snapshot_obj = null;
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
  filter_positions(document.getElementById('overview-shell'));
  setInterval(() => {
    check_expiry();
    update_clock();
  }, 1000);
  document.addEventListener('visibilitychange', check_expiry);
  window.addEventListener('pageshow', (event_obj) => {
    if (event_obj.persisted) mark_unknown('Refresh saved status.');
    else check_expiry();
  });
})();
