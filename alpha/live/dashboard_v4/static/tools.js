/* Tools copies fixed catalog commands. The preview/run controls below only
   contact synthetic demo endpoints; production execution is not connected. */
(() => {
  'use strict';
  const action_set = new Set(['tick', 'submit_vplan', 'post_execution_reconcile', 'eod_snapshot', 'compare_reference', 'manual_order']);
  const running_set = new Set(['queued', 'running']);
  const state_map = new WeakMap();
  const cancellation_map = new Map();
  let page_obj = null;
  let active_obj = null;

  function state_dict(row_obj) {
    if (!state_map.has(row_obj)) state_map.set(row_obj, {generation_int: 0, phase_str: 'idle', nonce_str: '',
      expires_ms: 0, timer_int: null, request_obj: null, poll_url_str: '', job_id_str: '',
      preview_line_list: []});
    return state_map.get(row_obj);
  }
  function current_bool(row_obj, generation_int) {
    return row_obj === active_obj && page_obj?.contains(row_obj) && state_dict(row_obj).generation_int === generation_int;
  }
  function enabled_bool(row_obj) {
    return page_obj?.getAttribute('data-demo') === 'true' && page_obj.getAttribute('data-actions-enabled') === 'true'
      && Boolean(page_obj.getAttribute('data-action-token')) && Boolean(page_obj.getAttribute('data-pod-id'))
      && action_set.has(row_obj.getAttribute('data-tool-action'));
  }
  function endpoint_str(row_obj, suffix_str) {
    return '/api/demo-tools/' + encodeURIComponent(page_obj.getAttribute('data-pod-id')) + '/'
      + encodeURIComponent(row_obj.getAttribute('data-tool-action')) + '/' + suffix_str;
  }
  function stop_request(row_obj) {
    const row_state_dict = state_dict(row_obj);
    row_state_dict.generation_int += 1;
    row_state_dict.request_obj?.abort(); row_state_dict.request_obj = null;
    if (row_state_dict.timer_int !== null) window.clearTimeout(row_state_dict.timer_int);
    row_state_dict.timer_int = null;
  }
  function show_result(row_obj, label_str, message_str) {
    row_obj.querySelector('[data-tool-result]').hidden = false;
    row_obj.querySelector('[data-tool-result-label]').textContent = label_str;
    row_obj.querySelector('[data-tool-result-message]').textContent = message_str;
  }
  function controls(row_obj) {
    const row_state_dict = state_dict(row_obj);
    const preview_obj = row_obj.querySelector('[data-tool-preview]');
    if (preview_obj) preview_obj.disabled = !enabled_bool(row_obj)
      || ['previewing', 'preview', 'asking', 'confirming', 'queued', 'running', 'unknown', 'cancelling'].includes(row_state_dict.phase_str);
    row_obj.querySelector('[data-tool-confirm]').disabled = !enabled_bool(row_obj) || row_state_dict.phase_str !== 'preview'
      || !row_state_dict.nonce_str || Date.now() >= row_state_dict.expires_ms;
    row_obj.querySelector('[data-tool-cancel]').disabled = row_state_dict.phase_str === 'confirming';
  }
  function quote_str(value_str) { return "'" + String(value_str).replaceAll("'", "''") + "'"; }
  function update_command(row_obj) {
    const command_obj = row_obj.querySelector('[data-tool-command]');
    const copy_obj = row_obj.querySelector('[data-tool-copy-button]');
    if (!command_obj || !copy_obj) return;
    let argument_list;
    try { argument_list = JSON.parse(row_obj.getAttribute('data-tool-argv')); } catch (error_obj) { argument_list = []; }
    let valid_bool = Array.isArray(argument_list) && argument_list.length > 0 && argument_list.every(value_str => typeof value_str === 'string');
    const parameter_list = [];
    for (const input_obj of row_obj.querySelectorAll('[data-tool-parameter]')) {
      const value_str = input_obj.value.trim();
      if ((input_obj.required && !value_str) || (value_str && !input_obj.checkValidity())) valid_bool = false;
      if (value_str) parameter_list.push(input_obj.getAttribute('data-flag'), value_str);
    }
    command_obj.value = valid_bool ? '& ' + [...argument_list, ...parameter_list].map(quote_str).join(' ') : '';
    copy_obj.disabled = !valid_bool || row_obj.getAttribute('data-tool-copy') !== 'true';
    row_obj.querySelector('[data-tool-copy-status]').textContent = valid_bool
      ? 'PowerShell · Copy does not run the command.' : 'Complete the required fields to copy this command.';
  }
  async function copy_command(row_obj) {
    update_command(row_obj);
    if (row_obj.querySelector('[data-tool-copy-button]').disabled) return;
    const command_obj = row_obj.querySelector('[data-tool-command]');
    const status_obj = row_obj.querySelector('[data-tool-copy-status]');
    try { await navigator.clipboard.writeText(command_obj.value); status_obj.textContent = 'Copied. Nothing was run.'; }
    catch (error_obj) { command_obj.focus(); command_obj.select(); status_obj.textContent = 'Command selected. Press Ctrl+C to copy. Nothing was run.'; }
  }
  async function request_dict(row_obj, url_str, body_dict = null) {
    if (!enabled_bool(row_obj) || !url_str.startsWith('/api/demo-tools/')) throw new Error('Demo execution unavailable.');
    const request_obj = new AbortController();
    const row_state_dict = state_dict(row_obj);
    row_state_dict.request_obj = request_obj;
    const timeout_int = window.setTimeout(() => request_obj.abort(), 20000);
    try {
      const response_obj = await window.fetch(url_str, {method: body_dict ? 'POST' : 'GET', credentials: 'same-origin', cache: 'no-store',
        signal: request_obj.signal, headers: body_dict ? {'Content-Type': 'application/json', 'X-Alpha-Action-Token': page_obj.getAttribute('data-action-token')} : {},
        ...(body_dict ? {body: JSON.stringify(body_dict)} : {})});
      const response_dict = await response_obj.json();
      if (!response_dict || typeof response_dict !== 'object' || Array.isArray(response_dict)) throw new Error('Invalid demo response.');
      return {response_dict, ok_bool: response_obj.ok};
    } finally { window.clearTimeout(timeout_int); if (row_state_dict.request_obj === request_obj) row_state_dict.request_obj = null; }
  }
  function expire_preview(row_obj, generation_int) {
    if (!current_bool(row_obj, generation_int)) return;
    const row_state_dict = state_dict(row_obj);
    const remaining_int = Math.max(0, Math.ceil((row_state_dict.expires_ms - Date.now()) / 1000));
    row_obj.querySelector('[data-tool-expires]').textContent = remaining_int
      ? Math.floor(remaining_int / 60) + ':' + String(remaining_int % 60).padStart(2, '0') + ' left · single use' : 'Expired · create a new preview';
    if (!remaining_int) { row_state_dict.phase_str = 'expired'; row_state_dict.nonce_str = ''; }
    controls(row_obj);
    if (remaining_int) row_state_dict.timer_int = window.setTimeout(() => expire_preview(row_obj, generation_int), 1000);
  }
  function update_manual(row_obj) {
    if (row_obj.getAttribute('data-tool-action') !== 'manual_order') return;
    const field_list = [...row_obj.querySelectorAll('[data-manual-field]')];
    const type_obj = field_list.find(input_obj => input_obj.name === 'broker_order_type_str');
    const limit_obj = field_list.find(input_obj => input_obj.name === 'limit_price_float');
    if (!type_obj || !limit_obj) return;
    limit_obj.disabled = type_obj.value !== 'LMT'; limit_obj.required = type_obj.value === 'LMT';
  }
  function manual_dict(row_obj) {
    const ticket_dict = {time_in_force_str: 'DAY'};
    for (const input_obj of row_obj.querySelectorAll('[data-manual-field]')) {
      if (input_obj.disabled) continue;
      const value_str = input_obj.value.trim();
      if (input_obj.name === 'limit_price_float' && !value_str) continue;
      ticket_dict[input_obj.name] = ['quantity_int', 'limit_price_float'].includes(input_obj.name) ? Number(value_str) : value_str;
    }
    return ticket_dict;
  }
  async function preview(row_obj) {
    const row_state_dict = state_dict(row_obj);
    if (!enabled_bool(row_obj) || row_obj.querySelector('[data-tool-preview]')?.disabled || !row_obj.querySelector('[data-tool-form]').reportValidity()) return;
    const body_dict = {confirmed_bool: true};
    if (row_obj.getAttribute('data-tool-action') === 'manual_order') {
      body_dict.manual_order_dict = manual_dict(row_obj);
      if (body_dict.manual_order_dict.confirmation_text_str !== 'SUBMIT MANUAL ORDER') {
        show_result(row_obj, 'Ticket needs confirmation', 'Type SUBMIT MANUAL ORDER exactly before requesting the preview.'); return;
      }
    }
    stop_request(row_obj);
    const generation_int = row_state_dict.generation_int;
    row_state_dict.phase_str = 'previewing'; row_state_dict.nonce_str = ''; row_state_dict.job_id_str = ''; row_state_dict.poll_url_str = '';
    row_obj.querySelector('[data-tool-preview-box]').hidden = true;
    show_result(row_obj, 'Preparing simulated preview', 'Nothing has run yet.'); controls(row_obj);
    try {
      // Server cancellation revokes every preview for this Pod, including a
      // different tool. Wait for its cancellation queue before issuing one.
      const cancellation_obj = cancellation_map.get(page_obj.getAttribute('data-pod-id'));
      if (cancellation_obj) await cancellation_obj;
      if (!current_bool(row_obj, generation_int)) return;
      const {response_dict, ok_bool} = await request_dict(row_obj, endpoint_str(row_obj, 'preview'), body_dict);
      if (!current_bool(row_obj, generation_int)) return;
      if (!ok_bool) throw new Error(response_dict.message || 'Preview unavailable. Nothing was run.');
      if (response_dict.demo_bool !== true || response_dict.pod_id_str !== page_obj.getAttribute('data-pod-id')
          || response_dict.action_name_str !== row_obj.getAttribute('data-tool-action')
          || typeof response_dict.confirmation_nonce_str !== 'string' || !response_dict.confirmation_nonce_str
          || !Number.isFinite(response_dict.expires_in_seconds_int) || response_dict.expires_in_seconds_int <= 0
          || !Array.isArray(response_dict.preview_line_list) || !response_dict.preview_line_list.every(line_str => typeof line_str === 'string')) throw new Error('Demo preview could not be verified.');
      row_state_dict.nonce_str = response_dict.confirmation_nonce_str;
      row_state_dict.expires_ms = Date.now() + Math.min(120, response_dict.expires_in_seconds_int) * 1000;
      row_state_dict.phase_str = 'preview';
      row_state_dict.preview_line_list = [...response_dict.preview_line_list];
      const list_obj = row_obj.querySelector('[data-tool-preview-lines]'); list_obj.replaceChildren();
      for (const line_str of response_dict.preview_line_list) { const item_obj = document.createElement('li'); item_obj.textContent = line_str; list_obj.appendChild(item_obj); }
      row_obj.querySelector('[data-tool-result]').hidden = true;
      row_obj.querySelector('[data-tool-preview-box]').hidden = false;
      expire_preview(row_obj, generation_int); row_obj.querySelector('[data-tool-confirm]').focus({preventScroll: true});
    } catch (error_obj) {
      if (!current_bool(row_obj, generation_int)) return;
      row_state_dict.phase_str = 'idle'; show_result(row_obj, 'Preview unavailable', error_obj.name === 'AbortError' ? 'Preview timed out. Nothing was run.' : error_obj.message || 'Nothing was run.'); controls(row_obj);
    }
  }
  function poll_url_str(value_str) {
    if (typeof value_str !== 'string') return '';
    const expected_str = '/api/demo-tools/' + encodeURIComponent(page_obj.getAttribute('data-pod-id')) + '/jobs/';
    try {
      const url_obj = new URL(value_str, window.location.origin);
      return url_obj.origin === window.location.origin && url_obj.pathname.startsWith(expected_str)
        && /^[A-Za-z0-9_-]+$/.test(url_obj.pathname.slice(expected_str.length)) && !url_obj.search && !url_obj.hash ? url_obj.pathname : '';
    } catch (error_obj) { return ''; }
  }
  function render_job(row_obj, response_dict) {
    const row_state_dict = state_dict(row_obj);
    const status_str = response_dict.status_str;
    const verified_bool = response_dict.demo_bool === true && response_dict.pod_id_str === page_obj.getAttribute('data-pod-id')
      && response_dict.action_name_str === row_obj.getAttribute('data-tool-action') && typeof response_dict.job_id_str === 'string'
      && Boolean(response_dict.job_id_str) && (!row_state_dict.job_id_str || response_dict.job_id_str === row_state_dict.job_id_str)
      && ['queued', 'running', 'succeeded', 'unknown', 'rejected'].includes(status_str);
    row_state_dict.phase_str = verified_bool ? status_str : 'unknown';
    if (verified_bool) {
      row_state_dict.job_id_str = response_dict.job_id_str;
      row_state_dict.poll_url_str = poll_url_str(response_dict.poll_url_str) || row_state_dict.poll_url_str;
    }
    const label_dict = {queued: 'Queued', running: 'Running', succeeded: 'Command completed', unknown: 'Outcome unconfirmed', rejected: 'Run rejected'};
    show_result(row_obj, 'Simulated · ' + label_dict[row_state_dict.phase_str], 'Demo result. No broker action. '
      + (verified_bool && typeof response_dict.message_str === 'string' ? response_dict.message_str : 'Check Activity before another attempt.'));
    controls(row_obj);
  }
  async function poll_job(row_obj, generation_int) {
    const row_state_dict = state_dict(row_obj);
    if (!current_bool(row_obj, generation_int) || !row_state_dict.poll_url_str) return;
    try {
      const {response_dict, ok_bool} = await request_dict(row_obj, row_state_dict.poll_url_str);
      if (!current_bool(row_obj, generation_int)) return;
      if (!ok_bool && !response_dict.status_str) throw new Error('Job unavailable.');
      render_job(row_obj, response_dict);
      if (running_set.has(row_state_dict.phase_str)) row_state_dict.timer_int = window.setTimeout(() => poll_job(row_obj, generation_int), 1500);
    } catch (error_obj) {
      if (!current_bool(row_obj, generation_int)) return;
      row_state_dict.phase_str = 'unknown'; show_result(row_obj, 'Demo outcome unconfirmed', 'The result could not be read. Check Activity before another attempt.'); controls(row_obj);
    }
  }
  async function confirm(row_obj) {
    const row_state_dict = state_dict(row_obj);
    if (row_obj !== active_obj || !enabled_bool(row_obj) || row_state_dict.phase_str !== 'preview' || !row_state_dict.nonce_str || Date.now() >= row_state_dict.expires_ms) { controls(row_obj); return; }
    const nonce_str = row_state_dict.nonce_str;
    const preview_generation_int = row_state_dict.generation_int;
    const preview_page_obj = page_obj;
    const pod_id_str = page_obj.getAttribute('data-pod-id');
    const action_str = row_obj.getAttribute('data-tool-action');
    const prompt_str = 'Are you sure?\n\nSIMULATION · ' + action_str
      + '\nPod: ' + (page_obj.getAttribute('data-pod-label') || pod_id_str) + '\n' + pod_id_str
      + '\nAccount: ' + (page_obj.getAttribute('data-pod-account') || 'See preview')
      + '\n\n' + row_state_dict.preview_line_list.join('\n') + '\n\nRun this simulation once?';
    // A popup can stay open past expiry. Declining never consumes the nonce
    // or sends a confirm request; accepting rechecks its selection and time.
    row_state_dict.phase_str = 'asking'; controls(row_obj);
    let accepted_bool = false;
    try { accepted_bool = window.confirm(prompt_str) === true; } catch (error_obj) { /* A blocked popup never confirms. */ }
    if (!current_bool(row_obj, preview_generation_int) || page_obj !== preview_page_obj) return;
    if (!enabled_bool(row_obj) || page_obj.getAttribute('data-pod-id') !== pod_id_str
        || row_obj.getAttribute('data-tool-action') !== action_str || row_state_dict.nonce_str !== nonce_str) {
      stop_request(row_obj); row_state_dict.nonce_str = ''; row_state_dict.phase_str = 'idle';
      row_obj.querySelector('[data-tool-preview-box]').hidden = true;
      show_result(row_obj, 'Preview changed', 'Nothing was run. Request a new preview.'); controls(row_obj); return;
    }
    if (Date.now() >= row_state_dict.expires_ms) { expire_preview(row_obj, preview_generation_int); return; }
    if (!accepted_bool) { row_state_dict.phase_str = 'preview'; controls(row_obj); return; }
    stop_request(row_obj);
    const generation_int = row_state_dict.generation_int;
    row_state_dict.nonce_str = ''; row_state_dict.phase_str = 'confirming';
    row_obj.querySelector('[data-tool-preview-box]').hidden = true;
    show_result(row_obj, 'Sending demo request', 'One simulated run requested.'); controls(row_obj);
    try {
      const {response_dict, ok_bool} = await request_dict(row_obj, endpoint_str(row_obj, 'confirm'),
        {confirmed_bool: true, browser_confirmed_bool: true, confirmation_nonce_str: nonce_str});
      if (!current_bool(row_obj, generation_int)) return;
      if (!ok_bool && !response_dict.status_str) throw new Error(response_dict.message || 'Demo result unavailable.');
      render_job(row_obj, response_dict);
      if (running_set.has(row_state_dict.phase_str)) {
        if (row_state_dict.poll_url_str) await poll_job(row_obj, generation_int);
        else throw new Error('No valid demo job link was returned.');
      }
    } catch (error_obj) {
      if (!current_bool(row_obj, generation_int)) return;
      row_state_dict.phase_str = 'unknown'; show_result(row_obj, 'Demo outcome unconfirmed', 'The request did not return a verified result. Check Activity before another attempt.'); controls(row_obj);
    }
  }
  function cancel_preview(row_obj, show_bool = true) {
    const row_state_dict = state_dict(row_obj);
    const cancel_bool = ['preview', 'previewing', 'expired', 'asking'].includes(row_state_dict.phase_str) && enabled_bool(row_obj);
    const url_str = cancel_bool ? endpoint_str(row_obj, 'cancel') : '';
    stop_request(row_obj); row_state_dict.nonce_str = ''; row_state_dict.expires_ms = 0;
    row_obj.querySelector('[data-tool-preview-box]').hidden = true;
    if (row_state_dict.phase_str === 'confirming') { row_state_dict.phase_str = 'unknown'; show_result(row_obj, 'Demo outcome unconfirmed', 'Check Activity before another attempt.'); }
    else if (!running_set.has(row_state_dict.phase_str) && row_state_dict.phase_str !== 'unknown') {
      row_state_dict.phase_str = 'idle'; if (show_bool) show_result(row_obj, 'Preview cancelled', 'Nothing was run.');
    }
    if (cancel_bool) {
      const pod_id_str = page_obj.getAttribute('data-pod-id');
      const token_str = page_obj.getAttribute('data-action-token');
      const previous_obj = cancellation_map.get(pod_id_str) || Promise.resolve();
      const cancellation_obj = previous_obj.then(async () => {
        const controller_obj = new AbortController();
        const timeout_int = window.setTimeout(() => controller_obj.abort(), 20000);
        try {
          await window.fetch(url_str, {method: 'POST', credentials: 'same-origin', signal: controller_obj.signal,
            headers: {'Content-Type': 'application/json', 'X-Alpha-Action-Token': token_str},
            body: JSON.stringify({confirmed_bool: true})});
        } catch (error_obj) { /* The server still enforces preview expiry. */ }
        finally { window.clearTimeout(timeout_int); }
      }).finally(() => { if (cancellation_map.get(pod_id_str) === cancellation_obj) cancellation_map.delete(pod_id_str); });
      cancellation_map.set(pod_id_str, cancellation_obj);
    }
    controls(row_obj);
  }
  function open_row(row_obj) {
    if (active_obj === row_obj) {
      cancel_preview(row_obj, false); row_obj.querySelector('[data-tool-open]').setAttribute('aria-expanded', 'false');
      row_obj.querySelector('[data-tool-body]').hidden = true; active_obj = null; return;
    }
    if (active_obj) {
      cancel_preview(active_obj, false); active_obj.querySelector('[data-tool-open]').setAttribute('aria-expanded', 'false');
      active_obj.querySelector('[data-tool-body]').hidden = true;
    }
    active_obj = row_obj; row_obj.querySelector('[data-tool-open]').setAttribute('aria-expanded', 'true');
    row_obj.querySelector('[data-tool-body]').hidden = false; update_command(row_obj); update_manual(row_obj); controls(row_obj);
    const row_state_dict = state_dict(row_obj);
    if (running_set.has(row_state_dict.phase_str) && row_state_dict.poll_url_str) poll_job(row_obj, row_state_dict.generation_int);
  }
  function install_page() {
    const next_obj = document.querySelector('[data-tools-page]');
    if (next_obj === page_obj) return;
    if (active_obj) cancel_preview(active_obj, false);
    active_obj = null; page_obj = next_obj;
    if (page_obj) {
      const row_list = [...page_obj.querySelectorAll('[data-tool]')];
      for (const row_obj of row_list) { update_command(row_obj); update_manual(row_obj); controls(row_obj); }
      const selected_obj = row_list.find(row_obj => row_obj.getAttribute('data-tool') === page_obj.getAttribute('data-selected-tool'));
      if (selected_obj) open_row(selected_obj);
    }
  }
  document.addEventListener('click', event_obj => {
    const button_obj = event_obj.target.closest?.('button'); const row_obj = button_obj?.closest('[data-tool]');
    if (!row_obj || !page_obj?.contains(row_obj)) return;
    if (button_obj.hasAttribute('data-tool-open')) open_row(row_obj);
    else if (button_obj.hasAttribute('data-tool-copy-button')) copy_command(row_obj);
    else if (button_obj.hasAttribute('data-tool-confirm')) confirm(row_obj);
    else if (button_obj.hasAttribute('data-tool-cancel')) cancel_preview(row_obj);
  });
  document.addEventListener('submit', event_obj => {
    if (event_obj.target.hasAttribute('data-tool-form')) { event_obj.preventDefault(); const row_obj = event_obj.target.closest('[data-tool]'); if (row_obj === active_obj) preview(row_obj); }
    else if (event_obj.target.hasAttribute('data-tools-pod-form') && page_obj?.contains(event_obj.target)) {
      event_obj.preventDefault();
      if (active_obj) cancel_preview(active_obj, false);
      const pod_id_str = event_obj.target.querySelector('[data-tools-pod]').value;
      const tool_key_str = page_obj.getAttribute('data-selected-tool');
      const query_obj = new URLSearchParams();
      if (pod_id_str) query_obj.set('pod', pod_id_str);
      if (tool_key_str) query_obj.set('tool', tool_key_str);
      window.location.assign('/tools' + (query_obj.size ? '?' + query_obj.toString() : ''));
    }
  });
  for (const event_str of ['input', 'change']) document.addEventListener(event_str, event_obj => {
    const row_obj = event_obj.target.closest?.('[data-tool]');
    if (!row_obj || !page_obj?.contains(row_obj)) return;
    if (event_obj.target.hasAttribute('data-tool-parameter') || event_obj.target.hasAttribute('data-manual-field')) {
      if (['preview', 'previewing', 'expired'].includes(state_dict(row_obj).phase_str)) cancel_preview(row_obj, false);
      update_command(row_obj); update_manual(row_obj);
    }
  });
  document.addEventListener('htmx:afterSwap', install_page);
  window.addEventListener('pagehide', () => { if (active_obj) cancel_preview(active_obj, false); });
  install_page();
})();
