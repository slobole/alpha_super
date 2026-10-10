/* Per-Pod console: a bounded, read-only tail of the serve operator lines.
   Its own fetch loop never touches the page refresh, so a console failure
   cannot mark operating status Unknown. Memory stays flat: at most MAX_ROW_INT
   rows, text nodes only, no copy of the log kept in JavaScript. */
(() => {
  'use strict';
  const panel_obj = document.getElementById('console-panel');
  if (!panel_obj) return;
  const MAX_ROW_INT = 1000;
  const CATCH_UP_LIMIT_INT = 4;
  const REQUEST_TIMEOUT_MS = 8000;
  const ERROR_DELAY_LIST = [5000, 10000, 30000, 60000];
  const QUIET_ACTION_RE = /(^|\.)wait$|^fill\.none$|^norgate\.sync\./;
  const LEVEL_LABEL_DICT = {C: 'CRIT', E: 'ERROR', W: 'WARN', I: 'INFO', O: '', M: ''};
  const tail_url_str = panel_obj.getAttribute('data-console-tail-url');
  const body_obj = panel_obj.querySelector('[data-console-body]');
  const status_obj = panel_obj.querySelector('[data-console-status]');
  const new_obj = panel_obj.querySelector('[data-console-new]');
  const find_obj = panel_obj.querySelector('[data-console-find]');
  const quiet_obj = panel_obj.querySelector('[data-console-quiet]');
  const pause_obj = panel_obj.querySelector('[data-console-pause]');
  const copy_obj = panel_obj.querySelector('[data-console-copy]');
  const time_formatter_obj = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
  });
  let cursor_str = '';
  let timer_int = 0;
  let controller_obj = null;
  let paused_bool = false;
  let following_bool = true;
  let catch_up_int = 0;
  let empty_poll_int = 0;
  let error_int = 0;
  let unseen_int = 0;
  let last_line_ms = Date.now();
  let checked_ms = 0;
  let retry_at_ms = 0;
  let state_str = 'connecting';
  let source_dict = {};
  let find_str = '';
  let find_timer_int = 0;

  function read_preference(key_str, default_str) {
    try { return window.localStorage.getItem('v4.console.' + key_str) || default_str; } catch (error_obj) { return default_str; }
  }

  function save_preference(key_str, value_str) {
    try { window.localStorage.setItem('v4.console.' + key_str, value_str); } catch (error_obj) { /* preferences only */ }
  }

  function time_str(iso_str) {
    if (!iso_str) return '';
    const value_ms = Date.parse(iso_str);
    return Number.isFinite(value_ms) ? time_formatter_obj.format(value_ms) : '';
  }

  function age_str(value_ms) {
    const seconds_int = Math.max(0, Math.round((Date.now() - value_ms) / 1000));
    if (seconds_int < 60) return seconds_int + ' s ago';
    if (seconds_int < 3600) return Math.floor(seconds_int / 60) + ' min ago';
    return Math.floor(seconds_int / 3600) + ' h ' + Math.floor((seconds_int % 3600) / 60) + ' min ago';
  }

  function size_str(bytes_int) {
    if (!(bytes_int >= 0)) return '';
    return bytes_int >= 1048576 ? (bytes_int / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(bytes_int / 1024)) + ' KB';
  }

  function line_key_str(line_dict) {
    // Repeated retries differ only by time; fold them into one row.
    return line_dict.l + '|' + line_dict.a + '|' + String(line_dict.x || '').replace(/\belapsed=\S+/g, '');
  }

  function span_obj(class_str, text_str) {
    const result_obj = document.createElement('span');
    result_obj.className = class_str;
    result_obj.textContent = text_str;
    return result_obj;
  }

  function row_text_str(row_obj) {
    return Array.from(row_obj.children).map((child_obj) => child_obj.textContent).filter(Boolean).join(' ');
  }

  function apply_find(row_obj) {
    row_obj.hidden = Boolean(find_str) && !row_obj.textContent.toLowerCase().includes(find_str);
  }

  function near_bottom_bool() {
    return body_obj.scrollHeight - body_obj.scrollTop - body_obj.clientHeight <= 40;
  }

  function scroll_to_bottom() {
    body_obj.scrollTop = body_obj.scrollHeight;
    unseen_int = 0;
    new_obj.hidden = true;
  }

  function append_line_list(line_list, reset_bool) {
    if (reset_bool) body_obj.textContent = '';
    if (!line_list.length) return;
    const fragment_obj = document.createDocumentFragment();
    let last_row_obj = body_obj.lastElementChild;
    for (const line_dict of line_list) {
      const key_str = line_key_str(line_dict);
      if (last_row_obj && line_dict.l !== 'M' && last_row_obj.getAttribute('data-key') === key_str) {
        const count_int = Number(last_row_obj.getAttribute('data-count')) + 1;
        last_row_obj.setAttribute('data-count', String(count_int));
        last_row_obj.querySelector('.cn').textContent = '×' + count_int + (line_dict.t ? ' · to ' + time_str(line_dict.t) : '');
        continue;
      }
      const row_obj = document.createElement('div');
      row_obj.className = 'cl l-' + line_dict.l + (line_dict.l === 'I' && QUIET_ACTION_RE.test(line_dict.a || '') ? ' q' : '');
      row_obj.setAttribute('data-key', key_str);
      row_obj.setAttribute('data-count', '1');
      row_obj.append(span_obj('ct', time_str(line_dict.t)), span_obj('cv', LEVEL_LABEL_DICT[line_dict.l] || ''),
        span_obj('ca', line_dict.a || ''), span_obj('cx', line_dict.x || ''), span_obj('cn', ''));
      apply_find(row_obj);
      fragment_obj.append(row_obj);
      last_row_obj = row_obj;
      unseen_int += 1;
    }
    const follow_bool = following_bool;
    body_obj.append(fragment_obj);
    while (body_obj.childElementCount > MAX_ROW_INT) body_obj.firstElementChild.remove();
    if (follow_bool) scroll_to_bottom();
    else if (unseen_int) {
      new_obj.hidden = false;
      new_obj.textContent = unseen_int + (unseen_int === 1 ? ' new line ↓' : ' new lines ↓');
    }
  }

  function render_status() {
    const part_list = [];
    if (state_str === 'paused') part_list.push('Paused');
    else if (state_str === 'hidden') part_list.push('Hidden tab · paused');
    else if (state_str === 'offline') part_list.push('Offline · retry in ' + Math.max(1, Math.ceil((retry_at_ms - Date.now()) / 1000)) + ' s');
    else if (state_str === 'busy') part_list.push('Busy · retry in ' + Math.max(1, Math.ceil((retry_at_ms - Date.now()) / 1000)) + ' s');
    else if (state_str === 'connecting') part_list.push('Connecting…');
    else part_list.push((following_bool ? 'Following' : 'Scrolled up') + (checked_ms ? ' · checked ' + age_str(checked_ms) : ''));
    if (source_dict.source_label_str) {
      let file_str = source_dict.source_label_str;
      if (source_dict.file_size_int >= 0) file_str += ' · ' + size_str(source_dict.file_size_int);
      if (source_dict.last_write_utc_str) file_str += ' · last write ' + age_str(Date.parse(source_dict.last_write_utc_str));
      part_list.push(file_str);
    }
    if (source_dict.note_str) part_list.push(source_dict.note_str);
    status_obj.textContent = part_list.join(' · ');
    status_obj.classList.toggle('is-bad', state_str === 'offline');
  }

  function next_delay_ms(line_count_int) {
    if (line_count_int > 0) { empty_poll_int = 0; last_line_ms = Date.now(); return 2000; }
    empty_poll_int += 1;
    const idle_ms = Date.now() - last_line_ms;
    if (idle_ms >= 300000) return 15000;
    if (idle_ms >= 60000) return 10000;
    return empty_poll_int >= 3 ? 5000 : 2000;
  }

  function schedule(delay_ms) {
    window.clearTimeout(timer_int);
    timer_int = 0;
    if (paused_bool || document.hidden) return;
    timer_int = window.setTimeout(poll, delay_ms);
  }

  async function poll() {
    timer_int = 0;
    if (paused_bool || document.hidden || controller_obj) return;
    controller_obj = new AbortController();
    const timeout_int = window.setTimeout(() => controller_obj && controller_obj.abort(), REQUEST_TIMEOUT_MS);
    let delay_ms = 2000;
    const sent_cursor_str = cursor_str;
    try {
      const url_str = tail_url_str + (sent_cursor_str ? '?cursor=' + encodeURIComponent(sent_cursor_str) : '');
      const response_obj = await fetch(url_str, {signal: controller_obj.signal, cache: 'no-store', headers: {Accept: 'application/json'}});
      if (response_obj.status === 429) {
        const retry_ms = Math.min(60, Math.max(1, Number(response_obj.headers.get('Retry-After')) || 5)) * 1000;
        state_str = 'busy';
        retry_at_ms = Date.now() + retry_ms;
        delay_ms = retry_ms;
      } else if (!response_obj.ok) {
        throw new Error('HTTP ' + response_obj.status);
      } else {
        const tail_dict = await response_obj.json();
        const line_list = Array.isArray(tail_dict.line_list) ? tail_dict.line_list : [];
        error_int = 0;
        source_dict = tail_dict;
        cursor_str = tail_dict.cursor_str || '';
        // A response to no cursor is a full first load: replace, never append.
        append_line_list(line_list, Boolean(tail_dict.reset_bool) || !sent_cursor_str);
        checked_ms = Date.now();
        state_str = paused_bool ? 'paused' : 'ok';
        catch_up_int = tail_dict.more_bool ? catch_up_int + 1 : 0;
        delay_ms = tail_dict.more_bool && catch_up_int <= CATCH_UP_LIMIT_INT ? 200 : next_delay_ms(line_list.length);
      }
    } catch (error_obj) {
      if (paused_bool || document.hidden) {
        // Our own abort (Pause or a hidden tab): no error, no longer backoff.
        state_str = paused_bool ? 'paused' : 'hidden';
      } else {
        delay_ms = ERROR_DELAY_LIST[Math.min(error_int, ERROR_DELAY_LIST.length - 1)];
        error_int += 1;
        state_str = 'offline';
        retry_at_ms = Date.now() + delay_ms;
      }
    } finally {
      window.clearTimeout(timeout_int);
      controller_obj = null;
    }
    render_status();
    schedule(delay_ms);
  }

  function set_level(level_str) {
    body_obj.classList.toggle('lv-warn', level_str === 'warn');
    body_obj.classList.toggle('lv-error', level_str === 'error');
    panel_obj.querySelectorAll('[data-console-level]').forEach((button_obj) => {
      const on_bool = button_obj.getAttribute('data-console-level') === level_str;
      button_obj.classList.toggle('on', on_bool);
      button_obj.setAttribute('aria-pressed', on_bool ? 'true' : 'false');
    });
    save_preference('level', level_str);
  }

  function set_paused(paused_value_bool) {
    paused_bool = paused_value_bool;
    pause_obj.textContent = paused_bool ? 'Resume' : 'Pause';
    pause_obj.setAttribute('aria-pressed', paused_bool ? 'true' : 'false');
    if (paused_bool) {
      window.clearTimeout(timer_int);
      if (controller_obj) controller_obj.abort();
      state_str = 'paused';
    } else {
      state_str = checked_ms ? 'ok' : 'connecting';
      schedule(0);
    }
    render_status();
  }

  panel_obj.querySelectorAll('[data-console-level]').forEach((button_obj) => {
    button_obj.addEventListener('click', () => set_level(button_obj.getAttribute('data-console-level')));
  });
  quiet_obj.addEventListener('change', () => {
    body_obj.classList.toggle('quiet', quiet_obj.checked);
    save_preference('quiet', quiet_obj.checked ? '1' : '0');
  });
  find_obj.addEventListener('input', () => {
    window.clearTimeout(find_timer_int);
    find_timer_int = window.setTimeout(() => {
      find_str = find_obj.value.trim().toLowerCase();
      for (const row_obj of body_obj.children) apply_find(row_obj);
    }, 150);
  });
  pause_obj.addEventListener('click', () => set_paused(!paused_bool));
  new_obj.addEventListener('click', scroll_to_bottom);
  body_obj.addEventListener('scroll', () => {
    following_bool = near_bottom_bool();
    if (following_bool) { unseen_int = 0; new_obj.hidden = true; }
  }, {passive: true});
  copy_obj.addEventListener('click', () => {
    const text_str = Array.from(body_obj.children).filter((row_obj) => !row_obj.hidden && row_obj.offsetParent !== null)
      .map(row_text_str).join('\n');
    const done_fn = () => { copy_obj.textContent = 'Copied'; window.setTimeout(() => { copy_obj.textContent = 'Copy'; }, 1500); };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text_str).then(done_fn, () => {});
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      window.clearTimeout(timer_int);
      if (controller_obj) controller_obj.abort();
      if (!paused_bool) state_str = 'hidden';
      render_status();
    } else if (!paused_bool) {
      state_str = checked_ms ? 'ok' : 'connecting';
      schedule(0);
    }
  });
  // Labels only; the polling chain above is the only network timer.
  window.setInterval(() => { if (!document.hidden) render_status(); }, 1000);

  set_level(read_preference('level', 'all'));
  quiet_obj.checked = read_preference('quiet', '1') === '1';
  body_obj.classList.toggle('quiet', quiet_obj.checked);
  render_status();
  schedule(0);
})();
