/* Order panel on the Pod page: "Complete" a missing trade, or a free "New order".
   One panel at a time, inside the page (no pop-out, no typed confirmation).
   check = one fresh read-only broker query + a one-use nonce; send = one manual ticket.
   The slots carry hx-preserve, so the 15-second page refresh never wipes an open panel. */
(() => {
  'use strict';
  const EXPIRY_MS = 120000;
  let active_obj = null;  // {slot, kind, asset, check, nonce, expires_ms, timer, type, side, busy}

  function el(tag_str, attr_dict, ...child_list) {
    const node_obj = document.createElement(tag_str);
    Object.entries(attr_dict || {}).forEach(([key_str, value_obj]) => {
      if (value_obj === null || value_obj === undefined || value_obj === false) return;
      if (key_str === 'class') node_obj.className = value_obj;
      else if (key_str === 'text') node_obj.textContent = value_obj;
      else node_obj.setAttribute(key_str, value_obj === true ? '' : String(value_obj));
    });
    child_list.flat().forEach((child_obj) => { if (child_obj) node_obj.append(child_obj); });
    return node_obj;
  }
  function slot_obj(kind_str) {
    return document.querySelector(`[data-order-panel="${kind_str === 'complete' ? 'complete' : 'new'}"]`);
  }
  function when_str(type_str, limit_str) {
    return type_str === 'MOC' ? 'at close' : type_str === 'MKT' ? 'now' : `at ${limit_str || '…'}`;
  }
  function close_all() {
    if (active_obj && active_obj.timer) clearTimeout(active_obj.timer);
    document.querySelectorAll('[data-order-panel]').forEach((node_obj) => { node_obj.hidden = true; node_obj.replaceChildren(); });
    active_obj = null;
  }
  async function post_json(url_str, body_dict, slot_node_obj) {
    const response_obj = await fetch(url_str, {method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json', 'X-Alpha-Action-Token': slot_node_obj.getAttribute('data-token')},
      body: JSON.stringify({confirmed_bool: true, ...body_dict})});
    const data_dict = await response_obj.json().catch(() => ({}));
    if (!response_obj.ok) return {status_str: 'stopped', line_str: data_dict.message || 'Request refused'};
    return data_dict;
  }

  function head(title_str, slot_node_obj) {
    return el('div', {class: 'panel-h'}, el('span', {class: 'panel-t', text: title_str}),
      el('span', {class: 'panel-m', text: slot_node_obj.getAttribute('data-subtitle') || ''}));
  }
  function fact(label_str, value_str, class_str) {
    return el('div', {class: class_str || ''}, el('span', {text: label_str}), el('b', {text: value_str}));
  }
  function line(text_str, kind_str) {
    return el('div', {class: 'oline' + (kind_str ? ' is-' + kind_str : ''), 'data-order-line': true, text: text_str});
  }
  function foot(...child_list) {
    return el('div', {class: 'nrow nfoot'}, el('span', {class: 'grow'}), ...child_list);
  }
  function cancel_button() {
    return el('button', {class: 'btn btn-q', type: 'button', 'data-order-act': 'cancel', text: 'Cancel'});
  }

  function render_stop(text_str, kind_str) {
    const state_obj = active_obj;
    const node_obj = state_obj.slot;
    const title_str = state_obj.kind === 'complete' ? `Complete ${state_obj.asset}` : 'New order';
    const body_list = [head(title_str, node_obj)];
    if (state_obj.kind === 'new') body_list.push(new_what_row(null));
    body_list.push(el('div', {class: 'nrow nfoot'}, line(text_str, kind_str),
      el('button', {class: 'btn btn-q', type: 'button', 'data-order-act': 'recheck', text: 'Check again'}), cancel_button()));
    node_obj.replaceChildren(...body_list);
  }

  function new_what_row(check_dict) {
    const side_str = active_obj.side || 'BUY';
    return el('div', {class: 'nrow nwhat'},
      el('label', {class: 'fld'}, el('span', {text: 'Symbol'}),
        el('input', {class: 'inp mono', 'data-order-field': 'asset', value: active_obj.asset || '', maxlength: 16,
          autocomplete: 'off', placeholder: 'AAPL', 'aria-label': 'Symbol'})),
      el('label', {class: 'fld'}, el('span', {text: 'Side'}), el('span', {class: 'segx seg2'},
        ['BUY', 'SELL'].map((value_str) => el('button', {type: 'button', class: value_str === side_str ? 'on' : '',
          'data-order-side': value_str, text: value_str})))),
      check_dict ? el('div', {class: 'facts3'}, fact('Broker now', check_dict.held_str), fact('Plan', check_dict.plan_str),
        fact('Open orders', check_dict.open_orders_str))
        : el('div', {}, el('button', {class: 'btn', type: 'button', 'data-order-act': 'check', text: 'Check broker'})));
  }

  function render_ticket() {
    const state_obj = active_obj;
    const check_dict = state_obj.check;
    const node_obj = state_obj.slot;
    const complete_bool = state_obj.kind === 'complete';
    const what_obj = complete_bool
      ? el('div', {class: 'nrow nwhat'},
          el('label', {class: 'fld'}, el('span', {text: 'Symbol'}), el('input', {class: 'inp mono', value: check_dict.asset_str, readonly: true})),
          el('label', {class: 'fld'}, el('span', {text: 'Side'}), el('span', {class: 'segx seg2 is-fixed'},
            el('button', {type: 'button', class: 'on', disabled: false, tabindex: '-1', text: check_dict.side_str}))),
          el('div', {class: 'facts3'}, fact('Plan', check_dict.plan_str), fact('Broker now', check_dict.held_str),
            fact('Missing', check_dict.missing_str, 'miss')))
      : new_what_row(check_dict);
    const max_int = complete_bool ? check_dict.max_quantity_int : (state_obj.side === 'SELL' ? check_dict.max_sell_int : null);
    const shares_label_str = complete_bool ? `Shares · max ${max_int} (missing)` : (state_obj.side === 'SELL' ? `Shares · max ${max_int} (held)` : 'Shares');
    const type_list = [['MOC', 'At close'], ['MKT', 'Now'], ['LMT', 'Limit']];
    const hint_list = [];
    const left_int = check_dict.minutes_left_int;
    const left_str = left_int >= 60 ? `${Math.floor(left_int / 60)}h ${left_int % 60}m` : `${left_int}m`;
    if (check_dict.moc_bool) hint_list.push(`MOC until ${check_dict.cutoff_str} · ${left_str} left`);
    else hint_list.push(`MOC closed at ${check_dict.cutoff_str} · Now until ${check_dict.close_str}`);
    hint_list.push(`broker checked ${check_dict.checked_str}`);
    hint_list.push(complete_bool ? `VPlan #${check_dict.vplan_id_int}` : 'leaves the pod off plan');
    const warn_bool = !check_dict.moc_bool || (complete_bool && check_dict.changed_bool);
    if (complete_bool && check_dict.changed_bool) hint_list.unshift(`Page showed ${check_dict.page_str}`);
    const order_obj = el('div', {class: 'nrow'}, el('div', {class: 'nsec', text: 'Order'}),
      el('div', {class: 'norder'},
        el('span', {class: 'segx seg3'}, type_list.map(([value_str, label_str]) => el('button', {type: 'button',
          class: value_str === state_obj.type ? 'on' : (value_str === 'MOC' && !check_dict.moc_bool ? 'is-off' : ''),
          disabled: value_str === 'MOC' && !check_dict.moc_bool,
          'data-order-type': value_str, text: label_str}))),
        el('label', {class: 'fld'}, el('span', {text: shares_label_str}),
          el('input', {class: 'inp mono', type: 'number', min: 1, step: 1, max: max_int, inputmode: 'numeric',
            'data-order-field': 'quantity', value: state_obj.quantity_str ?? (max_int || '')})),
        state_obj.type === 'LMT'
          ? el('label', {class: 'fld'}, el('span', {text: 'Limit price'}), el('input', {class: 'inp mono', type: 'number',
              min: 0, step: 'any', inputmode: 'decimal', 'data-order-field': 'limit', value: state_obj.limit_str || ''}))
          : (complete_bool ? el('span') : el('label', {class: 'fld'}, el('span', {text: 'Reason'}), el('input', {class: 'inp',
              maxlength: 200, 'data-order-field': 'reason', value: state_obj.reason_str || '', placeholder: 'Why this order'})))),
      (!complete_bool && state_obj.type === 'LMT') ? el('label', {class: 'fld'}, el('span', {text: 'Reason'}), el('input', {class: 'inp',
        maxlength: 200, 'data-order-field': 'reason', value: state_obj.reason_str || '', placeholder: 'Why this order'})) : null,
      el('div', {class: 'hint' + (warn_bool ? ' warn' : ''), text: hint_list.join(' · ')}));
    const send_obj = el('button', {class: 'btn btn-danger nbtn', type: 'button', 'data-order-act': 'send'}, 'Send · ', el('b', {'data-order-label': true}));
    node_obj.replaceChildren(head(complete_bool ? `Complete ${check_dict.asset_str}` : 'New order', node_obj), what_obj, order_obj,
      foot(cancel_button(), send_obj));
    refresh_label();
  }

  function read_fields() {
    const node_obj = active_obj.slot;
    const value_str = (name_str) => node_obj.querySelector(`[data-order-field="${name_str}"]`)?.value ?? '';
    active_obj.quantity_str = value_str('quantity');
    active_obj.limit_str = value_str('limit');
    active_obj.reason_str = value_str('reason');
  }
  function ticket_problem_str() {
    const check_dict = active_obj.check;
    const complete_bool = active_obj.kind === 'complete';
    const quantity_int = Number(active_obj.quantity_str);
    if (!Number.isInteger(quantity_int) || quantity_int < 1) return 'Enter shares';
    const max_int = complete_bool ? check_dict.max_quantity_int : (active_obj.side === 'SELL' ? check_dict.max_sell_int : null);
    if (max_int !== null && max_int !== undefined && quantity_int > max_int) return `Max ${max_int}`;
    if (active_obj.type === 'LMT' && !(Number(active_obj.limit_str) > 0)) return 'Enter a limit price';
    if (!complete_bool && !active_obj.reason_str.trim()) return 'Enter a reason';
    if (Date.now() >= active_obj.expires_ms) return 'Expired';
    return '';
  }
  function refresh_label() {
    if (!active_obj || !active_obj.check) return;
    read_fields();
    const node_obj = active_obj.slot;
    const side_str = active_obj.kind === 'complete' ? active_obj.check.side_str : active_obj.side;
    const label_obj = node_obj.querySelector('[data-order-label]');
    const send_obj = node_obj.querySelector('[data-order-act="send"]');
    if (!label_obj || !send_obj) return;
    label_obj.textContent = `${side_str} ${active_obj.quantity_str || '…'} ${active_obj.check.asset_str} ${when_str(active_obj.type, active_obj.limit_str)}`;
    send_obj.disabled = Boolean(ticket_problem_str()) || active_obj.busy;
  }

  async function check() {
    const state_obj = active_obj;
    if (!state_obj) return;
    if (state_obj.timer) clearTimeout(state_obj.timer);
    state_obj.check = null;
    state_obj.nonce = '';
    const node_obj = state_obj.slot;
    node_obj.hidden = false;
    if (state_obj.kind === 'new' && !state_obj.asset) {
      node_obj.replaceChildren(head('New order', node_obj), new_what_row(null), foot(cancel_button()));
      node_obj.querySelector('[data-order-field="asset"]')?.focus();
      return;
    }
    node_obj.replaceChildren(head(state_obj.kind === 'complete' ? `Complete ${state_obj.asset}` : 'New order', node_obj),
      el('div', {class: 'nrow nfoot'}, line('Checking the broker…', 'info')));
    let result_dict;
    try {
      result_dict = await post_json(node_obj.getAttribute('data-check-url'),
        {kind_str: state_obj.kind, asset_str: state_obj.asset}, node_obj);
    } catch (error_obj) {
      result_dict = {status_str: 'stopped', line_str: 'Broker check failed · nothing sent'};
    }
    if (active_obj !== state_obj) return;
    if (result_dict.status_str !== 'ok') { render_stop(result_dict.line_str || 'Stopped'); return; }
    state_obj.check = result_dict;
    state_obj.asset = result_dict.asset_str;
    state_obj.nonce = result_dict.confirmation_nonce_str;
    state_obj.expires_ms = Date.now() + Math.min(EXPIRY_MS, (result_dict.expires_in_seconds_int || 120) * 1000) - 2000;
    state_obj.type = result_dict.default_type_str;
    state_obj.quantity_str = undefined;
    state_obj.timer = setTimeout(() => {
      if (active_obj === state_obj && !state_obj.busy && !state_obj.done) render_stop('Expired · check again', 'info');
    }, state_obj.expires_ms - Date.now());
    render_ticket();
  }

  async function send() {
    const state_obj = active_obj;
    read_fields();
    if (!state_obj || state_obj.busy || ticket_problem_str()) return;
    state_obj.busy = true;
    refresh_label();
    const node_obj = state_obj.slot;
    const body_dict = {confirmation_nonce_str: state_obj.nonce, broker_order_type_str: state_obj.type,
      quantity_int: Number(state_obj.quantity_str)};
    if (state_obj.type === 'LMT') body_dict.limit_price_float = Number(state_obj.limit_str);
    if (state_obj.kind === 'new') Object.assign(body_dict, {side_str: state_obj.side, reason_str: state_obj.reason_str});
    let result_dict;
    try {
      result_dict = await post_json(node_obj.getAttribute('data-send-url'), body_dict, node_obj);
    } catch (error_obj) {
      result_dict = {status_str: 'unknown', line_str: 'Outcome unknown · check TWS before resending'};
    }
    if (active_obj !== state_obj) return;
    state_obj.busy = false;
    state_obj.done = true;
    if (state_obj.timer) clearTimeout(state_obj.timer);
    const footer_obj = node_obj.querySelector('.nfoot');
    const done_list = result_dict.status_str === 'sent'
      ? [line(result_dict.line_str, 'sent'), el('button', {class: 'btn', type: 'button', 'data-order-act': 'cancel', text: 'Close'})]
      : result_dict.status_str === 'stopped'
        ? [line(result_dict.line_str), el('button', {class: 'btn btn-q', type: 'button', 'data-order-act': 'recheck', text: 'Check again'}), cancel_button()]
        : [line(result_dict.line_str || 'Outcome unknown · check TWS before resending'), el('button', {class: 'btn', type: 'button', 'data-order-act': 'cancel', text: 'Close'})];
    footer_obj.replaceChildren(...done_list);
    node_obj.querySelectorAll('input, [data-order-type], [data-order-side]').forEach((input_obj) => { input_obj.disabled = true; });
  }

  document.addEventListener('click', (event_obj) => {
    const open_obj = event_obj.target.closest('[data-order-open]');
    if (open_obj) {
      const kind_str = open_obj.getAttribute('data-kind') === 'complete' ? 'complete' : 'new';
      const node_obj = slot_obj(kind_str);
      if (!node_obj || node_obj.getAttribute('data-enabled') !== 'true') return;
      close_all();
      active_obj = {slot: node_obj, kind: kind_str, asset: open_obj.getAttribute('data-asset') || '', side: 'BUY', type: 'MOC'};
      node_obj.hidden = false;
      node_obj.scrollIntoView({block: 'nearest'});
      check();
      return;
    }
    if (!active_obj || !active_obj.slot.contains(event_obj.target)) return;
    const act_obj = event_obj.target.closest('[data-order-act]');
    const type_obj = event_obj.target.closest('[data-order-type]');
    const side_obj = event_obj.target.closest('[data-order-side]');
    if (act_obj) {
      const act_str = act_obj.getAttribute('data-order-act');
      if (act_str === 'cancel') close_all();
      else if (act_str === 'send') send();
      else if (act_str === 'recheck' || act_str === 'check') {
        if (active_obj.kind === 'new') {
          const asset_input_obj = active_obj.slot.querySelector('[data-order-field="asset"]');
          if (asset_input_obj) active_obj.asset = asset_input_obj.value.trim().toUpperCase();
        }
        active_obj.done = false;
        check();
      }
    } else if (type_obj && !type_obj.disabled && active_obj.check && !active_obj.done) {
      read_fields();
      active_obj.type = type_obj.getAttribute('data-order-type');
      render_ticket();
    } else if (side_obj && !side_obj.disabled && !active_obj.done) {
      active_obj.side = side_obj.getAttribute('data-order-side');
      if (active_obj.check) { read_fields(); active_obj.quantity_str = active_obj.quantity_str || ''; render_ticket(); }
      else active_obj.slot.querySelectorAll('[data-order-side]').forEach((node_obj) => node_obj.classList.toggle('on', node_obj === side_obj));
    }
  });
  document.addEventListener('input', (event_obj) => {
    if (!active_obj || !active_obj.slot.contains(event_obj.target)) return;
    if (event_obj.target.getAttribute('data-order-field') === 'asset' && active_obj.check) {
      // A new symbol needs a new broker check.
      active_obj.asset = event_obj.target.value.trim().toUpperCase();
      active_obj.check = null;
      const facts_obj = active_obj.slot.querySelector('.facts3');
      if (facts_obj) facts_obj.replaceWith(el('div', {}, el('button', {class: 'btn', type: 'button', 'data-order-act': 'check', text: 'Check broker'})));
      active_obj.slot.querySelectorAll('.nrow:not(.nwhat)').forEach((node_obj) => node_obj.remove());
      active_obj.slot.append(foot(cancel_button()));
      return;
    }
    refresh_label();
  });
  document.addEventListener('keydown', (event_obj) => {
    if (event_obj.key !== 'Enter' || !active_obj || event_obj.target.getAttribute?.('data-order-field') !== 'asset') return;
    event_obj.preventDefault();
    active_obj.asset = event_obj.target.value.trim().toUpperCase();
    if (active_obj.asset) check();
  });
})();
