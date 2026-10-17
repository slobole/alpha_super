/* Positions page: the holdings map (treemap) and sortable tables.
   The server sends every number and label (data-map-json). This file only
   lays tiles out, shows a tooltip and keeps the operator's choices (tile size,
   selected holding, table sort) across the 15-second refresh. */
(() => {
  'use strict';
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const MAP_HEIGHT_PX = 470;
  const NARROW_HEIGHT_PX = 380;
  const NARROW_WIDTH_PX = 560;
  const BAND_PX = 22;
  const TILE_LIMIT_INT = 14;
  const NARROW_TILE_LIMIT_INT = 8;
  const MINUS_STR = '−';
  let size_mode_str = 'value';
  let selected_key_str = '';
  let selected_scope_str = '';
  const sort_state_dict = {};
  try {
    const saved_str = window.sessionStorage && window.sessionStorage.getItem('v4-positions-map-size');
    if (saved_str === 'value' || saved_str === 'exposure') size_mode_str = saved_str;
  } catch (error_obj) { /* Storage may be blocked; the default stays. */ }

  // ---------- pure helpers (exported to tests) ----------
  function squarify_list(item_list, x_float, y_float, width_float, height_float) {
    // Squarified treemap (Bruls et al.): rows that keep tiles near square.
    const out_list = [];
    const total_float = item_list.reduce((sum_float, item_obj) => sum_float + item_obj.size_float, 0);
    if (!(total_float > 0) || width_float <= 1 || height_float <= 1) return out_list;
    const scale_float = width_float * height_float / total_float;
    const rest_list = item_list.map((item_obj) => ({area_float: item_obj.size_float * scale_float, item_obj}));
    let rx = x_float, ry = y_float, rw = width_float, rh = height_float, row_list = [];
    function worst_float(list_obj, side_float) {
      let sum_float = 0, max_float = 0, min_float = Infinity;
      for (const entry_obj of list_obj) {
        sum_float += entry_obj.area_float;
        max_float = Math.max(max_float, entry_obj.area_float);
        min_float = Math.min(min_float, entry_obj.area_float);
      }
      return Math.max(side_float * side_float * max_float / (sum_float * sum_float), (sum_float * sum_float) / (side_float * side_float * min_float));
    }
    function lay(list_obj) {
      const sum_float = list_obj.reduce((total, entry_obj) => total + entry_obj.area_float, 0);
      if (rw >= rh) {
        const column_float = sum_float / rh;
        let yy = ry;
        for (const entry_obj of list_obj) {
          const tile_height_float = entry_obj.area_float / column_float;
          out_list.push({x: rx, y: yy, w: column_float, h: tile_height_float, item_obj: entry_obj.item_obj});
          yy += tile_height_float;
        }
        rx += column_float; rw -= column_float;
      } else {
        const row_height_float = sum_float / rw;
        let xx = rx;
        for (const entry_obj of list_obj) {
          const tile_width_float = entry_obj.area_float / row_height_float;
          out_list.push({x: xx, y: ry, w: tile_width_float, h: row_height_float, item_obj: entry_obj.item_obj});
          xx += tile_width_float;
        }
        ry += row_height_float; rh -= row_height_float;
      }
    }
    let index_int = 0;
    while (index_int < rest_list.length) {
      const side_float = Math.min(rw, rh);
      if (row_list.length === 0 || worst_float(row_list.concat([rest_list[index_int]]), side_float) <= worst_float(row_list, side_float)) {
        row_list.push(rest_list[index_int]);
        index_int += 1;
      } else {
        lay(row_list);
        row_list = [];
      }
    }
    if (row_list.length) lay(row_list);
    return out_list;
  }

  function bucket_str(return_float) {
    if (return_float === null || return_float === undefined || !Number.isFinite(return_float)) return '0';
    const size_float = Math.abs(return_float);
    if (size_float < 0.005) return '0';
    return (return_float > 0 ? 'p' : 'n') + (size_float < 0.03 ? '1' : size_float < 0.10 ? '2' : '3');
  }

  function compact_money_str(value_float, signed_bool = false) {
    if (!Number.isFinite(value_float)) return '—';
    const rounded_float = Math.round(value_float);
    const prefix_str = rounded_float < 0 ? MINUS_STR : signed_bool && rounded_float > 0 ? '+' : '';
    return prefix_str + '$' + Math.abs(rounded_float).toLocaleString('en-US');
  }

  function fits_bool(text_str, font_px, width_float, factor_float = 0.6) {
    return text_str.length * font_px * factor_float <= width_float;
  }

  function size_of_float(item_obj, mode_str) {
    return mode_str === 'exposure' ? item_obj.value_float * Math.abs(item_obj.multiplier_float || 1) : item_obj.value_float;
  }

  // One block's tiles: largest first, a "+N more" tile past the limit, cash last (value mode only).
  function block_item_list(pod_obj, mode_str, limit_int) {
    const item_list = pod_obj.item_list.slice().sort((a_obj, b_obj) => size_of_float(b_obj, mode_str) - size_of_float(a_obj, mode_str)
      || a_obj.symbol_str.localeCompare(b_obj.symbol_str));
    let result_list = item_list;
    if (item_list.length > limit_int) {
      const rest_list = item_list.slice(limit_int - 1);
      const value_float = rest_list.reduce((sum_float, item_obj) => sum_float + size_of_float(item_obj, mode_str), 0);
      const pnl_known_bool = rest_list.every((item_obj) => item_obj.pnl_float !== null && item_obj.pnl_float !== undefined);
      const pnl_float = pnl_known_bool ? rest_list.reduce((sum_float, item_obj) => sum_float + item_obj.pnl_float, 0) : null;
      result_list = item_list.slice(0, limit_int - 1).concat([{key_str: '', more_bool: true, symbol_str: '+' + rest_list.length + ' more',
        value_float, multiplier_float: 1, pnl_float, return_float: null, pnl_str: pnl_known_bool ? compact_money_str(pnl_float, true) : '',
        return_str: '', status_str: 'more', status_note_str: '',
        tip_list: [['Holdings', rest_list.length + ' smaller positions'], [mode_str === 'exposure' ? 'Exposure' : 'Value', compact_money_str(value_float)]]
          .concat(pnl_known_bool ? [['P&L', compact_money_str(pnl_float, true)]] : [])
          .concat([['Symbols', rest_list.slice(0, 8).map((item_obj) => item_obj.symbol_str).join(', ') + (rest_list.length > 8 ? ', …' : '')]])}]);
    }
    if (mode_str === 'value' && pod_obj.cash_float > 0) {
      const share_str = pod_obj.nav_float > 0 ? (pod_obj.cash_float / pod_obj.nav_float * 100).toFixed(1) + '% of Pod' : '';
      result_list = result_list.concat([{key_str: '', cash_bool: true, symbol_str: 'Cash', value_float: pod_obj.cash_float, multiplier_float: 1,
        pnl_float: null, return_float: null, pnl_str: '', return_str: share_str, status_str: 'cash', status_note_str: '',
        tip_list: [['Cash', pod_obj.cash_str || compact_money_str(pod_obj.cash_float)], ['Share of Pod', share_str]]}]);
    }
    return result_list;
  }

  function header_text_list(pod_obj, mode_str, block_size_float, room_float) {
    // Drop detail until the header fits its block: name always, then P&L, then size.
    const name_str = pod_obj.name_str.toUpperCase();
    const size_str = mode_str === 'exposure'
      ? 'exp ' + compact_money_str(block_size_float) + ' · ' + (pod_obj.nav_float > 0 ? (block_size_float / pod_obj.nav_float).toFixed(2) : '—') + 'x NAV'
      : pod_obj.nav_str;
    const pnl_long_str = pod_obj.pnl_str ? pod_obj.pnl_str + (pod_obj.return_str ? ' (' + pod_obj.return_str + ')' : '') : '';
    const candidate_list = [[name_str + '  ·  ' + size_str + (pnl_long_str ? '  ·  ' : ''), pnl_long_str],
      [name_str + '  ·  ' + size_str + (pod_obj.pnl_str ? '  ·  ' : ''), pod_obj.pnl_str || ''],
      [name_str + (pod_obj.pnl_str ? '  ·  ' : ''), pod_obj.pnl_str || ''], [name_str, '']];
    for (const candidate_obj of candidate_list) {
      if (fits_bool(candidate_obj[0] + candidate_obj[1], 11, room_float, 0.66)) return candidate_obj;
    }
    return candidate_list[candidate_list.length - 1];
  }

  function compare_rows(a_obj, b_obj, key_str, type_str, direction_int) {
    const a_str = a_obj.getAttribute('data-sort-' + key_str) || '';
    const b_str = b_obj.getAttribute('data-sort-' + key_str) || '';
    const index_diff_int = Number(a_obj.getAttribute('data-sort-index')) - Number(b_obj.getAttribute('data-sort-index'));
    // Missing values sort last in both directions; ties keep the server order.
    if (a_str === '' || b_str === '') return a_str === b_str ? index_diff_int : a_str === '' ? 1 : -1;
    let result_int;
    if (type_str === 'number') {
      const a_float = Number(a_str), b_float = Number(b_str);
      if (!Number.isFinite(a_float) || !Number.isFinite(b_float)) return Number.isFinite(a_float) ? -1 : Number.isFinite(b_float) ? 1 : index_diff_int;
      result_int = a_float === b_float ? 0 : a_float < b_float ? -1 : 1;
    } else {
      result_int = a_str.localeCompare(b_str, 'en', {sensitivity: 'base', numeric: true});
    }
    return result_int === 0 ? index_diff_int : result_int * direction_int;
  }

  // ---------- table sorting ----------
  function apply_sort(table_obj) {
    const table_key_str = table_obj.getAttribute('data-sortable-table');
    const state_obj = sort_state_dict[table_key_str];
    table_obj.querySelectorAll('th[aria-sort]').forEach((head_obj) => {
      const button_obj = head_obj.querySelector('[data-sort-key]');
      const active_bool = Boolean(state_obj && button_obj && button_obj.getAttribute('data-sort-key') === state_obj.key_str);
      head_obj.setAttribute('aria-sort', active_bool ? (state_obj.direction_int > 0 ? 'ascending' : 'descending') : 'none');
    });
    const body_obj = table_obj.querySelector('tbody');
    if (!body_obj) return;
    const row_list = Array.from(body_obj.querySelectorAll('tr[data-sort-row]'));
    const key_str = state_obj ? state_obj.key_str : 'index';
    const type_str = state_obj ? state_obj.type_str : 'number';
    const direction_int = state_obj ? state_obj.direction_int : 1;
    row_list.sort((a_obj, b_obj) => compare_rows(a_obj, b_obj, key_str, type_str, direction_int));
    for (const row_obj of row_list) body_obj.appendChild(row_obj);
  }

  function sort_click(button_obj) {
    const table_obj = button_obj.closest('[data-sortable-table]');
    if (!table_obj) return;
    const table_key_str = table_obj.getAttribute('data-sortable-table');
    const key_str = button_obj.getAttribute('data-sort-key');
    const type_str = button_obj.getAttribute('data-sort-type') === 'text' ? 'text' : 'number';
    const previous_obj = sort_state_dict[table_key_str];
    // First click: text A→Z, numbers largest first. Next clicks flip.
    const direction_int = previous_obj && previous_obj.key_str === key_str ? -previous_obj.direction_int : (type_str === 'text' ? 1 : -1);
    sort_state_dict[table_key_str] = {key_str, type_str, direction_int};
    apply_sort(table_obj);
  }

  // ---------- map ----------
  function svg_el(tag_str, attribute_dict) {
    const element_obj = document.createElementNS(SVG_NS, tag_str);
    for (const [key_str, value_obj] of Object.entries(attribute_dict || {})) element_obj.setAttribute(key_str, String(value_obj));
    return element_obj;
  }

  function show_tip(panel_obj, item_obj, client_x, client_y) {
    const tip_obj = panel_obj.querySelector('[data-map-tip]');
    const wrap_obj = panel_obj.querySelector('[data-map-wrap]');
    if (!tip_obj || !wrap_obj) return;
    tip_obj.textContent = '';
    const title_obj = document.createElement('div');
    title_obj.className = 't';
    title_obj.textContent = item_obj.cash_bool ? 'Cash' : item_obj.more_bool ? item_obj.symbol_str : item_obj.symbol_str + (item_obj.pod_name_str ? ' · ' + item_obj.pod_name_str : '');
    tip_obj.appendChild(title_obj);
    for (const row_list of item_obj.tip_list || []) {
      const row_obj = document.createElement('div');
      row_obj.className = 'kv';
      const key_obj = document.createElement('span');
      key_obj.textContent = row_list[0];
      const value_obj = document.createElement('b');
      value_obj.textContent = row_list[1];
      row_obj.append(key_obj, value_obj);
      tip_obj.appendChild(row_obj);
    }
    if (Math.abs(item_obj.multiplier_float || 1) !== 1 && size_mode_str === 'value') {
      const note_obj = document.createElement('div');
      note_obj.className = 'note';
      note_obj.textContent = 'Tile shows $ value; exposure is ' + Math.abs(item_obj.multiplier_float) + 'x. Switch Size to Exposure.';
      tip_obj.appendChild(note_obj);
    }
    tip_obj.hidden = false;
    const wrap_rect = wrap_obj.getBoundingClientRect();
    let left_float = client_x - wrap_rect.left + 14, top_float = client_y - wrap_rect.top + 14;
    if (left_float + tip_obj.offsetWidth > wrap_rect.width - 6) left_float = client_x - wrap_rect.left - tip_obj.offsetWidth - 14;
    if (top_float + tip_obj.offsetHeight > wrap_rect.height - 6) top_float = client_y - wrap_rect.top - tip_obj.offsetHeight - 14;
    tip_obj.style.left = Math.max(0, left_float) + 'px';
    tip_obj.style.top = Math.max(0, top_float) + 'px';
  }

  function hide_tip(panel_obj) {
    const tip_obj = panel_obj.querySelector('[data-map-tip]');
    if (tip_obj) tip_obj.hidden = true;
  }

  function draw_map(panel_obj, focus_key_str = '') {
    const canvas_obj = panel_obj.querySelector('[data-map-canvas]');
    const json_str = canvas_obj ? canvas_obj.getAttribute('data-map-json') : null;
    if (!canvas_obj || !json_str) return;
    // Keyboard focus stays on the same holding when the map is redrawn.
    const active_obj = document.activeElement;
    const kept_key_str = focus_key_str || (active_obj && canvas_obj.contains && canvas_obj.contains(active_obj)
      && active_obj.getAttribute ? active_obj.getAttribute('data-map-key') || '' : '');
    let map_obj;
    try {
      map_obj = JSON.parse(json_str);
    } catch (error_obj) {
      canvas_obj.textContent = 'The map could not be drawn.';
      return;
    }
    if (!map_obj || !map_obj.available_bool) return;
    const width_float = Math.max(280, Math.floor(canvas_obj.clientWidth || 0));
    const narrow_bool = width_float < NARROW_WIDTH_PX;
    const height_float = narrow_bool ? NARROW_HEIGHT_PX : MAP_HEIGHT_PX;
    const limit_int = narrow_bool ? NARROW_TILE_LIMIT_INT : TILE_LIMIT_INT;
    const block_list = map_obj.pod_list.map((pod_obj) => {
      const item_list = block_item_list(pod_obj, size_mode_str, limit_int);
      return {pod_obj, item_list, size_float: item_list.reduce((sum_float, item_obj) => sum_float + size_of_float(item_obj, size_mode_str), 0)};
    }).filter((block_obj) => block_obj.size_float > 0).sort((a_obj, b_obj) => b_obj.size_float - a_obj.size_float);
    const svg_obj = svg_el('svg', {width: width_float, height: height_float, viewBox: `0 0 ${width_float} ${height_float}`,
      class: 'positions-map-svg', role: 'group', 'aria-label': 'Holdings map, ' + (size_mode_str === 'exposure' ? 'sized by exposure' : 'sized by value')});
    const defs_obj = svg_el('defs');
    const pattern_obj = svg_el('pattern', {id: 'positions-map-hatch', patternUnits: 'userSpaceOnUse', width: 6, height: 6, patternTransform: 'rotate(45)'});
    pattern_obj.appendChild(svg_el('rect', {width: 2.2, height: 6, fill: '#ffffff', opacity: 0.5}));
    defs_obj.appendChild(pattern_obj);
    svg_obj.appendChild(defs_obj);
    for (const block_rect of squarify_list(block_list.map((block_obj) => ({size_float: block_obj.size_float, block_obj})), 0, 0, width_float, height_float)) {
      const block_obj = block_rect.item_obj.block_obj;
      const pod_obj = block_obj.pod_obj;
      svg_obj.appendChild(svg_el('rect', {x: block_rect.x + 1, y: block_rect.y + 1, width: Math.max(0, block_rect.w - 2),
        height: Math.max(0, block_rect.h - 2), class: 'pm-block'}));
      svg_obj.appendChild(svg_el('rect', {x: block_rect.x + 1, y: block_rect.y + 1, width: Math.max(0, block_rect.w - 2), height: 3,
        fill: pod_obj.color_str}));
      const text_pair_list = header_text_list(pod_obj, size_mode_str, block_obj.size_float, block_rect.w - 16);
      const head_obj = svg_el('text', {x: block_rect.x + 8, y: block_rect.y + 17, class: 'pm-head'});
      head_obj.textContent = text_pair_list[0];
      if (text_pair_list[1]) {
        const pnl_obj = svg_el('tspan', {class: (pod_obj.pnl_float || 0) >= 0 ? 'pm-pos' : 'pm-neg'});
        pnl_obj.textContent = text_pair_list[1];
        head_obj.appendChild(pnl_obj);
      }
      svg_obj.appendChild(head_obj);
      const tile_list = squarify_list(block_obj.item_list.map((item_obj) => ({size_float: size_of_float(item_obj, size_mode_str), item_obj})),
        block_rect.x + 3, block_rect.y + BAND_PX + 2, block_rect.w - 6, block_rect.h - BAND_PX - 5);
      for (const tile_rect of tile_list) {
        const item_obj = tile_rect.item_obj.item_obj;
        const tone_str = item_obj.cash_bool ? 'cash' : item_obj.more_bool ? 'more' : bucket_str(item_obj.return_float);
        const group_obj = svg_el('g', {class: 'pm-tile tm-' + tone_str + (item_obj.key_str && item_obj.key_str === selected_key_str ? ' sel' : ''),
          'data-map-key': item_obj.key_str || ''});
        if (item_obj.key_str) {
          group_obj.setAttribute('tabindex', '0');
          group_obj.setAttribute('role', 'button');
        }
        group_obj.setAttribute('aria-label', [item_obj.symbol_str, item_obj.pod_name_str || '', item_obj.return_str, item_obj.pnl_str]
          .filter(Boolean).join(', '));
        group_obj.appendChild(svg_el('rect', {x: tile_rect.x + 1, y: tile_rect.y + 1, width: Math.max(0, tile_rect.w - 2),
          height: Math.max(0, tile_rect.h - 2), class: 'pm-fill'}));
        if (item_obj.status_str === 'sold') {
          group_obj.appendChild(svg_el('rect', {x: tile_rect.x + 1, y: tile_rect.y + 1, width: Math.max(0, tile_rect.w - 2),
            height: Math.max(0, tile_rect.h - 2), fill: 'url(#positions-map-hatch)', class: 'pm-hatch'}));
        }
        const font_px = tile_rect.w > 90 && tile_rect.h > 46 ? 13 : 11.5;
        if (fits_bool(item_obj.symbol_str, font_px, tile_rect.w - 10, 0.7) && tile_rect.h >= font_px + 8) {
          const name_obj = svg_el('text', {x: tile_rect.x + 7, y: tile_rect.y + font_px + 5, class: 'pm-name', 'font-size': font_px});
          name_obj.textContent = item_obj.symbol_str;
          group_obj.appendChild(name_obj);
          const line_list = [item_obj.return_str, [item_obj.pnl_str, item_obj.status_note_str].filter(Boolean).join(' · ')];
          line_list.forEach((line_str, line_index_int) => {
            const line_y_float = tile_rect.y + font_px * (line_index_int + 2) + 6 + 2 * line_index_int;
            if (!line_str || tile_rect.h < line_y_float - tile_rect.y + 6) return;
            const fallback_str = line_index_int === 1 ? item_obj.pnl_str : '';
            const fitted_str = fits_bool(line_str, font_px - 1.5, tile_rect.w - 12) ? line_str
              : fallback_str && fits_bool(fallback_str, font_px - 1.5, tile_rect.w - 12) ? fallback_str : '';
            if (!fitted_str) return;
            const line_obj = svg_el('text', {x: tile_rect.x + 7, y: line_y_float, class: 'pm-sub', 'font-size': font_px - 1.5});
            line_obj.textContent = fitted_str;
            group_obj.appendChild(line_obj);
          });
        }
        group_obj.addEventListener('pointermove', (event_obj) => show_tip(panel_obj, item_obj, event_obj.clientX, event_obj.clientY));
        group_obj.addEventListener('pointerleave', () => hide_tip(panel_obj));
        group_obj.addEventListener('focus', () => {
          const box_obj = group_obj.getBoundingClientRect();
          show_tip(panel_obj, item_obj, box_obj.left + box_obj.width / 2, box_obj.top + box_obj.height / 2);
        });
        group_obj.addEventListener('blur', () => hide_tip(panel_obj));
        if (item_obj.key_str) {
          group_obj.addEventListener('click', () => select_key(item_obj.key_str, true));
          group_obj.addEventListener('keydown', (event_obj) => {
            if (event_obj.key === 'Enter' || event_obj.key === ' ') {
              event_obj.preventDefault();
              select_key(item_obj.key_str, true);
            }
          });
        }
        svg_obj.appendChild(group_obj);
      }
    }
    canvas_obj.textContent = '';
    canvas_obj.appendChild(svg_obj);
    if (kept_key_str) {
      const tile_obj = Array.from(svg_obj.querySelectorAll('[data-map-key]')).find((candidate_obj) => candidate_obj.getAttribute('data-map-key') === kept_key_str);
      if (tile_obj && typeof tile_obj.focus === 'function') tile_obj.focus({preventScroll: true});
    }
    const legend_obj = panel_obj.querySelector('[data-map-size-legend]');
    if (legend_obj) legend_obj.textContent = legend_obj.getAttribute(size_mode_str === 'exposure' ? 'data-exposure-legend' : 'data-value-legend') || '';
    panel_obj.querySelectorAll('[data-map-size]').forEach((button_obj) => {
      button_obj.setAttribute('aria-pressed', button_obj.getAttribute('data-map-size') === size_mode_str ? 'true' : 'false');
    });
  }

  function highlight_selection() {
    const symbol_str = selected_key_str.split('|')[0];
    document.querySelectorAll('[data-map-key]').forEach((tile_obj) => {
      tile_obj.classList.toggle('sel', Boolean(selected_key_str) && tile_obj.getAttribute('data-map-key') === selected_key_str);
    });
    document.querySelectorAll('[data-position-row]').forEach((row_obj) => {
      const key_attr_str = row_obj.getAttribute('data-position-key');
      const match_bool = Boolean(selected_key_str) && (key_attr_str ? key_attr_str === selected_key_str
        : row_obj.getAttribute('data-position-symbol') === symbol_str);
      row_obj.classList.toggle('hl', match_bool);
    });
  }

  function select_key(key_str, scroll_bool) {
    selected_key_str = selected_key_str === key_str ? '' : key_str;
    highlight_selection();
    if (!selected_key_str || !scroll_bool) return;
    // The page moves under a still pointer: close the tile facts first.
    document.querySelectorAll('[data-map-tip]').forEach((tip_obj) => { tip_obj.hidden = true; });
    const row_obj = Array.from(document.querySelectorAll('[data-position-row].hl')).find((candidate_obj) => !candidate_obj.hidden);
    if (row_obj && typeof row_obj.scrollIntoView === 'function') row_obj.scrollIntoView({behavior: 'smooth', block: 'center'});
  }

  function refresh_page(focus_key_str = '') {
    const panel_obj = document.querySelector('[data-positions-map]');
    const scope_str = panel_obj ? panel_obj.getAttribute('data-map-scope') || '' : '';
    // A different month or Pod filter is a new question: drop the old selection.
    if (scope_str !== selected_scope_str) {
      selected_scope_str = scope_str;
      selected_key_str = '';
      focus_key_str = '';
    }
    if (panel_obj) draw_map(panel_obj, focus_key_str);
    document.querySelectorAll('[data-sortable-table]').forEach(apply_sort);
    highlight_selection();
  }

  document.addEventListener('click', (event_obj) => {
    const target_obj = event_obj.target;
    if (!target_obj || !target_obj.closest) return;
    // The whole header cell sorts, not only its label.
    const head_obj = target_obj.closest('th[aria-sort]');
    const sort_obj = target_obj.closest('[data-sort-key]') || (head_obj ? head_obj.querySelector('[data-sort-key]') : null);
    if (sort_obj) {
      sort_click(sort_obj);
      return;
    }
    const size_obj = target_obj.closest('[data-map-size]');
    if (size_obj) {
      size_mode_str = size_obj.getAttribute('data-map-size') === 'exposure' ? 'exposure' : 'value';
      try {
        if (window.sessionStorage) window.sessionStorage.setItem('v4-positions-map-size', size_mode_str);
      } catch (error_obj) { /* Storage may be blocked; the choice lasts for this page. */ }
      const panel_obj = size_obj.closest('[data-positions-map]');
      if (panel_obj) draw_map(panel_obj);
      highlight_selection();
      return;
    }
    const mover_obj = target_obj.closest('[data-map-select]');
    if (mover_obj) {
      select_key(mover_obj.getAttribute('data-map-select'), true);
      return;
    }
    const row_obj = target_obj.closest('[data-position-row]');
    if (row_obj && !target_obj.closest('a, button, input')) {
      // Now rows merge Pods by symbol: select that symbol's first tile.
      const prefix_str = (row_obj.getAttribute('data-position-symbol') || '') + '|';
      const tile_obj = Array.from(document.querySelectorAll('[data-map-key]'))
        .find((candidate_obj) => (candidate_obj.getAttribute('data-map-key') || '').startsWith(prefix_str));
      const key_str = row_obj.getAttribute('data-position-key') || (tile_obj ? tile_obj.getAttribute('data-map-key') : '');
      if (key_str) select_key(key_str, false);
    }
  });
  // Same test as overview.js: the swapped target is the (old) shell or inside it.
  function shell_event_bool(event_obj) {
    const target_obj = event_obj.detail && (event_obj.detail.target || event_obj.detail.elt);
    return Boolean(target_obj && (target_obj.id === 'overview-shell' || (target_obj.closest && target_obj.closest('#overview-shell'))));
  }
  let swap_focus_key_str = '';
  document.addEventListener('htmx:beforeSwap', (event_obj) => {
    if (!shell_event_bool(event_obj)) return;
    const active_obj = document.activeElement;
    swap_focus_key_str = active_obj && active_obj.getAttribute ? active_obj.getAttribute('data-map-key') || '' : '';
  });
  document.addEventListener('htmx:afterSwap', (event_obj) => {
    if (!shell_event_bool(event_obj)) return;
    const focus_key_str = swap_focus_key_str;
    swap_focus_key_str = '';
    refresh_page(focus_key_str);
  });
  let resize_frame_int = 0;
  window.addEventListener('resize', () => {
    if (resize_frame_int) return;
    resize_frame_int = window.requestAnimationFrame(() => {
      resize_frame_int = 0;
      const panel_obj = document.querySelector('[data-positions-map]');
      if (panel_obj) draw_map(panel_obj);
      highlight_selection();
    });
  });
  if (typeof positions_test_hook === 'function') {
    // eslint-disable-next-line no-undef
    positions_test_hook({squarify_list, bucket_str, compact_money_str, fits_bool, block_item_list, header_text_list, compare_rows,
      apply_sort, sort_click, refresh_page, select_key, draw_map, highlight_selection,
      get_state: () => ({size_mode_str, selected_key_str, sort_state_dict})});
  }
  refresh_page();
})();
