/* Minimal SVG charts for the fund menu report. No dependencies.
   Colors come from CSS custom properties so both themes resolve from one token set.
   Every chart is paired with a table in the page (tooltips enhance, never gate). */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const el = (tag, attrs, parent) => {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  };
  const fmtPct = (v, d = 1) => (v === null || v === undefined || isNaN(v) ? "–" : (v * 100).toFixed(d) + "%");
  const fmtX = (v) => (v >= 10 ? v.toFixed(0) : v.toFixed(v >= 2 ? 1 : 2)) + "×";
  const parseDate = (s) => new Date(s + "T00:00:00Z");
  const yearOf = (t) => new Date(t).getUTCFullYear();

  function niceTicks(lo, hi, count) {
    const span = hi - lo;
    if (span <= 0) return [lo];
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(12));
    return out;
  }

  function logTicks(lo, hi) {
    const candidates = [0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50];
    const inRange = candidates.filter((v) => v >= lo && v <= hi);
    if (inRange.length <= 6) return inRange;
    const coarse = [1, 2, 4, 8, 16, 32, 64].filter((v) => v >= lo && v <= hi);
    return coarse.length >= 3 ? coarse : inRange.filter((_, i) => i % 2 === 0);
  }

  function makeTooltip(host) {
    const tip = document.createElement("div");
    tip.className = "viz-tip";
    tip.hidden = true;
    host.appendChild(tip);
    return tip;
  }

  function placeTip(tip, host, x, y) {
    const hostRect = host.getBoundingClientRect();
    const tipRect = tip.getBoundingClientRect();
    let left = x + 14;
    if (left + tipRect.width > hostRect.width - 4) left = x - tipRect.width - 14;
    let top = y - tipRect.height / 2;
    top = Math.max(4, Math.min(top, hostRect.height - tipRect.height - 4));
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  function tipRow(tip, color, value, label, dashed) {
    const row = document.createElement("div");
    row.className = "viz-tip-row";
    const key = document.createElement("span");
    key.className = "viz-tip-key" + (dashed ? " dashed" : "");
    key.style.borderColor = color;
    const val = document.createElement("strong");
    val.textContent = value;
    const lab = document.createElement("span");
    lab.className = "viz-tip-label";
    lab.textContent = label;
    row.append(key, val, lab);
    tip.appendChild(row);
  }

  /* Multi-series time line chart. opts: {series:[{name,colorVar,dashed,values:[...]}], dates:[...],
     yLog:bool, yFormat:'x'|'pct', height, endLabels:bool, yMax, yMin} */
  function lineChart(host, opts) {
    host.innerHTML = "";
    host.classList.add("viz-host");
    const width = Math.max(320, host.clientWidth || 720);
    const height = opts.height || 320;
    const margin = { top: 12, right: opts.endLabels ? 118 : 16, bottom: 28, left: 52 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const times = opts.dates.map((d) => parseDate(d).getTime());
    const t0 = times[0], t1 = times[times.length - 1];
    let vals = [];
    opts.series.forEach((s) => s.values.forEach((v) => { if (v !== null && isFinite(v)) vals.push(v); }));
    let lo = opts.yMin !== undefined ? opts.yMin : Math.min(...vals);
    let hi = opts.yMax !== undefined ? opts.yMax : Math.max(...vals);
    if (opts.yLog) { lo = Math.max(lo * 0.95, 1e-6); hi = hi * 1.05; }
    else { const pad = (hi - lo) * 0.06 || 0.01; lo = opts.yMin !== undefined ? lo : lo - pad; hi = opts.yMax !== undefined ? hi : hi + pad; }
    const xs = (t) => margin.left + ((t - t0) / (t1 - t0)) * innerW;
    const ys = opts.yLog
      ? (v) => margin.top + innerH - ((Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * innerH
      : (v) => margin.top + innerH - ((v - lo) / (hi - lo)) * innerH;
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height: height, role: "img", "aria-label": opts.ariaLabel || "line chart" }, host);
    const yTicks = opts.yLog ? logTicks(lo, hi) : niceTicks(lo, hi, 5);
    yTicks.forEach((v) => {
      const y = ys(v);
      el("line", { x1: margin.left, x2: margin.left + innerW, y1: y, y2: y, class: v === (opts.yLog ? 1 : 0) ? "viz-base" : "viz-grid" }, svg);
      const label = el("text", { x: margin.left - 8, y: y + 4, "text-anchor": "end", class: "viz-axis" }, svg);
      label.textContent = opts.yFormat === "pct" ? fmtPct(v, 0) : fmtX(v);
    });
    const firstYear = yearOf(t0), lastYear = yearOf(t1);
    const yearStep = lastYear - firstYear > 14 ? 3 : lastYear - firstYear > 7 ? 2 : 1;
    for (let yr = firstYear + 1; yr <= lastYear; yr += 1) {
      if ((yr - firstYear - 1) % yearStep !== 0) continue;
      const t = Date.UTC(yr, 0, 1);
      if (t < t0 || t > t1) continue;
      const x = xs(t);
      el("line", { x1: x, x2: x, y1: margin.top + innerH, y2: margin.top + innerH + 4, class: "viz-base" }, svg);
      const label = el("text", { x: x, y: height - 8, "text-anchor": "middle", class: "viz-axis" }, svg);
      label.textContent = String(yr);
    }
    opts.series.forEach((s) => {
      let d = "";
      s.values.forEach((v, i) => {
        if (v === null || !isFinite(v)) return;
        d += (d ? "L" : "M") + xs(times[i]).toFixed(1) + " " + ys(v).toFixed(1);
      });
      el("path", { d: d, fill: "none", stroke: `var(${s.colorVar})`, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dashed ? "5 4" : "none", class: "viz-line" }, svg);
    });
    if (opts.endLabels) {
      const ends = opts.series.map((s) => ({ s, v: s.values[s.values.length - 1] })).filter((e) => e.v !== null && isFinite(e.v));
      ends.sort((a, b) => ys(a.v) - ys(b.v));
      let lastY = -Infinity;
      ends.forEach((e) => {
        let y = ys(e.v);
        const natural = y;
        if (y < lastY + 13) y = lastY + 13;
        lastY = y;
        const xEnd = margin.left + innerW;
        if (Math.abs(y - natural) > 1) el("line", { x1: xEnd + 2, x2: xEnd + 8, y1: natural, y2: y, class: "viz-leader" }, svg);
        el("circle", { cx: xEnd, cy: natural, r: 3.5, fill: `var(${e.s.colorVar})`, class: "viz-endpoint" }, svg);
        const label = el("text", { x: xEnd + 10, y: y + 4, class: "viz-endlabel" }, svg);
        label.textContent = `${e.s.name} ${opts.yFormat === "pct" ? fmtPct(e.v) : fmtX(e.v)}`;
      });
    }
    const cross = el("line", { x1: 0, x2: 0, y1: margin.top, y2: margin.top + innerH, class: "viz-cross", visibility: "hidden" }, svg);
    const tip = makeTooltip(host);
    const overlay = el("rect", { x: margin.left, y: margin.top, width: innerW, height: innerH, fill: "transparent", tabindex: 0, "aria-label": "chart readout" }, svg);
    const show = (clientX) => {
      const rect = svg.getBoundingClientRect();
      const scale = width / rect.width;
      const px = (clientX - rect.left) * scale;
      const t = t0 + ((px - margin.left) / innerW) * (t1 - t0);
      let lo2 = 0, hi2 = times.length - 1;
      while (hi2 - lo2 > 1) { const mid = (lo2 + hi2) >> 1; if (times[mid] < t) lo2 = mid; else hi2 = mid; }
      const i = Math.abs(times[lo2] - t) < Math.abs(times[hi2] - t) ? lo2 : hi2;
      const x = xs(times[i]);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("visibility", "visible");
      tip.innerHTML = "";
      const head = document.createElement("div");
      head.className = "viz-tip-head";
      head.textContent = opts.dates[i];
      tip.appendChild(head);
      opts.series
        .map((s) => ({ s, v: s.values[i] }))
        .filter((r) => r.v !== null && isFinite(r.v))
        .sort((a, b) => b.v - a.v)
        .forEach((r) => tipRow(tip, `var(${r.s.colorVar})`, opts.yFormat === "pct" ? fmtPct(r.v) : fmtX(r.v), r.s.name, r.s.dashed));
      tip.hidden = false;
      placeTip(tip, host, x / scale, (margin.top + innerH / 3) / scale);
    };
    overlay.addEventListener("pointermove", (e) => show(e.clientX));
    overlay.addEventListener("pointerleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
    overlay.addEventListener("focus", () => { const r = svg.getBoundingClientRect(); show(r.right - 2); });
    overlay.addEventListener("blur", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
  }

  /* Scatter: opts {points:[{x,y,label,colorVar,kind:'product'|'bench'|'legacy'|'sleeve'}], xLabel, yLabel} */
  function scatterChart(host, opts) {
    host.innerHTML = "";
    host.classList.add("viz-host");
    const width = Math.max(320, host.clientWidth || 720);
    const height = opts.height || 360;
    const margin = { top: 14, right: 20, bottom: 44, left: 56 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const xsV = opts.points.map((p) => p.x), ysV = opts.points.map((p) => p.y);
    const xLo = 0, xHi = Math.max(...xsV) * 1.12;
    const yLo = Math.min(0, Math.min(...ysV) - 0.01), yHi = Math.max(...ysV) * 1.12;
    const xs = (v) => margin.left + ((v - xLo) / (xHi - xLo)) * innerW;
    const ys = (v) => margin.top + innerH - ((v - yLo) / (yHi - yLo)) * innerH;
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height: height, role: "img", "aria-label": opts.ariaLabel || "scatter chart" }, host);
    niceTicks(yLo, yHi, 5).forEach((v) => {
      el("line", { x1: margin.left, x2: margin.left + innerW, y1: ys(v), y2: ys(v), class: v === 0 ? "viz-base" : "viz-grid" }, svg);
      el("text", { x: margin.left - 8, y: ys(v) + 4, "text-anchor": "end", class: "viz-axis" }, svg).textContent = fmtPct(v, 0);
    });
    niceTicks(xLo, xHi, 6).forEach((v) => {
      el("line", { x1: xs(v), x2: xs(v), y1: margin.top, y2: margin.top + innerH, class: v === 0 ? "viz-base" : "viz-grid" }, svg);
      el("text", { x: xs(v), y: margin.top + innerH + 18, "text-anchor": "middle", class: "viz-axis" }, svg).textContent = fmtPct(v, 0);
    });
    el("text", { x: margin.left + innerW / 2, y: height - 6, "text-anchor": "middle", class: "viz-axis-title" }, svg).textContent = opts.xLabel;
    const yt = el("text", { x: 14, y: margin.top + innerH / 2, "text-anchor": "middle", class: "viz-axis-title", transform: `rotate(-90 14 ${margin.top + innerH / 2})` }, svg);
    yt.textContent = opts.yLabel;
    const tip = makeTooltip(host);
    const placed = [];
    opts.points.forEach((p) => {
      const cx = xs(p.x), cy = ys(p.y);
      const g = el("g", { class: "viz-pt", tabindex: 0 }, svg);
      if (p.kind === "legacy") el("circle", { cx, cy, r: 5, fill: "var(--surface)", stroke: `var(${p.colorVar})`, "stroke-width": 2 }, g);
      else if (p.kind === "bench") el("rect", { x: cx - 5, y: cy - 5, width: 10, height: 10, fill: `var(${p.colorVar})`, stroke: "var(--surface)", "stroke-width": 2 }, g);
      else if (p.kind === "sleeve") el("circle", { cx, cy, r: 3.5, fill: `var(${p.colorVar})`, "fill-opacity": 0.55 }, g);
      else el("circle", { cx, cy, r: 6.5, fill: `var(${p.colorVar})`, stroke: "var(--surface)", "stroke-width": 2 }, g);
      el("circle", { cx, cy, r: 13, fill: "transparent" }, g);
      if (p.label && p.kind !== "sleeve") {
        let ly = cy - 10;
        for (const q of placed) if (Math.abs(q.x - cx) < 70 && Math.abs(q.y - ly) < 13) ly = q.y - 14;
        placed.push({ x: cx, y: ly });
        const labelNode = el("text", { x: cx + 9, y: ly, class: "viz-pointlabel" }, svg);
        labelNode.textContent = p.label;
        const labelWidth = labelNode.getComputedTextLength ? labelNode.getComputedTextLength() : p.label.length * 6.5;
        if (cx + 9 + labelWidth > width - 4) {
          labelNode.setAttribute("x", cx - 9);
          labelNode.setAttribute("text-anchor", "end");
        }
      }
      const showTip = () => {
        tip.innerHTML = "";
        const head = document.createElement("div");
        head.className = "viz-tip-head";
        head.textContent = p.label || p.name;
        tip.appendChild(head);
        tipRow(tip, `var(${p.colorVar})`, fmtPct(p.y), "CAGR");
        tipRow(tip, `var(${p.colorVar})`, fmtPct(p.x), "volatility");
        if (p.extra) tipRow(tip, `var(${p.colorVar})`, p.extra.value, p.extra.label);
        tip.hidden = false;
        const rect = svg.getBoundingClientRect();
        const scale = width / rect.width;
        placeTip(tip, host, cx / scale, cy / scale);
      };
      g.addEventListener("pointerenter", showTip);
      g.addEventListener("focus", showTip);
      g.addEventListener("pointerleave", () => (tip.hidden = true));
      g.addEventListener("blur", () => (tip.hidden = true));
    });
  }

  /* Horizontal stacked bars: opts {rows:[{label, segments:[{key,value}]}], keys:[{key,label,colorVar}]} */
  function stackedBars(host, opts) {
    host.innerHTML = "";
    host.classList.add("viz-host");
    const width = Math.max(320, host.clientWidth || 720);
    const barH = 18, gap = 14;
    const margin = { top: 6, right: 12, bottom: 22, left: opts.labelWidth || 150 };
    const height = margin.top + margin.bottom + opts.rows.length * (barH + gap);
    const innerW = width - margin.left - margin.right;
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height: height, role: "img", "aria-label": opts.ariaLabel || "stacked bars" }, host);
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      const x = margin.left + v * innerW;
      el("line", { x1: x, x2: x, y1: margin.top, y2: height - margin.bottom, class: v === 0 ? "viz-base" : "viz-grid" }, svg);
      el("text", { x: x, y: height - 6, "text-anchor": "middle", class: "viz-axis" }, svg).textContent = (v * 100).toFixed(0) + "%";
    });
    const tip = makeTooltip(host);
    const colorOf = {};
    opts.keys.forEach((k) => (colorOf[k.key] = k));
    opts.rows.forEach((row, ri) => {
      const y = margin.top + ri * (barH + gap) + gap / 2;
      el("text", { x: margin.left - 10, y: y + barH / 2 + 4, "text-anchor": "end", class: "viz-rowlabel" }, svg).textContent = row.label;
      let acc = 0;
      const total = row.segments.reduce((a, s) => a + s.value, 0) || 1;
      row.segments.forEach((seg) => {
        const w = (seg.value / total) * innerW;
        if (w <= 0) return;
        const x = margin.left + acc;
        const rect = el("rect", { x: x + 1, y: y, width: Math.max(w - 2, 0.5), height: barH, fill: `var(${colorOf[seg.key].colorVar})`, rx: 2, class: "viz-seg", tabindex: 0 }, svg);
        if (w > 34) {
          const t = el("text", { x: x + w / 2, y: y + barH / 2 + 4, "text-anchor": "middle", class: "viz-seglabel" }, svg);
          t.textContent = ((seg.value / total) * 100).toFixed(0) + "%";
        }
        const showTip = () => {
          tip.innerHTML = "";
          const head = document.createElement("div");
          head.className = "viz-tip-head";
          head.textContent = row.label;
          tip.appendChild(head);
          tipRow(tip, `var(${colorOf[seg.key].colorVar})`, ((seg.value / total) * 100).toFixed(1) + "%", colorOf[seg.key].label);
          tip.hidden = false;
          const r = svg.getBoundingClientRect();
          const scale = width / r.width;
          placeTip(tip, host, (x + w / 2) / scale, (y + barH / 2) / scale);
        };
        rect.addEventListener("pointerenter", showTip);
        rect.addEventListener("focus", showTip);
        rect.addEventListener("pointerleave", () => (tip.hidden = true));
        rect.addEventListener("blur", () => (tip.hidden = true));
        acc += w;
      });
    });
  }

  /* Dot and whisker: opts {rows:[{label, lo, mid, hi, colorVar, point}], format:'pct'|'num', refValue, refLabel} */
  function dotWhisker(host, opts) {
    host.innerHTML = "";
    host.classList.add("viz-host");
    const width = Math.max(320, host.clientWidth || 720);
    const rowH = 30;
    const margin = { top: 8, right: 20, bottom: 26, left: opts.labelWidth || 150 };
    const height = margin.top + margin.bottom + opts.rows.length * rowH;
    const innerW = width - margin.left - margin.right;
    const all = [];
    opts.rows.forEach((r) => all.push(r.lo, r.hi, r.point ?? r.mid));
    if (opts.refValue !== undefined) all.push(opts.refValue);
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo) * 0.08;
    lo -= pad; hi += pad;
    const xs = (v) => margin.left + ((v - lo) / (hi - lo)) * innerW;
    const fmt = opts.format === "pct" ? (v) => fmtPct(v, 0) : (v) => v.toFixed(2);
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height: height, role: "img", "aria-label": opts.ariaLabel || "interval chart" }, host);
    niceTicks(lo, hi, 5).forEach((v) => {
      el("line", { x1: xs(v), x2: xs(v), y1: margin.top, y2: height - margin.bottom, class: v === 0 ? "viz-base" : "viz-grid" }, svg);
      el("text", { x: xs(v), y: height - 8, "text-anchor": "middle", class: "viz-axis" }, svg).textContent = fmt(v);
    });
    if (opts.refValue !== undefined) {
      el("line", { x1: xs(opts.refValue), x2: xs(opts.refValue), y1: margin.top, y2: height - margin.bottom, class: "viz-ref" }, svg);
    }
    const tip = makeTooltip(host);
    opts.rows.forEach((r, i) => {
      const y = margin.top + i * rowH + rowH / 2;
      el("text", { x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "viz-rowlabel" }, svg).textContent = r.label;
      el("line", { x1: xs(r.lo), x2: xs(r.hi), y1: y, y2: y, stroke: `var(${r.colorVar})`, "stroke-width": 2, "stroke-linecap": "round" }, svg);
      const g = el("g", { tabindex: 0 }, svg);
      el("circle", { cx: xs(r.point ?? r.mid), cy: y, r: 5, fill: `var(${r.colorVar})`, stroke: "var(--surface)", "stroke-width": 2 }, g);
      el("rect", { x: xs(r.lo) - 4, y: y - 12, width: Math.max(xs(r.hi) - xs(r.lo) + 8, 24), height: 24, fill: "transparent" }, g);
      const showTip = () => {
        tip.innerHTML = "";
        const head = document.createElement("div");
        head.className = "viz-tip-head";
        head.textContent = r.label;
        tip.appendChild(head);
        tipRow(tip, `var(${r.colorVar})`, fmt(r.point ?? r.mid), r.pointLabel || "estimate");
        tipRow(tip, `var(${r.colorVar})`, fmt(r.lo) + " to " + fmt(r.hi), "90% range");
        tip.hidden = false;
        const rect = svg.getBoundingClientRect();
        const scale = width / rect.width;
        placeTip(tip, host, xs(r.point ?? r.mid) / scale, y / scale);
      };
      g.addEventListener("pointerenter", showTip);
      g.addEventListener("focus", showTip);
      g.addEventListener("pointerleave", () => (tip.hidden = true));
      g.addEventListener("blur", () => (tip.hidden = true));
    });
  }

  window.FundCharts = { lineChart, scatterChart, stackedBars, dotWhisker, fmtPct };
})();
