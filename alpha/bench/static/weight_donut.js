/* Weight donut shared by the builder and the review page.
 *
 * Draws a basket as proportions plus a legend carrying the exact figures.
 * Slices step down one monochrome ramp rather than introducing a palette: the
 * console owns one ink and one red, and a basket chart is not a reason to
 * invent five more colours.
 */
(function (global) {
  var RADIUS = 46;
  var CIRCUMFERENCE = 2 * Math.PI * RADIUS;
  var SLICE_OPACITY_LIST = [1, 0.78, 0.6, 0.46, 0.35, 0.27, 0.2, 0.15];
  var SVG_NS = 'http://www.w3.org/2000/svg';

  /* itemList: [{label, weight}] — weights need not sum to 1; shares are taken
   * against their total so the ring always closes and the caller decides
   * separately whether an unbalanced total is worth flagging. */
  function render(slicesEl, legendEl, itemList) {
    slicesEl.innerHTML = '';
    legendEl.innerHTML = '';
    var total = itemList.reduce(function (sum, item) { return sum + item.weight; }, 0);
    var offset = 0;

    itemList.forEach(function (item, index) {
      var share = total > 0 ? item.weight / total : (itemList.length ? 1 / itemList.length : 0);
      var opacity = SLICE_OPACITY_LIST[index % SLICE_OPACITY_LIST.length];
      var length = share * CIRCUMFERENCE;

      var circle = document.createElementNS(SVG_NS, 'circle');
      circle.setAttribute('cx', '60');
      circle.setAttribute('cy', '60');
      circle.setAttribute('r', String(RADIUS));
      circle.setAttribute('class', 'donut-slice');
      circle.setAttribute('stroke-opacity', String(opacity));
      circle.setAttribute('stroke-dasharray', length + ' ' + (CIRCUMFERENCE - length));
      circle.setAttribute('stroke-dashoffset', String(-offset));
      slicesEl.appendChild(circle);
      offset += length;

      var row = document.createElement('div');
      row.className = 'legend-item';
      var swatch = document.createElement('span');
      swatch.className = 'legend-swatch';
      swatch.style.opacity = String(opacity);
      var name = document.createElement('span');
      name.className = 'legend-name';
      name.textContent = item.label;
      var value = document.createElement('span');
      value.className = 'legend-value';
      value.textContent = (share * 100).toFixed(1) + '%';
      row.appendChild(swatch);
      row.appendChild(name);
      row.appendChild(value);
      legendEl.appendChild(row);
    });
    return total;
  }

  global.BenchWeightDonut = { render: render };
})(window);
