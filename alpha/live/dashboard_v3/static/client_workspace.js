// Presentation only: switch already-rendered broker series; no requests or math.
document.querySelectorAll('[data-daily-panel]').forEach(panelObj => {
  const selectObj = panelObj.querySelector('[data-daily-select]');
  selectObj.addEventListener('change', () => {
    panelObj.querySelectorAll('[data-daily-scope]').forEach(scopeObj => {
      scopeObj.hidden = scopeObj.dataset.dailyScope !== selectObj.value;
    });
  });
  panelObj.querySelectorAll('[data-daily-unit]').forEach(buttonObj => {
    buttonObj.addEventListener('click', () => {
      const unitStr = buttonObj.dataset.dailyUnit;
      panelObj.querySelectorAll('[data-daily-unit]').forEach(peerObj => peerObj.setAttribute('aria-pressed', String(peerObj === buttonObj)));
      panelObj.querySelectorAll('[data-daily-chart]').forEach(chartObj => { chartObj.hidden = chartObj.dataset.dailyChart !== unitStr; });
    });
  });
});

document.querySelectorAll('[data-account-panel]').forEach(panelObj => {
  panelObj.querySelectorAll('[data-account-unit]').forEach(buttonObj => {
    buttonObj.addEventListener('click', () => {
      panelObj.querySelectorAll('[data-account-unit]').forEach(peerObj => peerObj.setAttribute('aria-pressed', String(peerObj === buttonObj)));
      panelObj.querySelectorAll('[data-account-series]').forEach(seriesObj => { seriesObj.hidden = seriesObj.dataset.accountSeries !== buttonObj.dataset.accountUnit; });
    });
  });
});

// The backend supplies every date/value. Pointer and keyboard only select a point.
document.querySelectorAll('.client-chart').forEach(chartObj => {
  const sourceObj = chartObj.querySelector('[data-chart-points]');
  if (!sourceObj) return;
  const pointList = JSON.parse(sourceObj.textContent);
  const plotObj = chartObj.querySelector('svg');
  let pointIndexInt = pointList.length - 1;
  const showFn = indexInt => {
    pointIndexInt = indexInt;
    const pointObj = pointList[indexInt];
    chartObj.querySelector('[data-chart-date]').textContent = pointObj.display_date_str;
    chartObj.querySelector('[data-chart-value]').textContent = pointObj.label_str;
    chartObj.querySelector('[data-chart-pnl]').textContent = pointObj.pnl_label_str;
    chartObj.querySelector('[data-chart-pnl]').dataset.sign = pointObj.pnl_sign_str;
  };
  plotObj.tabIndex = 0;
  plotObj.setAttribute('aria-label', plotObj.getAttribute('aria-label') + '. Use left and right arrows to inspect dates.');
  plotObj.addEventListener('pointermove', eventObj => {
    const boundsObj = plotObj.getBoundingClientRect();
    const horizontalFloat = (eventObj.clientX - boundsObj.left) / boundsObj.width * plotObj.viewBox.baseVal.width;
    let nearestInt = 0;
    pointList.forEach((pointObj, indexInt) => {
      if (Math.abs(pointObj.x_float - horizontalFloat) < Math.abs(pointList[nearestInt].x_float - horizontalFloat)) nearestInt = indexInt;
    });
    showFn(nearestInt);
  });
  plotObj.addEventListener('pointerleave', () => showFn(pointList.length - 1));
  plotObj.addEventListener('keydown', eventObj => {
    if (!['ArrowLeft', 'ArrowRight'].includes(eventObj.key)) return;
    eventObj.preventDefault();
    showFn(Math.max(0, Math.min(pointList.length - 1, pointIndexInt + (eventObj.key === 'ArrowLeft' ? -1 : 1))));
  });
});

document.querySelectorAll('[data-flow-link]').forEach(linkObj => {
  linkObj.addEventListener('click', () => {
    const detailObj = document.getElementById(linkObj.hash.slice(1));
    if (detailObj) detailObj.open = true;
  });
});
