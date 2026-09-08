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
