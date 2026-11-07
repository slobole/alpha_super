(function () {
  var fam = document.getElementById('f-family'), gr = document.getElementById('f-grade');
  var rows = document.querySelectorAll('#ledger-table tbody tr'), count = document.getElementById('f-count');
  if (!fam || !gr) return;
  function apply() {
    var n = 0;
    rows.forEach(function (r) {
      var ok = (!fam.value || r.dataset.family === fam.value) && (!gr.value || r.dataset.grade === gr.value);
      r.hidden = !ok; if (ok) n++;
    });
    count.textContent = n + ' registrations';
  }
  fam.addEventListener('change', apply); gr.addEventListener('change', apply);
})();
