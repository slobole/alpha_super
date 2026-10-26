// Monthly report page: print button, month picker, A4 fitting and the Programs panel.
const A4_HEIGHT_PX = 1123;

function fitPages() {
  const pageList = document.querySelectorAll(".pg");
  if (pageList.length < 2) return;
  // Page 1 is one A4 sheet: a long month table moves to page 2 instead of spilling.
  const months = pageList[0].querySelector("[data-months]");
  const slot = pageList[1].querySelector("[data-months-slot]");
  if (months && slot && pageList[0].offsetHeight > A4_HEIGHT_PX) slot.replaceWith(months);
  // A page 2 that would spill only its footer stays on one sheet.
  if (pageList[1].offsetHeight > A4_HEIGHT_PX) pageList[1].classList.add("pg-spill");
}

function setupPrograms() {
  const panel = document.querySelector("[data-programs]");
  if (!panel) return;
  const message = panel.querySelector("[data-programs-message]");
  document.querySelectorAll("[data-programs-toggle]").forEach((button) => button.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    document.querySelectorAll("[data-programs-toggle][aria-expanded]").forEach((item) => item.setAttribute("aria-expanded", String(!panel.hidden)));
  }));
  panel.querySelectorAll("[data-program-select]").forEach((select) => select.addEventListener("change", () => {
    const input = select.parentElement.querySelector("[data-program-name]");
    input.hidden = select.value !== "__other__";
    if (!input.hidden) input.focus();
  }));
  const save = panel.querySelector("[data-programs-save]");
  save.addEventListener("click", async () => {
    const programs = {};
    for (const row of panel.querySelectorAll("[data-pod-id]")) {
      const select = row.querySelector("[data-program-select]");
      const name = select.value === "__other__" ? row.querySelector("[data-program-name]").value.trim() : select.value;
      if (select.value === "__other__" && !name) {
        message.textContent = "Type a program name.";
        return;
      }
      programs[row.dataset.podId] = name;
    }
    save.disabled = true;
    message.textContent = "Saving…";
    try {
      const response = await fetch(panel.dataset.url, {method: "POST", credentials: "same-origin",
        headers: {"Content-Type": "application/json", "X-Alpha-Action-Token": panel.dataset.token},
        body: JSON.stringify({confirmed_bool: true, program_by_pod_dict: programs})});
      if (response.ok) {
        window.location.reload();
        return;
      }
      const data = await response.json().catch(() => ({}));
      message.textContent = data.message || "Not saved.";
    } catch (error) {
      message.textContent = "Network error. Try again.";
    }
    save.disabled = false;
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const printButton = document.querySelector("[data-print]");
  if (printButton) printButton.addEventListener("click", () => window.print());
  const monthSelect = document.querySelector("[data-month-nav]");
  if (monthSelect) {
    monthSelect.addEventListener("change", () => {
      const url = monthSelect.value;
      // Only same-origin report links are offered by the server.
      if (url.startsWith("/") && !url.startsWith("//")) window.location.assign(url);
    });
  }
  setupPrograms();
  document.fonts.ready.then(fitPages);
});
