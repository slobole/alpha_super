// Report layout is CSS-only, so the preview and frozen copy paginate identically.

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
  if (printButton) printButton.addEventListener("click", async () => {
    if (!printButton.dataset.archiveUrl) { window.print(); return; }
    const message = document.querySelector("[data-print-message]");
    let savedLink = null;
    printButton.disabled = true;
    message.textContent = "Saving report…";
    try {
      const response = await fetch(printButton.dataset.archiveUrl, {method: "POST", credentials: "same-origin",
        headers: {"Content-Type": "application/json", "X-Alpha-Action-Token": printButton.dataset.token},
        body: JSON.stringify({confirmed_bool: true, month_str: printButton.dataset.month, report_id_str: printButton.dataset.reportId})});
      const result = await response.json();
      if (!response.ok) { message.textContent = result.message || "Report not saved."; return; }
      const link = document.createElement("a");
      link.textContent = "Saved copy";
      link.href = result.archive_url_str;
      message.replaceChildren(link);
      savedLink = link;
      // Print the verified frozen document, never the older interactive preview.
      // Keep the frame alive after print(): browser print dialogs can be asynchronous.
      document.querySelector(".mr-print-frame")?.remove();
      const frame = document.createElement("iframe");
      frame.className = "mr-print-frame";
      frame.title = "Saved report for printing";
      frame.setAttribute("aria-hidden", "true");
      frame.tabIndex = -1;
      const loaded = new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("Saved copy did not load.")), 30000);
        frame.onload = () => { clearTimeout(timer); resolve(); };
        frame.onerror = () => { clearTimeout(timer); reject(new Error("Saved copy did not load.")); };
      });
      frame.src = result.archive_url_str;
      document.body.append(frame);
      await loaded;
      if (!frame.contentDocument.querySelector(".mr-pages") || frame.contentDocument.querySelector(".mr-draft")) {
        throw new Error("Saved copy could not be verified.");
      }
      await frame.contentDocument.fonts.ready;
      frame.contentWindow.focus();
      frame.contentWindow.print();
    } catch (error) {
      if (savedLink) message.replaceChildren(savedLink, document.createTextNode(" · Open the saved copy to print."));
      else message.textContent = "Report not saved. Please try again.";
    }
    finally { printButton.disabled = false; }
  });
  const monthSelect = document.querySelector("[data-month-nav]");
  if (monthSelect) {
    monthSelect.addEventListener("change", () => {
      const url = monthSelect.value;
      // Only same-origin report links are offered by the server.
      if (url.startsWith("/") && !url.startsWith("//")) window.location.assign(url);
    });
  }
  setupPrograms();
});
