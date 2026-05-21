// Tiny client-side helper: per-pod "new events since last visit" badge.
//
// Reads each visible pod row's data-latest-event-timestamp attribute,
// compares against localStorage["dashboard_v3.last_seen." + pod_id],
// and renders a small "● new" pill in the row if the row's timestamp is
// newer than the last-seen one. Clicking a row marks the pod as seen.
//
// Server-rendered timestamps come from the row dict — no server roundtrip
// needed to compute the badge. Re-runs after every HTMX swap so polled
// rows stay accurate.

(function () {
  const STORAGE_KEY_PREFIX_STR = "dashboard_v3.last_seen.";

  function lastSeenTimestampStr(podIdStr) {
    try {
      return window.localStorage.getItem(STORAGE_KEY_PREFIX_STR + podIdStr) || "";
    } catch (errorObj) {
      return "";
    }
  }

  function markPodSeen(podIdStr, timestampStr) {
    if (!podIdStr || !timestampStr) return;
    try {
      window.localStorage.setItem(STORAGE_KEY_PREFIX_STR + podIdStr, timestampStr);
    } catch (errorObj) { /* localStorage blocked — silent. */ }
  }

  function badgeElementOrNull(rowElement) {
    return rowElement.querySelector("[data-new-event-badge]");
  }

  function refreshBadgesFn() {
    const podRowElementList = document.querySelectorAll("[data-pod-id]");
    podRowElementList.forEach(function (rowElement) {
      const podIdStr = rowElement.getAttribute("data-pod-id");
      const latestTimestampStr = rowElement.getAttribute("data-latest-event-timestamp") || "";
      const lastSeenStr = lastSeenTimestampStr(podIdStr);
      const isNewBool = latestTimestampStr && latestTimestampStr > lastSeenStr;
      const badgeElement = badgeElementOrNull(rowElement);
      if (badgeElement) {
        badgeElement.style.display = isNewBool ? "inline-flex" : "none";
      }
    });
  }

  document.addEventListener("DOMContentLoaded", refreshBadgesFn);
  document.body.addEventListener("htmx:afterSwap", refreshBadgesFn);

  document.body.addEventListener("click", function (eventObj) {
    const rowElement = eventObj.target.closest("[data-pod-id]");
    if (!rowElement) return;
    const podIdStr = rowElement.getAttribute("data-pod-id");
    const latestTimestampStr = rowElement.getAttribute("data-latest-event-timestamp") || "";
    markPodSeen(podIdStr, latestTimestampStr);
    const badgeElement = badgeElementOrNull(rowElement);
    if (badgeElement) badgeElement.style.display = "none";
  });
})();
