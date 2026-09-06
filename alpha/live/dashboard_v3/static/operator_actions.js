/* Clipboard and visible action errors only. No execution or automatic retries. */
document.addEventListener('click', async (event_obj) => {
  const button_obj = event_obj.target.closest('[data-copy-command]');
  if (!button_obj) return;
  const text_obj = button_obj.closest('.ops-command').querySelector('textarea');
  try {
    await navigator.clipboard.writeText(text_obj.value);
    button_obj.textContent = 'Copied - not executed';
  } catch {
    text_obj.focus();
    text_obj.select();
    button_obj.textContent = 'Selected - press Ctrl+C';
  }
});
document.addEventListener('htmx:beforeSwap', (event_obj) => {
  if (event_obj.detail.xhr.getResponseHeader('X-Alpha-Action-Error') === 'true') {
    event_obj.detail.shouldSwap = true;
    event_obj.detail.isError = false;
  }
});
for (const event_name_str of ['htmx:sendError', 'htmx:timeout', 'htmx:responseError']) {
  document.addEventListener(event_name_str, (event_obj) => {
    const target_obj = event_obj.detail.target;
    if (!target_obj || !target_obj.id.startsWith('operator-tools-preview-')) return;
    target_obj.replaceChildren();
    const message_obj = document.createElement('p');
    message_obj.setAttribute('role', 'alert');
    message_obj.textContent = 'Outcome unknown: connection failed. Do not repeat an order. Inspect job and broker evidence before opening a new preview.';
    target_obj.append(message_obj);
  });
}
