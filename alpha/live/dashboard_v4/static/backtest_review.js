// "Reviewed" on a backtest difference: save the mark, then reload so the headline and count follow.
(function () {
  document.addEventListener('click', async (event_obj) => {
    const button_obj = event_obj.target.closest('[data-review-url]');
    if (!button_obj || button_obj.disabled) return;
    button_obj.disabled = true;
    button_obj.textContent = 'Saving…';
    try {
      const response_obj = await fetch(button_obj.getAttribute('data-review-url'), {method: 'POST', credentials: 'same-origin',
        headers: {'Content-Type': 'application/json', 'X-Alpha-Action-Token': button_obj.getAttribute('data-token')},
        body: JSON.stringify({confirmed_bool: true, key_str: button_obj.getAttribute('data-review-key')})});
      if (!response_obj.ok) {
        const data_dict = await response_obj.json().catch(() => ({}));
        button_obj.textContent = 'Not saved';
        button_obj.title = data_dict.message || 'Request refused';
        return;
      }
      window.location.reload();
    } catch (error_obj) {
      button_obj.textContent = 'Not saved';
      button_obj.title = 'Network error. Try again.';
      button_obj.disabled = false;
    }
  });
})();
