// Synthetic loopback QA. Real command executors/broker/config.env are never used.
const { chromium } = require('playwright');
const { spawn } = require('node:child_process');
const { randomBytes } = require('node:crypto');
const { mkdir } = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');

async function main() {
  const tokenStr = randomBytes(32).toString('hex');
  const originStr = 'http://127.0.0.1:18765';
  const outputStr = path.resolve('.codex_tmp/operator-ui');
  await mkdir(outputStr, { recursive: true });
  const codeStr = `
import sys
sys.path.insert(0, 'tests')
from dataclasses import replace
from pathlib import Path
from flask import jsonify
from test_dashboard_v3_routes import StubDataProvider
from test_live_dashboard import _build_release_obj
from alpha.live.dashboard import DashboardPodTarget
from alpha.live.dashboard_v3.app import create_app
provider_obj = StubDataProvider()
provider_obj.releases_root_path_str = 'C:/synthetic/releases'
provider_obj.get_target_for_pod = lambda pod_id_str: DashboardPodTarget(replace(_build_release_obj(pod_id_str), mode_str='live'), 'C:/synthetic/pod.sqlite3', False)
app_obj = create_app(provider_obj, journal_path_str='.codex_tmp/operator-ui/journal.jsonl', notification_webhook_url_str='')
@app_obj.get('/qa/calls')
def calls_fn():
    return jsonify(count_int=len(provider_obj.action_job_dict_list))
app_obj.run(host='127.0.0.1', port=18765, debug=False, use_reloader=False)
`;
  const serverObj = spawn(path.resolve('.venv/Scripts/python.exe'), ['-c', codeStr], {
    env: { ...process.env, ALPHA_OPS_OPERATOR_ACCESS_TOKEN_STR: tokenStr },
    windowsHide: true, stdio: 'ignore',
  });
  let browserObj;
  try {
    const headersDict = { Authorization: 'Basic ' + Buffer.from('operator:' + tokenStr).toString('base64') };
    let readyBool = false;
    for (let retryInt = 0; retryInt < 80; retryInt++) {
      try { if ((await fetch(originStr + '/qa/calls', { headers: headersDict })).ok) { readyBool = true; break; } } catch {}
      await new Promise(resolveFn => setTimeout(resolveFn, 200));
    }
    assert(readyBool, 'Synthetic server failed to start');
    browserObj = await chromium.launch({ channel: 'msedge', headless: true });
    const contextObj = await browserObj.newContext({ extraHTTPHeaders: headersDict });
    const pageObj = await contextObj.newPage();
    await pageObj.route('**/*', routeObj => routeObj.request().url().startsWith(originStr)
      ? routeObj.continue() : routeObj.abort());
    async function previewFn() {
      await pageObj.goto(originStr + '/fragments/action-preview/dv2_caspersky_live/submit_vplan');
      await pageObj.evaluate(() => {
        const contentStr = document.body.innerHTML;
        document.body.innerHTML = '<main id="operator-tools-preview-dv2_caspersky_live">' + contentStr + '</main>';
      });
      await pageObj.addStyleTag({ url: originStr + '/static/custom.css' });
      await pageObj.addScriptTag({ url: originStr + '/static/htmx.min.js' });
      await pageObj.addScriptTag({ url: originStr + '/static/operator_actions.js' });
      await pageObj.evaluate(() => htmx.process(document.body));
    }
    await previewFn();
    await pageObj.setViewportSize({ width: 768, height: 800 });
    await pageObj.screenshot({ path: path.join(outputStr, 'confirmation.png'), fullPage: true });
    const oldValuesDict = JSON.parse(await pageObj.getByRole('button', { name: 'Confirm and run' }).getAttribute('hx-vals'));
    await pageObj.getByRole('button', { name: 'Confirm and run' }).click();
    await pageObj.getByText('queued', { exact: false }).first().waitFor();
    assert.equal(await pageObj.getByRole('button', { name: 'Confirm and run' }).count(), 0);
    const replayObj = await contextObj.request.post(originStr + '/api/pods/dv2_caspersky_live/actions/submit_vplan', {
      headers: { Origin: originStr, 'X-Alpha-Action-Token': 'stub-token' }, data: oldValuesDict,
    });
    assert.equal(replayObj.status(), 409);
    assert.equal((await (await contextObj.request.get(originStr + '/qa/calls')).json()).count_int, 1);
    await previewFn();
    await contextObj.request.get(originStr + '/fragments/action-preview-cancel/dv2_caspersky_live');
    await pageObj.getByRole('button', { name: 'Confirm and run' }).click();
    await pageObj.getByRole('alert').waitFor();
    assert((await pageObj.getByRole('alert').innerText()).includes('already used'));
    assert.equal(await pageObj.getByRole('button', { name: 'Confirm and run' }).count(), 0);
    await pageObj.screenshot({ path: path.join(outputStr, 'confirmation-error.png'), fullPage: true });
    await previewFn();
    await pageObj.route('**/api/pods/*/actions/*', routeObj => routeObj.abort('failed'), { times: 1 });
    await pageObj.getByRole('button', { name: 'Confirm and run' }).click();
    await pageObj.getByRole('alert').waitFor();
    assert((await pageObj.getByRole('alert').innerText()).includes('Outcome unknown'));
    assert.equal(await pageObj.getByRole('button', { name: 'Confirm and run' }).count(), 0);
    await pageObj.goto(originStr + '/fragments/command-catalog/dv2_caspersky_live');
    await pageObj.addStyleTag({ url: originStr + '/static/custom.css' });
    await pageObj.addScriptTag({ url: originStr + '/static/operator_actions.js' });
    await pageObj.getByRole('button', { name: 'Copy command (does not run)', exact: true }).first().click();
    assert.equal((await (await contextObj.request.get(originStr + '/qa/calls')).json()).count_int, 1);
    assert.equal(await pageObj.locator('textarea').count(), 6);
    await pageObj.screenshot({ path: path.join(outputStr, 'catalog.png'), fullPage: true });
    console.log('PASS: real HTMX preview/confirm, replay, cancelled preview, network failure, visible recovery and copy without execution.');
  } finally {
    if (browserObj) await browserObj.close();
    serverObj.kill();
  }
}
main().catch(errorObj => { console.error(errorObj.message); process.exitCode = 1; });
