// Actual DashboardDataProvider, temporary existing-format releases/SQLite, no registry.
const { chromium } = require('playwright');
const { spawn } = require('node:child_process');
const { mkdir } = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');

async function main() {
  const originStr = 'http://127.0.0.1:18766';
  const serverObj = spawn(path.resolve('.venv/Scripts/python.exe'), [
    'scripts/review/serve_local_workspace_fixture.py', '--port', '18766',
  ], { env: process.env, windowsHide: true, stdio: 'pipe' });
  let browserObj;
  let errorStr = '';
  serverObj.stderr.on('data', chunkObj => { errorStr += chunkObj.toString(); });
  try {
    let readyBool = false;
    for (let attemptInt = 0; attemptInt < 120; attemptInt++) {
      if (serverObj.exitCode !== null) throw new Error(errorStr);
      try { if ((await fetch(originStr + '/healthz')).ok) { readyBool = true; break; } } catch {}
      await new Promise(resolveFn => setTimeout(resolveFn, 250));
    }
    assert(readyBool, errorStr);
    browserObj = await chromium.launch({ channel: 'msedge', headless: true });
    const contextObj = await browserObj.newContext();
    const pageObj = await contextObj.newPage();
    const failureList = [];
    pageObj.on('pageerror', errorObj => failureList.push(errorObj.message));
    await contextObj.route('**/*', routeObj => {
      if (!routeObj.request().url().startsWith(originStr + '/')) {
        failureList.push('Remote request: ' + routeObj.request().url());
        return routeObj.abort();
      }
      return routeObj.continue();
    });
    const outputDirStr = path.resolve('.codex_tmp/local-workspace-ui');
    await mkdir(outputDirStr, { recursive: true });
    const viewList = ['overview', 'performance', 'strategies', 'exposure', 'activity', 'diagnostics', 'report'];
    await pageObj.goto(originStr + '/');
    assert(pageObj.url().endsWith('/clients/local/overview'));
    for (const widthInt of [1440, 768, 390]) {
      await pageObj.setViewportSize({ width: widthInt, height: 1000 });
      for (const viewStr of viewList) {
        const responseObj = await pageObj.goto(originStr + `/clients/local/${viewStr}?from=2026-09-01&to=2026-09-01`);
        assert.equal(responseObj.status(), 200, viewStr);
        assert.equal(await pageObj.locator('nav a[href^="/clients/local/"]').count(), 7);
        assert.equal(await pageObj.getByText('Switch client', { exact: true }).count(), 0);
        assert.equal(await pageObj.getByRole('link', { name: 'Clients', exact: true }).count(), 0);
        if (viewStr === 'strategies') {
          assert.equal(await pageObj.locator('.client-pod-flow').count(), 3);
          const flowObj = pageObj.locator('.client-pod-flow').first();
          assert((await flowObj.locator('li').count()) >= 7);
          for (const stageObj of await flowObj.locator('li').all()) assert(await stageObj.isVisible());
        }
        const dimensionsObj = await pageObj.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
        assert(dimensionsObj.scroll <= dimensionsObj.width, `${viewStr} overflow ${JSON.stringify(dimensionsObj)}`);
        await pageObj.screenshot({ path: path.join(outputDirStr, `${viewStr}-${widthInt}.png`), fullPage: true });
      }
      await pageObj.goto(originStr + '/clients/local/activity');
      await pageObj.locator('input[name="from"]').fill('2026-08-31');
      await pageObj.locator('input[name="to"]').fill('2026-08-31');
      assert(await pageObj.locator('form.ops-toolbar').evaluate(formObj => formObj.checkValidity()));
      await pageObj.getByRole('button', { name: 'Apply period' }).click();
      assert(pageObj.url().includes('from=2026-08-31'));
    }
    for (const routeStr of ['/vps', '/pods/live', '/performance', '/exposure', '/events', '/diagnostics']) {
      const responseObj = await pageObj.goto(originStr + routeStr);
      assert.equal(responseObj.status(), 200, routeStr);
      assert(await pageObj.getByRole('link', { name: 'Live workspace', exact: true }).isVisible());
    }
    assert.deepEqual(failureList, []);
    const integrityObj = await (await fetch(originStr + '/__fixture_integrity')).json();
    assert.equal(integrityObj.unchanged_bool, true);
    assert.equal(integrityObj.network_attempt_count_int, 0);
    assert.equal((await fetch(originStr + '/api/pods/pod_a/actions/tick', { method: 'POST' })).status, 403);
    console.log(JSON.stringify({ result: 'PASS', provider: 'DashboardDataProvider', registry: false,
      views: viewList, widths: [1440, 768, 390], currentPods: 3, readonly: true, outputDir: outputDirStr }));
  } finally {
    if (browserObj) await browserObj.close();
    serverObj.kill(); // Only the synthetic child created above.
  }
}
main().catch(errorObj => { console.error(errorObj); process.exitCode = 1; });
