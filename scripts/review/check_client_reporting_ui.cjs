// Local synthetic end-to-end/visual QA. Requires Playwright in NODE_PATH.
// Private loopback preview; no application login or real data.
const { chromium } = require('playwright');
const { spawn } = require('node:child_process');
const { mkdir } = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');

async function main() {
  const portInt = 18764;
  const originStr = `http://127.0.0.1:${portInt}`;
  // The demo CLI never loads config.env or the real data provider.
  const serverObj = spawn(path.resolve('.venv/Scripts/python.exe'), [
    '-m', 'alpha.live.dashboard_v3', '--demo', '--port', String(portInt),
  ], { env: process.env, windowsHide: true, stdio: 'pipe' });
  let browserObj;
  let serverErrorStr = '';
  serverObj.stderr.on('data', chunkObj => { serverErrorStr += chunkObj.toString(); });
  try {
    let readyBool = false;
    for (let attemptInt = 0; attemptInt < 60; attemptInt++) {
      if (serverObj.exitCode !== null) throw new Error('Demo server exited: ' + serverErrorStr);
      try {
        const responseObj = await fetch(originStr + '/clients');
        if (responseObj.ok) { readyBool = true; break; }
      } catch {}
      await new Promise(resolveFn => setTimeout(resolveFn, 250));
    }
    assert(readyBool, 'Demo server not ready: ' + serverErrorStr);
    assert.equal((await fetch(originStr + '/clients')).status, 200);
    browserObj = await chromium.launch({ channel: 'msedge', headless: true });
    // The workspace is light-only even when the operator's OS prefers dark.
    const contextObj = await browserObj.newContext({ colorScheme: 'dark' });
    const pageObj = await contextObj.newPage();
    const pageErrorList = [];
    const remoteRequestList = [];
    pageObj.on('pageerror', errorObj => pageErrorList.push(errorObj.message));
    // Financial pages must remain usable without any remote CSS/scripts/fonts.
    await contextObj.route('**/*', routeObj => {
      if (!routeObj.request().url().startsWith(originStr)) {
        remoteRequestList.push(routeObj.request().url());
        return routeObj.abort();
      }
      return routeObj.continue();
    });
    const outputDirStr = path.resolve('.codex_tmp/client-ui');
    await mkdir(outputDirStr, { recursive: true });
    for (const widthInt of [1440, 768, 390]) {
      await pageObj.setViewportSize({ width: widthInt, height: 1000 });
      await pageObj.goto(originStr + '/clients');
      await pageObj.evaluate(() => document.fonts.ready);
      await pageObj.screenshot({ path: path.join(outputDirStr, `client-directory-${widthInt}.png`), fullPage: true });
      await pageObj.goto(originStr + '/clients/demo-client/report?from=2026-06-01&to=2026-09-04');
      await pageObj.getByRole('heading', { name: 'Report preview', exact: true }).waitFor();
      await pageObj.evaluate(() => document.fonts.ready);
      assert.equal(await pageObj.locator('body').evaluate(elementObj => getComputedStyle(elementObj).colorScheme), 'light');
      assert(await pageObj.getByText('Demo data', { exact: true }).isVisible());
      assert.equal(await pageObj.locator('tbody tr').count(), 4);
      assert(await pageObj.getByRole('heading', { name: 'Market comparison', exact: true }).isVisible());
      assert(await pageObj.getByRole('link', { name: 'Download investor PDF', exact: true }).isVisible());
      assert(await pageObj.locator('.client-evidence-strip').isVisible());
      assert.equal(await pageObj.locator('.client-source-details').count(), 0);
      const layoutObj = await pageObj.evaluate(() => ({ width: innerWidth, scrollWidth: document.documentElement.scrollWidth }));
      assert(layoutObj.scrollWidth <= layoutObj.width, `Horizontal page overflow: ${JSON.stringify(layoutObj)}`);
      await pageObj.screenshot({ path: path.join(outputDirStr, `client-report-${widthInt}.png`), fullPage: true });
    }
    const clientViewList = ['overview', 'performance', 'strategies', 'exposure', 'activity', 'diagnostics'];
    for (const widthInt of [1440, 768, 390]) {
      await pageObj.setViewportSize({ width: widthInt, height: 1000 });
      for (const viewStr of clientViewList) {
        const responseObj = await pageObj.goto(originStr + `/clients/demo-client/${viewStr}?from=2026-06-01&to=2026-09-04`);
        assert.equal(responseObj.status(), 200, viewStr);
        await pageObj.evaluate(() => document.fonts.ready);
        const paletteObj = await pageObj.evaluate(() => ({
          scheme: getComputedStyle(document.body).colorScheme,
          background: getComputedStyle(document.body).backgroundColor,
          panel: getComputedStyle(document.querySelector('.ops-panel')).backgroundColor,
          input: document.querySelector('input[type="date"]') ? getComputedStyle(document.querySelector('input[type="date"]')).colorScheme : 'light',
        }));
        assert.deepEqual(paletteObj, { scheme: 'light', background: 'rgb(246, 247, 249)', panel: 'rgb(255, 255, 255)', input: 'light' });
        assert(await pageObj.getByText('Demo data', { exact: true }).isVisible());
        assert(!(await pageObj.locator('main').innerText()).includes('financial selection never changes trading state'));
        if (['overview', 'performance'].includes(viewStr)) {
          assert(await pageObj.locator('.client-evidence-strip').isVisible());
          assert.equal(await pageObj.locator('.client-source-details').count(), 0);
          assert(!(await pageObj.locator('main').innerText()).includes('Unavailable: daily account NAV/TWR'));
          assert(await pageObj.locator('[data-client-twr]').getByText('Calculated · daily', { exact: true }).isVisible());
          assert(await pageObj.locator('.client-return-panel svg:visible').isVisible());
          if (viewStr === 'overview') {
            const accountObj = pageObj.locator('[data-account-panel]');
            assert(await accountObj.locator('[data-account-unit="pct"]').isEnabled());
            await accountObj.locator('[data-account-unit="usd"]').click();
            assert((await accountObj.locator('.client-chart:visible .client-y-axis').innerText()).includes('$'));
            assert(await accountObj.getByRole('heading', { name: 'Account value', exact: true }).isVisible());
            await accountObj.locator('[data-account-unit="pct"]').click();
            assert((await accountObj.locator('.client-chart:visible .client-y-axis').innerText()).includes('%'));
          }
          for (const chartObj of await pageObj.locator('.client-chart:visible').all()) {
            const tickList = await chartObj.locator('.client-y-axis span').all();
            assert.equal(tickList.length, 3);
            for (const tickObj of tickList) {
              assert(await tickObj.isVisible());
              assert((await tickObj.innerText()).includes('%'));
              const labelObj = await tickObj.evaluate(elementObj => ({ font: parseFloat(getComputedStyle(elementObj).fontSize), width: elementObj.getBoundingClientRect().width, left: elementObj.getBoundingClientRect().left }));
              assert(labelObj.font >= 10 && labelObj.width > 0 && labelObj.left >= 0, 'Y-axis must remain legible in screen pixels');
            }
            const axisAlignedBool = await chartObj.evaluate(elementObj => {
              const labelList = [...elementObj.querySelectorAll('.client-y-axis span')];
              const gridList = [...elementObj.querySelectorAll('.client-chart-grid')];
              return labelList.every((labelObj, indexInt) => {
                const labelRectObj = labelObj.getBoundingClientRect();
                const gridRectObj = gridList[indexInt].getBoundingClientRect();
                return Math.abs(labelRectObj.top + labelRectObj.height / 2 - gridRectObj.top) < 1;
              });
            });
            assert(axisAlignedBool, 'Y-axis labels must align with their gridlines');
          }
          const dailyObj = pageObj.locator('[data-daily-panel]');
          assert(await dailyObj.isVisible());
          assert((await dailyObj.locator('[data-daily-scope="0"] .client-bar').count()) > 0);
          assert.equal(await dailyObj.locator('[data-daily-select] option').count(), 5);
          for (const scopeStr of ['0', '1']) {
            await dailyObj.locator('[data-daily-select]').selectOption(scopeStr);
            const scopeObj = dailyObj.locator(`[data-daily-scope="${scopeStr}"]`);
            assert(await scopeObj.isVisible());
            const dayCountInt = await scopeObj.locator('tbody tr').count();
            assert(dayCountInt > 1);
            assert(!(await scopeObj.locator('tbody').innerText()).includes('SOD'));
            await dailyObj.locator('[data-daily-unit="usd"]').click();
            assert(await scopeObj.locator('[data-daily-chart="usd"]').isVisible());
            assert(!(await scopeObj.locator('[data-daily-chart="pct"]').isVisible()));
            assert((await scopeObj.locator('[data-daily-chart="usd"] .client-y-axis').innerText()).includes('$'));
            assert.equal(await scopeObj.locator('tbody tr').count(), dayCountInt);
            await dailyObj.locator('[data-daily-unit="pct"]').click();
          }
          await dailyObj.locator('[data-daily-select]').selectOption('0');
        }
        if (viewStr !== 'performance') {
          assert(await pageObj.getByText('Saved check ·', { exact: false }).isVisible());
          assert(await pageObj.locator('.client-status-summary .client-exception').first().isVisible());
        }
        assert(await pageObj.locator(`nav a[aria-current="page"]`).getAttribute('href').then(hrefStr => hrefStr.includes('/' + viewStr)));
        assert(!(await pageObj.locator('main').innerText()).includes('DEMO_0_'), 'Another client account leaked');
        if (viewStr === 'overview') {
          assert.equal(await pageObj.locator('.client-schedule-card').count(), 4);
          for (const cardObj of await pageObj.locator('.client-schedule-card').all()) {
            assert(await cardObj.isVisible());
            assert.equal(await cardObj.locator('time').count(), 3);
            assert.equal(await cardObj.locator('time[datetime]').count(), 3);
          }
          assert(await pageObj.getByRole('heading', { name: 'Recorded changes', exact: true }).isVisible());
          assert.equal(await pageObj.locator('.client-activity-panel .client-event').count(), 3);
          const activityHrefStr = await pageObj.getByRole('link', { name: 'View selected-period activity', exact: true }).getAttribute('href');
          assert.equal(activityHrefStr, '/clients/demo-client/activity?from=2026-06-01&to=2026-09-04');
        }
        if (viewStr === 'activity') {
          assert.equal(await pageObj.locator('.client-event').count(), 4);
          assert(!(await pageObj.locator('.client-activity-panel').innerText()).includes('Synthetic cycle: positions matched'));
          await pageObj.locator('.client-event summary').first().click();
          assert((await pageObj.locator('.client-activity-panel').innerText()).includes('Synthetic cycle: positions matched'));
          await pageObj.locator('.client-event summary').first().click();
        }
        if (viewStr === 'performance') {
          assert.equal(await pageObj.locator('.client-comparison').count(), 4);
          assert.equal(await pageObj.locator('.client-comparison[open]').count(), 0);
          await pageObj.locator('.client-comparison summary').first().click();
          assert(await pageObj.getByText('No saved comparison is pinned for this account period.', { exact: false }).first().isVisible());
          assert((await pageObj.locator('.client-comparison').first().innerText()).includes('not a certified replay'));
          await pageObj.locator('.client-comparison summary').first().click();
        }
        if (viewStr === 'exposure') assert(await pageObj.getByText('Reference values · not current', { exact: true }).isVisible());
        if (viewStr === 'strategies') {
          assert.equal(await pageObj.locator('.client-pod-flow').count(), 4);
          assert.equal(await pageObj.locator('details .client-pod-flow').count(), 0);
          for (const flowObj of await pageObj.locator('.client-pod-flow').all()) {
            assert(await flowObj.isVisible());
            assert.equal(await flowObj.locator('li').count(), 7);
            for (const stageObj of await flowObj.locator('li').all()) assert(await stageObj.isVisible());
          }
        }
        const layoutObj = await pageObj.evaluate(() => ({ width: innerWidth, scrollWidth: document.documentElement.scrollWidth }));
        assert(layoutObj.scrollWidth <= layoutObj.width, `${viewStr} overflow: ${JSON.stringify(layoutObj)}`);
        await pageObj.screenshot({ path: path.join(outputDirStr, `client-${viewStr}-${widthInt}.png`), fullPage: true });
      }
    }
    await pageObj.goto(originStr + '/clients/demo-owner/overview?from=2026-06-01&to=2026-09-04');
    assert(await pageObj.getByRole('heading', { name: 'No action required', exact: true }).isVisible());
    await pageObj.getByRole('link', { name: 'View selected-period activity', exact: true }).click();
    await pageObj.waitForURL('**/activity?from=2026-06-01&to=2026-09-04');
    await pageObj.goto(originStr + '/clients/demo-owner/activity?from=2026-08-01&to=2026-08-31');
    assert(await pageObj.getByText('No saved events for these dates.', { exact: true }).isVisible());
    for (const fromStr of ['2026-09-02', '2026-05-01']) {
      const responseObj = await pageObj.goto(originStr + `/clients/demo-owner/activity?from=${fromStr}&to=2026-09-01`);
      assert.equal(responseObj.status(), 400);
      assert(await pageObj.getByRole('alert').isVisible());
      assert.equal(await pageObj.locator('input[name="from"]').inputValue(), fromStr);
      await pageObj.locator('input[name="from"]').fill('2026-09-01');
      await pageObj.getByRole('button', { name: 'Apply period', exact: true }).click();
      await pageObj.waitForURL('**/activity?from=2026-09-01&to=2026-09-01');
      assert(await pageObj.getByRole('heading', { name: 'Selected-period activity', exact: true }).isVisible());
    }
    await pageObj.goto(originStr + '/clients/demo-client/report?from=2026-06-01&to=2026-09-04');
    await pageObj.locator('input[name="from"]').fill('2026-08-01');
    await pageObj.locator('input[name="to"]').fill('2026-08-31');
    await pageObj.getByRole('button', { name: 'Apply period', exact: true }).click();
    await pageObj.waitForURL('**/report?from=2026-08-01&to=2026-08-31');
    const downloadPromiseObj = pageObj.waitForEvent('download');
    await pageObj.getByRole('link', { name: 'Download investor PDF', exact: true }).click();
    const downloadObj = await downloadPromiseObj;
    assert.equal(await downloadObj.failure(), null);
    assert(downloadObj.suggestedFilename().endsWith('.pdf'));
    assert.equal((await contextObj.request.get(originStr + '/static/utilities.css')).status(), 200);
    const advancedPathList = ['/vps', '/pods/live', ...['status', 'lifecycle', 'events', 'provenance', 'operations'].map(viewStr => '/diagnostics?view=' + viewStr)];
    for (const widthInt of [1440, 768, 390]) {
      await pageObj.setViewportSize({ width: widthInt, height: 1000 });
      for (const pathStr of advancedPathList) {
        const responseObj = await pageObj.goto(originStr + pathStr);
        assert.equal(responseObj.status(), 200, pathStr);
        await pageObj.evaluate(() => document.fonts.ready);
        assert(await pageObj.locator('.client-badge').getByText('Read-only', { exact: true }).isVisible());
        assert.equal(await pageObj.getByRole('link', { name: 'VPS overview', exact: true }).getAttribute('href'), '/vps');
        if (pathStr === '/pods/live') {
          assert.equal(await pageObj.locator('.advanced-attention li').first().getAttribute('data-severity'), 'yellow');
          assert.equal(await pageObj.locator('.advanced-attention li').first().evaluate(elementObj => getComputedStyle(elementObj).borderLeftColor), 'rgb(183, 121, 9)');
          assert.equal(await pageObj.locator('.advanced-attention a').first().innerText(), 'Equity mean reversion');
          assert.equal(await pageObj.locator('.advanced-stage-strip').count(), 6);
          assert.equal(await pageObj.locator('.advanced-stage-strip').first().evaluate(elementObj => getComputedStyle(elementObj).display), 'flex');
          for (const flowObj of await pageObj.locator('.advanced-stage-strip').all()) {
            assert.equal(await flowObj.locator('span').count(), 7);
            for (const stageObj of await flowObj.locator('span').all()) assert(await stageObj.isVisible());
          }
        }
        const layoutObj = await pageObj.evaluate(() => ({ width: innerWidth, scrollWidth: document.documentElement.scrollWidth,
          overflowList: Array.from(document.querySelectorAll('main *')).filter(elementObj => elementObj.getBoundingClientRect().right > innerWidth).map(elementObj => ({ tag: elementObj.tagName, class: elementObj.className, text: elementObj.innerText?.slice(0, 80) })).slice(0, 12) }));
        await pageObj.screenshot({ path: path.join(outputDirStr, `advanced-${pathStr.replace(/[^a-z]+/g, '-')}-${widthInt}.png`), fullPage: true });
        assert(layoutObj.scrollWidth <= layoutObj.width, `${pathStr} overflow: ${JSON.stringify(layoutObj)}`);
      }
    }
    assert.deepEqual(remoteRequestList, []);
    assert.deepEqual(pageErrorList, []);
    console.log(JSON.stringify({ result: 'PASS', viewports: [1440, 768, 390], clientViews: clientViewList.concat('report'), advancedPaths: advancedPathList, strategies: 4, secondClientIsolation: 'PASS', periodForm: 'PASS', noLoginPdfDownload: 'PASS', remoteAssetsBlocked: true, outputDir: outputDirStr }));
  } finally {
    if (browserObj) await browserObj.close();
    serverObj.kill(); // Only the child created by this QA invocation.
  }
}
main().catch(errorObj => { console.error(errorObj); process.exitCode = 1; });
