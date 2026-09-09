// Actual DashboardDataProvider, temporary existing-format releases/SQLite, no registry.
const { chromium } = require('playwright');
const { spawn } = require('node:child_process');
const { mkdir } = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');

async function main() {
  const expandedBool = process.argv.includes('--expanded');
  const portStr = expandedBool ? '18768' : '18766';
  const originStr = 'http://127.0.0.1:' + portStr;
  const serverObj = spawn(path.resolve('.venv/Scripts/python.exe'), [
    'scripts/review/serve_local_workspace_fixture.py', '--port', portStr, ...(expandedBool ? ['--expanded'] : []),
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
    const outputDirStr = path.resolve('.codex_tmp/' + (expandedBool ? 'local-expanded-ui' : 'local-workspace-ui'));
    await mkdir(outputDirStr, { recursive: true });
    const viewList = ['overview', 'performance', 'strategies', 'exposure', 'activity', 'diagnostics', 'report'];
    await pageObj.goto(originStr + '/');
    assert(pageObj.url().endsWith('/clients/local/overview'));
    if (expandedBool) {
      assert.equal(await pageObj.locator('input[name="to"]').inputValue(), '2026-09-03');
      assert((await pageObj.locator('.client-source-delay').innerText()).includes('complete through 2026-09-03'));
      // Legacy Sep1 remains selected: a good final NAV must not unlock ALL returns.
      assert(await pageObj.locator('[data-account-unit="pct"]').isDisabled());
      await pageObj.locator('.client-data-issues summary').click();
      assert.equal(await pageObj.locator('.client-data-issues li').filter({ hasText: 'mtmAtPaxos' }).count(), 2);
    }
    for (const widthInt of [1440, 768, 390]) {
      await pageObj.setViewportSize({ width: widthInt, height: 1000 });
      await pageObj.goto(originStr + '/__fixture_chart_gap');
      const markerList = await pageObj.locator('.client-chart-point').all();
      assert.equal(markerList.length, 2);
      for (const markerObj of markerList) {
        assert.equal(await markerObj.getAttribute('data-isolated'), 'true');
        assert.equal(await markerObj.evaluate(elementObj => getComputedStyle(elementObj).opacity), '1');
      }
      const segmentList = await pageObj.locator('.client-chart-line').all();
      assert.equal(segmentList.length, 2);
      for (const segmentObj of segmentList) assert(!(await segmentObj.getAttribute('points')).includes(' '));
      for (const viewStr of viewList) {
        const periodStr = expandedBool ? 'from=2026-09-02&to=2026-09-03' : 'from=2026-09-01&to=2026-09-01';
        const responseObj = await pageObj.goto(originStr + `/clients/local/${viewStr}?${periodStr}`);
        assert.equal(responseObj.status(), 200, viewStr);
        assert.equal(await pageObj.locator('nav a[href^="/clients/local/"]').count(), 7);
        assert.equal(await pageObj.getByText('Switch client', { exact: true }).count(), 0);
        assert.equal(await pageObj.getByRole('link', { name: 'Clients', exact: true }).count(), 0);
        for (const selectorStr of ['body', 'h1', '.client-brand', '.client-rail nav a']) {
          assert((await pageObj.locator(selectorStr).first().evaluate(elementObj => getComputedStyle(elementObj).fontFamily)).includes('Segoe UI'), selectorStr);
        }
        if (viewStr === 'overview' || viewStr === 'strategies') {
          const statusObj = await (await fetch(originStr + '/clients/local/diagnostics?download=json')).json();
          const summaryObj = pageObj.locator('.client-status-summary');
          assert.equal(await summaryObj.locator('.client-exception').count(), statusObj.strategy_list.filter(rowObj => rowObj.severity_str !== 'green').length);
          assert.equal(await summaryObj.locator('details[open]').count(), 0);
          for (const warningObj of await summaryObj.locator('.client-exception').all()) {
            assert.equal(await warningObj.locator('p, ul, li').count(), 0);
            assert(await warningObj.isVisible());
          }
        }
        if (viewStr === 'overview') {
          const allocationReportObj = await (await fetch(originStr + `/clients/local/overview?${periodStr}&download=json`)).json();
          const allocationReadyBool = allocationReportObj.closing_nav_float !== null;
          assert.equal(await pageObj.locator('.client-allocation-panel .client-donut').count(), allocationReadyBool ? 1 : 0);
          assert.equal(await pageObj.locator('.client-allocation-panel .client-allocation-legend li').count(), allocationReadyBool ? allocationReportObj.valuation_account_list.length : 0);
          const axisObj = pageObj.locator('[data-account-panel] .client-chart:visible .client-y-axis');
          const plotObj = pageObj.locator('[data-account-panel] .client-chart:visible svg');
          if (await plotObj.count()) {
            assert(Math.abs((await axisObj.boundingBox()).height - (await plotObj.boundingBox()).height) < 1, 'Chart axis must match plotted height');
          } else {
            assert(await pageObj.locator('[data-account-panel] .client-chart-empty:visible').isVisible());
          }
          if (expandedBool) {
            const pnlObj = pageObj.locator('[data-account-panel] .client-chart:visible [data-chart-pnl]');
            assert.equal(await pnlObj.innerText(), '-$25.00');
            await plotObj.focus();
            await plotObj.press('ArrowLeft');
            assert.equal(await pnlObj.innerText(), '$60.00');
            await plotObj.press('ArrowLeft');
            assert.equal(await pnlObj.innerText(), 'Unavailable');
            await plotObj.press('ArrowRight');
            await plotObj.press('ArrowRight');
          }
          const chartObj = pageObj.locator('[data-account-panel]');
          assert.equal(await chartObj.locator('[data-account-unit="pct"]').isEnabled(), expandedBool);
          if (expandedBool) {
            assert((await chartObj.locator('.client-chart:visible .client-y-axis').innerText()).includes('%'));
            await chartObj.locator('[data-account-unit="usd"]').click();
            assert((await chartObj.locator('.client-chart:visible .client-y-axis').innerText()).includes('$'));
            await chartObj.locator('[data-account-unit="pct"]').click();
            assert.equal(await pageObj.locator('.client-data-issues').count(), 0);
            const reportObj = await (await fetch(originStr + `/clients/local/overview?${periodStr}&download=json`)).json();
            assert.equal(reportObj.pnl_float, 35);
            assert.equal(reportObj.capital_movement_float, 100);
            assert.equal(reportObj.closing_nav_float, 11135);
            assert.equal(reportObj.status_str, 'ready');
            assert((await chartObj.locator('.client-chart-dates:visible').innerText()).includes('Start'));
            assert(!(await chartObj.locator('.client-chart-dates:visible').innerText()).includes('SOD'));
            const methodObj = pageObj.locator('.client-method-note');
            await methodObj.focus();
            const tooltipObj = methodObj.locator('[role="tooltip"]');
            assert(await tooltipObj.isVisible());
            assert((await tooltipObj.innerText()).includes('end-of-day'));
            const tooltipBoxObj = await tooltipObj.boundingBox();
            assert(tooltipBoxObj.x >= 0 && tooltipBoxObj.x + tooltipBoxObj.width <= widthInt);
            await methodObj.evaluate(elementObj => elementObj.blur());
          } else {
            // The new Pod intentionally has no NAV; never manufacture a book curve.
            assert(await chartObj.locator('[data-account-series="usd"] .client-chart-empty').isVisible());
            assert.equal(await chartObj.locator('.client-chart:visible').count(), 0);
            assert(await pageObj.locator('.client-data-issues').isVisible());
            assert.equal(await pageObj.locator('.client-data-issues').getAttribute('open'), null);
            await pageObj.locator('.client-data-issues summary').click();
            assert((await pageObj.locator('.client-data-issues').innerText()).includes('pod_new / U300'));
            await pageObj.locator('.client-data-issues summary').click();
          }
          assert.equal(await pageObj.locator('.client-schedule-card').count(), expandedBool ? 2 : 3);
        }
        if (viewStr === 'overview' || viewStr === 'performance') {
          const dailyObj = pageObj.locator('[data-daily-panel]');
          assert(await dailyObj.isVisible());
          assert.equal(await dailyObj.locator('[data-daily-scope="0"] [data-daily-chart="pct"] .client-bar').count(), expandedBool ? 2 : 0);
          if (expandedBool) {
            const portfolioObj = dailyObj.locator('[data-daily-scope="0"]');
            assert((await portfolioObj.locator('tbody').innerText()).includes('-$25.00'));
            assert((await portfolioObj.locator('tbody').innerText()).includes('$60.00'));
            assert.equal(await portfolioObj.locator('.client-movement').count(), 1);
            assert((await portfolioObj.locator('tr').filter({ hasText: '2026-09-02' }).innerText()).includes('Movement'));
            await dailyObj.locator('[data-daily-unit="usd"]').click();
            assert((await portfolioObj.locator('[data-daily-chart="usd"] .client-y-axis').innerText()).includes('$'));
            assert.equal(await portfolioObj.locator('[data-daily-chart="usd"] .client-bar').count(), 2);
            await dailyObj.locator('[data-daily-unit="pct"]').click();
          }
          await dailyObj.locator('[data-daily-select]').selectOption('1');
          const accountObj = dailyObj.locator('[data-daily-scope="1"]');
          assert(await accountObj.isVisible());
          assert((await accountObj.locator('tbody').innerText()).includes(expandedBool ? '0.80%' : '1.00%'));
          assert.equal(await accountObj.locator('.client-movement').count(), expandedBool ? 1 : 0);
          assert.equal(await accountObj.locator('[data-daily-chart="pct"] .client-bar').count(), expandedBool ? 2 : 1);
          await dailyObj.locator('[data-daily-unit="usd"]').click();
          assert.equal(await accountObj.locator('[data-daily-chart="usd"] .client-bar').count(), expandedBool ? 2 : 0);
          if (!expandedBool) {
            assert(await accountObj.locator('[data-daily-chart="usd"] .client-chart-empty').isVisible());
            assert.equal(await accountObj.locator('[data-daily-chart="usd"] .client-chart-empty').innerText(), 'Incomplete IBKR data');
          }
          await dailyObj.locator('[data-daily-unit="pct"]').click();
          await dailyObj.locator('[data-daily-select]').selectOption('0');
        }
        if (viewStr === 'strategies') {
          assert.equal(await pageObj.getByRole('heading', { name: 'Data freshness', exact: true }).count(), expandedBool ? 2 : 3);
          assert.equal(await pageObj.locator('.client-holdings-allocation .client-donut').count(), 2);
          const stageLinkObj = pageObj.locator('[data-flow-link]').filter({ hasText: 'ACK' }).first();
          const targetStr = await stageLinkObj.getAttribute('href');
          await stageLinkObj.click();
          assert(await pageObj.locator(targetStr).evaluate(elementObj => elementObj.open));
          assert(await pageObj.locator(targetStr).innerText().then(textStr => textStr.includes('Broker acknowledgements')));
          await pageObj.locator(targetStr + ' > summary').click();
          assert.equal(await pageObj.locator('.client-pod-flow').count(), expandedBool ? 2 : 3);
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
    if (expandedBool) {
      await pageObj.goto(originStr + '/clients/local/overview?from=2026-09-01&to=2026-09-03');
      assert(await pageObj.locator('[data-account-unit="pct"]').isDisabled());
      assert.equal(await pageObj.locator('[data-daily-scope="0"] .client-bar').count(), 0);
      assert.equal(await pageObj.locator('.client-schedule-card').count(), 2);
      const reportObj = await (await fetch(originStr + '/clients/local/report?from=2026-09-02&to=2026-09-03&download=json')).json();
      const pdfObj = await fetch(originStr + '/clients/local/report?from=2026-09-02&to=2026-09-03&download=pdf&expected=' + reportObj.report_hash_str);
      assert.equal(pdfObj.status, 200);
      assert(pdfObj.headers.get('content-type').includes('application/pdf'));
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
      views: viewList, widths: [1440, 768, 390], currentPods: expandedBool ? 2 : 3,
      expandedFields: expandedBool, readonly: true, outputDir: outputDirStr }));
  } finally {
    if (browserObj) await browserObj.close();
    serverObj.kill(); // Only the synthetic child created above.
  }
}
main().catch(errorObj => { console.error(errorObj); process.exitCode = 1; });
