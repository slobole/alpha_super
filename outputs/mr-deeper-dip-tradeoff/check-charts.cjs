const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('C:/Users/User/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const outputPath = __dirname;
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage({ viewport: { width: 768, height: 1800 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(pathToFileURL(path.join(outputPath, 'preview.html')).href);
  const frame = page.frames().find(candidate => candidate.parentFrame());
  await frame.waitForSelector('#dip-tradeoff-v1[data-selected]', { timeout: 30000 });
  const checks = [];
  for (const width of [768,392,352,1056]) {
    await page.setViewportSize({ width, height: 1800 });
    for (const strategy of ['sector','hpi235','dv2']) {
      for (const policy of ['limit_0.5pct','limit_1pct','limit_2pct']) {
        await frame.locator('#dip-strategy').selectOption(strategy);
        await frame.locator('#dip-depth').selectOption(policy);
        await frame.waitForTimeout(380);
        const result = await frame.evaluate(() => {
          const root = document.getElementById('dip-tradeoff-v1');
          const clipped = [];
          const overlaps = [];
          for (const svg of root.querySelectorAll('svg')) {
            const boundary = svg.getBoundingClientRect();
            const labels = [...svg.querySelectorAll('text')].map(element => ({element, rect:element.getBoundingClientRect()}));
            for (const label of labels) {
              const box = label.rect;
              if (box.width && (box.left < boundary.left-1 || box.right > boundary.right+1 || box.top < boundary.top-1 || box.bottom > boundary.bottom+1)) {
                clipped.push({chart:svg.parentElement.id,text:label.element.textContent});
              }
            }
            for (let first = 0; first < labels.length; first++) {
              for (let second = first+1; second < labels.length; second++) {
                const a = labels[first].rect, b = labels[second].rect;
                if (a.width && b.width && a.left < b.right+3 && a.right+3 > b.left && a.top < b.bottom+3 && a.bottom+3 > b.top) {
                  overlaps.push({chart:svg.parentElement.id,first:labels[first].element.textContent,second:labels[second].element.textContent});
                }
              }
            }
          }
          return {selected:root.dataset.selected,identityError:Number(root.dataset.pairedIdentityError),clipped,overlaps,
            overflow:root.scrollWidth-root.clientWidth,svgCount:root.querySelectorAll('svg').length,
            fillLabel:root.querySelector('#dip-filled-label').textContent,pathDetail:root.querySelector('#dip-path-detail').textContent};
        });
        checks.push({width,strategy,policy,...result});
        if (width===768 && strategy==='sector' && policy==='limit_1pct') await frame.locator('#dip-tradeoff-v1').screenshot({path:path.join(outputPath,'sector-desktop.png')});
        if (width===768 && strategy==='hpi235' && policy==='limit_0.5pct') await frame.locator('#dip-tradeoff-v1').screenshot({path:path.join(outputPath,'hpi-desktop.png')});
        if (width===352 && strategy==='sector' && policy==='limit_1pct') await frame.locator('#dip-tradeoff-v1').screenshot({path:path.join(outputPath,'sector-mobile.png')});
      }
    }
  }
  const failures = checks.filter(check => check.clipped.length || check.overlaps.length || check.overflow>1 || check.svgCount!==5 || Math.abs(check.identityError)>1e-10 || check.selected!==check.strategy+'/'+check.policy);
  fs.writeFileSync(path.join(outputPath,'visual-qa.json'),JSON.stringify({errors,checks,failures},null,2)+'\n');
  await browser.close();
  console.log(JSON.stringify({states:checks.length,errors,failures},null,2));
  if (errors.length || failures.length) process.exitCode=1;
})().catch(error => { console.error(error); process.exitCode=1; });
