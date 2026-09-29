// Render only the isolated local artifact; no browser account or remote page.
const { chromium } = require('C:/Users/User/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const studyPath = path.resolve(__dirname, '../../../results/research/client_menu_independent_20260923');
(async () => {
  const browserObj = await chromium.launch({headless:true,executablePath:'C:/Users/User/AppData/Local/ms-playwright/chromium_headless_shell-1208/chrome-headless-shell-win64/chrome-headless-shell.exe'});
  const pageObj = await browserObj.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
  const errorList=[];pageObj.on('pageerror',errorObj=>errorList.push(errorObj.message));
  const verificationPath=path.join(studyPath,'verification');
  await pageObj.goto('file:///'+studyPath.replaceAll('\\','/')+'/REPORT.html');
  await pageObj.screenshot({path:path.join(verificationPath,'report_desktop.png')});
  const desktopState=await pageObj.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,imageCount:document.images.length,brokenImages:[...document.images].filter(imageObj=>!imageObj.complete||!imageObj.naturalWidth).length,formulaCount:document.querySelectorAll('.formula').length,cardCount:document.querySelectorAll('.card').length}));
  await pageObj.getByRole('heading',{name:'השוואה על אותם תאריכים',exact:true}).scrollIntoViewIfNeeded();
  await pageObj.screenshot({path:path.join(verificationPath,'report_results.png')});
  await pageObj.setViewportSize({width:430,height:932});
  await pageObj.evaluate(()=>window.scrollTo(0,0));
  await pageObj.screenshot({path:path.join(verificationPath,'report_mobile.png')});
  const mobileState=await pageObj.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}));
  await pageObj.setViewportSize({width:1440,height:1000});
  await pageObj.goto('file:///'+studyPath.replaceAll('\\','/')+'/REPORT_FULL.html');
  await pageObj.screenshot({path:path.join(verificationPath,'appendix_desktop.png')});
  const appendixState=await pageObj.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,sourceDetails:document.querySelectorAll('details').length}));
  const resultObj={desktopState,mobileState,appendixState,pageErrors:errorList,pass:desktopState.imageCount===4&&desktopState.brokenImages===0&&desktopState.formulaCount===2&&desktopState.cardCount===4&&desktopState.scrollWidth<=desktopState.width&&mobileState.scrollWidth<=mobileState.width&&appendixState.scrollWidth<=appendixState.width&&appendixState.sourceDetails===25&&errorList.length===0};
  fs.writeFileSync(path.join(verificationPath,'html_render_check.json'),JSON.stringify(resultObj,null,2));
  console.log(JSON.stringify(resultObj));await browserObj.close();if(!resultObj.pass)process.exitCode=1;
})();
