import json
R='results/research/portfolio/fund_products_20261005/report_equal_capital_snapshot/'
S=json.load(open(R+'study.json',encoding='utf-8'))
for k,v in S['margin'].items():
    vm=v['vol_matched']; print(k, vm['name'], 'cagr',vm['q']['cagr'],'tgt',v['target']['q']['cagr'],'s250',vm['spread_250']['cagr'],'plus5',vm['plus5']['cagr'],v['target']['plus5']['cagr'],'dd',vm['q']['dd'])
