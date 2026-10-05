import json
R='results/research/portfolio/fund_products_20261005/report/'
L=lambda f: json.load(open(R+f,encoding='utf-8'))
def short(o,n=400): return json.dumps(o,ensure_ascii=False)[:n]
S=L('study.json'); B=L('battery.json'); C=L('capacity.json'); E=L('exposure.json'); P=L('pm_confirm.json')
q=lambda k:S['books'][k]['q']
print('S8 gap', q('GR1')['cagr']-q('S8 GR1 75 / DEF 25')['cagr'])
print('GR3-GR2 step', q('GR3')['cagr']-q('GR2')['cagr'], S['products']['GR3'].get('step_over_product_below'), S['products']['GR3'].get('offered'))
print('GR2-GR1', q('GR2')['cagr']-q('GR1')['cagr'])
print('T1 gap', q('GR1')['cagr']-q('T1 MOM -> BIL')['cagr'])
pl=B['plateau']['GR1']; print('plateau keys', list(pl.keys())); print(short({k:v for k,v in pl.items() if k!='neighbours'},1500))
for k in ['GR1','GR2','GR3','S9 incumbent launch','old growth plus','S13 old monthly (2026-10-01)']:
    c=C['books'][k]; 
    if k=='GR1': print(list(c.keys()))
    print(k, short({kk:vv for kk,vv in c.items() if kk in ('house','capacity','aum_rule','routes','open','close','worked','label','ease')},700))
print('old monthly plus in capacity?', 'old monthly plus (2026-10-01)' in C['books'])
for k in ['GR1','GR2','GR3']:
    e=E['books'][k]; print(k, short(e,600))
print(short(E['tqqq_weight'],400))
for k,v in P['books'].items(): print(k, v['engine'], v['research_house_cash'])
print('dep products', short(B['dependence']['products'],1500))
print('net', short(S['books']['S9 incumbent launch']['net'],300), short(S['books']['old growth plus']['net'],300))
print('rungs S9', short(S['books']['S9 incumbent launch']['rungs'],900))
print('rungs MP', short(S['books']['old growth plus']['rungs'],900))
