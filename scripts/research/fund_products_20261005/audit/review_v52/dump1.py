import json
R='results/research/portfolio/fund_products_20261005/report/'
L=lambda f: json.load(open(R+f,encoding='utf-8'))
def short(o,n=400):
    s=json.dumps(o,ensure_ascii=False)
    return s[:n]
S=L('study.json'); B=L('battery.json'); V=L('versus.json'); M=L('monthly.json'); C=L('capacity.json'); E=L('exposure.json'); P=L('pm_confirm.json')
print('books keys', list(S['books'].keys()))
print('GR1 keys', list(S['books']['GR1'].keys()))
for k in ['GR1','GR2','GR3','S9 incumbent launch','old growth plus']:
    b=S['books'][k]
    print(k, {kk:(vv if not isinstance(vv,(dict,list)) else '..') for kk,vv in b.items()})
    print('  weights', short(b.get('weights') or b.get('w')))
print('products GR2', short(S['products']['GR2'],1500))
print('blends', short(S['blends'],3000))
print('gr2_vs_old_plus', short(S['gr2_vs_old_plus'],1500))
print('challenges[0]', short(S['challenges'][0],1200))
print('rev', [(c['challenger'],c['default'],round(c['share_xs'],3),round(c['share_cagr'],3),c['passed'],c['checks']) for c in S['challenges_reverse']])
print('slots', short(S['slots'],2500))
print('mr_be', S['mr_gate_breakeven'])
print('construction', short(S['construction'],1500))
print('dial', short(S['dial']['dial taa3x 40'],800))
