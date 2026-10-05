import json
R='results/research/portfolio/fund_products_20261005/report/'
L=lambda f: json.load(open(R+f+'.json',encoding='utf-8'))
S,B,C,X=L('study'),L('battery'),L('capacity'),L('exposure')
def short(o,n=400): return json.dumps(o,ensure_ascii=False)[:n]
m=S['margin']['GR1 -> GR2']
print(list(m.keys()))
vm=m['vol_matched']; print(list(vm.keys()))
for k,v in vm.items():
    if k not in('q','tails','nav'): print(' ',k,short(v,500))
print('GR2 cagr',S['books']['GR2']['q']['cagr'])
vm=S['margin']['GR1 -> GR3']['vol_matched']
for k,v in vm.items():
    if k not in('q','tails','nav'): print(' ',k,short(v,500))
# capacity participation
for n in ['GR1','GR2','GR3']:
    b=C['books'][n]; print(n,short(b['participation'],500)); print('  ease',short(b['ease'],400),'fee',short(b['fee_income'],200)); print('  byleg',short(b['participation_by_leg'],700)); print('  wall',short(b['btal_wall'],200))
print('tbill', {n:X['books'][n]['tbill_like_share'] for n in ['GR1','GR2','GR3']})
print('rolling share',short(B['rolling_share_above']['GR1'],500))
print('plateau GR1', short({k:v for k,v in B['plateau']['GR1'].items() if k!='rows'},900))
print('reset GR1 other', short(B['reset']['GR1'].get('start_months') or {k:v for k,v in B['reset']['GR1'].items() if k!='policies'},900))
print('rolling', short(B['rolling']['GR1'],500), short(B['loyo']['GR1'],300))
print('startyears',short(B['start_years']['GR1'],300))
print('gr2_vs_old_plus',S['gr2_vs_old_plus']['share_xs'])
