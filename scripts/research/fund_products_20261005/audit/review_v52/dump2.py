import json
R='results/research/portfolio/fund_products_20261005/report/'
L=lambda f: json.load(open(R+f,encoding='utf-8'))
def short(o,n=400):
    return json.dumps(o,ensure_ascii=False)[:n]
S=L('study.json'); B=L('battery.json'); V=L('versus.json'); M=L('monthly.json')
ed=B['edge_decay']
print('scenario keys', list(ed['scenarios'].keys()))
sc=ed['scenarios']['all at 0.75']
print('0.75 keys', list(sc.keys())[:60])
for k in ['GR1','GR2','GR3','S9 incumbent launch','old growth plus','S13 old monthly (2026-10-01)','S0 equal capital (registered default)']:
    print(k, short(sc.get(k),500))
print('headline keys', list(ed['headline'].keys()))
for k in ['GR1','GR2','GR3','S9 incumbent launch','old growth plus']:
    print(k, short(ed['headline'].get(k),1500))
for s in ed['scenarios']:
    print(s, {k: short(ed['scenarios'][s].get(k),160) for k in ['GR1','S9 incumbent launch','old growth plus','GR3']})
print('worst_case', {k: short(ed['worst_case'].get(k),300) for k in ['GR1','GR3','S9 incumbent launch','old growth plus']})
print('minimax', short(ed['minimax'],800))
print('edge_margin', short(ed['edge_margin'],1500))
print('believe S9', short(B['believe']['S9 incumbent launch'],2500))
print('believe GR1', short(B['believe']['GR1'],2500))
print('after', short(B['after_window'],3000))
