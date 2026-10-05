import json
R='results/research/portfolio/fund_products_20261005/report/'
D=json.load(open(R+'defensive.json',encoding='utf-8'))
g=D['more_return']['gr1']
print(list(g.keys()))
for r in g['frontier_launch'][:8]:
    print({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items() if k!='weights'})
p=g['pick']; print(json.dumps({k:v for k,v in p.items() if k not in('weights',)})[:1500])
