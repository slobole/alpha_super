import json
R='results/research/portfolio/fund_products_20261005/report/'
L=lambda f: json.load(open(R+f+'.json',encoding='utf-8'))
S,B,V,C,X=L('study'),L('battery'),L('versus'),L('capacity'),L('exposure')
K=S['books']
def short(o,n=400): return json.dumps(o,ensure_ascii=False)[:n]
for n in ['GR1','S0 equal capital (registered default)','GR2','GR3','S8 GR1 75 / DEF 25','dial taa3x 33','dial taa3x 40','dial taa3x_1n 40']:
    b=K[n]; print(n, {k:round(v,5) for k,v in b['q'].items() if isinstance(v,float)}, 'p20',b['tails'].get('p20'),'w',short(b.get('weights'),200))
print('GR1 keys',list(K['GR1'].keys()))
print('GR1 frames keys',list(K['GR1']['frames'].keys()))
for f in K['GR1']['frames']: print(' ',f, short(K['GR1']['frames'][f],200))
print('--- versus')
for k,v in V.items():
    print(k)
    for f,d in v['frames'].items(): print('   frame',f, short(d,300))
    for f,d in v['blocks'].items(): print('   block',f, short(d,200))
