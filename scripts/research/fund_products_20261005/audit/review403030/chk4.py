import json,re,io,math
R='results/research/portfolio/fund_products_20261005/report/'
L=lambda f: json.load(open(R+f+'.json',encoding='utf-8'))
S,D=L('study'),L('defensive')
def short(o,n=400): return json.dumps(o,ensure_ascii=False)[:n]
def show(o,d=0):
    for k,v in o.items():
        if isinstance(v,dict) and d<2: print('  '*d+k); show(v,d+1)
        else: print('  '*d+k, short(v,260))
show(D)
for k,v in S['margin'].items():
    print(k,[kk for kk in v.keys()]); 
    for kk in v:
        if kk not in('target','vol_matched','cagr_matched'): print('   ',kk,short(v[kk],600))
    vm=v['vol_matched']; print('   vm keys',list(vm.keys())); print('   ',short({a:b for a,b in vm.items() if a not in('q','tails')},900))
