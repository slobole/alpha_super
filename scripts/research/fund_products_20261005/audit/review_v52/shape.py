import json,sys
R='results/research/portfolio/fund_products_20261005/report/'
def show(o,d=0,maxd=3,pre=''):
    if isinstance(o,dict):
        ks=list(o.keys())
        for k in ks[:40]:
            v=o[k]
            t=type(v).__name__
            s=''
            if isinstance(v,(int,float,str,bool)) or v is None: s=repr(v)[:60]
            elif isinstance(v,list): s='list[%d]'%len(v)
            else: s='dict[%d]'%len(v)
            print('  '*d+str(k)+': '+s)
            if d<maxd and isinstance(v,dict): show(v,d+1,maxd)
        if len(ks)>40: print('  '*d+'... %d keys'%len(ks))
for f,md in [('study.json',1),('battery.json',1),('versus.json',1),('monthly.json',1),('capacity.json',1),('exposure.json',1),('pm_confirm.json',2)]:
    print('=====',f); show(json.load(open(R+f,encoding='utf-8')),0,md)
