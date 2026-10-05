import json,sys
R='results/research/portfolio/fund_products_20261005/report/'
def show(o,d=0,maxd=2,pre=''):
    if isinstance(o,dict):
        ks=list(o.keys())
        for k in ks[:45]:
            v=o[k]
            t=type(v).__name__
            s=''
            if isinstance(v,(int,float,str,bool)) or v is None: s=repr(v)[:60]
            elif isinstance(v,list): s='list[%d]'%len(v)
            else: s='dict[%d]'%len(v)
            print('  '*d+str(k),':',s)
            if d<maxd and isinstance(v,dict): show(v,d+1,maxd)
        if len(ks)>45: print('  '*d+'... +%d'%(len(ks)-45))
for f in sys.argv[2:]:
    print('=====',f)
    show(json.load(open(R+f+'.json',encoding='utf-8')),0,int(sys.argv[1]))
