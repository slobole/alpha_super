import re,io,math
h=io.open('results/research/portfolio/fund_products_20261005/report/fund_products_v5.html',encoding='utf-8').read()
pies=re.findall(r'<div class="pie">(.*?)</ul></div>',h,flags=re.S)
print('n pies',len(pies))
defined=set(re.findall(r'(--[a-z0-9-]+)\s*:',h))
for p in pies:
    lab=re.search(r'aria-label="([^"]*)"',p).group(1)
    paths=re.findall(r'<path d="M ([\d.]+) ([\d.]+) A 46.0 46.0 0 (\d) 1 ([\d.]+) ([\d.]+) L [^"]*" fill="([^"]+)"',p)
    tot=0;segs=[]
    for x0,y0,big,x1,y1,col in paths:
        a0=math.atan2(float(x0)-50,50-float(y0))%(2*math.pi); a1=math.atan2(float(x1)-50,50-float(y1))%(2*math.pi)
        d=(a1-a0)%(2*math.pi)
        if d<1e-3: d=2*math.pi
        if (d>math.pi)!=(big=='1') and abs(d-math.pi)>0.01: print('  LARGE-ARC FLAG MISMATCH',d,big)
        tot+=d; segs.append((round(d/(2*math.pi)*100,1),col))
    leg=re.findall(r'<li><i style="background:([^"]+)"></i><b>(\d+)%</b> ([^<]*)</li>',p)
    cols=set(c for _,c in segs)|set(c for c,_,_ in leg)
    undefined=[c for c in cols if not c.startswith('var(') or c[4:-1] not in defined]
    print(lab,'| arcs',segs,'sum%.2f'%(tot/(2*math.pi)*100),'| legend',[(b,n) for c,b,n in leg],'sum',sum(int(b) for _,b,_ in leg),'| leg/arc colour match',[c for c,_,_ in leg]==[c for _,c in segs],'| undefined',undefined, '| circles',p.count('<circle'))
print(sorted(d for d in defined if re.match(r'--(s\d|warn|accent|bad|ink-3|bench|surface|ink-2)$',d)))
