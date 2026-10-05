import json
R='results/research/portfolio/fund_products_20261005/report/'
L=lambda f: json.load(open(R+f,encoding='utf-8'))
S=L('study.json'); B=L('battery.json'); V=L('versus.json'); M=L('monthly.json'); C=L('capacity.json'); E=L('exposure.json')
g=M['grid']
rows=[]
for k,v in g.items():
    if v['momentum']>0:
        base=g[k.rsplit(' mom',1)[0]+' mom 0%']
        rows.append((k, v['vs_no_momentum']['share_xs'], v['vs_no_momentum']['share_cagr'], v['q']['cagr']-base['q']['cagr'], (v['q']['dd']-base['q']['dd'])*100, v['q']['crises']['bear_2022']-base['q']['crises']['bear_2022'], v['q']['xs']-base['q']['xs'], v.get('rung_growth')))
for r in rows: print('%-28s sx %.3f sc %.3f dCAGR %+.4f dDD %+.2f d2022 %+.4f dxs %+.3f'%r[:7])
sx=[r[1] for r in rows]; print('share range',min(sx),max(sx))
print('1N below 50:', sum(1 for r in rows if '1n' in r[0] and r[1]<0.5), 'of', sum(1 for r in rows if '1n' in r[0]))
print('cagr <= :', sum(1 for r in rows if r[3]<=0.0005), 'strictly lower', sum(1 for r in rows if r[3]<0), 'rounded-equal-or-lower', sum(1 for r in rows if round((r[3]),3)<=0))
print('dd shallower range', min(r[4] for r in rows), max(r[4] for r in rows))
print('2022 worse', sum(1 for r in rows if r[5]<0))
print('xs higher w/ mom', sum(1 for r in rows if r[6]>0))
for k,v in M['ladder'].items(): print(k, round(v['q']['cagr'],4), round(v['q']['xs'],3), round(v['q']['dd'],4), v.get('rung_growth'), v.get('rung_growth_plus'), round(v['q']['crises']['gfc'],3), round(v['q']['crises']['bear_2022'],3), round(v['tails']['p25'],3))
print(M['corr'])
print('==== versus')
for k,v in V.items():
    print(k)
    print('  ', {f:(round(x['share_xs'],3),round(x['share_cagr'],3)) for f,x in v['frames'].items()})
    for b,x in v['blocks'].items():
        m=x['main'] if 'main' in x else x
        print('   ',b,round(m['share_xs'],3),round(m['share_cagr'],3), 'a', round(m['a']['cagr'],4), round(m['a']['xs'],3), 'b', round(m['b']['cagr'],4), round(m['b']['xs'],3))
