import json
R='results/research/portfolio/fund_products_20261005/report/'
L=lambda f: json.load(open(R+f,encoding='utf-8'))
def short(o,n=400):
    return json.dumps(o,ensure_ascii=False)[:n]
S=L('study.json'); B=L('battery.json'); V=L('versus.json'); M=L('monthly.json'); C=L('capacity.json'); E=L('exposure.json')
aw=B['after_window']['books']
for k in aw:
    y=S['books'][k]['years']
    y26 = y.get('2026') if isinstance(y,dict) else y[-1]
    print(k, 'years2026', y26, 'after', aw[k]['after_window'], 'ytd', aw[k]['ytd_2026'], 'implied', (1+(y26 if isinstance(y26,(int,float)) else 0))*(1+aw[k]['after_window'])-1)
print(short(S['books']['GR1']['years'],600))
# blends
for k,v in S['blends'].items():
    if 'q' in v and 'diluted' in v:
        q=v['q'];d=v['diluted']['q']
        print(k, 'cagr %.4f xs %.4f vol %.4f dd %.4f | dil cash %.2f cagr %.4f xs %.4f dd %.4f | dC %.2f dX %.3f dDD %.2f | p10 %.3f p15 %.3f p20 %.4f gfc %.3f b22 %.3f'%(q['cagr'],q['xs'],q['vol'],q['dd'],v['diluted']['cash'],d['cagr'],d['xs'],d['dd'],(q['cagr']-d['cagr'])*100,q['xs']-d['xs'],(q['dd']-d['dd'])*100,v['tails']['p10'],v['tails']['p15'],v['tails']['p20'],q['crises']['gfc'],q['crises']['bear_2022']), 'dilvol %.4f'%d['vol'])
    else: print(k, short(v,600))
print(S['corr_gr1_defensive'])
print('weights 80', S['books']['GR1 80 / defensive 19']['weights'], S['books']['GR1 20 / defensive 80']['weights'], S['books']['defensive launch']['weights'])
print('GR1 xs', S['books']['GR1']['long']['xs'] if 'xs' in S['books']['GR1']['long'] else short(S['books']['GR1']['long']))
# versus
for k,v in V.items():
    fr=v['frames']; bl=v['blocks']
    print(k, {f:(round(x['share_xs'],3),round(x['share_cagr'],3)) for f,x in fr.items()}, {b:(round(x['share_xs'],3),round(x['share_cagr'],3)) for b,x in bl.items()})
print(short(V['GR1 | S9 incumbent launch']['blocks'],1500))
