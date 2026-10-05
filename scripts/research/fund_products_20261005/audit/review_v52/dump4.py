import json
R='results/research/portfolio/fund_products_20261005/report/'
L=lambda f: json.load(open(R+f,encoding='utf-8'))
def short(o,n=400):
    return json.dumps(o,ensure_ascii=False)[:n]
S=L('study.json'); B=L('battery.json'); V=L('versus.json'); M=L('monthly.json'); C=L('capacity.json'); E=L('exposure.json')
k='GR1 | S9 incumbent launch'
print(short(V[k],1800))
print('---- monthly')
g=M['grid']
print(short(g['taa3x_1n 50%:50% mom 0%'],900))
print(short(g['taa3x_1n 50%:50% mom 30%'],1200))
print(short(M['ladder']['1N 65% / CORE5 35%'],700))
print(short(M['ladder']['1N 50 / CORE5 50 x1.25'],700))
print(short(M['reference'],1500))
