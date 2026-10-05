import re,html,io
p='results/research/portfolio/fund_products_20261005/report/fund_products_v5.html'
s=io.open(p,encoding='utf-8').read()
print('len',len(s))
# header with P(
i=s.find('P(Excess'); print(repr(s[i-200:i+500]))
# cards
cards=re.findall(r'<div class="card[^"]*">.*?(?=<div class="card|</div>\s*<!--|<div class="tbl|<ul class="notes|<p class="cap)',s,flags=re.S)
print('ncards',len(cards))
for c in cards:
    title=re.search(r'<h3[^>]*>(.*?)</h3>',c,flags=re.S)
    lis=re.findall(r'<li>(.*?)</li>',c,flags=re.S)
    vals=[]
    for li in lis:
        t=re.sub(r'<[^>]+>','',li); m=re.search(r'([\d.]+)%',t); vals.append(float(m.group(1)) if m else None)
    # svg arcs
    svg=re.search(r'<svg.*?</svg>',c,flags=re.S)
    nseg=len(re.findall(r'<(path|circle)',svg.group(0))) if svg else 0
    cols_svg=re.findall(r'(?:stroke|fill)="(#[0-9a-fA-F]{3,8}|var\([^)]+\))"',svg.group(0)) if svg else []
    cols_li=re.findall(r'<i style="background:([^"]+)"',c)
    print(re.sub(r'<[^>]+>','',title.group(1)) if title else '?', '| legend', vals, 'sum', sum(v for v in vals if v), '| svg segs', nseg, '|', cols_svg, '| li', cols_li)
c=cards[5]; svg=re.search(r'<svg.*?</svg>',c,flags=re.S).group(0); print(svg[:1500])
