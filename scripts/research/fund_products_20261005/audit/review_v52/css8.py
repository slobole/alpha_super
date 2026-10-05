import re,io
s=io.open('results/research/portfolio/fund_products_20261005/report/fund_products_v5.html',encoding='utf-8').read()
for m in re.finditer(r'<style.*?</style>',s,flags=re.S):
    css=m.group(0)
    print('STYLE len',len(css))
    for line in css.split('\n'):
        if re.search(r'overflow|\.tbl|\.chip|\.tip|\.code|\.svgwrap|\.chart|\.legend|main\b|body|html|nowrap|min-width|@media|table',line):
            print(line[:330])
