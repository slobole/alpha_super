import re,html,sys,io
p='results/research/portfolio/fund_products_20261005/report/fund_products_v5.html'
s=io.open(p,encoding='utf-8').read()
s=re.sub(r'<script.*?</script>','',s,flags=re.S)
s=re.sub(r'<style.*?</style>','',s,flags=re.S)
s=re.sub(r'<svg.*?</svg>','[SVG]',s,flags=re.S)
s=re.sub(r'</(p|li|tr|h\d|div|table|figcaption|summary|details|section)>','\n',s)
s=re.sub(r'</t[dh]>',' | ',s)
s=re.sub(r'<[^>]+>','',s)
s=html.unescape(s)
s=re.sub(r'[ \t]+',' ',s)
s=re.sub(r'\n\s*\n+','\n',s)
io.open(sys.argv[1],'w',encoding='utf-8').write(s)
print(len(s), s.count('\n'))
