import re,sys,html,io
src=sys.argv[1];dst=sys.argv[2]
t=io.open(src,encoding='utf-8').read()
t=re.sub(r'<script.*?</script>','',t,flags=re.S)
t=re.sub(r'<style.*?</style>','',t,flags=re.S)
t=re.sub(r'<svg.*?</svg>','[SVG]',t,flags=re.S)
t=re.sub(r'</(p|div|tr|h\d|li|table|section|figcaption|details|summary)>','\n',t)
t=re.sub(r'</t[dh]>',' | ',t)
t=re.sub(r'<[^>]+>','',t)
t=html.unescape(t)
t=re.sub(r'[ \t]+',' ',t)
t=re.sub(r'\n\s*\n+','\n',t)
io.open(dst,'w',encoding='utf-8').write(t)
print(len(t), t.count('\n'))
