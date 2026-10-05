import re,html,io
p='results/research/portfolio/fund_products_20261005/report/fund_products_v5.html'
s=io.open(p,encoding='utf-8').read()
s=re.sub(r'<script.*?</script>','',s,flags=re.S); s=re.sub(r'<style.*?</style>','',s,flags=re.S); s=re.sub(r'<svg.*?</svg>','',s,flags=re.S)
txt=lambda h: html.unescape(re.sub(r'<[^>]+>','',h)).strip()
heb=re.compile(r'[\u0590-\u05FF]')
RLM='\u200f'
out=[]
# block elements
blocks=[]
for tag,pat in [('p',r'<p\b([^>]*)>(.*?)</p>'),('li',r'<li\b([^>]*)>(.*?)</li>'),('td',r'<td\b([^>]*)>(.*?)</td>'),('div.why',r'<div class="why"()>(.*?)</div>'),('th',r'<th\b([^>]*)>(.*?)</th>'),('summary',r'<summary()>(.*?)</summary>'),('h',r'<h[1-4]\b([^>]*)>(.*?)</h[1-4]>')]:
    for m in re.finditer(pat,s,flags=re.S):
        blocks.append((tag,m.group(1),m.group(2)))
print('blocks',len(blocks))
print('=== long (>300 chars)')
for tag,attr,h in blocks:
    t=txt(h)
    if tag in('p','li','div.why','td') and len(t)>300 and '<li' not in h and '<p' not in h:
        print(tag,len(t),'|',t[:90].replace('\n',' '))
print('=== hebrew blocks starting with latin/digit without RLM')
n=0
for tag,attr,h in blocks:
    t=txt(h)
    if not t or not heb.search(t): continue
    raw=html.unescape(h).lstrip()
    # first visible char
    first=t[0]
    # element starts with <b> etc; check first text char
    if re.match(r'[A-Za-z0-9+\-−$(]',first):
        # is the whole element ltr-wrapped? check if starts with span class ltr and rest hebrew
        n+=1
        print(tag,attr.strip()[:30],'|',t[:80].replace('\n',' '),'|| raw:',raw[:60].replace('\n',' '))
print('count',n)
print('=== raw identifiers')
vis=txt(s)
for pat in [r'S9 incumbent',r'old growth plus',r'nb GR1',r'\bGR[123]\b',r'\bS\d{1,2}\b',r'incumbent',r'old monthly',r'planning',r'Planning',r'[Ff]loor',r'\bM[12]\b',r'T[1-4] ',r'dial taa',r'taa3x',r'ndx_',r'dv2_g',r'hpi_g',r'S13',r'G3\b',r'report_equal',r'k = ?0']:
    ms=[m.start() for m in re.finditer(pat,vis)]
    if ms:
        print(pat,len(ms),'e.g.',[vis[max(0,i-40):i+50].replace('\n',' ') for i in ms[:3]])
