import io,re,sys
S=sys.argv[1]
o=set(l.strip() for l in io.open(S+'/old.txt',encoding='utf-8'))
for i,l in enumerate(io.open(S+'/new.txt',encoding='utf-8'),1):
    l=l.strip()
    if l in o and re.search(r'[\u0590-\u05ff]',l) and len(l)>60 and i>93:
        print(i,l[:260].replace('\u200f',''))
