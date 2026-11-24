import json,collections as C
from pathlib import Path
def diagnostic(fn):
 rows=[x for x in map(json.loads,open(fn)) if x.get('diff')]
 for x in sorted(rows,key=lambda x:(x['actions'],x['boardSize'],x['key'])):
  d=[]
  for i,(a,b) in enumerate(zip(x['actual']['board'],x['result']['board'])):
   if a!=b:
    if a and b and a[0]==b[0]:d.append(f'{i} {a[0]} real'+str([(k,a[k],b[k]) for k in range(1,8) if a[k]!=b[k]]))
    else:d.append(f'{i} '+str(a)+' vs '+str(b))
  if 'frozen' in x['diff']:d.append('frozen '+str(x['actual']['frozen'])+' vs '+str(x['result']['frozen']))
  print(x['actions'],x['boardSize'],x['key'][0][:8],x['key'][1][:8],x['key'][2],x['distinct'],x['pets'],x['foods'],' '.join(d)[:520],(x.get('error') or '')[:130])
if __name__=='__main__':
 import sys
 diagnostic(sys.argv[1])
