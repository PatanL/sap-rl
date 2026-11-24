"""Pair primary and all-shopping-inventory witnesses at a common maximum of 1,024 seeds."""
import json
from collections import Counter
from pathlib import Path
from receipts import P
def main():
 d={};curve=[]
 for cap in (1,64,1024):
  path=P/f'full-{cap}.jsonl'
  if not path.exists():break
  for l in open(path):
   r=json.loads(l);d[tuple(r['key'])]=r
  curve.append([cap,sum(r.get('reproduced') is True for r in d.values())])
 result={}
 for sp in ('all','dev','heldout'):
  xs=[r for r in d.values() if not r.get('skip') and (sp=='all' or sp==r['split'])]
  result[sp]=dict(attempted=len(xs),core=sum('core joint' in r['ever'] for r in xs),
              content=sum('inventory content joint' in r['ever'] for r in xs),
              petOrder=sum('pet order joint' in r['ever'] for r in xs),foodOrder=sum('food order joint' in r['ever'] for r in xs),full=sum(r['reproduced'] for r in xs))
 json.dump(dict(curve=curve,summary=result),open(P/'full-shop-summary.json','w'),indent=2)
 print(curve,result['all'])
if __name__=='__main__':main()
