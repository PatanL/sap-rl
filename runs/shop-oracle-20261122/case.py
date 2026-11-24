import argparse,json
from replay import compile_turn,engine_turn,expected,settle
def load(key):
 m,p,n=key
 for line in open('seats.jsonl'):
  row=json.loads(line)
  if row['match']==m and row['seat']==p:return row,int(n)
 raise KeyError(key)
def details(row,n,seed=None):
 k=str(n);prev=row['states'][str(n-1)];end=row['states'][k]
 if seed is None:
  import zlib
  seed=zlib.crc32('/'.join(map(str,[row['match'],row['seat'],n])).encode())
 steps,names,implied,issues,problems=compile_turn(row,n,'near-right')
 kinds={int(i):v[0] for i,v in row['known'].items()};kinds.update({i:'pet' for i in implied})
 run=engine_turn(prev,end,row['actions'][k],names,kinds,seed,trace=True)
 return dict(key=[row['match'],row['seat'],n],before=prev,after=end,steps=steps,actions=row['actions'][k],
             names={i:names.get(i) for a in row['actions'][k] for i in [a.get('id'),a.get('target')] if i is not None},
             inferred=implied,run=run)
def compact(d):
 before=d['before'];after=d['after'];print('case',d['key'],'base',before['unc'],'endUNC',after['unc'])
 print('prev',[(v['x'],v['id'],v['name'],v['lvl'],v['a'],v['h'],v['perk'],v['sellBonus']) for v in before['board'] if v])
 print('prev shop',[(v['id'],v['name'],v['f'],v['cost']) for v in before['shop']+before['food']])
 for j,(a,step) in enumerate(zip(d['actions'],d['steps'])):
  lg=d['run']['log'][j] if j<len(d['run']['log']) else {}
  print(j,step,a.get('id'),d['names'].get(a.get('id')),a.get('x'),a.get('target'),a.get('orders'),a.get('freeze'),(a.get('spell') or {}).get('cost'),
        '|',lg.get('gold'),lg.get('cost'),lg.get('unc'),lg.get('uids'))
 print('after',[(v['x'],v['id'],v['name'],v['lvl'],v['a'],v['h'],v['ta'],v['th'],v['perk']) for v in after['board'] if v], 'shop',[(v['id'],v['name'],v['f'],v['cost']) for v in after['shop']+after['food']],'budget',after['gold'],after['gain'],after['spent'])
 print('engine',d['run'].get('error'),d['run'].get('state') or d['run'].get('partial'),d['run'].get('genIssues'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('key',nargs=3);p.add_argument('--seed',type=int);p.add_argument('--save');a=p.parse_args()
 row,n=load(a.key);d=details(row,n,a.seed)
 if a.save:json.dump(d,open(a.save,'w'),indent=2)
 else:compact(d)
