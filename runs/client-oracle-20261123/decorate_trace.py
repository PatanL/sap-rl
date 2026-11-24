"""Materialize the observed final outcome and final attack alias in both machine-readable and image output."""
from pathlib import Path
import json,os
from physical import physical
def decorate(d,v):
 rows=v['readout']
 for r in rows:
  for side in ('player','opponent'):
   for p in r[side]:p['slot_front_0']=physical(p['x']) if p.get('x') is not None else None
 last=rows[-1]
 if not last['player'] and not last['opponent']:out='draw'
 elif last['player'] and not last['opponent']:out='player'
 elif last['opponent'] and not last['player']:out='opponent'
 else:out='unresolved'
 v['observed_outcome']=out;v['outcome_basis']='occupied sides in the terminal frame; see frame if badge recognition is uncertain'
 if len(rows)>1 and rows[0]['stage']=='start' and rows[-1]['stage']=='final' and not any(p.get('terminal_alias') for p in rows):
  new=dict(last,stage=f'after-attack-{len(rows)-1:02d}',terminal_alias=True);target=d/'terminal-round.jpg'
  if not target.exists():os.link(d/last['frame'],target)
  new['frame']='terminal-round.jpg';rows.insert(-1,new)
 return v
if __name__=='__main__':
 import sys
 for f in sys.argv[1:]:
  d=Path(f);p=d/'trace.json';v=decorate(d,json.loads(p.read_text()));p.write_text(json.dumps(v,indent=2))
