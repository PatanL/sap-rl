"""Attach physical slots and separately attributed human frame readings to every published pair."""
import json
from pathlib import Path
from physical import physical
from paired import pair
from read_frames import D,badges
import cv2
def audit(d,manual):
 p=d/'trace.json';j=json.loads(p.read_text());r=j['readout']
 for x in r:
  for side in ('player','opponent'):
   for z in x[side]:z['slot_front_0']=physical(z['x']) if z.get('x') is not None else None
  key=(d.name[4:],x['stage'])
  if key in manual:
   path,ps,os=manual[key]
   if path!='-':
    x['frame']=path
    src=json.load(open(d/'run.json'));x['time']=next(t for f,t in src['frames'] if f==path)
   im=cv2.imread(str(d/x['frame']));bb=badges(im)
   for side,values in [('player',ps),('opponent',os)]:
    boxes=[v for v in bb if (v['x']<960)==(side=='player')]
    if side=='player':boxes.reverse()
    rows=[]
    for i,v in enumerate(values.split(',') if values!='empty' else []):
     nm,stat=v.rsplit(':',1);at,hp=map(int,stat.split('/'));pos=boxes[i]['x'] if len(boxes)==len(values.split(',')) else None
     orig=next((p for p in x[side] if pos is not None and p.get('x')==pos),{})
     rows.append(dict(pet=nm,attack=at,health=hp,perk=orig.get('perk'),x=pos,slot_front_0=physical(pos) if pos is not None else None,identity_basis='human reading of the linked frame'))
    x[side]=rows
   x['status']='human-read board at indicated boundary'
 if len(r)>=2 and r[0]['stage']=='start' and r[-1]['stage']=='final' and not any(x.get('terminal_alias') for x in r):
  final=dict(r[-1]);final['stage']=f'after-attack-{len(r)-1:02d}';final['terminal_alias']=True;r.insert(-1,final)
 from decorate_trace import decorate
 j=decorate(d,j);p.write_text(json.dumps(j,indent=2));pair(d)
def main():
 m={}
 for l in (D/'manual-boards.tsv').read_text().splitlines():
  k,stage,fn,pl,opp=l.split('\t');m[k,stage]=(fn,pl,opp)
 for d in (D/'captures').glob('dev-*'):
  if (d/'refined.done').exists():audit(d,m)
if __name__=='__main__':main()
