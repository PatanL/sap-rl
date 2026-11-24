"""Pair rendered candidate boundaries with native attack rounds. Keep unreadable fields explicit."""
import json,collections,sys
from pathlib import Path
D=Path(__file__).resolve().parent/'session2'
def fmt(side):
 if not side:return 'empty'
 def token(p):
  n=p['pet'] or '?';v='? ' if p.get('attack') is None else str(p['attack']);hp='?' if p.get('health') is None else p['health']
  perk=p.get('perk')
  slot=p.get('slot_front_0',p.get('slot'));prefix=f'#{slot} ' if slot is not None else ''
  return f'{prefix}{n} {v}/{hp}'+(f' [{perk}]' if perk and perk!='none' else '')
 return ', '.join(token(p) for p in side)
def difference(real,eng):
 diff=[];unread=[]
 for side in ('player','opponent'):
  rs,es=real[side],eng[side]
  if len(rs)!=len(es):diff.append(f'{side} pet counts real {len(rs)}, engine {len(es)}')
  for i,(x,y) in enumerate(zip(rs,es)):
   for field in ('pet','attack','health'):
    if x[field] is None:unread.append(f'{side}/{i}/{field}');continue
    if x[field]!=y[field]:diff.append(f'{side} slot {i} {field}: {x[field]} / {y[field]}')
 return diff,unread
def pair(dest):
 real=json.loads((dest/'trace.json').read_text())
 ep=D/'engine-traces'/(real['id']+'.json')
 if not ep.exists():return
 eng=json.loads(ep.read_text());es={r['stage']:r for r in eng['steps']}
 rr=real['readout']
 if rr and rr[-1]['stage']=='final' and len(rr)>1 and rr[0]['stage']=='start' and not any(v.get('terminal_alias') for v in rr):
  rr=rr[:-1]+[dict(rr[-1],stage=f'after-attack-{len(rr)-1:02d}'),rr[-1]]
 pairs=[];first=None
 for r in rr:
  e=es.get(r['stage'])
  if e is None:
   dif=['engine had already finished'];u=[]
  else:dif,u=difference(r,e)
  row=dict(stage=r['stage'],real=r,engine=e,differences=dif,unread=u)
  pairs.append(row)
  if first is None and dif:first=r['stage']
 earliest=next((r for r in pairs if r['differences']),None)
 text=['# '+real['id'],'','Recorded seed '+str(real['seed'])+'. Both sides are shown front to back. Question marks denote unverified readings. Frames are direct evidence; a paired-damage marker is a candidate attack delimiter. An earlier missing field or a random target can change attribution.','',
    '| Stage and real frame | Real player | Engine player | Real opponent | Engine opponent |','|---|---|---|---|---|']
 for p in pairs:
  r=p['real'];e=p['engine']
  text.append(f'| [{r["stage"]}]({r["frame"]}) | {fmt(r["player"])} | {fmt(e["player"]) if e else "ended"} | {fmt(r["opponent"])} | {fmt(e["opponent"]) if e else "ended"} |')
 if earliest:
  text.extend(['','First machine-flagged boundary **'+earliest['stage']+'**. '+'; '.join(earliest['differences'])+'.'])
 else:text.extend(['','No difference identified in the readable fields.'])
 text.extend(['','[All native steps](../../engine-traces/'+real['id']+'.json) · [All real keyframe readings](trace.json) · [Full frame timeline](run.json)'])
 if (dest/'battle.webm').exists():text.append('· [Native canvas movie](battle.webm)')
 if (dest/'rounds-contact.jpg').exists():text.append('· [Visual round-by-round sheet](rounds-contact.jpg)')
 (dest/'paired.md').write_text('\n'.join(text)+'\n');(dest/'paired.json').write_text(json.dumps(dict(first_candidate=first,rows=pairs),indent=2))
if __name__=='__main__':
 for p in sys.argv[1:]:pair(Path(p))
