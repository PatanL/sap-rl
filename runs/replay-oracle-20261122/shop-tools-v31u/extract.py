"""Recover player-supplied inputs independently of the engine. No network."""
import json,hashlib,collections as C
from pathlib import Path
from collections import Counter,defaultdict
P=Path(__file__).resolve().parent
ROOT=P.parent.parent
raw=P.parent/'replay-oracle-20261122/raw'
archive=P.parent/'expert-replay-ingest-20260926-v1/corpus/matches'
cal=ROOT/'SAP-Calculator/src/assets/data'
def read(p):return json.loads(p.read_text())
def enum(n):
 return {int(v['Id']):v['Name'] for v in read(cal/(n+'.json'))}
pets,foods,perks=map(enum,['pets','food','perks'])
perks[41]='Bread'
foods[10]='Bread'
def num(d,k):return int((d or {}).get(k) or 0)
def uid(d):return num(d,'Uni')
def name(d,tab):return tab.get(num(d,'Enu'),'#'+str(num(d,'Enu')))
def coord(d):return num(d,'x')
def de(x):return json.loads(x) if isinstance(x,str) else x or {}
def pet(d):
 if d is None:return None
 return dict(id=uid(d['Id']),name=name(d,pets),x=coord(d.get('Poi')),lvl=d.get('Lvl') or 1,xp=num(d,'Exp'),
             a=num(d.get('At'),'Perm'),h=num(d.get('Hp'),'Perm'),ta=num(d.get('At'),'Temp'),th=num(d.get('Hp'),'Temp'),
             perk=perks.get(d.get('Perk'),'#'+str(d['Perk'])) if d.get('Perk') is not None else None,
             uses=max([num(y,'TrCo') for y in d.get('Abil') or []] or [0]),sellBonus=num(d,'SeV'),
             f=bool(d.get('Fro')),cost=num(d,'Pri'),reward=bool(d.get('Rwrd')),links=[uid(v) for v in d.get('Link') or []])
def food(d):
 return dict(id=uid(d['Id']),name=name(d,foods),f=bool(d.get('Fro')),cost=num(d,'Pri'),power=num(d,'Pow'),stock=bool(d.get('Sto')))
def state(b):
 return dict(turn=num(b,'Tur'),tier=num(b,'Ti'),board=[pet(v) for v in b['Mins']['Items']],
             shop=[pet(v) for v in b.get('MiSh') or [] if v],food=[food(v) for v in b.get('SpSh') or [] if v],
             unc=num(b,'UNC'),spent=num(b,'GoSp'),gain=num(b,'GoGa'),gold=num(b,'Go'),rolls=num(b,'Rold'),
             buffs=b.get('MSBo') or [],result=num(b,'PrOu'),wins=num(b,'Vic'),losses=num(b,'Los'),
             pslots=num(b,'MiSC'),fslots=num(b,'SpSC'))
def act(a):
 t=a['Type'];q=de(a.get('Request'));r=de(a.get('Response'))
 d=q.get('Data') or {};out=dict(t=t,seed=(r.get('Data') or {}).get('Seed'))
 if 'BoardOrders' in d:out['orders']=[[uid(z['MinionId']),coord(z.get('Point'))] for z in d['BoardOrders']]
 if 'BoardFreezes' in d:out['freeze']=[[uid(z['ItemId']),bool(z.get('Freeze'))] for z in d['BoardFreezes']]
 if t==6:out.update(id=uid(q['MinionId']),x=coord(q.get('Point')))
 if t==7:out.update(id=uid(q['SourceMinionId']),target=uid(q['TargetMinionId']))
 if t==8:
  spell=((r.get('Event') or {}).get('Event') or {}).get('Spell')
  out.update(id=uid(q['SpellId']),target=uid(q['Aim']) if q.get('Aim') else None,spell=food(spell) if spell else None)
 if t==9:out.update(id=uid(q['Minion']))
 return out
def split(m):return 'heldout' if int(hashlib.sha256(m.encode()).hexdigest(),16)%5<2 else 'dev'
def extract():
 files=sorted(raw.glob('*.json'))+sorted(archive.glob('*/BUNDLE.json'))
 seats=set();matches=set();stat=Counter();kinds=Counter();grammar=Counter();issues=Counter(); genesis=[];pairs=Counter()
 with (P/'seats.jsonl').open('w') as fp:
  for path in files:
   source='raw' if path.parent==raw else 'archive'
   rec=read(path)
   for per in rec['perspectives']['perspectives']:
    r=per['raw'];m=r['MatchId'];pid=r['ParticipationId']
    if (m,pid) in seats:stat['duplicate seats']+=1;continue
    seats.add((m,pid));matches.add(m)
    gen=state(de(r['GenesisBuildModel'])['Bor'])
    turns=defaultdict(list);states={};known={};links=[];origin={};seen_battles=set()
    def observe(v,kind,why):
     if v is None:return
     val=[kind,v['name']];idx=v['id']
     assert known.get(idx,val)==val,(idx,known[idx],val)
     known[idx]=val;origin.setdefault(idx,why)
    def readstate(s,why):
     for v in s['board']+s['shop']:observe(v,'pet',why)
     for v in s['food']:observe(v,'food',why)
    readstate(gen,'genesis')
    for p in gen['shop']:genesis.append([m,p['name']])
    for a in r['Actions']:
     t=a['Turn'];typ=a['Type'];kinds[typ]+=1
     if typ==0 and a.get('Build'):
      b=de(a['Build'])['Bor'];s=state(b);states[t]=s;readstate(s,'snapshot');stat['snapshots '+source]+=1
      battle=de(a.get('Battle'))
      if battle.get('UserBoard'):
       assert battle['User']['Id']==r['UserId'];assert battle['Outcome']==s['result']
       ub=state(battle['UserBoard'])
       for f in ('board','shop','food','unc','spent','gain','gold'):
        if s[f]!=ub[f]:issues['postbattle vs prebattle '+f]+=1
      continue
     if typ in (1,2,3):continue
     action=act(a);turns[t].append(action)
     if action.get('orders'):grammar['order batches']+=1
     if action.get('freeze'):grammar['freeze batches']+=1
     if typ==7:links.append((action['id'],action['target']))
     if typ==8 and action['spell']:
      assert action['id']==action['spell']['id'];observe(action['spell'],'food','food response')
    # A legal explicit merge requires identical species. Propagate through both directions.
    while True:
     added=False
     for a,b in links:
      if a in known and b in known:
       assert known[a]==known[b],(m,a,b)
      elif a in known or b in known:
       x,y=(a,b) if a in known else (b,a);known[y]=known[x];origin[y]='explicit merge';added=True
     if not added:break
    for n in states:
     if n>=2:stat['eligible '+source]+=1
    row=dict(file=path.name if source=='raw' else str(path.parent.name)+'/BUNDLE.json',source=source,match=m,seat=pid,split=split(m),
             start=gen,actions=turns,states=states,known=known,origin=origin)
    fp.write(json.dumps(row,separators=(',',':'))+'\n')
    for t in set(turns)-set(states):stat['action-only turns without terminal snapshot']+=1
 print(stat,kinds,grammar,issues,len(matches),len(seats))
 (P/'corpus.json').write_text(json.dumps(dict(counts=stat,actionTypes=kinds,grammar=grammar,prePostDifferences=issues,matches=len(matches),seats=len(seats),
        rawFiles=len(list(raw.glob('*.json'))),archiveFiles=len(list(archive.glob('*/BUNDLE.json')))),indent=2))
if __name__=='__main__':extract()
