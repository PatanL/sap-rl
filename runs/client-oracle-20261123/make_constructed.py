"""Local experimental input fixtures for the three placement questions."""
from pathlib import Path
import json,copy,hashlib,uuid
from select_fights import D,R,h,dev
def main():
 base=json.loads((D/'fixtures/validation-01.json').read_text());catalog={int(v['Id']):v['Name'] for v in json.load(open(D.parent.parent.parent/'SAP-Calculator/src/assets/data/pets.json'))};templates={}
 for path in sorted((R/'raw').glob('*.json')):
  roles=json.loads(path.read_text())['perspectives']['perspectives']
  if not dev(roles[0]['raw']['MatchId']):continue
  for a in roles[0]['raw']['Actions']:
   if not a.get('Battle'):continue
   b=json.loads(a['Battle'])
   for p in b['UserBoard']['Mins']['Items']:
    if p and p.get('Lvl',1)==1 and not p.get('Exp') and not p.get('Perk') and not any(x.get('TrCo') for x in p['Abil']):
     templates.setdefault(catalog[p.get('Enu',0)],p)
  if {'Deer','Fish','Pig','Beaver','Duck','Rabbit','Hedgehog','Dolphin'}<=templates.keys():break
 perk=next(int(v['Id']) for v in json.load(open(D.parent.parent.parent/'SAP-Calculator/src/assets/data/perks.json')) if v['Name']=='Mushroom')
 plans=[
 ('front-deer',[(4,'Deer',2,1,perk)],[(4,'Fish',2,28,None)]),
 ('blast-middle',[(0,'Duck',1,18,None),(1,'Rabbit',2,17,None),(2,'Deer',6,2,perk),(3,'Beaver',3,2,None),(4,'Pig',4,2,None)],[(3,'Fish',1,32,None),(4,'Hedgehog',1,1,None)]),
 ('snipe-gaps',[(0,'Pig',4,18,None),(2,'Deer',3,1,perk),(4,'Beaver',5,19,None)],[(3,'Fish',1,30,None),(4,'Dolphin',2,25,None)])]
 man=[]
 for name,left,right in plans:
  b=copy.deepcopy(base);b['Id']=str(uuid.uuid5(uuid.NAMESPACE_DNS,'client-oracle-'+name));b['Seed']=870113;b['Outcome']=3
  for k,specs in [('UserBoard',left),('OpponentBoard',right)]:
   bor=b[k];bor['Mins']['Items']=[None]*5
   for x,n,atk,hp,pk in specs:
    p=copy.deepcopy(templates[n]);p.update(Poi={'x':x},At={'Perm':atk},Hp={'Perm':hp},Id={'BoId':bor['Id'],'Uni':40+x},Own=1,Lvl=1,Loc=1)
    if pk is not None:p['Perk']=pk
    for v in p['Abil']:
     for t in ('TrCo','AcCo','UsCo'):v.pop(t,None)
    bor['Mins']['Items'][x]=p
   bor['UNC']=100
  b['ResultingBoard']['Mins']['Items']=[None]*10
  s=json.dumps(b,separators=(',',':'));file='custom-'+name+'.json';(D/'fixtures'/file).write_text(s);man.append(dict(file=file,id=b['Id'],sha256=h(s),inputs=[left,right]))
 (D/'custom-manifest.json').write_text(json.dumps(man,indent=2))

# ---------------------------------------------------------------------------------------------
# Session 3: generic spec-driven builder.  `make_constructed.py spec <spec.json> <batch>` writes
# one fixture per (test, seed) to session2/fixtures/c3-<test>-<seedtag>.json and a batch manifest
# session2/<batch>-manifest.json usable with `client.py --replay-manifest <batch>`.
# Boards are written front to back: index 0 is the front pet (JSON Poi.x = 4), None is an empty slot.
# A pet is [species, attack, health] or [species, attack, health, {lvl, perk, uni, exp, tatk, thp}] (tatk/thp = temporary part, added on top).
# Templates are real recorded pets of the same species and level, stripped of shop-only fields.
SEEDS=[870113,-1714808783,1203948571,-402918337,1999999937,-2000000011,55555,-987654321,314159265,-271828182,1618033988,-1414213562]
def load_templates():
 import collections
 cache=D.parent/'session3/templates.json'
 if cache.exists():return {tuple(json.loads(k)):v for k,v in json.loads(cache.read_text()).items()}
 catalog={int(v['Id']):v['Name'] for v in json.load(open(D.parent.parent.parent/'SAP-Calculator/src/assets/data/pets.json'))};out={}
 def score(p):return (len(p),bool(p.get('Perk')),any(x.get('TrCo') or x.get('AcCo') for x in p['Abil']))
 for sub in ('raw','raw-fresh','raw-fresh2'):
  for path in sorted((R/sub).glob('*.json')):
   for pers in json.loads(path.read_text())['perspectives']['perspectives']:
    for a in pers['raw']['Actions']:
     if not a.get('Battle'):continue
     b=json.loads(a['Battle'])
     for side in ('UserBoard','OpponentBoard'):
      for p in b[side]['Mins']['Items']:
       if not p:continue
       n=catalog.get(p.get('Enu',0))
       if n is None or n=='Parrot':continue
       k=(n,p.get('Lvl',1))
       if k not in out or score(p)<score(out[k]):out[k]=p
 cache.parent.mkdir(exist_ok=True);cache.write_text(json.dumps({json.dumps(k):v for k,v in out.items()},indent=1));return out
def construct(name,left,right,seed,templates=None,perks=None):
 templates=templates or load_templates()
 perks=perks or {v['Name']:int(v['Id']) for v in json.load(open(D.parent.parent.parent/'SAP-Calculator/src/assets/data/perks.json'))}
 base=json.loads((D/'fixtures/validation-01.json').read_text())
 b=copy.deepcopy(base);b['Id']=str(uuid.uuid5(uuid.NAMESPACE_DNS,f'client-oracle-s3-{name}-{seed}'));b['Seed']=seed;b['Outcome']=3
 for k,specs,ubase in [('UserBoard',left,40),('OpponentBoard',right,60)]:
  bor=b[k];bor['Mins']['Items']=[None]*5
  assert len(specs)<=5
  for i,sp in enumerate(specs):
   if sp is None:continue
   n,atk,hp=sp[:3];o=sp[3] if len(sp)>3 else {};lvl=o.get('lvl',1);x=4-i
   t=templates[(n,lvl)]
   p={q:copy.deepcopy(t[q]) for q in ('Own','Enu','Loc','Lvl','Abil','Pri') if q in t}
   p.update(Poi={'x':x},At={'Perm':atk},Hp={'Perm':hp},Id={'BoId':bor['Id'],'Uni':o.get('uni',ubase+x)},Own=1,Lvl=lvl,Loc=1)
   if o.get('tatk'):p['At']['Temp']=o['tatk']
   if o.get('thp'):p['Hp']['Temp']=o['thp']
   exp=o.get('exp',{1:0,2:2,3:5}[lvl])
   if exp:p['Exp']=exp
   if o.get('perk'):p['Perk']=perks[o['perk']]
   for v in p['Abil']:
    for q in ('TrCo','AcCo','UsCo','Dur'):v.pop(q,None)
    v['Lvl']=lvl
   if o.get('trco'):p['Abil'][0]['TrCo']=o['trco']   # uses already spent this turn (as recorded after a shop phase)
   bor['Mins']['Items'][x]=p
  bor['UNC']=100
 b['ResultingBoard']['Mins']['Items']=[None]*10
 return json.dumps(b,separators=(',',':'))
def spec_main(specfile,batch,nseeds=None):
 tests=json.loads(Path(specfile).read_text());templates=load_templates();man=[]
 for t in tests:
  seeds=t.get('seeds') or SEEDS[:nseeds or t.get('nseeds',2)]
  for seed in seeds:
   tag=('m' if seed<0 else 'p')+str(abs(seed));s=construct(t['name'],t['left'],t['right'],seed,templates)
   file=f"c3-{t['name']}-{tag}.json";dst=D/'fixtures'/file
   if dst.exists() and dst.read_text()!=s:raise SystemExit('fixture exists with different content: '+file)
   dst.write_text(s);man.append(dict(file=file,id=json.loads(s)['Id'],seed=seed,sha256=h(s),test=t['name'],left=t['left'],right=t['right']))
 (D/(batch+'-manifest.json')).write_text(json.dumps(man,indent=1));print(batch,len(man),'fixtures')
if __name__=='__main__':
 import sys
 if len(sys.argv)==1:main()
 elif len(sys.argv)>2 and sys.argv[1]=='spec':
  spec_main(sys.argv[2],sys.argv[3],int(sys.argv[4]) if len(sys.argv)>4 else None)
