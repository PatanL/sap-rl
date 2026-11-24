import json
from pathlib import Path
from case import details,load
from receipts import P,load_rows,size
READINGS={
 ('01a1c61f',6):'Two Salad Bowls and one Rabbit. Permuting notifications within an already-selected Salad pair reproduces the real health split.',
 ('01a1d3c6',6):'Two Salad Bowls reach a Rabbit’s three-use cap. The same recipients can generate the recorded health if the capped notification goes to the other pet.',
 ('01a1b9fd',10):'The two Canned Foods consumed the native Rabbit’s uses on shop pets. The real Tiger received the level-two Rabbit’s +2 HP when it ate Bread.',
 ('01a1ccee',7):'Ignoring Canned Food’s shop pets for Rabbit’s budget accounts for the three later +1 HP adjustments on the actual board.',
 ('01a1ce58',12):'The real frozen Scorpion was bare. The native constructor attached Peanut while the pet was still in the shop.',
 ('01a1d3c6',8):'Canned Food precedes the Squirrel buy-merge and the Camel’s Cupcake. The native shop Squirrel picked up an extra HP that survived the merge; the real Camel received that HP. This matches Rabbit spending a use on shop stock.',
 ('01a1d859',12):'The recorded frozen Scorpion had no perk while its native counterpart already had Peanut.',
 ('01a1db4e',12):'Steak consumes one Rabbit use before Sushi. Changing which two Sushi recipients receive the remaining health bonuses accounts for the difference, with the Cat and Sushi stat buffs unchanged.',
 ('01a1e117',7):'Two Rabbits and two Salad Bowls. Independent recipient order for each Rabbit can split the last health points as recorded. A single shared fixed order does not reproduce this split.',
 ('01a194ec',10):'Better Apple consumes a Rabbit use before Sushi. Changing the order of the Sushi notifications accounts for the Cat/Shark health difference.'
}
def signature(v):return v['key'][0][:8],v['key'][2]
def diff(r):
 out=[]
 for i,(a,b) in enumerate(zip(r['actual']['board'],r['result']['board'])):
  if a==b:continue
  if not a or not b or a[0]!=b[0]:out.append(f'x{i} {a} / {b}');continue
  fields=['species','level','XP','attack','HP','temp attack','temp HP','perk']
  for j in range(1,len(a)):
   if a[j]!=b[j]:out.append(f'{a[0]} x{i} {fields[j]} {a[j]} / {b[j]}')
 for a in r['actual']['frozen']:
  b=next((q for q in r['result']['frozen'] if q[0]==a[0]),None)
  if b!=a:
   if b and a[1]=='pet' and a[:-2]==b[:-2]:out.append(f'Frozen {a[2]} #{a[0]} perk {a[-2] or "none"} / {b[-2] or "none"}')
   else:out.append(f'Frozen {a} / {b}')
 return '; '.join(out)
def main():
 rows,_=load_rows();miss=sorted([r for r in rows.values() if r.get('diff')],key=size)
 probes={tuple(r['key']):r for r in json.load(open(P/'food-probes.json'))}
 dest=P/'cases';dest.mkdir(exist_ok=True);summ=[]
 for rank,v in enumerate(miss[:10],1):
  row,n=load(v['key']);d=details(row,n,v['seed']);label=f'{rank:02d}-{row["match"][:8]}-t{n}'
  ops=[]
  for a,(op,target) in zip(d['actions'],d['steps']):
   if op in ('start','end'):continue
   a_name=d['names'].get(a.get('id'))
   if op=='food':ops.append(a['spell']['name'])
   elif op=='roll':ops.append('roll')
   else:ops.append(op+' '+str(a_name))
  data=dict(split=v['split'],replayFile=row['file'],why=READINGS[signature(v)],headlineDifference=diff(v),seed=v['seed'],result=v,fixture=d,foodTargetProbe=probes.get(tuple(v['key'])))
  (dest/(label+'.json')).write_text(json.dumps(data,indent=2))
  summ.append(dict(key=v['key'],case=label,split=v['split'],actions=ops,beforePets=v['boardSize'],diff=diff(v),mechanism=READINGS[signature(v)],tested=v['tries'],distinct=v['distinct']))
 for label,key in [('dragon-pig',('01a195bf-22cf-78f0-9a27-52538721a2cb','01a195bf-22d1-7e75-8e30-14317986b4f9',11)),
                   ('scorpion-purchase',next(tuple(v['key']) for v in miss if v['key'][0].startswith('01a1b081') and v['key'][2]==10))]:
  row,n=load(key);(dest/(label+'.json')).write_text(json.dumps(details(row,n,rows[key]['seed']),indent=2))
 (P/'smallest.json').write_text(json.dumps(summ,indent=2))
if __name__=='__main__':main()
