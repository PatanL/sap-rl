"""Independent real-record bookkeeping, no engine imports or simulated gold."""
from collections import Counter,defaultdict
from pathlib import Path
import csv,json
from identity import compile_turn
P=Path(__file__).resolve().parent
def fresh(d):return {'name':d.get('name'),'xp':d.get('xp',0),'bonus':d.get('sellBonus',0)}
def level(xp):return 1 if xp<2 else 2 if xp<5 else 3
def sales_ledger(before,acts,steps,names):
    stock={v['id']:fresh(v) for v in before['shop'] if v['f']}
    board={v['id']:fresh(v) for v in before['board'] if v}
    income=sum(level(v['xp']) for v in board.values() if v['name']=='Swan')
    for a,(op,dest) in zip(acts,steps):
        i=a.get('id');d=a.get('target')
        if op in ('buy','buy merge','shop merge','board merge'):
            if op=='board merge':src=board.pop(i,None)
            else:src=stock.get(i,dict(name=names.get(i),xp=0,bonus=0))
            if op=='buy':board[i]=src;continue
            target=dest if op=='buy merge' else d
            p=board.get(target)
            if p is None or src is None:board[target]=None;continue
            # Experience represents extra copies. A previous Cake sale bonus is kept.
            p['xp']=min(5,p['xp']+src['xp']+1);p['bonus']=max(p['bonus'],src['bonus'])
        elif op=='food':
            food=a.get('spell',{}).get('name')
            if food=='Sleeping Pill':board.pop(d,None)
            elif food=='Chocolate' and board.get(d):board[d]['xp']=min(5,board[d]['xp']+1)
            elif food and food.startswith('#') and board.get(d):board[d]=None
        elif op=='sell':
            p=board.pop(i,None)
            if not p or p['name'] is None:return None,'sale identity/level unavailable'
            lvl=level(p['xp']);income+=lvl+p['bonus']+(lvl if p['name']=='Pig' else 0)
    return income,None
def main():
 roll_cohorts=defaultdict(Counter);c=Counter();g={};m={}
 wallet=[];ambiguity=[];rtypes=Counter();inc_miss=[]
 for row in map(json.loads,open(P/'seats.jsonl')):
    for pet in row['start']['shop']:roll_cohorts[('opening',1)][pet['name']]+=1
    for t in set(row['actions'])-set(row['states']):c[f'incomplete action-only turn {t}']+=1
    for k,s in row['states'].items():
        n=int(k)
        if n<2 or str(n-1) not in row['states']:continue
        before=row['states'][str(n-1)];acts=row['actions'][k];tr,names,im,issues,problems=compile_turn(row,n,'near-right')
        for a,(op,dest) in zip(acts,tr):rtypes[op]+=1
        count=Counter(a['t'] for a in acts)
        prices=[a['spell']['cost'] for a in acts if a['t']==8]
        gross=count[5]+3*count[6]+sum(prices)
        inc,why=sales_ledger(before,acts,tr,names);split=row['split']
        new=roll_cohorts['snapshot',s['tier']]
        for v in s['shop']:
            if v['id']>before['unc'] and not v['reward']:new[v['name']]+=1
        values=dict(match=row['match'],seat=row['seat'],turn=n,split=split,gross=gross,GoSp=s['spent'],GoGa=s['gain'],Go=s['gold'],
                    inferred_income=inc,income_skip=why,rolls=count[5],Rold=s['rolls'],food_cost=sum(prices),pet_cost=3*count[6])
        wallet.append(values)
        if inc is not None and inc!=s['gain']:inc_miss.append(values)
        for splitkey in (split,'all'):
          v=g.setdefault(splitkey,Counter());v['turns']+=1;v['action cost equals GoSp']+=gross==s['spent']
          v['budget equals remaining gold']+=10+s['gain']-s['spent']==s['gold'];v['roll action count equals Rold']+=count[5]==s['rolls']
          v['spent over ten']+=s['spent']>10;v['left gold']+=s['gold']>0;v['income independently computable']+=inc is not None
          if inc is not None:v['independent income matches']+=inc==s['gain']
          v['UID inferred turns']+=bool(im);v['UIDs inferred']+=len(im)
    m[row['match']]=row['split']
 with (P/'gold-ledger.csv').open('w') as fp:
    w=csv.DictWriter(fp,fieldnames=list(wallet[0]));w.writeheader();w.writerows(wallet)
 (P/'raw-gold.json').write_text(json.dumps(dict(stats=g,extra=c,actions=rtypes,matches=Counter(m.values()),residuals=inc_miss),indent=2))
 (P/'raw-odds.json').write_text(json.dumps([[str(group),int(tier),dict(counts)] for (group,tier),counts in sorted(roll_cohorts.items())],indent=2))
 print(g,c,rtypes,'income residuals',len(inc_miss))
if __name__=='__main__':main()
