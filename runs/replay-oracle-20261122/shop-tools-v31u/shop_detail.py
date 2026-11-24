"""Same-witness checks of the unbought inventory and its displayed order."""
import collections as C,json,multiprocessing as mp
from receipts import load_rows
from replay import expected_pet
from case import details
def obs(s):
 return [dict(uid=x['id'],kind='pet',body=expected_pet(x),cost=x['cost'],frozen=x['f'],reward=x['reward']) for x in s['shop']]+[
        dict(uid=x['id'],kind='food',body=[x['name']],cost=x['cost'],frozen=x['f'],reward=False) for x in s['food']]
def one(arg):
 row,rr=arg;out=[]
 for n,r in rr.items():
  if not r.get('completed'):continue
  d=details(row,n,r['seed']);native=d['run']['shopDetail'];actual=obs(d['after'])
  a={v['uid']:v for v in actual};b={v['uid']:v for v in native};ids=a.keys()==b.keys()
  order=lambda vs,kind:[v['uid'] for v in vs if v['kind']==kind]
  issues=[];scorp=False
  for i in a.keys()&b.keys():
   if a[i]!=b[i]:
    issues.extend(k for k in ('kind','body','cost','frozen','reward') if a[i][k]!=b[i][k])
    if a[i]['kind']=='pet' and a[i]['body'][0]=='Scorpion' and a[i]['body'][-1] is None and b[i]['body'][-1]=='Peanut':scorp=True
  po=order(actual,'pet')==order(native,'pet');fo=order(actual,'food')==order(native,'food')
  out.append(dict(key=r['key'],split=r['split'],primary=r['reproduced'],ids=ids,content=ids and not issues,petOrder=po,foodOrder=fo,
                  fullyOrdered=ids and not issues and po and fo,issues=sorted(set(issues)),scorpionPeanutMismatch=scorp,
                  bonusMatches=[d['run']['log'][-1]['unc']==d['after']['unc']]))
 return out
def main():
 by=C.defaultdict(dict)
 for (m,p,n),r in load_rows()[0].items():by[m,p][n]=r
 rows=list(map(json.loads,open('seats.jsonl')));result=[]
 with mp.Pool(6) as pool:
  for sub in pool.imap_unordered(one,[(r,by[r['match'],r['seat']]) for r in rows]):result+=sub
 with open('remaining-shop.jsonl','w') as f:
  for r in sorted(result,key=lambda r:r['key']):f.write(json.dumps(r)+'\n')
 for cond in (True,False):
  z=[r for r in result if r['primary']==cond];print(cond,len(z),{k:sum(r[k] for r in z) for k in ['ids','content','petOrder','foodOrder','fullyOrdered','scorpionPeanutMismatch']})
if __name__=='__main__':main()
