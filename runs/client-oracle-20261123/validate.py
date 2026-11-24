"""Compare terminal frames read left-to-right with the recorded result. No endpoint data in image read."""
from pathlib import Path
import collections,json,re,statistics
D=Path(__file__).resolve().parent/'session2'
pets={int(v['Id']):v['Name'] for v in json.loads((D.parent.parent.parent/'SAP-Calculator/src/assets/data/pets.json').read_text())}
def totals(p,k):return sum((p.get(k) or {}).get(x,0) or 0 for x in ('Perm','Temp'))
def summary():
 out=[]
 for line in (D/'terminal-manual.tsv').read_text().splitlines():
  ix,side,raw=line.split('\t')
  disp=[]
  for x in raw.split(',') if raw!='-' else []:
   nm,stats=x.rsplit(':',1);a,h=stats.split('/');disp.append((nm,int(a),int(h)))
  name='validation-'+ix
  b=json.loads((D/'fixtures'/(name+'.json')).read_text());rec=json.loads((D/'captures'/name/'run.json').read_text())
  expected=[(pets.get(v.get('Enu',0),'?'),totals(v,'At'),totals(v,'Hp')) for v in b['ResultingBoard']['Mins']['Items'] if v]
  win=side=={1:'L',2:'R',3:'D'}[b['Outcome']]
  out.append(dict(fight=name,id=b['Id'],side=side,read=disp,expected=expected,outcome_match=win,multiset_match=collections.Counter(disp)==collections.Counter(expected),order_match=disp==expected,
      seconds=rec['seconds_to_terminal'],source=f'captures/{name}/'+rec['terminal_frame']))
 (D/'validation-results.json').write_text(json.dumps(out,indent=2))
 print('n',len(out),'outcome/identities/stats',sum(r['outcome_match'] and r['multiset_match'] for r in out),'order',sum(r['order_match'] for r in out))
 for r in out:
  if not(r['outcome_match'] and r['order_match']):print(r)
if __name__=='__main__':summary()
