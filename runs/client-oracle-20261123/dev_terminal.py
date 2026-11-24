import json,collections as C
from pathlib import Path
from physical import physical
D=Path(__file__).resolve().parent/'session2'
pet={int(x['Id']):x['Name'] for x in json.load(open(D.parent.parent.parent/'SAP-Calculator/src/assets/data/pets.json'))}
def stat(x,k):return sum((x.get(k) or {}).get(j,0) or 0 for j in ['Perm','Temp'])
def summarize():
 human={}
 src=D/'dev-endpoint-manual.tsv'
 if src.exists():
  for l in src.read_text().splitlines():
   key,side,fields=l.split('\t');v=[]
   for entry in fields.split(',') if fields!='-' else []:
    name,stats=entry.rsplit(':',1);a,h=map(int,stats.split('/'));v.append((name,a,h))
   human[key]=(side,v)
 out=[]
 for job in json.load(open(D/'dev-manifest.json')):
  d=D/'captures'/Path(job['file']).stem
  if not (d/'trace.json').exists():continue
  b=json.load(open(D/'fixtures'/job['file']));real=json.load(open(d/'trace.json'))['readout'][-1]
  real=list(reversed(real['player']))+real['opponent'];exp=[(i,pet[x.get('Enu',0)],stat(x,'At'),stat(x,'Hp')) for i,x in enumerate(b['ResultingBoard']['Mins']['Items']) if x]
  seen=[(4-physical(q['x']) if q['x']<960 and physical(q['x']) is not None else 5+physical(q['x']) if physical(q['x']) is not None else None,q['pet'],q['attack'],q['health']) for q in real]
  manual=human.get(job['id'][:8]);review=None
  if manual:
   side,v=manual
   review=[side=={1:'L',2:'R',3:'D'}[b['Outcome']],v==[tuple(row[1:]) for row in exp]]
  exact=exp==seen
  unknown=len(exp)==len(seen) and all(all(y==x or y is None for y,x in zip(s,t)) for s,t in zip(seen,exp))
  out.append(dict(id=job['id'],slots_and_stats=len(exp)==len(seen) and all((s[0],s[2:])==(x[0],x[2:]) for s,x in zip(seen,exp)),flags='exact' if exact else 'partially read' if unknown else 'requires visual review',observed=seen,recorded=exp,manual=manual,manual_winner_and_board=review))
 (D/'dev-terminal-comparison.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':print(C.Counter(v['flags'] for v in summarize()))
