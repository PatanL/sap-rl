"""Preselect validation across dev matches; keep the recorded JSON text byte-for-byte."""
from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent/'session2'; R=D.parent.parent/'replay-oracle-20261122'
def h(v):return hashlib.sha256(v.encode()).hexdigest()
def dev(m):return int(h(m),16)%5>=2
def main():
 (D/'fixtures').mkdir(exist_ok=True)
 failed={x['battleId'] for x in json.load(open(R/'v31p-result-340.json'))['rows'] if x.get('outcomePossible') is False or x.get('statePossible') is False}
 candidates=[]
 for f in sorted((R/'raw').glob('*.json')):
  doc=json.loads(f.read_text())
  raw=doc['perspectives']['perspectives'][0]['raw'];m=raw['MatchId']
  if not dev(m):continue
  for a in raw['Actions']:
   if a['Type']!=0 or not a.get('Battle'):continue
   s=a['Battle'];b=json.loads(s)
   if b['Id'] not in failed:candidates.append((h('client-validation-20261123/'+b['Id']),m,b,s))
 chosen=[];seen=set()
 for _,m,b,s in sorted(candidates,key=lambda v:v[0]):
  if m in seen:continue
  seen.add(m);i=len(chosen)+1;name=f'validation-{i:02d}.json';(D/'fixtures'/name).write_bytes(s.encode())
  chosen.append(dict(file=name,id=b['Id'],match=m,sha256=h(s),seed=b['Seed'],outcome=b['Outcome']))
  if len(chosen)==20:break
 (D/'validation-manifest.json').write_text(json.dumps(chosen,indent=2))
 print('validation:',len(chosen),'distinct dev matches')
if __name__=='__main__':main()
