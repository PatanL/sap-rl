"""Package a user-supplied battle .json or JSON-string as a local viewer input, without changing battle fields."""
import argparse,hashlib,json,re
from pathlib import Path
D=Path(__file__).resolve().parent/'session2'
def main():
 p=argparse.ArgumentParser();p.add_argument('battle',type=Path);p.add_argument('--label',required=True);a=p.parse_args()
 if not re.fullmatch(r'[a-zA-Z0-9_-]+',a.label):p.error('Use a filename-safe label')
 raw=a.battle.read_bytes();obj=json.loads(raw)
 if isinstance(obj,str):raw=obj.encode('utf-8');obj=json.loads(raw)
 for k in ('Id','Seed','UserBoard','OpponentBoard','Outcome','ResultingBoard'):
  if k not in obj:p.error('Missing battle field '+k)
 fn='user-'+a.label+'.json';dst=D/'fixtures'/fn
 if dst.exists() and dst.read_bytes()!=raw:p.error('Choose a new label for a different input')
 dst.write_bytes(raw);m=D/(a.label+'-manifest.json')
 m.write_text(json.dumps([dict(id=obj['Id'],seed=obj['Seed'],file=fn,sha256=hashlib.sha256(raw).hexdigest())],indent=2))
 print(m)
if __name__=='__main__':main()
