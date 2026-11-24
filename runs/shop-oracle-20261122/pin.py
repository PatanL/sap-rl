import hashlib, json, os
from pathlib import Path
here=Path(__file__).resolve().parent
source=Path(os.environ.get('SAP_ENGINE_ROOT',here.parent/'replay-oracle-20261122/engine-v31')).resolve()
tag=os.environ.get('PIN_LABEL','v31')
assert '/' not in tag and tag not in ('.','..')
target=here/f'pinned-{tag}'
def manifest():
 return {p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((source/'sapai').rglob('*.py'))}
a=manifest()
pinfile=here/('engine-pin.json' if tag=='v31' else f'engine-pin-{tag}.json')
if pinfile.exists():
 assert json.loads(pinfile.read_text())['files']==a,'Choose a fresh PIN_LABEL for the changed engine' 
for name,digest in a.items():
 blob=(source/name).read_bytes();assert hashlib.sha256(blob).hexdigest()==digest
 dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
assert a==manifest()
pinfile.write_text(json.dumps({'source':str(source),'files':a,'sha256':hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest()},indent=2))
