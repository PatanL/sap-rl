from pathlib import Path
from hashlib import sha256
import json
P=Path(__file__).resolve().parent;R=P.parent
def digest(path):return sha256(path.read_bytes()).hexdigest()
def manifest(root):
 paths={str(f.relative_to(root)):digest(f) for f in sorted((root/'sapai').rglob('*.py'))}
 return dict(files=paths,sha256=sha256(json.dumps(paths,sort_keys=True).encode()).hexdigest())
v30=R/'search-throughput-5x-20260912-v1/reference-source/sapai';live=R/'replay-oracle-20261122/engine-v31'
old=manifest(v30);pinned=manifest(P/'pinned-v31');refresh=manifest(P/'pinned-v31-refresh');latest=manifest(live)
files=sorted((R/'replay-oracle-20261122/raw').glob('*.json'))+sorted((R/'expert-replay-ingest-20260926-v1/corpus/matches').glob('*/BUNDLE.json'))
sources={str(f.relative_to(R)):digest(f) for f in files}
changed=[f for f,h in pinned['files'].items() if old['files'].get(f)!=h]
info=dict(v31=pinned,v31_refresh=refresh,v30=old,live_sha256_at_report=latest['sha256'],refresh_matches_live=refresh['sha256']==latest['sha256'],v30_root=str(v30),v31_source=str(live),source_still_matches_pin=latest['sha256']==pinned['sha256'],
          differing_files=changed,input_files=sources,corpus_hash=sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),seats_jsonl=digest(P/'seats.jsonl'))
(P/'provenance.json').write_text(json.dumps(info,indent=2))
print(pinned['sha256'],old['sha256'],info['source_still_matches_pin'],changed)
