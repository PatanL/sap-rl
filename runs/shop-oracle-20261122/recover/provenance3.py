from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
R=P.parent.parent/'replay-oracle-20261122'
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def m(o):return hashlib.sha256(json.dumps(o,sort_keys=True).encode()).hexdigest()
pin=json.load(open(P.parent/'engine-pin-recover.json'))
files={p.relative_to(R/'engine-v31').as_posix():h(p) for p in sorted((R/'engine-v31/sapai').rglob('*.py'))}
fresh={p.name:h(p) for p in sorted((R/'raw-fresh').glob('*.json'))}
out=dict(pinned=pin['sha256'],live_at_verification=m(files),pin_matches_live=m(files)==pin['sha256'],patch_file_sha256=h(R/'engine-v31.patch'),
     patch_hunks=sum(x.startswith('@@') for x in (R/'engine-v31.patch').read_text().splitlines()),
     original_seats=h(P.parent/'seats.jsonl'),fresh_seats=h(P/'fresh/seats.jsonl'),fresh_file_hashes=fresh,fresh_corpus_sha256=m(fresh))
(P/'provenance.json').write_text(json.dumps(out,indent=2))
print(out['patch_hunks'],out['pin_matches_live'],out['fresh_corpus_sha256'])
