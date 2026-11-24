"""Copy two independently pinned imports and assign a distinct, content-bound v31 revision."""
import argparse,hashlib,json,shutil,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest(root):
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
def build(source,gym,dest):
    if dest.exists():raise FileExistsError(dest)
    pkg=source/'sapai'
    original={'sapai/'+k:v for k,v in manifest(pkg).items()}
    gym_hashes=manifest(gym)
    revision=f'turtle-v31-replay-{digest(original)[:12]}-20261123'
    if digest(gym_hashes)!='3291838db6765842d63dfc4cf8966e713cffbd40230c26ebd07014736b96ba86':
        revision+='-gym-'+digest(gym_hashes)[:12]
    tmp=Path(tempfile.mkdtemp(prefix=dest.name+'.building-',dir=dest.parent))
    try:
        shutil.copytree(pkg,tmp/'sapai/sapai',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(gym,tmp/'sapai-gym',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        assert manifest(tmp/'sapai')==original and manifest(tmp/'sapai-gym')==gym_hashes
        rev=tmp/'sapai/sapai/turtle_pack.py'
        s=rev.read_text();old='SIMULATOR_REVISION = "turtle-v30-food-right-reorder10-v1-20260911"'
        if s.count(old)!=1:raise ValueError('Unexpected upstream revision: inspect source before packaging')
        addition='SOURCE_OF_TRUTH["replay_engine_sha256"] = '+repr(digest(original))+'\nSIMULATOR_REVISION = '+repr(revision)
        rev.write_text(s.replace(old,addition))
        files=manifest(tmp)
        if any(Path(k).suffix in ('.so','.dll','.dylib') for k in files):raise ValueError('Unexpected compiled engine/scorer')
        doc=dict(simulator_revision=revision,upstream_engine_sha256=digest(original),upstream_gym_sha256=digest(gym_hashes),files=files,
                 payload_sha256=digest(files),engine_source=str(source.resolve()),gym_source=str(gym.resolve()),battle_backend='sapai.battle.Battle',native_rust_scorer=False)
        (tmp/'MANIFEST.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
        tmp.rename(dest)
    except BaseException:
        if tmp.is_dir():shutil.rmtree(tmp)
        raise
    print(revision,digest(files))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--gym',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();build(a.source,a.gym,a.out)
