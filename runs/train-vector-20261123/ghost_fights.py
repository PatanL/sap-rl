"""Outcome-only oracle adapter for the original ghost program, using the pinned v31g ABI."""
import argparse,ctypes as C,json,multiprocessing as mp
from functools import lru_cache
from pathlib import Path
PIN=Path('/sap/engine-rs-pinned/v31g')
def init():
    global lib
    lib=C.CDLL(str(PIN/'libsap_engine.so'))
    lib.sap_battle.argtypes=[C.POINTER(C.c_int32),C.c_int,C.POINTER(C.c_int32),C.c_int,C.c_uint32]
    assert lib.sap_engine_init(str(PIN/'data').encode())==0
@lru_cache(maxsize=32768)
def encode(specs):
    cells=[]
    for pet,a,h,xp,status,count,copied in specs:
        kind=lib.sap_kind(pet.encode());perk=lib.sap_status(status.encode())
        c=lib.sap_kind(copied.encode()) if isinstance(copied,str) else copied
        if min(kind,perk)<0 or isinstance(copied,str) and c<0:raise ValueError(specs)
        cells.extend((kind,a,h,xp,perk,count,c))
    return (C.c_int32*len(cells))(*cells)
def pack(team):
    return tuple((p['id'],p['attack'],p['health'],p['experience'],p.get('status','none'),
        p.get('triggersConsumed',0),p.get('parrotCopies',-2) if p.get('parrotCopies',-2) is not None else -1) for p in team)
def score(pair):
    lhs,rhs=pair
    a,b=encode(lhs),encode(rhs);counts=[0,0,0]
    for seed in SEEDS:
        v=lib.sap_battle(a,len(lhs),b,len(rhs),seed)
        if v<0:return {'skipped':'engine error'}
        counts[v]+=1
    return {'simOutcomes':dict(zip(('player','opponent','draw'),counts))}
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['fixtures','out','data','samples','threads']:p.add_argument('--'+k,required=True)
    a=p.parse_args();SEEDS=[(1234567+i*2654435761)%2**32 for i in range(int(a.samples))]
    fixtures=json.loads(Path(a.fixtures).read_text())['fixtures']
    pairs=[(pack(f['player']),pack(f['opponent'])) for f in fixtures];del fixtures
    with mp.get_context('fork').Pool(int(a.threads),initializer=init) as pool:rows=list(pool.imap(score,pairs,chunksize=256))
    Path(a.out).write_text(json.dumps({'rows':rows}))
