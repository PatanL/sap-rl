"""Session 3 engine side.  Private immutable copy of the *current* engine-v31 (the session-2 pin predates
later engine edits), battle-JSON -> engine team conversion through the replay oracle's own converter,
round-by-round traces and multi-seed sweeps.

  engine3.py constructed <batch>        engine outcome distribution for every test of a constructed batch
  engine3.py trace <fixture.json> [seed]  one round-by-round trace for a battle JSON
"""
import sys,json,hashlib,collections,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE/'session2';S3=HERE/'session3';O=HERE.parent/'replay-oracle-20261122'
CALC=HERE.parent.parent/'SAP-Calculator'
def pin():
    root=S3/'engine-pin'
    if (root/'MANIFEST.json').exists():return root
    real=O/'engine-v31';files={}
    for p in sorted((real/'sapai').rglob('*.py')):
        b=p.read_bytes();rel=p.relative_to(real);q=root/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);files[str(rel)]=hashlib.sha256(b).hexdigest()
    (root/'MANIFEST.json').write_text(json.dumps(dict(source=str(real),files=files,hash=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()),indent=1));return root
_mods={}
def mods():
    if _mods:return _mods
    root=pin();sys.path.insert(0,str(root))
    old=sys.argv[:];sys.argv=['run_sapai.py',str(root),'-','-','1','1']
    try:
        spec=importlib.util.spec_from_file_location('snap_ref3',O/'run_sapai.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    finally:sys.argv=old
    spec=importlib.util.spec_from_file_location('extract3',O/'extract_battles.py');ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
    import sapai
    _mods.update(oracle=oracle,ex=ex,Battle=sapai.Battle,tables=ex.load_tables(CALC));return _mods
def from_battle(b):
    m=mods();pets,perks=m['tables'];problems=[]
    f=dict(battleId=b['Id'],seed=b['Seed'],player=m['ex'].convert_board(b['UserBoard'],pets,perks,problems),opponent=m['ex'].convert_board(b['OpponentBoard'],pets,perks,problems),problems=problems)
    return f
def trace(f,seed=None):
    m=mods();battle=m['Battle'](m['oracle'].make_team(f['player']),m['oracle'].make_team(f['opponent']),seed=(f['seed'] if seed is None else seed)%2**32)
    def board(team):return [dict(pet=s.pet.name[4:].replace('-',' ').title(),attack=int(s.pet.attack),health=int(s.pet.health),perk=s.pet.status,slot=i) for i,s in enumerate(team) if not s.empty]
    def state(stage):return dict(stage=stage,player=board(battle.t0),opponent=board(battle.t1))
    rows=[state('init')];battle.start();rows.append(state('start'));i=0
    while any(not s.empty and s.pet.health>0 for s in battle.t0) and any(not s.empty and s.pet.health>0 for s in battle.t1):
        battle.pet_priority=battle.update_pet_priority(battle.t0,battle.t1,battle.rs)
        more=battle.attack(i);i+=1;rows.append(state(f'after-attack-{i:02d}'))
        if not more:break
    rows.append(state('final'));return rows
def short(side):return ', '.join(f"{p['pet']} {p['attack']}/{p['health']}" for p in side) or 'empty'
def line(r):return f"{r['stage']}: P[{short(r['player'])}] O[{short(r['opponent'])}]"
def sweep(f,n=200):
    seqs=collections.Counter()
    for k in range(n):
        try:rows=trace(f,(1234567+k*2654435761)%2**32)
        except Exception as e:seqs[('ENGINE ERROR '+repr(e)[:160],)]+=1;continue
        seqs[tuple(line(r) for r in rows[1:])]+=1
    return seqs
def constructed(batch,n=200):
    man=json.loads((D/(batch+'-manifest.json')).read_text());out={};seen=set()
    for j in man:
        if j['test'] in seen:continue
        seen.add(j['test']);b=json.loads((D/'fixtures'/j['file']).read_text());f=from_battle(b);sq=sweep(f,n)
        out[j['test']]=[dict(count=c,steps=list(s)) for s,c in sq.most_common()]
        print('==',j['test'],f.get('problems') or '')
        for s,c in sq.most_common(6):
            print(f'  {c}/{n}:');[print('     ',x) for x in s]
    (S3/'engine-constructed').mkdir(exist_ok=True);(S3/'engine-constructed'/(batch+'.json')).write_text(json.dumps(out,indent=1))
if __name__=='__main__':
    if sys.argv[1]=='constructed':constructed(sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 200)
    elif sys.argv[1]=='trace':
        b=json.loads(Path(sys.argv[2]).read_text())
        if isinstance(b,str):b=json.loads(b)
        f=from_battle(b)
        for r in trace(f,int(sys.argv[3]) if len(sys.argv)>3 else None):print(line(r))
