"""Rebuild the actual Python END teams from a Rust run and rerun all battle seeds independently."""
from pathlib import Path
import sys,json,copy,multiprocessing as mp
from runtime import Battle
from sapai import Team,Pet
from sapai.data import data
def check(r):
    seed,*boards,expected=r;teams=[]
    for b in boards:
        t=Team()
        for pos,n,a,h,ta,th,xp,status,count,borrowed in b:
            p=Pet(n);p.set_total_experience(xp);p._attack=a;p._health=h
            p._until_end_of_battle_attack_buff=ta;p._until_end_of_battle_health_buff=th;p.status=status;p.ability_counter=count
            if borrowed:
                ab=copy.deepcopy(data['pets'][borrowed][f'level{p.level}Ability']);ab['_copiedFrom']=borrowed;p.set_ability(ab)
            t[pos]=p
        teams.append(t)
    real=Battle(*teams,seed=seed,fast_training=True).battle()
    return real==expected
if __name__=='__main__':
    root=Path(sys.argv[1]);paths=sorted(root.glob('battles-*.jsonl'))
    if not paths:raise ValueError('Missing captured battle inputs')
    rows=[json.loads(l) for p in paths for l in p.open()]
    with mp.Pool(6) as pool:matches=sum(pool.imap_unordered(check,rows,chunksize=32))
    fallbacks=sum(json.loads(p.read_text())['python_fallback'] for p in root.glob('battles-*-counts.json'))
    calls=sum(json.loads(p.read_text())['rust'] for p in root.glob('battles-*-counts.json'))
    assert len(rows)==calls,'Incomplete audit capture'
    (root/'battle-parity.json').write_text(json.dumps(dict(battles=len(rows),matches=matches,python_fallbacks=fallbacks),indent=2))
    if matches!=len(rows):raise SystemExit('Battle parity failure')
