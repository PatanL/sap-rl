import json
from pathlib import Path
P=Path(__file__).resolve().parent
caps=(1,64,1024,8192,65536)
def load_rows(variant='v31'):
    merged={};curve=[]
    for k in caps:
        path=P/f'strict-{variant}-{k}.jsonl'
        if not path.exists():break
        hits=0
        for l in path.read_text().splitlines():
            if not l:continue
            row=json.loads(l);idx=tuple(row['key'])
            if idx in merged:assert not merged[idx].get('reproduced') and not merged[idx].get('skip')
            merged[idx]=row;hits+=bool(row.get('reproduced'))
        curve.append((k,hits,sum(bool(v.get('reproduced')) for v in merged.values())))
    return merged,curve
def size(row):return row['actions'],row['boardSize'],row['key']
