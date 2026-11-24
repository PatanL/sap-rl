import collections,hashlib,json,os,pickle,random,sys
from pathlib import Path
import numpy as np
from plan_common import install
install(cuda=False)
sys.path.insert(0,'/scorer')
from native_scorer import encode,pet_kind,score_grid
from sapai.shop import get_shop_rules
from sapai import Pet
from sapai.data import data
pool=get_shop_rules(11,pack='StandardPack')[4]
omitted={};quiet=[];special=[]
for name in sorted(pool):
    entries=[]
    for level in (1,2,3):
        p=Pet(name);p.level=level
        if pet_kind(name,level)<0:entries.append([level,p.ability.get('trigger'),p.ability.get('description')])
    if entries:omitted[name]=entries
    elif pet_kind(name,1)==1:quiet.append(name)
    else:special.append(name)
print('omitted',omitted)
# Trusted, previously captured planner-boundary corpus; no game/network execution.
path=Path('/performance/calls.pkl')
calls=pickle.loads(path.read_bytes())
by_reason=collections.Counter(); dynamic=collections.Counter()
for states,opps,k,rngstate in calls:
    rng=random.Random();rng.setstate(rngstate)
    seeds=np.array([rng.getrandbits(31) for _ in range(k)],np.uint32)
    a,b=np.stack(list(map(encode,states))),np.stack(list(map(encode,opps)))
    result=score_grid(a,b,seeds)
    for i,j in np.ndindex(result.shape[:2]):
        names=tuple(sorted(set(p['pet']['name'] for src,enc in [(states[i],a[i]),(opps[j],b[j])] for p,slot in zip(src['team'],enc) if slot[0]<0)))
        for v in result[i,j]:
            if v>=0:by_reason['native']+=1
            elif names:by_reason['rejected on input: '+','.join(names)]+=1
            else:
                spider=any(s['pet']['name']=='pet-spider' for side in (states[i],opps[j]) for s in side['team'])
                by_reason['during fight, Spider present' if spider else 'during fight, no Spider']+=1
out=dict(rollable=len(pool),quiet=quiet,special=special,omitted=omitted,corpus=str(path),corpusSHA=hashlib.sha256(path.read_bytes()).hexdigest(),counts=by_reason)
Path('fallback.json').write_text(json.dumps(out,indent=2))
print(by_reason)
