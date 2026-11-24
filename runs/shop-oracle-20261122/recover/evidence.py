"""Constraints inspectable without simulating the unknown pet's effects on the endpoint."""
import collections as C,itertools as I,math
from order_evidence import clues,TIER

def pair_ok(x,y):
    return x=='Sloth' or y=='Sloth' or TIER[x]>=TIER[y]
def ordered(p,names):
    edges,ordinary=p['clues']
    for i in ordinary.intersection(p['b']):
        if TIER[names[i]]>p['end']['tier']:return False
    for i,j in edges:
        if i not in p['b'] and j not in p['b']:continue
        x=names.get(i);y=names.get(j)
        if x in TIER and y in TIER and not pair_ok(x,y):return False
    return True
def input_domains(p,use_receipts=True,use_wallet=True,use_order=True):
    """Enumerate sufficient category signatures, then expand their members into marginal sets."""
    cats=[];b=p['b'];w=p['wallet'] if use_wallet else None
    edges,ordinary=p['clues'];rel={i for edge in edges for i in edge if set(edge)&set(b)}
    clause=p['clauses'] if use_receipts else []
    tags={name for name,ids in clause}
    for i,ns in zip(b,p['domain']):
        buckets=C.defaultdict(list)
        for name in ns:
            # The only uses of a bucket member in the raw constraints are tier, exceptional Sloth, Pig pay or stock source.
            if use_order and i in ordinary and TIER[name]>p['end']['tier']:continue
            tag=(name if name in tags or (w and name=='Pig') or (use_order and i in rel and name=='Sloth') else None,
                 TIER[name] if use_order and i in rel else None)
            buckets[tag].append(name)
        cats.append(list(buckets.values()))
    possible=[set() for _ in b];count=0
    for groups in I.product(*cats):
        names={**p['names'],**{i:g[0] for i,g in zip(b,groups)}}
        if use_order and not ordered(p,names):continue
        if w and w[0]+sum(c for c,i in zip(w[1],b) if names[i]=='Pig')!=w[2]:continue
        if not all(any(names.get(i)==name for i in ids) for name,ids in clause):continue
        count+=math.prod(map(len,groups))
        for dest,g in zip(possible,groups):dest.update(g)
    return count,possible

def summon_certificate(p,marginals):
    """Only certify a name when a logged Pill and published summon/perk rules leave a single species.
    No simulated board, resulting stats, native RNG, ID assignment order, or candidate label is used.
    """
    if not p['e']:return {}
    names={**p['names'],**{i:next(iter(g)) for i,g in zip(p['b'],marginals) if len(g)==1}}
    board={v['id']:[v['name'],v['perk']] for v in p['before']['board'] if v}
    frozen={v['id']:v for v in p['before']['shop'] if v['f']}
    allowed=set(); safe=True
    perks={'Honey','Mushroom','Melon','Garlic','Steak','Chili','Meat Bone','Cake','Bread'}
    native={'Sheep':{'Ram'},'Deer':{'Bus'},'Cricket':{'Zombie Cricket'},'Rooster':{'Chick'},'Mammoth':set(),'Ant':set(),'Flamingo':set(),
            'Spider':{x for x,t in TIER.items() if t==3}}
    for a,(op,dst) in zip(p['acts'],p['tr']):
        t=a['t'];i=a.get('id');j=a.get('target')
        if t==6 and op=='buy':board[i]=[names.get(i),frozen.get(i,{}).get('perk')]
        if (t==7 or op=='buy merge'):
            dst=j if t==7 else dst
            src=board.pop(i,None) if t==7 else None
            if (src and src[1]) or frozen.get(i,{}).get('perk'):
                # No assumed winner for competing perks.
                if dst in board:board[dst][1]='?'
        if t==9:board.pop(i,None)
        if t!=8:continue
        food=a['spell']['name']
        if j in board and food in perks:board[j][1]=food
        if food!='Sleeping Pill':continue
        if any(v[0] in ('Fly',None) for k,v in board.items() if k!=j):safe=False
        value=board.pop(j,None)
        if value is None and len(allowed)==1:value=[next(iter(allowed)),None]
        if value is None or value[0] not in native or value[1]=='?':safe=False;continue
        nm,perk=value
        allowed.update(native[nm])
        if perk=='Honey':allowed.add('Bee')
        if perk=='Mushroom':allowed.add(nm)
    if safe and len(allowed)==1:return dict.fromkeys(p['e'],next(iter(allowed)))
    return {}
