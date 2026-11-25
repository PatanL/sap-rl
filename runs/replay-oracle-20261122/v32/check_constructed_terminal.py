#!/usr/bin/env python3
"""Terminal-board check of the constructed client tests for two trace files.

Usage: check_constructed_terminal.py <v31 constructed trace.jsonl> <v32 constructed trace.jsonl>
For every session 3 capture: is the real terminal reading (attack/health of every pet, front to
back, both sides; "?" matches anything) the end board of some engine seed, and of every seed.
Weaker than check_constructed.py (which follows the whole movie) but not affected by the client's
stale labels in the middle of a fight.
"""
import json,sys,collections
HERE='/home/dzliu/pzliu/sap-rl/runs/replay-oracle-20261122/v32/'
doc=json.load(open(HERE+'constructed-fixtures.json'))
find=json.load(open('/home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/session3/findings.json'))
def parse(side):
    if side=='empty': return []
    out=[]
    for tok in side.split(' | '):
        a,h=tok.split()[1].split('/'); out.append((None if a=='?' else int(a), None if h=='?' else int(h)))
    return out
def eq(real,eng):
    return len(real)==len(eng) and all((a is None or a==ea) and (h is None or h==eh) for (a,h),(ea,eh) in zip(real,eng))
res={}
for tag,fn in (('v31',sys.argv[1]),('v32',sys.argv[2])):
    by=collections.defaultdict(list)
    for line in open(fn):
        r=json.loads(line); fin=r['trace'][-1]
        by[r['battleId']].append([[(s[1],s[2]) for s in side if s] for side in fin])
    for c in doc['captures']:
        rp,ro=parse(c['terminal']['P']),parse(c['terminal']['O'])
        n=sum(eq(rp,e[0]) and eq(ro,e[1]) for e in by[c['fixture']])
        res.setdefault(c['capture'],{})[tag]=(n,len(by[c['fixture']]))
for tag in ('v31','v32'):
    print(tag,'captures whose real terminal board is reached by some seed:',sum(v[tag][0]>0 for v in res.values()),'of',len(res),'; by every seed:',sum(v[tag][0]==v[tag][1] for v in res.values()))
for rule in find['rules']:
    caps=[c for c in rule['supporting_captures'] if c in res]
    if caps: print('  %-4s captures %3d  terminal reached by some seed: v31 %3d  v32 %3d ; by every seed: v31 %3d  v32 %3d'%(rule['id'],len(caps),sum(res[c]['v31'][0]>0 for c in caps),sum(res[c]['v32'][0]>0 for c in caps),sum(res[c]['v31'][0]==res[c]['v31'][1] for c in caps),sum(res[c]['v32'][0]==res[c]['v32'][1] for c in caps)))
print('terminal not reached under v32:',[c for c,v in res.items() if v['v32'][0]==0])
print('lost (reached under v31, not under v32):',[c for c,v in res.items() if v['v31'][0]>0 and v['v32'][0]==0])
