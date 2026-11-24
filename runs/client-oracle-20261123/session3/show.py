"""usage: show.py <capture-name> [first_frame last_frame]  -> fixture summary, paired summary, timeline slice"""
import json,sys
sys.path.insert(0,'/home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123')
import engine3
name=sys.argv[1];d=engine3.D/'captures'/name;run=json.load(open(d/'run.json'));b=json.loads((engine3.D/'fixtures'/run['fixture']).read_text())
f=engine3.from_battle(b)
for side in ('player','opponent'):
    print(side,'; '.join(f"{p['name']}{'' if p['level']==1 else ' L'+str(p['level'])} {p['attack']}/{p['health']}"+(f" [{p['perk']}]" if p['perk'] else '')+(f" (parrot:{p.get('parrotCopies')})" if p['name']=='Parrot' else '')+(f" tc{p['triggersConsumed']}" if p['triggersConsumed'] else '') for p in f[side]))
for k in ('UserBoard','OpponentBoard'):
    xs=[(p.get('Poi') or {}).get('x',i) for i,p in enumerate(b[k]['Mins']['Items']) if p]
    if sorted(xs)!=list(range(5-len(xs),5)):print('  GAPS',k,'occupied x =',sorted(xs))
if (d/'pair3.json').exists():
    r=json.load(open(d/'pair3.json'));print({k:r[k] for k in ('engine_distinct','matched_stages','total_stages','first_difference','terminal_matches_archive','best_variants')})
    for e in r.get('engine_at_difference',[])[:3]:print('  ENG',e)
lo=int(sys.argv[2]) if len(sys.argv)>2 else 0;hi=int(sys.argv[3]) if len(sys.argv)>3 else 10**6
for l in open(d/'timeline.txt'):
    fr=int(l[2:6])
    if lo<=fr<=hi and 'P:   ||  O: \n' not in l:print(l.rstrip()[:230])
