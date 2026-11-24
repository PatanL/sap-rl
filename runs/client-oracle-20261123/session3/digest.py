"""usage: digest.py <id8> [maxlines] -> compact digest of a paired real capture (s3f-<id8>)"""
import json,sys
sys.path.insert(0,'/home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123')
import engine3
for id8 in sys.argv[1].split(','):
    d=engine3.D/'captures'/('s3f-'+id8)
    if not (d/'pair3.json').exists():print(id8,'not paired yet');continue
    b=json.loads((engine3.D/'fixtures'/('s3f-'+id8+'.json')).read_text());f=engine3.from_battle(b);r=json.load(open(d/'pair3.json'))
    print('#####',id8,'seed',b['Seed'],'outcome',b['Outcome'])
    for side in ('player','opponent'):
        print(' ',side[:2],'; '.join(f"{p['name']}{'' if p['level']==1 else ' L'+str(p['level'])} {p['attack']}/{p['health']}"+(f" [{p['perk']}]" if p['perk'] else '')+(f" (={p.get('parrotCopies')})" if p['name']=='Parrot' else '')+(f" tc{p['triggersConsumed']}" if p['triggersConsumed'] else '') for p in f[side]))
    for k in ('UserBoard','OpponentBoard'):
        xs=[(p.get('Poi') or {}).get('x',i) for i,p in enumerate(b[k]['Mins']['Items']) if p]
        if xs and sorted(xs)!=list(range(5-len(xs),5)):print('  GAPS',k,'occupied x =',sorted(xs))
    print('  ',{k:r.get(k) for k in ('engine_distinct','matched_stages','total_stages','first_difference','lag_tolerant_first_difference','lag_tolerant_matched','terminal_matches_archive')})
    if r.get('lag_last_agreed'):print('   LAST AGREED',r['lag_last_agreed']['stage'],'f',r['lag_last_agreed']['f'],'|',r['lag_last_agreed']['P'],' || ',r['lag_last_agreed']['O'])
    if r.get('lag_engine_at_difference'):print('   ENG(lag)',r['lag_engine_at_difference']['P'],' || ',r['lag_engine_at_difference']['O'])
    if not r['terminal_matches_archive']:print('   TERMINAL real',r['real_terminal'],'archive',r['archive_terminal'])
    if not r.get('lag_engine_at_difference'):
        for e in r.get('engine_at_difference',[])[:3]:print('   ENG',e['P'],' || ',e['O'])
    if r.get('lag_tolerant_first_difference') is None:
        tr=engine3.trace(f);print('   ENGFINAL',engine3.line(tr[-1]),'| ARCH',r['archive_terminal'])
    n=int(sys.argv[2]) if len(sys.argv)>2 else 30;prev=None;k=0
    for w in (r.get('lag_window') or r.get('window',[])):
        txt='P '+' | '.join(w['P'])+'  || O '+' | '.join(w['O'])
        if txt.replace('?','')==(prev or '').replace('?',''):continue
        if '?' in txt and not w['dmg'] and w['n']<2:continue
        prev=txt;k+=1
        if k>n:break
        print('   ',str(w['f']).rjust(4),str(w['n']).rjust(2),'DMG' if w['dmg'] else '   ',txt)
