"""Session-3 traffic accounting from session2/network.jsonl lines after the recorded offset."""
import json,collections
from pathlib import Path
H=Path(__file__).resolve().parent;D=H.parent/'session2'
off=int((H/'network-offset.txt').read_text().split()[0])
rows=[json.loads(l) for l in (D/'network.jsonl').read_text().splitlines()[off:]]
def op(r):return (r.get('op') or '').split('/api/')[-1]
c=collections.Counter()
for r in rows:
    a=r['action'];o=op(r);m=r.get('method');mode=r.get('mode')
    if a=='local':c['battle_get_fulfilled_locally']+=1
    if a=='response' and o=='arena/queue' and r['status']==200:c['arena_queue_200 (games started)']+=1
    if a=='response' and o=='build/end' and r['status']==200:c['build_end_200 (turns ended)']+=1
    if a=='response' and o=='build/play-minion' and r['status']==200:c['build_play_minion_200 (pets bought)']+=1
    if a=='allow' and m=='POST' and o=='arena/watch':c[f'arena_watch_POST_forwarded_mode_{mode}']+=1
    if a=='block' and m=='POST' and o=='arena/watch':c[f'arena_watch_POST_blocked_mode_{mode}']+=1
    if a=='allow' and m=='GET' and o.startswith('battle/get') :c[f'battle_get_forwarded_mode_{mode}']+=1
    if a=='block':c['blocked_total']+=1
    if a=='allow' and m=='POST':c[f'POST_forwarded {o} [{mode}]']+=1
    if a=='block' and r.get('host')=='api.teamwood.games':c[f'blocked {m} {o.split("/")[0]+"/"+o.split("/")[1] if "/" in o else o} [{mode}]']+=1
    if a=='block' and r.get('host')!='api.teamwood.games':c[f'blocked non-game host {r.get("host")}']+=1
out=dict(lines_from=off,lines=len(rows),counts=dict(sorted(c.items())))
(H/'ACCOUNTING.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
