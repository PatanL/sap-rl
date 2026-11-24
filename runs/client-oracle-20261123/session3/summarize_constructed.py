"""Machine summary of every constructed capture: settled boards (start, after each exchange, terminal).
Writes session3/constructed-readings.json and session3/constructed-readings.md."""
import json,glob,collections
from pathlib import Path
H=Path(__file__).resolve().parent;D=H.parent/'session2'
man={}
for m in sorted(D.glob('c3*-manifest.json')):
    for j in json.loads(m.read_text()):man[j['file'][:-5]]=j
eng={}
for p in (H/'engine-constructed').glob('*.json'):eng.update(json.loads(p.read_text()))
def side(ts):return ' | '.join((f"s{t[0]} " if t[0]>=0 else '~')+t[2] for t in ts) or 'empty'
out=[];md=['| capture | seed | start (before first exchange) | terminal | frames |','|---|---:|---|---|---|']
for name,j in sorted(man.items()):
    d=D/'captures'/name
    if not (d/'timeline.json').exists():continue
    rows=json.loads((d/'timeline.json').read_text())
    settled=[r for r in rows if r['n']>=2 and (r['P'] or r['O']) and all('?' not in t[2] for s in 'PO' for t in r[s])]
    firstd=next((i for i,r in enumerate(rows) if r['dmg']),None)
    start=None
    if firstd is not None:
        c=[r for r in rows[:firstd] if r['n']>=2 and r['P'] and r['O']]
        start=c[-1] if c else None
    last=rows[-1]
    for r in reversed(rows):
        if all('?' not in t[2] for s in 'PO' for t in r[s]) and [t[2].split('/')[0] for t in r['P']]==[t[2].split('/')[0] for t in last['P']] and len(r['O'])==len(last['O']):last=r;break
    rec=dict(capture=name,test=j['test'],seed=j['seed'],left=j['left'],right=j['right'],
             start=None if start is None else dict(P=side(start['P']),O=side(start['O']),frame=start['frame']),
             terminal=dict(P=side(last['P']),O=side(last['O']),frame=last['frame']),
             engine=[dict(count=e['count'],steps=e['steps'][:3]) for e in eng.get(j['test'],[])[:4]])
    out.append(rec)
    md.append(f"| {name} | {j['seed']} | {'—' if start is None else 'P: '+rec['start']['P']+' / O: '+rec['start']['O']} | P: {rec['terminal']['P']} / O: {rec['terminal']['O']} | [timeline](../session2/captures/{name}/timeline.txt) |")
(H/'constructed-readings.json').write_text(json.dumps(out,indent=1));(H/'constructed-readings.md').write_text('\n'.join(md)+'\n');print(len(out))
