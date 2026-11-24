"""Session 3 pairing of a real-client capture with the pinned current engine.

The real side is the collapsed movie timeline (timeline.json).  The engine side is a set of seeds
(the two random generators are not equivalent, so one same-seed trace is not a valid pair for a
fight with random effects).  Every engine seed's round-by-round boards are searched, in order, in the
real readings; the seed(s) that follow the real fight longest are kept, and the first engine round
that cannot be found in the real readings is the first difference.

Writes <capture>/paired.md (the report's table format), <capture>/pair3.json.
Usage: pair3.py <capture dir> [n_seeds]
"""
import json,sys,collections
from pathlib import Path
import engine3
D=engine3.D
def parse(tok):
    a,h=tok.split('/');return (None if a=='?' else int(a),None if h=='?' else int(h))
def real_board(row):
    out={}
    for side in ('P','O'):
        out[side]=[parse(t[2]) for t in row[side]]
    return out
def alive(b):return all(h is None or h>0 for side in ('P','O') for a,h in b[side])
def eq(real,eng,lag=False):
    """lag=True tolerates a stale label on a pet: the client shows only the first of two quick hits (or buffs)
    until that pet's next change, so the real label may lag the engine by a hit (health higher, attack equal)
    or by a buff (attack and health both lower or equal)."""
    if len(real)!=len(eng):return False
    for (a,h),(ea,eh) in zip(real,eng):
        ea,eh=min(ea,50),min(eh,50)
        if not lag:
            if a is not None and a!=ea:return False
            if h is not None and h!=eh:return False
        else:
            if a is None or h is None:continue
            if (a,h)==(ea,eh):continue
            if a==ea and h>eh:continue          # a hit not yet on the label
            if a<=ea and h<=eh and h>0:continue   # a buff not yet on the label
            return False
    return True
def eboard(step):return dict(P=[(p['attack'],p['health']) for p in step['player']],O=[(p['attack'],p['health']) for p in step['opponent']])
def follow(real,steps,lag=False):
    """Greedy in-order search. Returns list of real indices (None from first failure on)."""
    pos=0;hits=[]
    for st in steps:
        eb=eboard(st);found=None
        for i in range(pos,len(real)):
            rb=real[i]
            if eq(rb['P'],eb['P'],lag) and eq(rb['O'],eb['O'],lag) and (not lag or alive(rb)):found=i;break
        if found is None:break
        hits.append(found);pos=found  # the same reading may serve consecutive identical stages
    return hits
def fmt_side(side,eng=None):
    if not side:return 'empty'
    out=[]
    for i,(a,h) in enumerate(side):
        name='?'
        if eng and i<len(eng) and (a,h)==(min(eng[i]['attack'],50),min(eng[i]['health'],50)):name=eng[i]['pet']
        out.append(f"{name} {'?' if a is None else a}/{'?' if h is None else h}")
    return ', '.join(out)
def fmt_eng(side):return ', '.join(f"{p['pet']} {p['attack']}/{p['health']}"+(f" [{p['perk'][7:]}]" if p.get('perk') and p['perk']!='none' else '') for p in side) or 'empty'
def survivors(b):
    m=engine3.mods();pets,_=m['tables'];sv=m['ex'].survivors(b.get('ResultingBoard'),pets)
    return sv
def analyse(d,n=300):
    d=Path(d);run=json.loads((d/'run.json').read_text());b=json.loads((D/'fixtures'/run['fixture']).read_text())
    if isinstance(b,str):b=json.loads(b)
    rows=json.loads((d/'timeline.json').read_text());real=[real_board(r) for r in rows]
    f=engine3.from_battle(b)
    seeds=[f['seed']%2**32]+[(1234567+k*2654435761)%2**32 for k in range(n-1)]
    best=[];bestlen=-1;distinct=collections.Counter();errors=0
    cache={}
    for sd in seeds:
        try:steps=engine3.trace(f,sd)[1:]   # drop init
        except Exception as e:errors+=1;continue
        key=tuple(engine3.line(s) for s in steps);distinct[key]+=1
        if key in cache:continue
        hits=follow(real,steps);cache[key]=(steps,hits)
        full=len(hits)==len(steps)
        score=len(hits)+(1000 if full else 0)
        if score>bestlen:bestlen=score;best=[key]
        elif score==bestlen:best.append(key)
    steps,hits=cache[best[0]];full=len(hits)==len(steps)
    # second pass: tolerate stale labels (see eq)
    lagbest=None;lagscore=-1
    for key,(st2,_) in cache.items():
        h2=follow(real,st2,True);sc=len(h2)+(1000 if len(h2)==len(st2) else 0)
        if sc>lagscore:lagscore=sc;lagbest=(st2,h2)
    lsteps,lhits=lagbest;lagfull=len(lhits)==len(lsteps)
    # terminal check against the archive
    sv=survivors(b);last=real[-1];arch_side={1:'P',2:'O',3:None}.get(b.get('Outcome'))
    # ResultingBoard slots 0-4 player (4 = front), 5-9 opponent
    arch=[(s['attack'],s['health'],s['slot']) for s in sv]
    archP=[(a,h) for a,h,s in sorted([x for x in arch if x[2]<5],key=lambda x:-x[2])];archO=[(a,h) for a,h,s in sorted([x for x in arch if x[2]>=5],key=lambda x:x[2])]
    def same(realside,archside):
        return eq(realside,archside) or eq(realside,archside[::-1])
    terminal_ok=same(last['P'],archP) and same(last['O'],archO)
    res=dict(id=b['Id'],seed=b['Seed'],engine_seeds=len(seeds),engine_errors=errors,engine_distinct=len(distinct),
             matched_stages=len(hits),total_stages=len(steps),engine_reproduces_all_stages=full,
             first_difference=None if full else steps[len(hits)]['stage'],
             lag_tolerant_reproduces=lagfull,lag_tolerant_first_difference=None if lagfull else lsteps[len(lhits)]['stage'],
             lag_tolerant_matched=len(lhits),
             best_variants=len(best),terminal_matches_archive=terminal_ok,
             real_terminal=dict(P=last['P'],O=last['O']),archive_terminal=dict(P=archP,O=archO,outcome=b.get('Outcome')))
    text=['# '+b['Id'],'',f"Recorded seed {b['Seed']}. Both sides are shown front to back. The real columns are machine readings of the client's canvas movie (species are filled in only where attack/health agree with the engine pet in the same place; `?` otherwise). "
          f"Engine columns: the pinned current engine-v31, {len(seeds)} seeds, {len(distinct)} distinct round sequences; shown is the sequence that follows the real fight longest ({len(best)} such). "
          "A real stage is the first movie reading, in order, that equals the engine board for that round.",'',
          '| Stage and real frame | Real player | Engine player | Real opponent | Engine opponent |','|---|---|---|---|---|']
    for i,st in enumerate(steps):
        if i<len(hits):
            r=rows[hits[i]];rb=real[hits[i]];fr=r['frame'] or f"movie frame {r['lastf']}"
            link=f"[{st['stage']}]({r['frame']})" if r['frame'] else f"{st['stage']} (movie frame {r['lastf']}, t={r['t']})"
            text.append(f"| {link} | {fmt_side(rb['P'],st['player'])} | {fmt_eng(st['player'])} | {fmt_side(rb['O'],st['opponent'])} | {fmt_eng(st['opponent'])} |")
        else:
            text.append(f"| {st['stage']} (not found in the real movie) | — | {fmt_eng(st['player'])} | — | {fmt_eng(st['opponent'])} |")
            if i==len(hits):
                lo=hits[-1] if hits else 0
                window=[r for r in rows[lo:lo+40]]
                res['window']=[dict(f=r['f'],t=r['t'],n=r['n'],dmg=r['dmg'],P=[t[2] if t[0]>=0 else '~'+t[2] for t in r['P']],O=[t[2] if t[0]>=0 else '~'+t[2] for t in r['O']],frame=r['frame']) for r in window]
            break
    text.append('')
    if full:text.append('The engine reproduces every round of the real movie with at least one seed.')
    else:
        text.append(f"**First difference: {res['first_difference']}.** No engine seed reaches the real board at this round. The real readings after the last agreed round are listed in `timeline.txt` from movie frame {rows[hits[-1]]['f'] if hits else 0}.")
        alts=[cache[k][0][len(hits)] for k in best[:4]]
        res['engine_at_difference']=[dict(P=fmt_eng(a['player']),O=fmt_eng(a['opponent'])) for a in alts]
    if not full:
        if lagfull:text.append('Allowing for the client\'s label lag (a second quick hit or buff reaches the label only at the pet\'s next change), an engine seed does follow every round; the strict difference above may be a display artefact and needs the frames.')
        else:
            text.append(f"Allowing for label lag, the first round no engine seed reaches is **{res['lag_tolerant_first_difference']}**.")
            res['lag_engine_at_difference']=dict(P=fmt_eng(lsteps[len(lhits)]['player']),O=fmt_eng(lsteps[len(lhits)]['opponent']))
            lo=lhits[-1] if lhits else 0
            res['lag_window']=[dict(f=r['f'],t=r['t'],n=r['n'],dmg=r['dmg'],P=[t[2] if t[0]>=0 else '~'+t[2] for t in r['P']],O=[t[2] if t[0]>=0 else '~'+t[2] for t in r['O']],frame=r['frame']) for r in rows[lo:lo+45]]
            if lhits:res['lag_last_agreed']=dict(stage=lsteps[len(lhits)-1]['stage'],P=fmt_eng(lsteps[len(lhits)-1]['player']),O=fmt_eng(lsteps[len(lhits)-1]['opponent']),f=rows[lhits[-1]]['f'])
    text.append('')
    text.append(f"Client terminal board {'agrees' if terminal_ok else '**DISAGREES**'} with the archived ResultingBoard (real P {fmt_side(last['P'])} / O {fmt_side(last['O'])}; archive P {archP} / O {archO}).")
    text.extend(['','[Full movie timeline](timeline.txt) · [Machine summary](pair3.json) · [Capture receipt](run.json) · [Canvas movie](battle.webm)'])
    (d/'paired.md').write_text('\n'.join(text)+'\n');(d/'pair3.json').write_text(json.dumps(res,indent=1))
    return res
if __name__=='__main__':
    n=int(sys.argv[2]) if len(sys.argv)>2 else 300
    r=analyse(sys.argv[1],n);print(json.dumps({k:v for k,v in r.items() if k not in ('window',)},indent=1))
