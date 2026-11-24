"""Render the measured tables. Rates use wall time and explicit denominators."""
import json,statistics,sys
from pathlib import Path
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/(n+'.json')).read_text())
S=['small','mid','medium','parent']; B=[1,4,16,64,256,1024]
names=dict(zip(S,['0.390M','1.350M','4.467M','14.347M']))
micro=read('micro')
def actor(k,b,p):
    return statistics.median(x['ms'] for x in read('actor-exact') if (x['model'],x['batch'],x['precision'])==(k,b,p))
def update(k,b,p):
    return next(x['ms'] for x in micro if (x['kind'],x['model'],x['batch'],x['precision'])==('update',k,b,p))
def harmonic(c,l):return 1/(1/c+1/l)
order=['4c-narrow-parent','4c-wide-parent','wide-parent','wide-parent-random','wide-small','wide-mid','wide-medium',
       'pipe-parent','pipe-mid','pipe96-small','pipe96-mid','pipe96-medium','pipe96-parent']
cases={n:read(n) for n in order if (P/(n+'.json')).exists()}
def actual(n):
    if (P/(n+'-update.json')).exists():
        d=read(n+'-update');return [next((v.get('decisions_s') for v in d if v.get('batchedLoss')==b),None) for b in (False,True)]
    return [None,None]
def micro_rate(n):return 1024000/update(cases[n]['model'],1024,'bf16')
def L(n):return actual(n)[0] or micro_rate(n)
def rate(n):return harmonic(cases[n]['accepted_s'],L(n))
def split(r):
    t=r['seconds'];f=r['forward_s'];e=r['critical_engine_s'];o=r['critical_obs_s']
    w=r['coordinator_wait_s'] if 'coordinator_wait_s' in r else r['totals']['coordwait']
    return [100*v/t for v in (e,o,f,w-e-o,t-w-f)]
def gpu_tables():
    rows=[]
    for label,fn in [('Policy forward',actor),('One optimizer step',update)]:
        for prec in ['fp32','bf16']:
            rows.extend([f'### {label}, {prec.upper()}','','| Parameters | '+' | '.join(map(str,B))+' |','|---|'+ '---:|'*6])
            for k in S:rows.append('| '+names[k]+' | '+' | '.join(f'{fn(k,b,prec):.2f} / {1000*b/fn(k,b,prec):,.0f}' for b in B)+' |')
            rows.append('')
    return '\n'.join(rows)
def collect_table():
    rows=['| Network / run | Actors × games / cohorts | Wave / policy batch | Matches/s | Seat decisions/s | Accepted/s | Learner decisions/s | Projected learned/s |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for n,r in cases.items():
        star='*' if actual(n)[0] is None else ''
        rows.append(f"| {names[r['model']]} / {n} | {r['workers']} × {r['gamesPerActor']} / {r.get('cohorts',1)} | {r['mean_items_per_wave']:.0f} / {r['mean_policy_batch']:.0f} | {r['matches_s']:.2f} | {r['seat_s']:,.0f} | {r['accepted_s']:,.0f} | {L(n):,.0f}{star} | {rate(n):,.0f}{star} |")
    return '\n'.join(rows)
def wall_table():
    rows=['| Run | Engine | Observation encoding | Policy forward | Remaining wait / IPC | Routing, drain and other host work |',
          '|---|---:|---:|---:|---:|---:|']
    for n in cases:rows.append('| '+n+' | '+' | '.join(f'{x:.1f}%' for x in split(cases[n]))+' |')
    return '\n'.join(rows)
def caps_table():
    rows=['| Action tape | Observations | Matches/s | Seat decisions/s | Eligible/s | Battles/s |','|---|---|---:|---:|---:|---:|']
    for n in ('wide-parent','wide-mid'):
        for mode in ('env','core'):
            r=read(f'{n}-{mode}-cap')
            rows.append(f"| {n} | {'Encoded' if mode=='env' else 'Suppressed'} | {r['matches_s']:.2f} | {r['seat_s']:,.0f} | {r['eligible_s']:,.0f} | {r['battles_s']:.1f} |")
    return '\n'.join(rows)
if __name__=='__main__':
    print(collect_table());print();print(wall_table())
