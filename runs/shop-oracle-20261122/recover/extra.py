import csv,json,collections as C,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
def keyed(n):return {tuple(r['key']):r for r in map(json.loads,open(P/(n+'.jsonl')))}
def dumpcsv(n,rows):
 with (P/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def run():
 z=keyed('verified');q=keyed('input-certificates');diagnostics=keyed('negative');competitors=keyed('contrast')
 allrows=[];neg=[];c=C.defaultdict(C.Counter)
 for k,r in sorted(z.items()):
    pr=q[k];comp=competitors.get(k,{})
    if pr['certified']:source=pr['certificate']
    elif r['n']==1 and comp.get('action_wallet_competitor'):source='terminal effects discriminate'
    elif r['n']==1:source='engine reconstruction; no action/wallet competitor witnessed'
    elif r['n']>1:source='multiple endpoint-compatible identities'
    elif r['unmapped_food']:source='unmapped food; no witness'
    else:source='no witness; identity unresolved'
    for sp in ('all',r['split']):c[sp][(source,'match' if r['n'] else 'miss')]+=1
    allrows.append(dict(match=k[0],seat=k[1],turn=k[2],split=r['split'],witnesses=r['n'],source=source,
        record_labels=json.dumps(pr['record_labels']),full_endpoint_identities=json.dumps(r['marginals']),
        competing_identity_fits_actions_and_gold=json.dumps(comp.get('action_wallet_competitor')),competing_endpoint_disagreement=comp.get('failed_fields','')))
    if r['n']:continue
    d=diagnostics.get(k,{}) or {};best=d.get('best') or {};w=d.get('expected',{});s=best.get('state',{});why=best.get('why','');err=best.get('error') or ''
    if r['unmapped_food']:reason='Unmapped free food after Chocolate'
    elif why=='frozen' and any(v[1:3]==['pet','Scorpion'] and v[-2]=='Peanut' for v in s.get('frozen',[]) if v not in w['frozen']):reason='Frozen Scorpion perk differs'
    elif ('gold' in err.lower()) and any(n in ('Ram','Zombie Fly','Zombie Cricket') for n in best.get('endogenous',{}).values()):reason='Unaffordable action after selling summoned pets'
    elif not best or err:reason='Recorded action or UID generation mismatch'
    else:reason='Terminal board stats/perk differ'
    delta=[]
    if best and 'action' not in why:
      for j,(a,b) in enumerate(zip(w['board'],s['board'])):
        if a!=b:delta.append(dict(x=j,real=a,native=b))
      if w['frozen']!=s['frozen']:delta.append(dict(real_frozen=w['frozen'],native_frozen=s['frozen']))
    neg.append(dict(match=k[0],seat=k[1],turn=k[2],split=r['split'],source=source,category=reason,seed_cap=r['seedCap'],record_labels=json.dumps(pr['record_labels']),
             example_hypothesis=json.dumps(best.get('names')),example_summons=json.dumps(best.get('endogenous')),action_index=best.get('at'),error=err,
             component=why or str(d.get('diagnostic_counts',{})),real_vs_native_terminal=json.dumps(delta),real_turn_gold_gain=w.get('gain'),native_complete_or_partial_gain=s.get('gain')))
 dumpcsv('identity-evidence.csv',allrows);dumpcsv('no-match.csv',neg)
 (P/'evidence-summary.json').write_text(json.dumps({sp:[dict(source=k[0],result=k[1],n=v) for k,v in d.items()] for sp,d in c.items()},indent=2))
 print(C.Counter(v['category'] for v in neg))
if __name__=='__main__':run()
