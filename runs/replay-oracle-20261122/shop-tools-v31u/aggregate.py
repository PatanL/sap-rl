"""Aggregate by unique (match, perspective, turn); associations have intersecting denominators."""
from collections import Counter,defaultdict
from pathlib import Path
import csv,json,random
from receipts import P,load_rows,size,caps
def writecsv(path,rows):
 with (P/path).open('w') as fp:
  w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summary(rows):
 ok=[r for r in rows if not r.get('skip')]
 hit=sum(r['reproduced'] for r in ok)
 available=sum(r['completed'] for r in ok)
 return dict(eligible=len(rows),attempted=len(ok),skipped=len(rows)-len(ok),matched=hit,missed=len(ok)-hit,complete=available,
        board=sum(r['boardMatch'] for r in ok),wallet=sum('economy' in r['ever'] for r in ok),
        frozenIds=sum('frozen IDs' in r['ever'] for r in ok),frozenDetail=sum('frozen' in r['ever'] for r in ok),
        retention=sum('remaining IDs' in r['ever'] for r in ok),skips=Counter(r['skip'] for r in rows if r.get('skip')),
        errors=Counter('native exception' if not (r.get('error') or '').startswith('Divergence:') else 'recorded action could not be mapped' for r in ok if not r['completed']))
def unambiguous_ops(r):
 ops=set(r['ops'])
 # The two post-Pill sources missing in the symbolic ledger were free explicit board merges.
 if 'shop merge' in ops:ops.remove('shop merge');ops.add('board merge')
 return ops
def entities(r,group):
 if group=='pet':return set(r['pets'])|set(r['frozenPetNames'])
 if group=='board_pet':return set(r['pets'])
 if group=='affected_action_pet':return set(r['direct'])
 if group=='food':return set(r['foods'])
 return unambiguous_ops(r)
def strata(rows):
 for div in ['all','dev','heldout']:yield div,[r for r in rows if div=='all' or r['split']==div]
def ci(rows):
 by=defaultdict(lambda:[0,0])
 for r in rows:
  if r.get('skip'):continue
  z=by[r['key'][0]];z[0]+=r['reproduced'];z[1]+=1
 v=list(by.values());rng=random.Random(1001);rates=[]
 for _ in range(4000):
  xs=rng.choices(v,k=len(v));rates.append(sum(x[0] for x in xs)/sum(x[1] for x in xs))
 rates.sort();return [rates[100],rates[3899]]
def main():
 reg,curve=load_rows('v31');old,oc=load_rows('v30');fresh,fc=load_rows('v31-refresh');rows=list(reg.values())
 for line in open(P/'seats.jsonl'):
  row=json.loads(line)
  for k,actions in row['actions'].items():
   rr=reg.get((row['match'],row['seat'],int(k)))
   if not rr:continue
   if any(a.get('orders') for a in actions):rr['ops'].append('board order request')
   if any(a.get('freeze') for a in actions):rr['ops'].append('freeze/unfreeze request')
   if any(a['t']==6 for a in actions):rr['ops'].append('type 6 pet purchase')
 header={sp:summary(sub) for sp,sub in strata(rows)}
 assoc=[]
 for typ in ['pet','board_pet','affected_action_pet','food','action']:
  for div,drows in strata(rows):
   index=defaultdict(list)
   for r in drows:
    for e in entities(r,typ):index[e].append(r)
   for e,sub in sorted(index.items()):
    a=summary(sub);assoc.append(dict(group=typ,split=div,entity=e,eligible=a['eligible'],attempted=a['attempted'],skipped=a['skipped'],matched=a['matched'],failed=a['missed'],failure_rate=a['missed']/a['attempted'] if a['attempted'] else None))
 writecsv('association-rates.csv',assoc)
 trajectories=[];capsums=[];comparisons={}
 for label in ('v31','v30','v31-refresh'):
  d,c=load_rows(label)
  left={}
  for k,h,total in c:
   for l in open(P/f'strict-{label}-{k}.jsonl'):
    r=json.loads(l);left[tuple(r['key'])]=r
   for split,ss in strata(list(left.values())):
    a=summary(ss);capsums.append(dict(version=label,split=split,cap=k,matched=a['matched'],attempted=a['attempted']))
  comparisons[label]={sp:summary(ss) for sp,ss in strata(list(d.values()))}
 writecsv('seed-convergence.csv',capsums)
 for k,r in sorted(reg.items()):
  before=old.get(k,{});f=fresh.get(k,{})
  trajectories.append(dict(match=k[0],perspective=k[1],turn=k[2],split=r['split'],source=r['source'],skipped=r.get('skip',''),
        v31_exact=r.get('reproduced'),v30_exact=before.get('reproduced'),v31_refresh_exact=f.get('reproduced'),
        board_exact=r.get('boardMatch'),reached_end=r.get('completed'),frozen_id_match='frozen IDs' in r.get('ever',[]) if not r.get('skip') else None,
        native_economy_match='economy' in r.get('ever',[]) if not r.get('skip') else None,seeds_tested=r.get('tries'),best_seed=r.get('seed'),distinct=r.get('distinct'),
        diff=','.join(r.get('diff',[])),error=r.get('error','') or '',items_inferred=r.get('inferred',0)))
 writecsv('turn-results.csv',trajectories)
 co=Counter();co2=Counter()
 for k,r in reg.items():
    co[(r.get('reproduced'),old.get(k,{}).get('reproduced'))]+=1;co2[(r.get('reproduced'),fresh.get(k,{}).get('reproduced'))]+=1
 bymatch=Counter()
 for r in rows:bymatch[r['key'][0],r['split']]+=int(bool(r.get('diff')))
 inputs={}
 for split,ss in strata(rows):
    inputs[split]={}
    for name,grp in [('no implicit source identities',[r for r in ss if not r.get('inferred')]),('implicit source identity used',[r for r in ss if r.get('inferred')]),('no pet purchase',[r for r in ss if not {'buy','buy merge'}&set(r['ops'])])]:
     inputs[split][name]=summary(grp)
 signatures=json.load(open('mechanic-signatures.json'));sg={}
 for sp,ss in strata(signatures):sg[sp]=Counter(t for r in ss for t in r['signatures'])
 worst=[r for r in rows if r.get('diff')]
 diagnostic=dict(terminalDifferencePatterns=Counter(', '.join(r['diff']) for r in worst),signatures=sg,
         replaySources={str((src,sp)):summary([r for r in rows if r['source']==src and r['split']==sp]) for src in ('archive','raw') for sp in ('dev','heldout')},
         matchesWithMiss=Counter(sp for (m,sp),v in bymatch.items() if v),coverageChecks=inputs,
         noGenIssuesAmongExact=sum(bool(r.get('genIssues')) for r in rows if r.get('reproduced'))==0)
 out=dict(summary=header,versions=comparisons,pairedV31V30={str(k):v for k,v in co.items()},pairedV31Refresh={str(k):v for k,v in co2.items()},
          diagnostics=diagnostic,heldoutMatchBootstrap95=ci([r for r in rows if r['split']=='heldout']))
 (P/'SUMMARY.json').write_text(json.dumps(out,indent=2))
 print(json.dumps(header),dict(co),dict(co2))
if __name__=='__main__':main()
