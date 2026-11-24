"""Turn denominators, conservative seller associations, and sources of identifiability."""
import csv,json,collections as C,sys
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
from identity import compile_turn
def read(p):return [json.loads(l) for l in open(p)]
def keyed(p):return {tuple(x['key']):x for x in read(p)}
def strict(fresh=False):
    d={};curve=[]
    for k in (1,64,1024,8192,65536):
        fn=P/'fresh'/f'strict-{k}.jsonl' if fresh else P/f'original-{k}.jsonl'
        if not fn.exists():continue
        for x in read(fn):d[tuple(x['key'])]=x
        curve.append((k,sum(bool(v.get('reproduced')) for v in d.values())))
    return d,curve
def rate(fail,n):return fail/n if n else None
def savecsv(name,rows):
    with (P/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def stat(d):
    result={}
    for sp in ['all','dev','heldout']:
        rows=[v for v in d.values() if sp=='all' or v['split']==sp];skips=C.Counter(v['skip'] for v in rows if v.get('skip'))
        attempted=[v for v in rows if not v.get('skip')]
        result[sp]=dict(completed=len(rows),attempted=len(attempted),reproduced=sum(v['reproduced'] for v in attempted),skips=skips)
    return result
def main():
    original,oc=strict();fresh,fc=strict(True);ver=keyed(P/'verified.jsonl');evidence=keyed(P/'input-certificates.jsonl');neg=keyed(P/'negative.jsonl')
    ranges={};summary={};actual=[];grouped=C.defaultdict(C.Counter);unknown_modes=C.Counter();sellrows=[]
    for r in map(json.loads,open(P.parent/'seats.jsonl')):
        for k in r['states']:
            n=int(k);key=(r['match'],r['seat'],n)
            if key not in original:continue
            q=original[key];sp=r['split'];tr,nam,_,_,_=compile_turn(r,n,'near-right');actions=r['actions'][k]
            names=dict(nam);ops={op for op,_ in tr};sells=[a['id'] for a in actions if a['t']==9]
            if any(a.get('orders') for a in actions):ops.add('board order request')
            if any(a.get('freeze') for a in actions):ops.add('freeze/unfreeze request')
            if any(a['t']==6 for a in actions):ops.add('any pet purchase')
            if 'shop merge' in ops:ops.remove('shop merge');ops.add('board merge') # explicit post-Pill board source
            if not q.get('skip'):
                universe='original strict';hit=q['reproduced'];cert=True;modelable=True
                margs={};full=False
            else:
                z=ver[key];pr=evidence[key];cnt=z['n'];hit=cnt>0;cert=pr['certified'];modelable=not bool(z['unmapped_food']);universe='new pet/summon'
                names.update({int(i):s for i,s in pr['record_labels'].items()});margs={int(i):s for i,s in z['marginals'].items()}
                if cert and cnt:assert cnt==1 and all(v==[names[i]] for i,v in margs.items()),key
                bucket='unmapped food, no witness' if not modelable else 'no candidate witnessed' if not cnt else 'unique' if cnt==1 else 'small 2-5' if cnt<=5 else 'broad 6+'
                source=pr['certificate'] if cert else ('engine-dependent unique' if cnt==1 else 'multiple assignments' if cnt else 'unresolved')
                for div in ('all',sp):
                    ranges.setdefault(div,C.Counter())[bucket]+=1
                    for klass in ('all 710', 'pet/summon' if modelable else 'unmapped foods',q['skip'],source):
                        ks=summary.setdefault(div,{}).setdefault(klass,C.Counter());ks['turns']+=1;ks['matches']+=hit
                actual.append(dict(match=key[0],seat=key[1],turn=n,split=sp,original_skip=q['skip'],status=bucket,matching_assignments=cnt,
                    identity_source=source,record_certified=cert,independently_identified=json.dumps(pr['record_labels'],sort_keys=True),
                    uid_candidate_sets=json.dumps(z['marginals'],sort_keys=True),purchases_missing=json.dumps(z['buy']),summons_missing=json.dumps(z['endogenous']),
                    sell_count=len(sells),comparison_seed_cap=z['seedCap'],concrete_witness_rechecks=z['individual_rechecks'],
                    diagnostic=(neg.get(key,{}).get('best') or {}).get('why','') if not hit else ''))
            if not modelable:continue
            named=set(names.get(i) for i in sells)-{None}
            if any(names.get(i) is None for i in sells):named.add('(unidentified seller)')
            cond=['all turns']+(['at least one sale'] if sells else ['no sale'])+(['sell-heavy (2+)'] if len(sells)>=2 else [])
            for div in ('all',sp):
                for cohort in [universe]+(['record-certified new'] if q.get('skip') and cert else []):
                    for window in cond:
                        for kind,items in [('overall',{'(all)'}),('seller before endpoint',named),('action',ops)]:
                            for name in items:
                                st=grouped[div,cohort,window,kind,name];st['turns']+=1;st['fail']+=not hit
            if q.get('skip') and sells:
                sellrows.append(dict(match=key[0],seat=key[1],turn=n,split=sp,sales=len(sells),endpoint_hit=hit,
                           sellers_record_known=json.dumps({i:names.get(i) for i in sells}),possible_sellers_given_hit=json.dumps({i:margs.get(i,[names[i]] if i in names else []) for i in sells})))
    assoc=[]
    for (sp,cs,win,kind,name),ns in sorted(grouped.items()):
        assoc.append(dict(split=sp,cohort=cs,turn_selection=win,category=kind,name=name,attempted=ns['turns'],misses=ns['fail'],failure_rate=rate(ns['fail'],ns['turns'])))
    savecsv('turns.csv',actual);savecsv('sell-action-rates.csv',assoc);savecsv('sell-turns.csv',sellrows)
    origmatches={v['match'] for v in map(json.loads,open(P.parent/'seats.jsonl'))};freshmatches={v['match'] for v in map(json.loads,open(P/'fresh'/'seats.jsonl'))}
    concrete=read(P/'concrete.jsonl')
    out=dict(inference=ranges,by_source=summary,strict_original=stat(original),strict_fresh=stat(fresh),original_convergence=oc,fresh_convergence=fc,
             training_and_fresh_match_overlap=len(origmatches&freshmatches),concrete_seed0_vectors=sum(x['combos'] for x in concrete),concrete_calls=sum(x['engineCalls'] for x in concrete),
             matching_assignments=sum(x['n'] for x in ver.values()),excluded_by_offer_order=sum(x['excluded_by_order'] for x in ver.values()),
             failed_concrete_rechecks=sum(x['failed_individual_recheck'] for x in ver.values()))
    (P/'SUMMARY.json').write_text(json.dumps(out,indent=2))
    print(ranges,summary['all'])
if __name__=='__main__':main()
