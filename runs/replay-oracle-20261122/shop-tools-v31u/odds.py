"""Compare true opening inventories, selected visible offers, and one-gold ordinary rolls."""
import csv,json,collections as C, multiprocessing as mp
from scipy.stats import chisquare
from case import details
from receipts import P, load_rows
def scan(arg):
 row,keys=arg;counter=C.defaultdict(C.Counter);counts=C.Counter();examples=[]
 for k in row['states']:
    n=int(k);r=keys.get(n)
    if not r or r.get('skip') or not r.get('completed'):continue
    prev=row['states'].get(str(n-1));end=row['states'][k]
    d=details(row,n,r['seed']);rr=d['run']
    if rr.get('error') or rr['unc']!=end['unc'] or rr['genIssues'] or rr['costErrors']:continue
    if rr['state']['spent']!=end['spent'] or rr['state']['gold']!=end['gold']:continue
    counts['validated eligible turns']+=1
    old=0;paid_zero=False
    for event in rr['log']:
       generation=event['gen'][old:];old=len(event['gen'])
       fresh=[(i,name) for i,kind,how,name in generation if kind=='pet' and how=='roll']
       if not fresh:continue
       tier=end['tier'];zero=(event['op']=='roll' and event['gold0']==1)
       for namegroup in ['all_traced_draws']+(['one_gold'] if zero else []):
           key=(row['split'],namegroup,tier)
           counts[namegroup+' waves']+=1
           for i,name in fresh:
               counter[key][name or '?']+=1
               if name=='Sloth':examples.append([row['match'],row['seat'],n,i,event['op']])
 return dict(counter),counts,examples
def main():
 rows,_=load_rows();by=C.defaultdict(dict)
 for (m,p,n),r in rows.items():by[m,p][n]=r
 agg=C.defaultdict(C.Counter);counts=C.Counter();sloth=set()
 seats=[json.loads(l) for l in open('seats.jsonl')]
 # These raw cohorts use all 680 perspectives and are independent of engine replayability.
 for row in seats:
    for v in row['start']['shop']:agg[row['split'],'opening',1][v['name']]+=1
    for k,s in row['states'].items():
        if int(k)<2:continue
        before=row['states'][str(int(k)-1)]
        for v in s['shop']:
            if v['id']>before['unc'] and not v['reward']:agg[row['split'],'snapshot',s['tier']][v['name']]+=1
 with mp.Pool(6) as pool:
    for d,c,examples in pool.imap_unordered(scan,[(s,by[s['match'],s['seat']]) for s in seats]):
        for key,co in d.items():agg[key].update(co)
        counts.update(c);sloth.update(tuple(x) for x in examples)
 for (split,group,tier),c in list(agg.items()):agg['all',group,tier].update(c)
 probs=json.loads((P/'engine-odds.json').read_text());results=[];tests=[]
 for (split,group,tier),c in sorted(agg.items()):
    expected=probs[str(tier)];total=sum(c.values());known=total-c['?']
    unknown=c['?'];outside=sum(v for k,v in c.items() if k!='?' and k not in expected)
    ns=sorted(expected);supported=sum(c[n] for n in ns)
    if min(supported*expected[n] for n in ns)>=5 and supported:
        x,p=chisquare([c[n] for n in ns],[expected[n]*supported for n in ns]);result=[float(x),len(ns)-1,float(p)]
    else:result=[None,None,None]
    tests.append(dict(split=split,cohort=group,tier=tier,total=total,identified=known,outside=outside,chi2=result[0],df=result[1],p_conditionally_on_known_and_supported=result[2]))
    for name in sorted(set(expected)|set(c)-{'?'}):
      results.append(dict(split=split,cohort=group,tier=tier,species=name,observed=c[name],identified=known,
             all_counted=total,missing_identity=unknown,observed_fraction=c[name]/known if known else None,
             missing_lower=c[name]/total,missing_upper=(c[name]+unknown)/total,engine_probability=expected.get(name,0.)))
 with open('roll-odds.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 (P/'roll-odds-summary.json').write_text(json.dumps(dict(counts=counts,cohorts=tests,sloth=sorted(sloth)),indent=2))
 print(counts)
 for t in tests:
    if t['split']=='all' and t['cohort'] in ('opening','one_gold'):print(t)
if __name__=='__main__':main()
