"""Check record-side pruning against sold pet UIDs whose true species was witnessed."""
from infer import *
from evidence import *
from profile_cases import inspect
def main():
    counts=C.Counter();mismatch=[]
    # Retain only controls in the current ordinary tier range (unidentified ORIGINALLY bought pets cannot be
    # previously frozen, while a deliberately blinded control may have been a frozen next-tier reward).
    for row in map(json.loads,open(P.parent/'seats.jsonl')):
        for k,acts in row['actions'].items():
            n=int(k)
            if n<2 or k not in row['states'] or str(n-1) not in row['states']:continue
            t=row['states'][k]['tier'];bought={a['id'] for a in acts if a['t']==6};sold={a['id'] for a in acts if a['t']==9}
            for i in bought&sold:
                name=row['known'].get(str(i),[None,None])[1]
                if name not in TIER or TIER[name]>t or row['origin'].get(str(i)) not in ('genesis','snapshot'):continue
                if any(a['t']==7 and i in (a['id'],a['target']) for a in acts):continue
                masked=dict(row,known=dict(row['known']),states=dict(row['states']))
                del masked['known'][str(i)]
                b0=masked['states'][str(n-1)];masked['states'][str(n-1)]=dict(b0,shop=[dict(v,name='(masked)') if v['id']==i else v for v in b0['shop']])
                p=prep(masked,n)
                if i not in p['b'] or len(p['b'])!=1 or p['e']:continue
                p['clauses']=stock_clauses(p);p['clues']=clues(masked,n);size,sets=input_domains(p)
                counts['tested']+=1;counts[row['split']]+=1
                if name in sets[0]:counts['retained true name']+=1
                else:mismatch.append([row['match'],row['seat'],n,i,name,list(sets[0])])
                if size==1:counts['predicted exactly one']+=1
    (P/'masked-controls.json').write_text(json.dumps(dict(counts=counts,removed=mismatch),indent=2))
    print(counts,'misses',len(mismatch))
if __name__=='__main__':main()
