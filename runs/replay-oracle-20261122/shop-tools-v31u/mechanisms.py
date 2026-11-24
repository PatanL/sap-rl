"""Diagnostic signatures in real-vs-native endpoint differences. No engine modifications."""
import collections as C,json
from receipts import load_rows
from case import details,load
from replay import data,expected
PETS={v['name']:v for v in data['pets'].values()}
def scorpion(r):
 if not r.get('completed') or not r.get('actual'):return False
 actual,native=r['actual'],r['result']
 if r['diff']==['frozen'] or 'frozen' in r['diff']:
    if any(v[1]=='pet' and v[2]=='Scorpion' and v[-2] is None and any(w[0]==v[0] and w[:-2]==v[:-2] and w[-2]=='Peanut' for w in native['frozen']) for v in actual['frozen']):return True
 return any(a and b and a[0]==b[0]=='Scorpion' and a[-1]=='Peanut' and b[-1] is None for a,b in zip(actual['board'],native['board']))
def dragon(d):
    run=d['run'];logs=run['log'];s=run.get('state')
    if not s:return False,[]
    end=expected(d['after']); final=logs[-1]['uids']
    if s['frozen']!=end['frozen'] or s['spent']!=end['spent'] or s['gold']!=end['gold']:return False,[]
    if any(a!=b for a,b in zip([v[:3]+v[5:] if v else None for v in s['board']],[v[:3]+v[5:] if v else None for v in end['board']])):return False,[]
    delta=C.Counter();counts=C.Counter();cause=[]
    for prev,cur,a,(op,dest) in zip(logs,logs[1:],d['actions'][1:],d['steps'][1:]):
        if op not in ('buy','buy merge'):continue
        i=a['id'];name=d['names'][i]
        if name not in PETS or PETS[name]['tier']!=1:continue
        recipient=i if op=='buy' else dest
        for rid,v in zip(prev['uids'],prev['state']['board']):
            if not v or v[0]!='Dragon' or counts[rid]>=4:continue
            counts[rid]+=1;lv=v[1]
            delta[rid]-=lv;delta[recipient]+=lv;cause.append([rid,recipient,lv])
    if not cause:return False,[]
    # A strict endpoint signature, no projection across an ambiguous intervening merge.
    for i,slot in enumerate(final):
        if slot is None:continue
        v=s['board'][i];w=end['board'][i]
        if any(v[h]+delta[slot]!=w[h] for h in (3,4)):return False,cause
    return any(delta[i] for i in final),cause
def spider_receipt():
    key=['01a1a46a-73d2-738f-8311-24d777755fee','01a1a46c-607a-74ac-a213-4c1c9196d7c4',7]
    row,n=load(key);rates={};outcomes=C.Counter()
    for seed in range(256):
        d=details(row,n,seed);logs=d['run']['log'];summon=logs[3];j=summon['uids'].index(118);pet=summon['state']['board'][j]
        sale=logs[5];v=(pet[0],pet[1],sale['gold']-sale['gold0']);rates[v]=rates.get(v,0)+1
        outcomes[d['run'].get('error') or 'complete']+=1
    return dict(key=key,saleUid=118,possibleTier3=sorted(v['name'] for v in PETS.values() if isinstance(v.get('packs'),list) and 'StandardPack' in v['packs'] and v.get('tier')==3),nativeSellOutcomes=[[list(k),v] for k,v in sorted(rates.items())],errors=outcomes,
             actualIncome=d['after']['gain'],otherSale='level-one Dolphin, 1 gold',inputPolicy='unresolved summon species; excluded from primary native replay denominator')
def main():
 r,_=load_rows();probes={tuple(v['key']):v for v in json.load(open('food-probes.json'))};out=[]
 for k,v in r.items():
    if not v.get('diff'):continue
    labels=[];evidence={};f=probes.get(k,{})
    if scorpion(v):labels.append('Scorpion Peanut timing')
    if f.get('explainedBy'):
        found=f['explainedBy']
        if 'permute food notifications' in found or 'per-observer ordering' in found:labels.append('Rabbit multi-target notification order')
        if 'omit shop pets from Rabbit' in found:labels.append('Rabbit counts Canned Food shop targets')
        if 'both' in found and len(found)==1:labels.append('Rabbit food notification order and shop targets')
    if 'Dragon' in v['pets'] and v['diff']==['permanent']:
        row,n=load(k);d=details(row,n,v['seed']);ok,causes=dragon(d)
        if ok:labels.append('Dragon excludes tier-one buyer');evidence['dragonTransfers']=causes
    out.append(dict(key=k,split=v['split'],signatures=labels,evidence=evidence))
 json.dump(out,open('mechanic-signatures.json','w'),indent=2)
 json.dump(spider_receipt(),open('spider-sale.json','w'),indent=2)
 print(C.Counter(y for x in out for y in x['signatures']))
if __name__=='__main__':main()
