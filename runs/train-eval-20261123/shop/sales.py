import json,gzip,os,sys
from pathlib import Path
S=Path('/sap/runs/shop-oracle-20261122');sys.path[:0]=[str(S/'recover'),str(S)];os.environ['SAP_ENGINE_ROOT']='/engine/sapai'
from infer import prep,jobs
import replay
def main():
 rows={tuple(v['key']):v for v in map(json.loads,open('shop/rechecked.jsonl'))}
 prev={tuple(x['key']):x for x in map(json.loads,gzip.open(S/'recover/assignments.jsonl.gz','rt'))}
 out=[]
 for row,n,o in jobs():
    k=tuple(o['key']);r=rows[k]
    if not(r['lost_assignments'] or (not r['old_matches'] and r['passes']) or (k[0].startswith('01a1a19a') and n==9)):continue
    p=prep(row,n)
    lab,si,path=(r['witness'] or (prev[k]['witnesses'] or [[[],0,[]]])[0])
    names={**p['names'],**dict(zip(p['b'],lab))}
    actual=replay.engine_turn(p['before'],p['end'],p['acts'],names,p['kinds'],(p['seed']+si*2654435761)%2**32,trace=True)
    events=[]
    for i,(a,step) in enumerate(zip(p['acts'],actual['log'])):
        if a['t']==9:
            last=actual['log'][i-1];j=last['uids'].index(a['id']);pet=last['state']['board'][j]
            events.append(dict(op='sell',pet=pet[0],level=pet[1],uid=a['id'],income=step['gold']-step['gold0']))
        if a['t']==8 and a['spell']['name']=='Sleeping Pill':
            last=actual['log'][i-1];pet=last['state']['board'][last['uids'].index(a['target'])]
            events.append(dict(op='pill',pet=pet[0],level=pet[1],perk=pet[-1]))
    state=actual.get('state') or actual['partial']
    out.append(dict(key=k,fix=r['passes'],before=r['old_matches'],events=events,real_gain=p['end']['gain'],engine_gain=state['gain'],error=actual.get('error'),diff=replay.differences(p['want'],actual),real=p['want'],native=state))
 json.dump(out,open('shop/sale-cases.json','w'),indent=2)
if __name__=='__main__':main()
