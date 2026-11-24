import json
from pathlib import Path

def read(name):return json.loads(Path(name+'.json').read_text())
def aggregate(blocks):return {k:sum(b[k] for b in blocks) for k in blocks[0]}
def describe(t):
    secs=t['wall']; decisions=t['decisions']
    parts={k:{'ms':1000*t[k]/decisions,'pct':100*t[k]/secs} for k in ('shop','battle','obs','forward','ipc','other')}
    engine=t['shop']+t['battle']
    return dict(totals=t,parts=parts,ms=1000*secs/decisions,enginePct=100*engine/secs,
                mechanicsCeiling=secs/(secs-engine),withEncodingCeiling=secs/(secs-engine-t['obs']))
out={}
for name in ('actor','four-actor'):
    meta=read(name+'-summary'); t=aggregate(meta['blocks'])
    out[name]=describe(t)
    out[name]['enginePctBlockRange']=[min(describe(b)['enginePct'] for b in meta['blocks']),max(describe(b)['enginePct'] for b in meta['blocks'])]
    rows=read(name+'-cycles')[meta['warmup']:]
    events=read(name+'-forwards')
    other=0.; i=0
    for row in rows:
        t0,t1=row['sent'],row['received']
        while i<len(events) and events[i][1]<=t0:i+=1
        j=i
        while j<len(events) and events[j][0]<t1:
            a,b,_,_,*focal=events[j]
            if focal==[0]:other+=max(0,min(b,t1)-max(a,t0))
            j+=1
    remainder=t['ipc']-other
    assert remainder>=0
    out[name]['forwardOtherGroupsWhileWaiting']={'ms':other*1000/t['decisions'],'pct':other*100/t['wall']}
    out[name]['remainingIPC']={'ms':remainder*1000/t['decisions'],'pct':remainder*100/t['wall']}
for name in ('engine-only','engine-core-only'):
    if not Path(name+'.json').is_file():continue
    out[name]={}
    for mode,blocks in read(name)['results'].items():
        total=aggregate(blocks)
        out[name][mode]={'totals':total,**{key+'PerS':total[key]/total['seconds'] for key in ('matches','decisions','expanded','battles')}}
if 'engine-core-only' in out:
    for mode in ('primitive','joint'):
        for a,b in zip(read('engine-only')['results'][mode],read('engine-core-only')['results'][mode]):
            assert all(a[k]==b[k] for k in ('matches','decisions','expanded','battles'))
out['production']=read('population-log');out['fallback']=read('fallback')['counts']
Path('measurements.json').write_text(json.dumps(out,indent=2))
print([(name,out[name]['forwardOtherGroupsWhileWaiting'],out[name]['remainingIPC']) for name in ('actor','four-actor')])
for name in ('engine-only','engine-core-only'):
    if name in out:print(name,out[name])
