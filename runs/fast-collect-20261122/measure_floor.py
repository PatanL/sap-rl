"""CUPTI diagnostic, distinct from the unprofiled latency sweep."""
import torch,statistics,os,json,tempfile
from common import save
from pathlib import Path
def diagnose(models,data):
    from torch.profiler import profile,record_function,ProfilerActivity
    rows=[]
    if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        save('fixed-profile.json',dict(available=False,reason='CUDA profiler activity unavailable'));return
    for name in ('parent','mid'):
        for batch in (1,16):
            m=models[name];mem=m.initial_memory(batch,'cuda');obs=data[batch]
            def forward():
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):m.act(obs,memory=mem,deterministic=False)
            for _ in range(3):forward()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    with record_function('AUDIT_FORWARD'):
                        forward();torch.cuda.synchronize()
            with tempfile.TemporaryDirectory(dir='/audit/tmp') as tmp:
                path=Path(tmp)/'trace.json'
                prof.export_chrome_trace(str(path))
                events=json.loads(path.read_text())['traceEvents']
            windows=[e for e in events if e.get('name')=='AUDIT_FORWARD' and e.get('cat')=='user_annotation' and e.get('ph')=='X']
            device=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and e.get('ph')=='X']
            if len(windows)!=5 or not device:raise RuntimeError('Missing CPU ranges or GPU kernel activity')
            for w in windows:
                lo,hi=w['ts'],w['ts']+w['dur']
                spans=sorted((max(lo,e['ts']),min(hi,e['ts']+e['dur'])) for e in device if e['ts']+e['dur']>lo and e['ts']<hi)
                if not spans:raise RuntimeError('Device and CPU time ranges could not be aligned')
                busy=0.;end=lo
                for start,stop in spans:
                    busy+=max(0.,stop-max(end,start));end=max(end,stop)
                rows.append(dict(model=name,batch=batch,profiledWallMs=(hi-lo)/1000,deviceActiveMs=busy/1000,deviceEvents=len(spans)))
    save('fixed-profile.json',dict(available=True,rows=rows))

if __name__=='__main__':
    from common import setup_gpu,model_for,examples
    import numpy as np
    setup_gpu()
    models={k:model_for(k).cuda().eval().requires_grad_(False) for k in ('parent','mid')}
    rows=[o for x,_,_ in examples() for o in x]
    data={b:{k:torch.from_numpy(np.stack([rows[i][k] for i in range(b)])).cuda() for k in rows[0]} for b in (1,16)}
    diagnose(models,data)
