import gc,random,statistics,time
from common import *
setup_gpu(); turns=examples(); rows=[o for x,_,_ in turns for o in x]
models={k:model_for(k).cuda().eval() for k in SIZES}
data={b:{k:torch.from_numpy(np.stack([rows[i%len(rows)][k] for i in range(b)])).cuda() for k in rows[0]} for b in BATCHES}
result=[]
with torch.inference_mode():
    for block in range(3):
        cases=[(k,b,prec) for k in models for b in BATCHES for prec in ('fp32','bf16')]
        random.Random(169+block).shuffle(cases)
        for k,b,prec in cases:
            m=models[k]; obs=data[b]; mem=m.initial_memory(b,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=prec=='bf16'):
                m.act(obs,memory=mem)
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(9):m.act(obs,memory=mem)
                torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000/9
            result.append(dict(block=block,model=k,batch=b,precision=prec,ms=ms))
        save('latency.json',result);print('block',block,flush=True)
    stages=[]
    for name in SIZES:
        m=models[name]
        for b in (1,16,256):
            obs=data[b];mem=m.initial_memory(b,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                f=m.encode_decision(obs,memory=mem,forecast=False)
                for label,fn in [('encode',lambda:m.encode_decision(obs,memory=mem,forecast=False)),('decode',lambda:m.sample_joint_action(f,deterministic=False))]:
                    fn();torch.cuda.synchronize();start=time.perf_counter()
                    for _ in range(16):fn()
                    torch.cuda.synchronize();stages.append(dict(model=name,batch=b,phase=label,ms=(time.perf_counter()-start)*1000/16))
    save('stages.json',stages)
