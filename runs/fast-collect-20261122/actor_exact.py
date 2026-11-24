"""Native actor inference timing: frozen weights, one autocast context per stochastic call."""
import random,statistics,time
from common import *
setup_gpu();models={k:model_for(k).cuda().eval().requires_grad_(False) for k in SIZES}
rows=[o for x,_,_ in examples() for o in x];bs=sorted(set(BATCHES+[7,13]))
data={b:{k:torch.from_numpy(np.stack([rows[j%len(rows)][k] for j in range(b)])).cuda() for k in rows[0]} for b in bs}
out=[]
try:
    from measure_floor import diagnose
    diagnose(models,data)
except Exception as error:
    save('fixed-profile.json',dict(available=False,reason=str(error)))
with torch.no_grad():
    for block in range(3):
        configs=[(m,b,p) for m in SIZES for b in bs for p in ('fp32','bf16') if b in BATCHES or (m=='parent' and p=='bf16')]
        random.Random(108+block).shuffle(configs)
        for m,b,p in configs:
            model=models[m];mem=model.initial_memory(b,'cuda')
            def call():
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=p=='bf16'):
                    return model.act(data[b],memory=mem,deterministic=False)
            for _ in range(2):call()
            torch.cuda.synchronize();t=time.perf_counter()
            for _ in range(11):call()
            torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000/11
            out.append(dict(block=block,model=m,batch=b,precision=p,ms=ms))
        save('actor-exact.json',out);print('block',block,flush=True)
