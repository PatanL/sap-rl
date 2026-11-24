"""Group active seats by their fixed policy identity. GRU memory stays in a GPU seat bank."""
import numpy as np,torch
import collections,time
from opponents import LIVE
class WaveActor:
    def __init__(self,model,pool,n):
        self.model=model;self.pool=pool;self.n=n
        self.memory=model.initial_memory(2*n,'cuda')
        self.actions=np.zeros((n,2,3),'i4');self.masks=np.zeros((n,2),'i4')
        self.live_log=np.empty(2*n,'f4');self.batches=[];self.timing=collections.Counter()
    @torch.no_grad()
    def reset(self,ids):
        if not len(ids):return
        ix=np.stack((ids*2,ids*2+1),axis=1).ravel();self.memory.index_fill_(0,torch.as_tensor(ix,device='cuda'),0.)
        self.pool.reset_games(ids)
    @torch.no_grad()
    def act(self,o):
        begin=time.perf_counter();events=[]
        ids=np.flatnonzero(o['active'].ravel());owners=self.pool.owners[ids]
        active_obs={k:v.reshape(-1,*v.shape[2:]) for k,v in o['obs'].items()}
        # Each slot receives exactly one active policy. Return only the compact action/log result to host.
        result=torch.empty((len(ids),5),device='cuda',dtype=torch.float32)
        for owner in np.unique(owners):
            local=np.flatnonzero(owners==owner);slots=ids[local]
            policy=self.model if owner==LIVE else self.pool.models[owner]
            inp={k:torch.from_numpy(v[slots]).to('cuda',non_blocking=True) for k,v in active_obs.items()}
            idx=torch.as_tensor(slots,device='cuda')
            before=torch.cuda.Event(enable_timing=True);after=torch.cuda.Event(enable_timing=True);before.record()
            with torch.autocast('cuda',dtype=torch.bfloat16):out=policy.act(inp,memory=self.memory.index_select(0,idx))
            after.record();events.append((before,after))
            self.memory.index_copy_(0,idx,out.memory.float())
            values=torch.cat((out.actions,out.end_freeze_mask[:,None],out.log_prob[:,None]),dim=1)
            result.index_copy_(0,torch.as_tensor(local,device='cuda'),values)
            self.batches.append(len(slots))
        self.masks.fill(0);result=result.cpu().numpy();self.actions.reshape(-1,3)[ids]=result[:,:3];self.masks.ravel()[ids]=result[:,3];self.live_log[ids]=result[:,4]
        self.timing['policy_stream_s']+=sum(a.elapsed_time(b) for a,b in events)/1000
        self.timing['service_s']+=time.perf_counter()-begin
        return ids,ids[owners==LIVE]
