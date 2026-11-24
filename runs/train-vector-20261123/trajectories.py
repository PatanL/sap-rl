"""Copy only live-seat rows before vector step; drain complete turns before explicit reset."""
import numpy as np,torch
from sapai_gym1.rl.dense_trajectory import OwnedTrajectoryBatch
from opponents import LIVE

class Turns:
    def __init__(self,o,n,width):
        seats=2*n;self.lengths=np.zeros(seats,'i4');self.limit=52
        shape=(seats,self.limit)
        self.obs={k:np.empty((*shape,*v.shape[2:]),v.dtype) for k,v in o['obs'].items()}
        self.actions=np.empty((*shape,3),'i8');self.freeze=np.empty(shape,'i2')
        self.log=np.empty(shape,'f4');self.version=np.empty(shape,'i8')
        self.initial=np.zeros((seats,width),'f4')
    def push(self,o,actor,ids,version):
        times=self.lengths[ids]
        if np.any(times>=self.limit):raise ValueError('Turn exceeded environment move budget')
        first=ids[times==0]
        if len(first):
            # Captured before inference by the caller when a live turn starts.
            self.initial[first]=self.new_initial
        for k,v in o['obs'].items():self.obs[k][ids,times]=v.reshape(-1,*v.shape[2:])[ids]
        self.actions[ids,times]=actor.actions.reshape(-1,3)[ids]
        self.freeze[ids,times]=actor.masks.ravel()[ids]
        self.log[ids,times]=actor.live_log[ids];self.version[ids,times]=version;self.lengths[ids]+=1
    def start_memory(self,actor,ids):
        starts=ids[self.lengths[ids]==0]
        if len(starts):self.new_initial=actor.memory.index_select(0,torch.as_tensor(starts,device='cuda')).cpu().numpy()
    def finish(self,o,pool):
        slots=np.flatnonzero(np.repeat(o['battle_resolved'],2)&(pool.owners==LIVE))
        if not len(slots):return []
        lens=self.lengths[slots];assert np.all(lens>0)
        seats=np.repeat(slots,lens);offset=lens.cumsum()-lens;ticks=np.arange(len(seats))-np.repeat(offset,lens)
        ht=torch.from_numpy;ends=lens.cumsum()-1;reward=np.zeros(len(seats),'f4');discount=np.ones(len(seats),'f4')
        term=o['terminated'][slots//2];reward[ends]=o['rewards'].ravel()[slots];discount[ends]=np.where(term,0,.99)
        result=o['battle_result'][slots//2];label=np.where(result==2,2,(result!=(slots%2)).astype('i8'))
        b=OwnedTrajectoryBatch({k:ht(v[seats,ticks]) for k,v in self.obs.items()},ht(self.actions[seats,ticks]),ht(self.log[seats,ticks]),ht(self.version[seats,ticks]),ht(reward),ht(discount),ht(self.freeze[seats,ticks]),
            tuple(map(int,lens)),ht(self.initial[slots]),{k:ht(v.reshape(-1,*v.shape[2:])[slots]) for k,v in o['obs'].items()},ht(term),ht(label),tuple(map(int,slots)))
        self.lengths[slots]=0
        return b.segments()
