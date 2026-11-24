"""Episode-stable assignments and bounded recent-snapshot sampling."""
from collections import Counter,deque
import copy
import numpy as np
import torch
from config import PARENT_HASH
LIVE=-1
PARENT=-2
class Opponents:
    def __init__(self,actor,parent_path,share,size,slots,seed,start,info,resume):
        self.share=share;self.size=size;self.actor=actor
        self.models={};self.recent=deque(maxlen=size);self.assignments=Counter();self.decisions=Counter()
        self.owners=np.full(slots,LIVE,dtype=np.int64)
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,start,20261123]))
        if not share:return
        self.models[PARENT]=self.frozen(parent_path)
        if resume:
            for f in sorted(resume.parent.glob('update-*.pt')):
                version=int(f.stem.split('-')[-1])
                if version<=start:self.recent.append((version,f))
            for v,f in self.recent:self.models[v]=self.frozen(f,info)
    def frozen(self,path,info=None):
        ck=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
        if info and (ck.get('engine_payload_sha256')!=info['payload_sha256'] or ck.get('wide_arm')!='parent'):raise ValueError('Mismatched historical opponent')
        model=copy.deepcopy(self.actor)
        model.load_state_dict(ck['model'],strict=True)
        return model.eval().requires_grad_(False)
    def snapshot(self,version,path):
        if not self.share:return
        self.recent.append((version,path))
        self.models[version]=copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.retire()
    def retire(self):
        retained={PARENT,*[v for v,_ in self.recent],*self.owners.tolist()}
        for v in list(self.models):
            if v not in retained:del self.models[v]
    def reset_games(self,indices):
        for g in indices:
            self.owners[2*g:2*g+2]=LIVE
            key='mirror'
            if self.share and self.rng.random()<self.share:
                frozen=PARENT
                if self.recent and self.rng.random()<.5:
                    frozen=self.recent[int(self.rng.integers(len(self.recent)))][0]
                seat=int(self.rng.integers(2));self.owners[2*g+seat]=frozen
                key=f'parent:seat{seat}' if frozen==PARENT else f'snapshot-{frozen}:seat{seat}'
            self.assignments[key]+=1
        self.retire()
