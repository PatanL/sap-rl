"""Wide paired self-play with two CPU cohorts, shared pages, and per-action policy versions."""
import mmap,multiprocessing as mp,os,traceback
from multiprocessing.reduction import DupFd
from contextlib import contextmanager
import numpy as np
import torch
import sys
sys.path.insert(0,"/sap/runs/train-v31-prep-20261123")
from opponents import LIVE
from runtime import selfplay_env
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai_gym1.rl.dense_trajectory import OwnedTrajectoryBatch
from sapai_gym1.rl.end_freeze import step_with_end_freeze

class Page:
    """One lazily committed Linux memfd per worker, shared across a clean spawn."""
    def __init__(self,games,example,width):
        self.slots=2*games
        cap=self.slots*52  # native shop forces END after 50 primitive actions
        self.layout={};size=0
        def field(name,shape,dtype):
            nonlocal size
            size=(size+63)//64*64
            dt=np.dtype(dtype);self.layout[name]=(tuple(shape),dt.str,size)
            size+=int(np.prod(shape))*dt.itemsize
        for group,n in [('obs',self.slots),('bootstrap',self.slots),('history',cap)]:
            for key,arr in example.items():field((group,key),(n,*arr.shape),arr.dtype)
        for name,shape,dtype in [
            ('ids',(self.slots,),'i4'),('reset',(self.slots,),'?'),('actions',(self.slots,4),'i8'),
            ('owner',(self.slots,),'i8'),('logs',(self.slots,),'f4'),('hidden',(self.slots,width),'f4'),('initial',(self.slots,width),'f4'),
            ('moves',(cap,4),'i8'),('versions',(cap,),'i8'),('values',(cap,3),'f4'),('meta',(self.slots,4),'i8')]:
            field(name,shape,dtype)
        self.fd=os.memfd_create('sap-wide-page');os.ftruncate(self.fd,size);self.attach()
    def attach(self):
        self.mapping=mmap.mmap(self.fd,0,flags=mmap.MAP_SHARED)
        self.obs={};self.history={};self.bootstrap={}
        for name,(shape,dtype,offset) in self.layout.items():
            arr=np.ndarray(shape,dtype,buffer=self.mapping,offset=offset)
            if isinstance(name,tuple):getattr(self,name[0])[name[1]]=arr
            else:setattr(self,name,arr)
    def __getstate__(self):
        return self.slots,self.layout,DupFd(self.fd)
    def __setstate__(self,info):
        self.slots,self.layout,handle=info;self.fd=handle.detach();self.attach()
    def close(self):
        self.mapping.close();os.close(self.fd)

    def write_turn(self,buf,start,segment,initial,bootstrap,reward,terminal,label,slot):
        n=len(buf);section=slice(start,start+n)
        if n>52 or buf[-1][1][0]!=0:raise RuntimeError('Incomplete or overlong native shop turn')
        for k in self.history:
            np.stack([x[0][k] for x in buf],out=self.history[k][section])
            self.bootstrap[k][segment]=bootstrap[k]
        np.stack([x[1] for x in buf],out=self.moves[section])
        self.values[section,0]=[x[2] for x in buf]
        self.versions[section]=[x[3] for x in buf]
        self.values[section,1]=0
        self.values[section,2]=1
        self.values[start+n-1,1]=reward
        self.values[start+n-1,2]=0 if terminal else .99
        self.initial[segment]=initial
        self.meta[segment]=(n,terminal,label,slot)
        return n

    def drain(self,nseg,nrows):
        if not nseg:
            assert not nrows
            return []
        def owned(arr):return torch.from_numpy(arr.copy())
        meta=self.meta[:nseg];lengths=tuple(map(int,meta[:,0]))
        assert sum(lengths)==nrows
        v=self.values[:nrows];a=self.moves[:nrows]
        batch=OwnedTrajectoryBatch({k:owned(x[:nrows]) for k,x in self.history.items()},
            owned(a[:,:3]),owned(v[:,0]),owned(self.versions[:nrows]),owned(v[:,1]),owned(v[:,2]),owned(a[:,3].astype('i2')),
            lengths,owned(self.initial[:nseg]),{k:owned(x[:nseg]) for k,x in self.bootstrap.items()},
            owned(meta[:,1].astype(bool)),owned(meta[:,2]),tuple(map(int,meta[:,3])))
        return batch.segments()

def worker(w,games,width,page,pipe,seed,start_version):
    try:
        torch.set_num_threads(1)
        envs=[selfplay_env.ParallelSelfPlayEnv() for _ in range(games)]
        gens=[0]*games
        def reset(i):
            seq=np.random.SeedSequence([seed,start_version,w,i,gens[i]])
            gens[i]+=1
            world=int(seq.generate_state(1,dtype=np.uint64)[0])%(2**63)
            return envs[i].reset(seed=world,paired_tape=True)
        obs=[reset(i) for i in range(games)]
        locked=np.full((games,2),-99,dtype=np.int64)
        fresh=np.ones((games,2),dtype=bool)
        mem=np.zeros((games,2,width),'f4');initial=mem.copy()
        turns=[[[],[]] for _ in envs]
        nseg=nrows=matches=0
        eligible=frozen=0
        while True:
            active=[(i,s) for i,env in enumerate(envs) for s,yes in enumerate(env.active_players()) if yes]
            n=len(active)
            for k,a in page.obs.items():np.stack([obs[i][s][k] for i,s in active],out=a[:n])
            page.ids[:n]=[i*2+s for i,s in active]
            page.reset[:n]=[fresh[i,s] for i,s in active]
            pipe.send((n,nseg,nrows,matches,eligible,frozen,sum(len(buf) for pair in turns for buf in pair)))
            version=pipe.recv()
            if version is None:break
            moves=[[None,None] for _ in envs];masks=[[0,0] for _ in envs]
            eligible=frozen=0
            for j,(i,s) in enumerate(active):
                action=page.actions[j].copy()
                owner=int(page.owner[j])
                if fresh[i,s]:locked[i,s]=owner
                if locked[i,s]!=owner:raise RuntimeError('Opponent changed inside a match')
                if owner==LIVE:
                    turns[i][s].append((obs[i][s],action,float(page.logs[j]),version));eligible+=1
                else:frozen+=1
                moves[i][s]=AutoregressiveAction(*map(int,action[:3]));masks[i][s]=int(action[3])
                mem[i,s]=page.hidden[j];fresh[i,s]=False
            nseg=nrows=matches=0
            for i,env in enumerate(envs):
                step=step_with_end_freeze(env,*moves[i],mask_ids=masks[i]).result
                obs[i]=step.observations
                if not step.battle_resolved:continue
                for s in (0,1):
                    if locked[i,s]!=LIVE:
                        if turns[i][s]:raise RuntimeError('Frozen actor produced a learner trajectory')
                        continue
                    label=2 if step.battle_result==2 else int(step.battle_result!=s)
                    nrows+=page.write_turn(turns[i][s],nrows,nseg,initial[i,s],obs[i][s],
                        step.rewards[s],step.terminated,label,w*page.slots+i*2+s)
                    nseg+=1;turns[i][s].clear()
                initial[i]=mem[i]
                if step.terminated:
                    matches+=1;obs[i]=reset(i);fresh[i]=True;mem[i]=0;initial[i]=0
    except BaseException:
        try:pipe.send(('error',traceback.format_exc()))
        except (EOFError,BrokenPipeError):pass
        raise
    finally:pipe.close()

class Collector:
    def __init__(self,pages,pipes,actor,width,device,pool):
        self.pool=pool;self.live_steps=0;self.frozen_steps=0;self.pending={}
        self.pages=pages;self.pipes=pipes;self.actor=actor;self.device=device
        self.groups=np.array_split(range(len(pages)),2)
        self.ready={}
        self.bank=torch.zeros(len(pages)*pages[0].slots,width,device=device)
        self.matches=0;self.seat_decisions=0;self.delivered=0

    def receive(self,group):
        segments=[]
        for i in group:
            if not self.pipes[i].poll(120):raise TimeoutError(f'Worker {i} did not return a page')
            msg=self.pipes[i].recv()
            if msg[0]=='error':raise RuntimeError(msg[1])
            n,nseg,nrows,matches,live,frozen,pending=msg
            self.live_steps+=live;self.frozen_steps+=frozen;self.pending[i]=pending
            self.ready[i]=n
            delivered=self.pages[i].drain(nseg,nrows)
            if any(self.pool.owners[s.slot_id]!=LIVE for s in delivered):raise RuntimeError('Frozen row reached learner')
            segments.extend(delivered)
            self.delivered+=nrows;self.matches+=matches
        return segments

    @torch.no_grad()
    def dispatch(self,group,version):
        rows=[(int(i),self.pages[i],self.ready.pop(i)) for i in group]
        obs={k:torch.from_numpy(np.concatenate([p.obs[k][:n] for _,p,n in rows])).to(self.device) for k in rows[0][1].obs}
        ids=np.concatenate([p.ids[:n]+i*p.slots for i,p,n in rows])
        reset=np.concatenate([p.reset[:n] for _,p,n in rows])
        self.pool.reset_games(np.unique(ids[reset]//2))
        owners=self.pool.owners[ids]
        index=torch.from_numpy(ids).to(self.device,dtype=torch.long)
        h=self.bank.index_select(0,index)
        h.masked_fill_(torch.from_numpy(reset).to(self.device)[:,None],0.)
        if self.pool.share:
            moves_gpu=torch.empty((len(ids),4),device=self.device,dtype=torch.long)
            prob_gpu=torch.empty(len(ids),device=self.device);nxt=torch.empty_like(h)
            for key in np.unique(owners):
                model=self.actor if key==LIVE else self.pool.models[key]
                idx=torch.from_numpy(np.flatnonzero(owners==key)).to(self.device)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device=='cuda'):
                    out=model.act({k:v.index_select(0,idx) for k,v in obs.items()},memory=h.index_select(0,idx))
                moves_gpu.index_copy_(0,idx,torch.cat((out.actions,out.end_freeze_mask[:,None]),1))
                prob_gpu.index_copy_(0,idx,out.log_prob.float());nxt.index_copy_(0,idx,out.memory.float())
                self.pool.decisions[int(key)]+=len(idx)
        else:
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device=='cuda'):
                out=self.actor.act(obs,memory=h)
            moves_gpu=torch.cat((out.actions,out.end_freeze_mask[:,None]),1)
            prob_gpu=out.log_prob.float();nxt=out.memory.float()
            self.pool.decisions[LIVE]+=len(ids)
        self.bank.index_copy_(0,index,nxt)
        moves=moves_gpu.cpu().numpy();prob=prob_gpu.cpu().numpy();hidden=nxt.cpu().numpy()
        offset=0
        for i,page,n in rows:
            section=slice(offset,offset+n);page.owner[:n]=owners[section];page.actions[:n]=moves[section]
            page.logs[:n]=prob[section];page.hidden[:n]=hidden[section];offset+=n
            self.pipes[i].send(version)
        self.seat_decisions+=offset

@contextmanager
def actors(workers,games,width,seed,start_version):
    # Spawn keeps workers independent of the coordinator's CUDA runtime.
    ctx=mp.get_context('spawn')
    example=selfplay_env.ParallelSelfPlayEnv().reset(seed=42,paired_tape=True)[0]
    pages=[];pipes=[];processes=[]
    try:
        for i in range(workers):
            p=Page(games,example,width);pages.append(p)
            parent,child=ctx.Pipe()
            proc=ctx.Process(target=worker,args=(i,games,width,p,child,seed,start_version))
            proc.start();child.close();processes.append(proc);pipes.append(parent)
        yield pages,pipes
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (BrokenPipeError,EOFError):pass
        for proc in processes:
            proc.join(3)
            if proc.is_alive():proc.terminate();proc.join()
        for pipe in pipes:pipe.close()
        for page in pages:page.close()
