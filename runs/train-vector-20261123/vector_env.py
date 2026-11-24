"""Vector ABI. Caller owns all output arrays. Terminal games are held until selectively reset."""
import ctypes,multiprocessing as mp,traceback
import numpy as np
from runtime import selfplay_env
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai_gym1.rl.end_freeze import step_with_end_freeze
from sapai_gym1.rl.distributed import _completed_match_result
from rust_g import install

def allocate(n,example,shared=False):
    def arr(shape,dtype):
        dtype=np.dtype(dtype)
        if not shared:return np.zeros(shape,dtype)
        raw=mp.RawArray(ctypes.c_ubyte,int(np.prod(shape))*dtype.itemsize)
        return np.ndarray(shape,dtype,buffer=raw)
    obs={k:arr((n,2,*v.shape),v.dtype) for k,v in example.items()}
    out={'obs':obs}
    for k,dt,two in [('rewards','f4',True),('battle_resolved','?',False),('battle_result','i1',False),
                       ('terminated','?',False),('winner','i1',False),('lives','i4',True),('wins','i4',True),('active','?',True)]:
        out[k]=arr((n,2) if two else (n,),dt)
    return out

def copy_output(src,dst):
    if src is dst:return
    for k,v in src.items():
        if k=='obs':
            for key,x in v.items():np.copyto(dst[k][key],x)
        else:np.copyto(dst[k],v)

def worker(pipe,lo,hi,out,actions,masks):
    try:
        selfplay_env.Battle=install()
        envs={i:selfplay_env.ParallelSelfPlayEnv() for i in range(lo,hi)}
        ready=set()
        def emit(i,obs,step=None):
            env=envs[i]
            for k,a in out['obs'].items():a[i,0]=obs[0][k];a[i,1]=obs[1][k]
            term=bool(step and step.terminated)
            out['rewards'][i]=step.rewards if step else (0,0)
            out['battle_resolved'][i]=bool(step and step.battle_resolved)
            out['battle_result'][i]=step.battle_result if step and step.battle_resolved else -1
            out['terminated'][i]=term
            out['winner'][i]=_completed_match_result(env.core) if term else -1
            out['lives'][i]=(env.core.agent.lives,env.core.opponent.lives)
            out['wins'][i]=(env.core.agent.wins,env.core.opponent.wins)
            out['active'][i]=(False,False) if term else env.active_players()
        pipe.send(None)
        while True:
            cmd=pipe.recv()
            if cmd is None:break
            if cmd[0]=='reset':
                for i,seed in cmd[1]:
                    ready.add(i);emit(i,envs[i].reset(seed=int(seed),paired_tape=True))
            else:
                for i,env in envs.items():
                    if i not in ready or out['terminated'][i]:
                        out['battle_resolved'][i]=False;out['battle_result'][i]=-1;out['rewards'][i]=0;continue
                    acts=[AutoregressiveAction(*map(int,actions[i,s])) if yes else None for s,yes in enumerate(out['active'][i])]
                    step=step_with_end_freeze(env,*acts,mask_ids=[int(masks[i,s]) if yes else 0 for s,yes in enumerate(out['active'][i])]).result
                    emit(i,step.observations,step)
            pipe.send(None)
    except BaseException:
        pipe.send(traceback.format_exc())
    finally:pipe.close()

class PythonVector:
    def __init__(self,n,out,workers):
        self.n=n
        # Ordinary caller arrays need a shared staging page across the Python worker processes.
        if not isinstance(out['active'].base,ctypes.Array):
            out=allocate(n,{k:v[0,0] for k,v in out['obs'].items()},True)
        self.out=out
        ctx=mp.get_context('fork');self.procs=[];self.pipes=[];self.parts=np.array_split(np.arange(n),min(workers,n))
        self.actions=np.frombuffer(ctx.RawArray('i',n*6),dtype='i4').reshape(n,2,3)
        self.masks=np.frombuffer(ctx.RawArray('i',n*2),dtype='i4').reshape(n,2)
        for ids in self.parts:
            a,b=ctx.Pipe();p=ctx.Process(target=worker,args=(b,int(ids[0]),int(ids[-1]+1),out,self.actions,self.masks))
            p.start();b.close();self.pipes.append(a);self.procs.append(p)
        self.wait()
    def wait(self):
        for p in self.pipes:
            r=p.recv()
            if r is not None:raise RuntimeError(r)
    def reset(self,indices,seeds,out):
        if len(indices)!=len(seeds) or len(set(indices))!=len(indices) or any(i<0 or i>=self.n for i in indices):raise ValueError('Invalid reset selection')
        for part,p in zip(self.parts,self.pipes):p.send(('reset',[(int(i),int(s)) for i,s in zip(indices,seeds) if part[0]<=i<=part[-1]]))
        self.wait();copy_output(self.out,out)
    def step(self,actions,freeze,out):
        if actions.shape!=(self.n,2,3) or freeze.shape!=(self.n,2):raise ValueError('Action dimensions')
        np.copyto(self.actions,actions,casting='same_kind');np.copyto(self.masks,freeze,casting='same_kind')
        for p in self.pipes:p.send(('step',))
        self.wait();copy_output(self.out,out)
    def close(self):
        for p in self.pipes:
            try:p.send(None)
            except BrokenPipeError:pass
        for p in self.procs:p.join(3)
        for p in self.procs:
            if p.is_alive():p.terminate();p.join()
        for p in self.pipes:p.close()

class RecordedVector:
    """Copies actual contiguous steps of complete games. Actions do not affect taped outcomes."""
    def __init__(self,n,tape):
        from pathlib import Path
        self.n=n;p=Path(tape);self.data={f.stem:np.load(f,mmap_mode='r') for f in p.glob('*.npy')}
        self.starts=self.data.pop('starts');self.ends=self.data.pop('ends');self.pos=np.zeros(n,'i4');self.last=np.zeros(n,'i4');self.ready=np.zeros(n,'?')
    def transfer(self,ids,out):
        ix=self.pos[ids]
        for k,buf in out['obs'].items():buf[ids]=self.data['obs_'+k][ix]
        for k,buf in out.items():
            if k!='obs':buf[ids]=self.data[k][ix]
    def reset(self,indices,seeds,out):
        ix=np.asarray(indices,dtype=np.intp);self.ready[ix]=True;ids=np.asarray(seeds,dtype='uint64')%len(self.starts)
        self.pos[ix]=self.starts[ids];self.last[ix]=self.ends[ids];self.transfer(ix,out)
    def step(self,actions,freeze,out):
        held=(self.pos==self.last)|~self.ready
        self.pos=np.minimum(self.pos+1,self.last)
        self.transfer(slice(None) if self.ready.all() else np.flatnonzero(self.ready),out)
        if held.any():
            out['battle_resolved'][held]=False;out['battle_result'][held]=-1;out['rewards'][held]=0
    def close(self):pass
