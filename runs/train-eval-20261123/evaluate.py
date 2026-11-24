"""Paired public self-play scoring on one immutable v31 artifact."""
import argparse,hashlib,json,os,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from config import ENGINE,REV,PAYLOAD,PARENT_HASH,PANEL
sys.path.insert(0,'/sap/runs/train-v31-prep-20261123')
from runtime import verify_engine
from sapai_gym1.rl.model import EntityTransformerConfig,EntityTransformerPolicy
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai_gym1.rl.distributed import _completed_match_result
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.joint_action import JointAction,step_decisions
from sapai_gym1.rl.observation_contract import OBSERVATION_REVISION
from sapai_gym1.rl.causal_transport import validate_checkpoint_context
import datetime
def sha(p):
    with Path(p).open('rb') as fp:return hashlib.file_digest(fp,'sha256').hexdigest()
def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def load(path,device):
    digest=sha(path)
    ck=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    transfer=digest in {PARENT_HASH,*[v[1] for v in PANEL.values()]}
    if not transfer and (ck.get('simulator_revision')!=REV or ck.get('engine_payload_sha256')!=PAYLOAD or ck.get('observation_revision')!=OBSERVATION_REVISION):
        raise ValueError('Checkpoint not stamped for the scoring engine or an exact allowed baseline')
    validate_checkpoint_context(ck,allow_observation_transfer=transfer)
    model=EntityTransformerPolicy(EntityTransformerConfig(**ck['config']))
    model.load_state_dict(ck['model'],strict=True)
    return model.eval().requires_grad_(False).to(device),dict(sha256=digest,update=ck.get('update'),source_revision=ck.get('simulator_revision'),joint_freeze=model.config.end_freeze)
def world_ids(mode,count,candidate,opponent):
    code={'screen':1,'confirm':2,'panel':3,'sanity':4}[mode]
    salt=f'{candidate}/{opponent}' if mode=='confirm' else 'fixed'
    seeds=[(code<<56)|int.from_bytes(hashlib.sha256(f'sap-v31-ranking-v1/{mode}/{salt}/{i}'.encode()).digest()[:7],'big') for i in range(count)]
    assert len(set(seeds))==count
    return seeds
def interval(pairs):
    x=np.asarray(pairs,dtype=float)
    rng=np.random.default_rng(20261123)
    boots=x[rng.integers(0,len(x),size=(20000,len(x)))].mean(1)
    return [float(v) for v in np.quantile(boots,[.025,.975])]
@torch.inference_mode()
def play(seeds,models,device,batch):
    tasks=[(i,s,side) for i,s in enumerate(seeds) for side in (0,1)]
    records=[]
    for start in range(0,len(tasks),batch):
        group=tasks[start:start+batch];env=[];obs=[];done=set()
        memories=[m.initial_memory(len(group)*2,device) for m in models]
        for _,seed,_ in group:
            e=ParallelSelfPlayEnv();obs.append(e.reset(seed=seed,paired_tape=True));env.append(e)
        for tick in range(2500):
            if len(done)==len(group):break
            req=[[],[]];actions=[[None,None] for _ in group]
            for k,(_,_,seat) in enumerate(group):
                if k in done:continue
                for s,active in enumerate(env[k].active_players()):
                    if active:req[int(s!=seat)].append((k,s))
            for j,requests in enumerate(req):
                if not requests:continue
                inp={name:torch.from_numpy(np.stack([obs[k][s][name] for k,s in requests])).to(device) for name in obs[0][0]}
                indices=torch.tensor([k*2+s for k,s in requests],device=device)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                    out=models[j].act(inp,memory=memories[j].index_select(0,indices),deterministic=True)
                memories[j].index_copy_(0,indices,out.memory.float())
                moves=out.actions.cpu().tolist()
                masks=[None]*len(requests) if out.end_freeze_mask is None else out.end_freeze_mask.cpu().tolist()
                for (k,s),move,mask in zip(requests,moves,masks):actions[k][s]=JointAction(AutoregressiveAction(*move),mask)
            for k,(i,seed,cs) in enumerate(group):
                if k in done:continue
                step=step_decisions(env[k],actions[k]).result;obs[k]=step.observations
                if step.terminated:
                    winner=_completed_match_result(env[k].core);done.add(k)
                    records.append(dict(pair=i,seed=seed,seat=cs,score=.5 if winner==2 else float(winner==cs),last_battle=env[k].core.turn-1))
        if len(done)!=len(group):raise RuntimeError('Incomplete match; no score emitted')
    return sorted(records,key=lambda r:(r['pair'],r['seat']))
def main():
    p=argparse.ArgumentParser()
    for a in ('candidate','opponent','out'):p.add_argument(a,type=Path)
    p.add_argument('--mode',choices=['screen','confirm','panel','sanity'],required=True)
    p.add_argument('--pairs',type=int,required=True);p.add_argument('--batch',type=int,default=128)
    a=p.parse_args()
    if a.pairs<2 or not 2<=a.batch<=256:p.error('At least two pairs, batch 2..256')
    if a.out.exists():raise FileExistsError(a.out)
    meta=verify_engine()
    if meta['payload_sha256']!=PAYLOAD or meta['simulator_revision']!=REV:raise ValueError('Wrong engine')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda'
    _,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction((2*1024**3)/total)
    m,c=load(a.candidate,device);n,o=load(a.opponent,device);started=time.monotonic()
    seeds=world_ids(a.mode,a.pairs,c['sha256'],o['sha256']);rows=play(seeds,[m,n],device,a.batch)
    assert [(r['seed'],r['seat']) for r in rows]==[(s,seat) for s in seeds for seat in (0,1)]
    pairs=np.array([r['score'] for r in rows]).reshape(-1,2).mean(1)
    atomic(a.out,dict(protocol='paired-greedy-v31-v1',mode=a.mode,candidate=c,opponent=o,engine=PAYLOAD,revision=REV,
        score=float(pairs.mean()),ci95=interval(pairs),pairs=len(pairs),games=len(rows),wdl=[sum(r['score']==v for r in rows) for v in (1,.5,0)],
        seconds=time.monotonic()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved(),records=rows,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
if __name__=='__main__':main()
