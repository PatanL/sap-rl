import hashlib,json,os,sys
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch
ROOT=Path(os.environ['SAP_ENGINE_ROOT']).resolve()
sys.path[:0]=[str(ROOT/'sapai'),str(ROOT/'sapai-gym')]
from sapai_gym1.rl.model import EntityTransformerPolicy,EntityTransformerConfig
from sapai_gym1.rl.training import Transition,TurnSegment
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.joint_action import JointAction,step_decisions
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai.turtle_pack import SIMULATOR_REVISION
assert Path(sys.modules['sapai'].__file__).resolve().is_relative_to(ROOT)
assert Path(sys.modules['sapai_gym1'].__file__).resolve().is_relative_to(ROOT)
PARENT='0669109e9b6ee2d4f0309a3fec83039fc8a7d74f74e3c9f629d2f7026e515d49'
SIZES={'small':(80,2,4,96,32),'mid':(128,4,4,192,48),'medium':(224,5,8,320,80),'parent':(384,6,8,512,128)}
BATCHES=[1,4,16,64,256,1024]
def setup_gpu():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,float(os.environ.get('BENCH_GPU_GIB','1.5'))*1024**3/torch.cuda.get_device_properties(0).total_memory))
def model_for(name,seed=502,random_parent=False):
    torch.manual_seed(seed)
    path=Path('/weights/parent.pt')
    ck=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    d,l,h,m,o=SIZES[name]
    cfg=replace(EntityTransformerConfig(**ck['config']),d_model=d,num_layers=l,nhead=h,dim_feedforward=4*d,memory_dim=m,opponent_history_dim=o)
    model=EntityTransformerPolicy(cfg)
    if name=='parent' and not random_parent:
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==PARENT
        model.load_state_dict(ck['model'])
    return model
def save(file,data):
    file=Path(file);tmp=file.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(file)
def examples():
    turns=[]
    for f in sorted(Path('/examples').glob('game-*-seat-*.json')):
        r=json.loads(f.read_text())
        acts=[p[r['side']] for p in r['pairs'] if p[r['side']] is not None]
        obs=[{k:np.asarray(v['data'],dtype=v['dtype']) for k,v in row.items()} for row in r['publicHistory']]
        assert len(acts)==len(obs)
        start=0
        for i,a in enumerate(acts):
            if a['primitive'][0]==0:
                if i+1<len(obs):turns.append((obs[start:i+1],acts[start:i+1],obs[i+1]))
                start=i+1
    return turns
