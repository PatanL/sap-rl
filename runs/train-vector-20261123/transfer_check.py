"""Strict CPU initializer transfer across c, f and g. Never modify the original checkpoint."""
import os,sys,subprocess,json
from pathlib import Path
def check(label):
    os.environ['SAP_ENGINE_ROOT']=f'/sap/runs/replay-oracle-20261122/packaged/v31-20261123{label}'
    from runtime import verify_engine
    from learner import prepare,optimizer_for,read_checkpoint
    import torch
    torch.set_num_threads(1)
    ck=Path('/sap/checkpoints/update-025772.pt');info=verify_engine()
    policy,st,rates,sem,update,rows,primitives=prepare(ck,None,'parent',info,6163)
    opt=optimizer_for(policy,st,rates,1);source=read_checkpoint(ck)
    assert all(torch.equal(t,source['model'][k]) for k,t in policy.state_dict().items())
    restored=opt.state_dict();previous=source['optimizer']
    assert restored['param_groups']==previous['param_groups']
    assert restored['state'].keys()==previous['state'].keys()
    for key,state in restored['state'].items():
        for n,v in state.items():
            other=previous['state'][key][n];assert torch.equal(v,other) if torch.is_tensor(v) else v==other
    print(json.dumps(dict(package=label,revision=info['simulator_revision'],gym=info['upstream_gym_sha256'],tensors=len(source['model']),optimizer_states=len(st['state']),rates=rates,update=update)))
if __name__=='__main__':
    if len(sys.argv)>1:check(sys.argv[1])
    else:
        out=[json.loads(subprocess.check_output([sys.executable,__file__,label],text=True)) for label in 'cfg']
        Path('results/transfer.json').write_text(json.dumps(out,indent=2))
