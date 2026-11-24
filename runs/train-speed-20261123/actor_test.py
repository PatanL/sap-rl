import argparse,copy,json,torch
from fixture import get_fixture,PARENT
from equivalence import settings,verify_engine
from learner import prepare
from fast_actor import enable
from learner_fast import pack,plan
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args();device=a.device;settings(device=='cuda')
    m,*_=prepare(PARENT,None,'parent',verify_engine(),487);del _;m.to(device).eval().requires_grad_(False);n=enable(copy.deepcopy(m))
    tests=rows=0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
        for group in plan(get_fixture(512),256):
            packed,_=pack(group,device);obs={k:v.flatten(0,1) for k,v in packed.observations.items()}
            for deterministic in (False,True):
                h=m.initial_memory(len(obs['type_mask']),device)
                def call(model):
                    torch.distributions.Distribution.set_default_validate_args(model is m)
                    torch.manual_seed(8383);return model.act(obs,memory=h,deterministic=deterministic)
                x=call(m);y=call(n)
                for name in ('actions','end_freeze_mask','log_prob','memory'):assert torch.equal(getattr(x,name),getattr(y,name)),name
                tests+=1;rows+=len(h)
    print(json.dumps(dict(device=device,comparisons=tests,padded_rows=rows,bitwise_equal=True)))
if __name__=='__main__':main()
