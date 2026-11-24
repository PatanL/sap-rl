"""Measure the Rust-only path up to the missing pre-battle payload; never emit an inferred board index."""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from evaluate import load
from rust_adapter import RustVector,PIN

from runtime import verify_engine

def public_signature(pet):
    return (pet['id'],pet['attack'],pet['health'],pet['level'],pet['status'])

def signatures(o,g,side,items,statuses):
    # At settlement each seat sees the last battle *opponent*, front-first.
    view=(g,1-side,-1)
    mask=o['obs']['opponent_history_mask'][view]
    species=o['obs']['opponent_history_item_ids'][view]
    perk=o['obs']['opponent_history_status_ids'][view]
    stats=o['obs']['opponent_history_numeric'][view]
    return [(items[int(species[j])],round(float(stats[j,0])*50),round(float(stats[j,1])*50),
             round(float(stats[j,2])*3),statuses[int(perk[j])]) for j in np.flatnonzero(mask)]

@torch.inference_mode()
def main(args):
    verify_engine();torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
    model,meta=load(args.checkpoint,'cuda')
    ref={}
    for line in args.reference.open():
        row=json.loads(line);key=(int(row['match'].rsplit('-',1)[1]),row['turn'])
        if key in ref:raise ValueError('Duplicate archived battle')
        ref[key]=row
    seeds=[(20261123*1000003+i*7919)%2**31 for i in range(1000)]
    names=json.loads((PIN/'data/turtle_v31_env.json').read_text())
    items={v:k for k,v in names['itemIds'].items()};statuses={v:k for k,v in names['statusIds'].items()}
    ticks=collections.Counter();battle_count=0;decisions=0;matched=0;missing=0;finished=0;seen=set()
    out=args.output.with_suffix('.rehearsal-jsonl')
    with out.open('x') as archive:
        begin=time.perf_counter()
        for start in range(0,len(seeds),128):
            worlds=seeds[start:start+128];n=len(worlds);env=RustVector(n,10)
            try:
                o=env.output;env.reset(np.arange(n),worlds,o)
                memory=model.initial_memory(n*2,'cuda');turn=np.ones(n,dtype=np.int32)
                moves=np.zeros((n,2,3),'i8');freeze=np.zeros((n,2),'i8')
                while np.any(o['active']):
                    t=time.perf_counter();ids=np.flatnonzero(o['active'].ravel())
                    ix=torch.from_numpy(ids).to('cuda')
                    ob={k:torch.from_numpy(v.reshape(-1,*v.shape[2:])[ids]).to('cuda') for k,v in o['obs'].items()}
                    with torch.autocast('cuda',dtype=torch.bfloat16):a=model.act(ob,memory=memory.index_select(0,ix),deterministic=True)
                    memory.index_copy_(0,ix,a.memory.float());moves.reshape(-1,3)[ids]=a.actions.cpu().numpy();freeze.ravel()[ids]=a.end_freeze_mask.cpu().numpy()
                    ticks['act_s']+=time.perf_counter()-t;t=time.perf_counter()
                    env.step(moves,freeze,o);decisions+=len(ids);ticks['step_s']+=time.perf_counter()-t;t=time.perf_counter()
                    for g in np.flatnonzero(o['battle_resolved']):
                        key=(worlds[g],int(turn[g]));turn[g]+=1;battle_count+=1
                        row=ref.get(key)
                        if row is None:missing+=1;continue
                        seen.add(key)
                        equal=all(signatures(o,g,s,items,statuses)==[public_signature(p) for p in row['teams'][s]] for s in (0,1))
                        matched+=int(equal)
                        # Actual full JSON encoding/writing, but hidden values come from the prior independent archive.
                        archive.write(json.dumps(row)+'\n')
                    ticks['verify_projection_and_encode_s']+=time.perf_counter()-t
                assert o['terminated'].all();finished+=n
            finally:env.close()
        archive.flush();total=time.perf_counter()-begin
    result=dict(checkpoint=meta,batch=128,matches=finished,battles=battle_count,decisions=decisions,
        matched_visible_boards_both_seats=matched,missing_reference=missing,unused_reference=len(ref)-len(seen),wall=total,
        phases=ticks,includes_reference_projection_check=True,hidden_fields_from_archive=True,
        outputs_measured='Rust rollout and serialization rehearsal only; no new independent ghost index')
    args.output.write_text(json.dumps(result,indent=2))
    if matched!=battle_count or len(seen)!=len(ref):
        print('Public-board correspondence incomplete. Treat as a timing workload only.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('checkpoint','reference','output'):p.add_argument(k,type=Path)
    main(p.parse_args())
