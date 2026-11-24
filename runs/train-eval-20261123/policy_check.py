"""Read-only greedy policy drift on archived public states; no optimization and no GPU."""
from collections import Counter
import json
from pathlib import Path
from evaluate import load,verify_engine,torch,np
from config import PARENT,LIVE
@torch.inference_mode()
def main():
    verify_engine();torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    original,_=load(Path('/sap')/PARENT,'cpu')
    candidate,_=load(Path('/sap')/LIVE/'update-026022.pt','cpu')
    examples=[json.load(open(p))['publicHistory'] for p in sorted(Path('/sap/runs/population-league-25772-v1/inputs').glob('game-*-seat-*.json'))]
    policies=(original,candidate);mem=[p.initial_memory(len(examples),'cpu') for p in policies]
    types=[Counter(),Counter()];pairs=Counter();moves=0;same=0
    for t in range(max(map(len,examples))):
        ix=[i for i,seq in enumerate(examples) if t<len(seq)]
        obs={k:torch.from_numpy(np.stack([np.asarray(examples[i][t][k]['data'],dtype=examples[i][t][k]['dtype']) for i in ix])) for k in examples[ix[0]][t]}
        outputs=[]
        for j,policy in enumerate(policies):
            out=policy.act(obs,memory=mem[j][ix],deterministic=True)
            mem[j][ix]=out.memory;outputs.append(torch.cat((out.actions,out.end_freeze_mask[:,None]),dim=1))
            types[j].update(map(int,out.actions[:,0].tolist()))
        same+=int(outputs[0].eq(outputs[1]).all(-1).sum());moves+=len(ix)
        pairs.update(zip(outputs[0][:,0].tolist(),outputs[1][:,0].tolist()))
    path=Path('public-state-drift.json')
    path.write_text(json.dumps(dict(states=moves,identical_joint_actions=same,parent_types=types[0],candidate_types=types[1],transitions=[[list(k),n] for k,n in pairs.most_common()]),indent=2))
    print(moves,same,types)
if __name__=='__main__':main()
