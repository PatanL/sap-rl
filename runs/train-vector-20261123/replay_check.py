import json
import numpy as np
from vector_env import RecordedVector,allocate,selfplay_env
def main():
    root='results/tape';r=json.load(open(root+'/receipt.json'));n=r['games']
    d=RecordedVector(n,root);ex=selfplay_env.ParallelSelfPlayEnv().reset(seed=0,paired_tape=True)[0];o=allocate(n,ex)
    d.reset(np.arange(n),np.arange(n),o);count=np.zeros(n,'i4');ends=np.zeros(n,'i4')
    moves=np.zeros((n,2,3),'i4');masks=np.zeros((n,2),'i4')
    for i in range(max(r['steps'])+1):
        count+=o['active'].sum(axis=1).astype('i4')
        previous=o['terminated'].copy();d.step(moves,masks,o)
        ends+=o['terminated']&~previous
        assert not o['battle_resolved'][previous].any() and not o['rewards'][previous].any()
    assert np.array_equal(count,r['seat_decisions']) and (ends==1).all()
    print(json.dumps(dict(games=n,decisions=int(count.sum()),terminals=int(ends.sum()))))
if __name__=='__main__':main()
