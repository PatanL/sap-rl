"""Check one shard and multiple shards against independent scalar engines, including terminal hold and partial reset."""
from vector_env import *
import json
def main():
    n=6
    refs=[selfplay_env.ParallelSelfPlayEnv() for i in range(n)]
    from sapai_gym1.rl.action_schema import FlatActionCodec
    rng=np.random.default_rng(338)
    seeds=np.arange(n)+736;current=[e.reset(seed=int(s),paired_tape=True) for e,s in zip(refs,seeds)]
    o=allocate(n,current[0][0]);env=PythonVector(n,o,3)
    try:
        env.reset(np.arange(n),seeds,o);done=set();ticks=0;battles=0;steps=0
        while len(done)<n and ticks<2500:
            actions=np.zeros((n,2,3),'i4');masks=np.zeros((n,2),'i4');expected={}
            for i,e in enumerate(refs):
                for side in (0,1):
                    for k,v in current[i][side].items():np.testing.assert_array_equal(o['obs'][k][i,side],v)
                if i in done:continue
                moves=[]
                for s,yes in enumerate(e.active_players()):
                    if yes:
                        a=FlatActionCodec.decode(int(rng.choice(tuple(e.core._avail_actions(s)))))
                        actions[i,s]=[a.action_type,a.source,a.target];moves.append(a);steps+=1
                    else:moves.append(None)
                st=step_with_end_freeze(e,*moves,mask_ids=masks[i]).result;expected[i]=st;current[i]=st.observations
            env.step(actions,masks,o);ticks+=1
            for i,st in expected.items():
                assert st.terminated==o['terminated'][i] and st.battle_resolved==o['battle_resolved'][i]
                np.testing.assert_array_equal(o['rewards'][i],np.asarray(st.rewards,dtype='f4'))
                np.testing.assert_array_equal(o['active'][i],(False,False) if st.terminated else refs[i].active_players())
                assert o['winner'][i]==(_completed(refs[i]) if st.terminated else -1)
                assert list(o['lives'][i])==[refs[i].core.agent.lives,refs[i].core.opponent.lives]
                assert list(o['wins'][i])==[refs[i].core.agent.wins,refs[i].core.opponent.wins]
                if st.battle_resolved:assert o['battle_result'][i]==st.battle_result;battles+=1
                if st.terminated:done.add(i)
        assert len(done)==n
        final={k:v.copy() for k,v in o['obs'].items()};env.step(actions,masks,o)
        assert not o['battle_resolved'].any() and not o['rewards'].any()
        env.reset([0],[56986],o)
        for k,v in final.items():np.testing.assert_array_equal(o['obs'][k][1:],v[1:])
        assert not o['terminated'][0] and o['terminated'][1:].all()
        print(json.dumps(dict(matches=len(done),ticks=ticks,seat_steps=steps,battles=battles,partial_reset=True,terminal_hold=True)))
    finally:env.close()
def _completed(e):
    from sapai_gym1.rl.distributed import _completed_match_result
    return _completed_match_result(e.core)
if __name__=='__main__':main()
