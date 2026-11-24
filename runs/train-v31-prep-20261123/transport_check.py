"""CPU check for terminal/continuing segments, slot identity and ownership across page reuse."""
import json
import numpy as np
from runtime import verify_engine,selfplay_env
from wide import Page
from sapai import Pet,Team
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai_gym1.rl.end_freeze import step_with_end_freeze
def check():
    verify_engine()
    rows=[]
    for lives in (1,2):
        env=selfplay_env.ParallelSelfPlayEnv();env.reset(seed=993,paired_tape=True);env.core.turn=5
        for pl,name in [(env.core.agent,'fish'),(env.core.opponent,'ant')]:
            pl.team=Team([Pet(name)]);pl._connect_objects();pl.gold=0;pl.turn=5
        env.core.agent.team[0].pet._attack=50;env.core.agent.team[0].pet._health=50
        env.core.opponent.lives=lives
        env.core._agent_actions_cache=env.core._opponent_actions_cache=None
        obs=env.observations();end=AutoregressiveAction(0)
        step=step_with_end_freeze(env,end,end,mask_ids=(0,0)).result
        assert step.battle_resolved and step.terminated==(lives==1)
        page=Page(1,obs[0],3);initial=np.array([1,2,3],'f4')
        for s in (0,1):
            a=np.asarray([*end.as_tuple(),0],dtype='i8')
            page.write_turn([(obs[s],a,-.25,19)],s,s,initial,step.observations[s],step.rewards[s],step.terminated,int(step.battle_result!=s),17+s)
        segments=page.drain(2,2)
        page.initial.fill(0);page.values.fill(0);page.versions.fill(-1)
        for s,segment in enumerate(segments):
            assert segment.slot_id==17+s and segment.terminal==step.terminated and segment.battle_outcome==s
            assert segment.initial_memory.tolist()==[1.,2.,3.] and segment.behavior_versions.tolist()==[19]
            assert np.isclose(segment.discounts[-1],0 if step.terminated else .99)
            assert np.isclose(segment.rewards[-1],step.rewards[s]) and segment.behavior_log_probs.tolist()==[-.25]
            for k,v in step.observations[s].items():assert np.array_equal(segment.bootstrap_observation[k],v)
        page.close();rows.append(dict(terminal=step.terminated,rewards=step.rewards,discounts=[float(s.discounts[-1]) for s in segments]))
    return rows
if __name__=='__main__':print(json.dumps(check()))
