import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import numpy as np
import pytest
import torch
from sapai import Pet,Team,Shop
from sapai_gym1.rl.battle_equity import BattleEquityConfig,BattleEquityEstimator,resolve_config,reward_table
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.joint_action import JointAction,step_decisions
from sapai_gym1.rl.action_schema import ActionType,AutoregressiveAction


def reference_class():
    path=Path('/reference/sapai-gym/sapai_gym1/rl/selfplay_env.py')
    if not path.exists():pytest.skip('Explicit frozen reference mount required')
    name='sapai_gym1.rl._equity_reference'
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module.ParallelSelfPlayEnv


def same_obs(a,b):
    assert a.keys()==b.keys()
    for key in a:assert np.array_equal(a[key],b[key]),key


def same_rng(a,b):
    for left,right in ((a.core.rs_random,b.core.rs_random),(a.core.agent.shop.rs,b.core.agent.shop.rs),
                       (a.core.opponent.shop.rs,b.core.opponent.shop.rs)):
        x,y=left.get_state(),right.get_state()
        assert x[0]==y[0] and np.array_equal(x[1],y[1]) and x[2:]==y[2:]


@pytest.mark.parametrize('samples',[1,2])
@pytest.mark.parametrize('seed',[71957001,71957002,71957003])
def test_complete_fixed_action_trajectory_exact_against_frozen_reference(samples,seed):
    reference=reference_class()();config=BattleEquityConfig(samples,71957901 if samples==2 else None)
    candidate=ParallelSelfPlayEnv(battle_equity_config=config,battle_equity_owner='fixture')
    a=reference.reset(seed=seed);b=candidate.reset(seed=seed);rng=np.random.default_rng(seed+1)
    for _ in range(2500):
        for side in (0,1):same_obs(a[side],b[side])
        assert candidate.active_players()==reference.active_players()
        actions=[]
        for side,active in enumerate(reference.active_players()):
            if not active:actions.append(None);continue
            t=int(rng.choice(np.flatnonzero(a[side]['type_mask'])))
            s=int(rng.choice(np.flatnonzero(a[side]['source_mask'][t])))
            u=int(rng.choice(np.flatnonzero(a[side]['target_mask'][t,s])))
            actions.append(JointAction(AutoregressiveAction(t,s,u),0))
        left=step_decisions(reference,actions).result;right=step_decisions(candidate,actions).result
        assert (left.battle_resolved,left.battle_result,left.terminated)==(right.battle_resolved,right.battle_result,right.terminated)
        same_rng(reference,candidate)
        assert right.rewards[1]==-right.rewards[0]
        if samples==1 or not left.battle_resolved:assert left.rewards==right.rewards
        else:
            info=candidate._battle_equity.last
            terminal_bonus=left.rewards[0]-info['actualComponent']
            assert right.rewards[0]==info['meanComponent']+terminal_bonus
            assert info['actualOutcome']==left.battle_result
        a,b=left.observations,right.observations
        if left.terminated:break
    else:raise AssertionError('Did not finish full reference match')
    for side in (0,1):same_obs(a[side],b[side])


def test_m1_does_not_prepare_or_sample_extra_battles(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('M1 has no estimator work')
    monkeypatch.setattr(BattleEquityEstimator,'prepare',forbidden)
    env=ParallelSelfPlayEnv();env.reset(seed=5)
    for p in (env.core.agent,env.core.opponent):p.gold=0
    result=step_decisions(env,[JointAction(AutoregressiveAction(ActionType.END_TURN),0)]*2).result
    assert result.battle_resolved and env._battle_equity is None


def test_m2_includes_actual_result_not_two_new_samples(monkeypatch):
    import sapai_gym1.rl.battle_equity as module
    env=ParallelSelfPlayEnv();env.reset(seed=9);env.core.turn=7
    estimator=BattleEquityEstimator(BattleEquityConfig(2,31),'owner');estimator.reset()
    calls=[]
    class FakeBattle:
        def __init__(self,*args,**kwargs):calls.append(kwargs)
        def battle(self):return 2
    monkeypatch.setattr(module,'Battle',FakeBattle)
    prepared=estimator.prepare(env.core)
    assert estimator.estimate(prepared,0,.1)==.05
    assert len(calls)==1 and calls[0]['fast_training'] is True
    assert estimator.last['actualOutcome']==0 and estimator.last['extraOutcome']==2


def test_rng_state_owner_reset_clone_redeterminize_and_global_isolation():
    config=BattleEquityConfig(2,51)
    a=ParallelSelfPlayEnv(battle_equity_config=config,battle_equity_owner='a')
    b=ParallelSelfPlayEnv();a.reset(seed=100);b.reset(seed=100)
    py=random.getstate();global_np=np.random.get_state();torch_rng=torch.get_rng_state().clone()
    prepared=a._battle_equity.prepare(a.core)
    assert a._battle_equity.estimate(prepared,2,0.)==0. # actual/extra empty-team draws
    saved=json.loads(json.dumps(a._battle_equity.state_dict()))
    other=BattleEquityEstimator(config,'a');other.restore_state(saved)
    assert other._rng.randrange(2**32-1)==a._battle_equity._rng.randrange(2**32-1)
    clone=a.clone();assert clone._battle_equity is not a._battle_equity
    assert clone._battle_equity._rng.randrange(2**32-1)==a._battle_equity._rng.randrange(2**32-1)
    a.redeterminize(321);b.redeterminize(321);same_rng(a,b)
    assert random.getstate()==py
    assert torch.equal(torch.get_rng_state(),torch_rng)
    now=np.random.get_state();assert now[0]==global_np[0] and np.array_equal(now[1],global_np[1]) and now[2:]==global_np[2:]
    with pytest.raises(ValueError):BattleEquityEstimator(config,'foreign').restore_state(saved)
    previous=a._battle_equity.state_dict();a.reset(seed=100)
    assert a._battle_equity.generation==previous['generation']+1 and a._battle_equity.state_dict()['rng']!=previous['rng']


@pytest.mark.parametrize('turn',[1,2,7,25])
@pytest.mark.parametrize('lives',[(6,6),(5,6),(6,5),(1,1)])
def test_reward_table_uses_actual_outcome_rule_including_healing(turn,lives):
    env=ParallelSelfPlayEnv();env.reset(seed=77);env.core.turn=turn
    env.core.agent.lives,env.core.opponent.lives=lives;table=reward_table(env.core)
    for outcome in (0,1,2):
        core=copy.deepcopy(env.core);core._player_fight_outcome(outcome)
        assert table[outcome]==.1*((core.agent.lives-lives[0])-(core.opponent.lives-lives[1]))


def test_explicit_checkpoint_transfer_and_inherited_resume():
    assert resolve_config({})==BattleEquityConfig()
    with pytest.raises(ValueError):resolve_config({},samples=2,seed=4)
    config=resolve_config({},samples=2,seed=4,allow_transfer=True)
    ckpt={'training_semantics':{'reward_estimator':config.to_dict()}}
    assert resolve_config(ckpt)==config
    with pytest.raises(ValueError):resolve_config(ckpt,samples=1)
    with pytest.raises(ValueError):resolve_config(ckpt,seed=5)
    assert resolve_config(ckpt,samples=1,allow_transfer=True)==BattleEquityConfig()


@pytest.mark.parametrize('value',[{'samples':2}, {'samples':True,'seed':None,'schema':'bad'},
    {'samples':2,'seed':None,'schema':'sampled-or-mc-life-swing-v1-20260909'},
    {'samples':1,'seed':4,'schema':'sampled-or-mc-life-swing-v1-20260909'}])
def test_invalid_estimator_contract_rejected(value):
    with pytest.raises(ValueError):BattleEquityConfig.parse(value)
