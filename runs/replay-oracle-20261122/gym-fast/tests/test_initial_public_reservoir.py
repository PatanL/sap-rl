"""CPU routing/privacy tests; no claim about forecaster or playing strength."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sapai import Pet,Team
from sapai_gym1.rl import ParallelSelfPlayEnv,AutoregressiveAction,ActionType
from sapai_gym1.rl.opponent_belief import OpponentBoardBelief,OpponentBeliefMix
from sapai_gym1.rl.opponent_reservoir import MatchedOpponentTrajectory,OpponentTrajectoryReservoir
from sapai_gym1.rl.public_battle_board import public_records
from sapai_gym1.rl.public_opponent_model import public_state_from_one_step_reservoir,PublicOpponentSupportMissing


class UniformBelief(OpponentBoardBelief):
    def __init__(self):pass
    def reservoir_probabilities(self,teams,**kwargs):return np.ones(len(teams))/len(teams)


def environment(side=0):
    env=ParallelSelfPlayEnv();env.reset(seed=715001)
    c=env.core;c.turn=c.agent.turn=c.opponent.turn=2;c.agent.gold=c.opponent.gold=0
    enemy=Pet('fish');enemy._attack=2;enemy._health=3;enemy.level=2;enemy.sell_value=7
    enemy._until_end_of_battle_attack_buff=3;enemy._until_end_of_battle_health_buff=4
    enemy.ability_counter=3;enemy.friends_hurt_counter=2
    own=Team(['pet-ant'])
    c.previous_agent_state=own if side==0 else Team([enemy])
    c.previous_opponent_state=Team([enemy]) if side==0 else own
    c.previous_previous_agent_state=c.previous_previous_opponent_state=None
    c._agent_actions_cache=c._opponent_actions_cache=None;c._avail_actions(0);c._avail_actions(1)
    return env


class RecordingReservoir:
    def __init__(self):
        pet=Pet('fish');pet._attack=6;pet._health=7;pet.level=2;pet.sell_value=9
        pet._until_end_of_battle_attack_buff=4;pet._until_end_of_battle_health_buff=4;pet.ability_counter=8
        self.future=Team([pet]);self.requests=[]
    def match(self,observed,**kwargs):
        self.requests.append((observed,kwargs))
        return [MatchedOpponentTrajectory('source-test-next',0.,(self.future,),source_index='test')]
    def opening_prior(self,**kwargs):
        self.requests.append((None,kwargs))
        return [MatchedOpponentTrajectory('source-test-opening',0.,(self.future,),source_index='test')]


def reservoir_only(state):
    state.enable_learned_opponent_belief(OpponentBeliefMix(0,0,1))
    state.set_opponent_board_belief(UniformBelief())
    return state


def old_builder():
    path=Path('/old-selector.py')
    if not path.exists():pytest.skip('Explicit immutable old builder mount required')
    spec=importlib.util.spec_from_file_location('old_selector_routing_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.public_state


@pytest.mark.parametrize('side',[0,1])
def test_old_actual_builder_duplicates_persistence_on_first_battle(side):
    env=environment(side);reservoir=RecordingReservoir()
    state=reservoir_only(old_builder()(env,side,reservoir))
    observed=public_records(state.core.previous_opponent_state)
    state.step(AutoregressiveAction(ActionType.END_TURN),np.random.default_rng(123))
    assert reservoir.requests==[]
    assert public_records(state.core.previous_opponent_state)==observed
    state.core.agent.gold=0;state.core._agent_actions_cache=None
    state.step(AutoregressiveAction(ActionType.END_TURN),np.random.default_rng(124))
    assert len(reservoir.requests)==1 and reservoir.requests[0][1]['horizon']==1


@pytest.mark.parametrize('side',[0,1])
def test_first_battle_uses_current_target_without_changing_observed_history(side):
    env=environment(side);reservoir=RecordingReservoir()
    before=env.observations()[side]
    state,meta=public_state_from_one_step_reservoir(env.core,side,reservoir)
    assert len(reservoir.requests)==1 and reservoir.requests[0][1]['current_turn']==2
    assert reservoir.requests[0][1]['horizon']==1 and meta['retrievalHorizon']==1
    assert meta['candidateIds']==['source-test-next']
    for key,value in before.items():np.testing.assert_array_equal(value,state.observation()[key],err_msg=key)
    reservoir_only(state).step(AutoregressiveAction(ActionType.END_TURN),np.random.default_rng(123))
    assert state.core.previous_opponent_state[0].pet.attack==10
    assert state.core.previous_opponent_state[0].pet.health==11
    assert 'source-test-next' in state.selected_opponent_trajectory_ids


def test_matching_inputs_and_hypotheses_are_public_and_owned():
    env=environment();reservoir=RecordingReservoir()
    state,_=public_state_from_one_step_reservoir(env.core,0,reservoir)
    observed=reservoir.requests[0][0]
    for pet in (observed[0].pet,state.opponent_trajectories[0][0][0].pet):
        assert pet._until_end_of_battle_attack_buff==pet._until_end_of_battle_health_buff==0
        assert pet.ability_counter==0 and pet.sell_value==pet.level
    assert public_records(observed)[0]['temporaryAttack']
    original=public_records(state.opponent_trajectories[0][0])
    reservoir.future[0].pet._attack=49
    assert public_records(state.opponent_trajectories[0][0])==original
    env.core.opponent.team=Team(['pet-dragon']);env.core.opponent.shop.roll();env.core.opponent.gold=27
    env.core.previous_opponent_state[0].pet.sell_value=77
    env.core.previous_opponent_state[0].pet.ability_counter=44
    other=RecordingReservoir();again,_=public_state_from_one_step_reservoir(env.core,0,other)
    assert public_records(other.requests[0][0])==public_records(observed)
    assert public_records(again.opponent_trajectories[0][0])==original


@pytest.mark.parametrize('side',[0,1])
def test_opening_uses_explicit_seed_and_one_step_without_forged_history(side):
    env=ParallelSelfPlayEnv();env.reset(seed=715001);reservoir=RecordingReservoir()
    state,meta=public_state_from_one_step_reservoir(env.core,side,reservoir,opening_seed=715002)
    assert reservoir.requests==[(None,{'seed':715002,'horizon':1,'limit':32})]
    assert state.core.previous_opponent_state is None and meta['targetTurn']==1
    for key,value in env.observations()[side].items():np.testing.assert_array_equal(value,state.observation()[key])


def test_terminal_turn_next_board_is_not_censored_by_longer_future():
    env=environment()
    trajectories=[{'id':'ended-on-current','boards':[{'turn':1,'team':Team(['fish'])},{'turn':2,'team':Team(['ant'])}]},
                  {'id':'survived-longer','boards':[{'turn':1,'team':Team(['fish'])},{'turn':2,'team':Team(['beaver'])},{'turn':3,'team':Team(['dog'])}]}]
    reservoir=OpponentTrajectoryReservoir(trajectories)
    _,meta=public_state_from_one_step_reservoir(env.core,0,reservoir,limit=2)
    assert set(meta['candidateIds'])=={'ended-on-current','survived-longer'}
    assert not meta['futureSurvivalRequiredBeyondCurrentTurn']


def test_missing_or_malformed_support_is_explicit():
    env=environment();reservoir=RecordingReservoir();reservoir.match=lambda *a,**kw:[]
    with pytest.raises(PublicOpponentSupportMissing):public_state_from_one_step_reservoir(env.core,0,reservoir)
    reservoir.match=lambda *a,**kw:[MatchedOpponentTrajectory('bad',0.,(Team(['fish']),Team(['ant'])))]
    with pytest.raises(ValueError,match='horizon'):public_state_from_one_step_reservoir(env.core,0,reservoir)


def test_real_missing_opening_is_specific_but_broken_data_is_not_hidden():
    env=ParallelSelfPlayEnv();env.reset(seed=715001)
    reservoir=OpponentTrajectoryReservoir([{'id':'no-opening','boards':[{'turn':2,'team':Team(['fish'])}]}])
    with pytest.raises(PublicOpponentSupportMissing):
        public_state_from_one_step_reservoir(env.core,0,reservoir,opening_seed=1)
    def broken(**kwargs):raise ValueError('broken data')
    reservoir.opening_prior=broken
    with pytest.raises(ValueError,match='broken data') as error:
        public_state_from_one_step_reservoir(env.core,0,reservoir,opening_seed=1)
    assert not isinstance(error.value,PublicOpponentSupportMissing)


@pytest.mark.parametrize('kwargs',[{'side':2},{'side':True},{'side':0,'limit':0},{'side':0,'opening_seed':4}])
def test_invalid_public_contract_arguments_fail_closed(kwargs):
    with pytest.raises(ValueError):public_state_from_one_step_reservoir(environment().core,reservoir=RecordingReservoir(),**kwargs)
