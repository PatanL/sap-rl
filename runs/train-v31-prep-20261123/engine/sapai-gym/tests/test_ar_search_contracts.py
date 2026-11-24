import copy,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest,torch
from sapai import Pet,Team,Battle
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import ParallelSelfPlayEnv
from sapai_gym1.rl.rollout_search import PublicSearchState,NeuralSearchEvaluator
from sapai_gym1.rl.opponent_belief import OpponentBeliefMix
from sapai_gym1.rl.ar_decoder_readout import BeliefReadout
from sapai_gym1.rl.ar_search_belief import ARBeliefProvider,ParticleOpponentBelief
from sapai_gym1.rl.opponent_distribution import AutoregressiveOpponentBoard,OpponentDistributionConfig
from sapai_gym1.rl.public_battle_board import public_records,from_public_records,to_state,from_state,flags
from sapai_gym1.rl.search_work_budget import SearchWorkBudget,SearchBudgetExceeded
from sapai_gym1.rl.puct_search import StochasticPUCTSearch,PUCTConfig
from test_end_freeze_policy import policies

def state_and_policy(side=0):
    env,_,policy=policies();c=env.core;c.agent.turn=c.opponent.turn=3;c.turn=3
    c.agent.gold=c.opponent.gold=0
    p=Pet('fish');p._attack=4;p._health=5;p._until_end_of_battle_attack_buff=2;p._until_end_of_battle_health_buff=3
    p.ability_counter=3;p.friends_hurt_counter=2
    c.previous_agent_state=Team([p]);c.previous_opponent_state=Team([copy.deepcopy(p)])
    c.previous_previous_agent_state=Team(['pet-ant']);c.previous_previous_opponent_state=Team(['pet-ant'])
    c._agent_actions_cache=c._opponent_actions_cache=None
    return env,PublicSearchState.from_core(c,side),policy

def provider():
    from sapai.data import data
    from sapai_gym1.rl.entity_observation import EntityObservationEncoder
    torch.manual_seed(190)
    model=AutoregressiveOpponentBoard(OpponentDistributionConfig(('none',*sorted(k for k in data['pets'] if k!='pet-none')),
        tuple(EntityObservationEncoder().status_to_id),width=32))
    readout=BeliefReadout(model,'a'*64,'/uniforms.npy',history_only=True,rules=SIMULATOR_REVISION)
    return ARBeliefProvider(readout)

@pytest.mark.parametrize('side',[0,1])
def test_public_sanitization_preserves_observation_but_not_hidden_decomposition(side):
    env,state,_=state_and_policy(side)
    expected=env.observations()[side];actual=state.observation()
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)
    pet=state.core.previous_opponent_state[0].pet
    assert pet._attack==6 and pet._health==8 and pet._until_end_of_battle_attack_buff==pet._until_end_of_battle_health_buff==0
    assert pet.ability_counter==pet.friends_hurt_counter==0
    assert flags(state.core.previous_opponent_state)[0]==(True,True)
    enemy=env.core.opponent if side==0 else env.core.agent
    observed=env.core.previous_opponent_state if side==0 else env.core.previous_agent_state
    observed[0].pet._attack=2;observed[0].pet._until_end_of_battle_attack_buff=4
    observed[0].pet._health=1;observed[0].pet._until_end_of_battle_health_buff=7
    observed[0].pet.ability_counter=99;enemy.team=Team(['pet-dragon']);enemy.shop.roll();enemy.gold=27
    other=PublicSearchState.from_core(env.core,side)
    assert public_records(state.core.previous_opponent_state)==public_records(other.core.previous_opponent_state)
    assert state.core.previous_opponent_state.state==other.core.previous_opponent_state.state
    for key in actual:np.testing.assert_array_equal(actual[key],other.observation()[key])
    clone=state.clone()
    for key in actual:np.testing.assert_array_equal(actual[key],clone.observation()[key])
    restored=from_state(json.loads(json.dumps(to_state(state.core.previous_opponent_state))))
    assert public_records(restored)==public_records(state.core.previous_opponent_state)

def test_effective_battle_stats_and_public_flags_are_separate():
    _,state,_=state_and_policy();team=state.core.previous_opponent_state
    for seed in range(8):
        plain=Team(['pet-fish']);plain[0].pet._attack=6;plain[0].pet._health=8
        assert Battle(team,Team(['pet-beaver','pet-cricket']),seed=seed,fast_training=True).battle()==Battle(plain,Team(['pet-beaver','pet-cricket']),seed=seed,fast_training=True).battle()
    assert public_records(team)[0]['temporaryHealth'] is True

@pytest.mark.parametrize('level',[1,2,3])
def test_public_parrot_copy_matches_native_effect_without_double_scaling(level):
    from sapai.effects import TransferAbility
    actual=Team(['pet-deer','pet-none','pet-parrot','pet-giraffe'])
    actual[2].pet.level=level
    TransferAbility(actual[2].pet,[0,2],[actual,Team()])
    expected=copy.deepcopy(actual[2].pet.ability)
    board=public_records(actual);restored=from_public_records(board)
    assert restored[2].pet.ability==expected
    assert public_records(restored)==board
    assert restored[2].pet.ability['_copiedFrom']=='pet-deer'

def test_live_future_rng_erased_but_sampled_world_clone_continues_same_stream():
    env,a,_=state_and_policy()
    env.core.agent.shop.rs=np.random.RandomState(98123);env.core.agent._connect_objects()
    b=PublicSearchState.from_core(env.core,0)
    assert a.core.agent.shop.rs.get_state()[1].tolist()==b.core.agent.shop.rs.get_state()[1].tolist()
    a.redeterminize(199);a.core.agent.shop.rs.random(11)
    clone=a.clone()
    np.testing.assert_array_equal(a.core.agent.shop.rs.random(10),clone.core.agent.shop.rs.random(10))

def test_joint_particle_provider_survives_neural_refresh_clone_and_puct_nodes():
    _,state,policy=state_and_policy();p=provider();state.enable_learned_opponent_belief(OpponentBeliefMix(1.,0.,0.))
    evaluator=NeuralSearchEvaluator(policy,opponent_belief_provider=p)
    memory=policy.initial_memory(1)[0]
    first=evaluator.evaluate(state,memory);assert isinstance(first.opponent_belief,ParticleOpponentBelief)
    pool=first.opponent_belief.pool;evaluator.choose(state,memory)
    assert isinstance(state.opponent_board_belief,ParticleOpponentBelief) and state.opponent_board_belief.pool is pool
    clone=state.clone();assert clone.opponent_board_belief.pool is pool
    rng=np.random.default_rng(911);before=copy.deepcopy(rng.bit_generator.state)
    a=first.opponent_belief.sample_synthetic_team(rng);b=first.opponent_belief.sample_synthetic_team(rng)
    assert rng.bit_generator.state==before and a._search_particle_id==b._search_particle_id
    assert a is not b and a.state==b.state
    if not a[0].empty:a[0].pet._attack=49
    again=first.opponent_belief.sample_synthetic_team(rng);assert again.state==b.state
    search=StochasticPUCTSearch(evaluator,PUCTConfig(max_simulations=3,max_simulator_transitions=30,horizon_battles=1,
        tree_leaf_rollout_steps=2,root_screening_rollouts=2,root_screening_rounds=1,validation_rollouts=2,max_validation_rollouts=2,audit_rollouts=0))
    result=search.search(state,memory,seed=100,validation_seed=101)
    assert isinstance(search.root_node.opponent_belief,ParticleOpponentBelief)
    assert result.policy_action in search.root_node.priors
    assert p.metrics()['cacheHits']>0

def test_invalid_particles_retain_mass_and_do_not_renormalize():
    _,state,policy=state_and_policy();state.enable_learned_opponent_belief(OpponentBeliefMix(1,0,0));p=provider()
    evaluator=NeuralSearchEvaluator(policy,opponent_belief_provider=p);belief=evaluator.evaluate(state,policy.initial_memory(1)[0]).opponent_belief
    assert len(belief.pool.states)==64 and any(x is None for x in belief.pool.states)
    assert p.metrics()['invalidParticles']>0 and p.metrics()['minimumRetainedMass']<1
    seen=set()
    for seed in range(100):seen.add(belief.sample_synthetic_team(np.random.default_rng(seed))._search_belief_source)
    assert 'belief-ar' in seen and 'belief-ar-invalid-fallback' in seen

def test_shared_budget_bounds_cloned_steps_and_neural_work():
    _,state,policy=state_and_policy();now=[0.];budget=SearchWorkBudget(max_decisions=1,max_neural_states=1,seconds=2,clock=lambda:now[0])
    state.work_budget=budget;clone=state.clone();assert clone.work_budget is budget
    evaluator=NeuralSearchEvaluator(policy);memory=policy.initial_memory(1)[0]
    decision=evaluator.choose(clone,memory)[0]
    with pytest.raises(SearchBudgetExceeded):evaluator.choose(state,memory)
    clone.step(decision,np.random.default_rng(9));assert budget.decisions==1
    with pytest.raises(SearchBudgetExceeded):state.step(decision,np.random.default_rng(9))
    now[0]=3
    with pytest.raises(SearchBudgetExceeded,match='deadline'):budget.check()
