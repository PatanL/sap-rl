"""Constructor RNG isolation without changing normal game/reset semantics."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from sapai_gym1.SuperAutoPetsEnv import SuperAutoPetsEnv
from sapai_gym1.rl.entity_observation import EntityObservationEncoder
from sapai_gym1.rl.action_schema import AutoregressiveAction,ActionType,FlatActionCodec
from sapai_gym1.rl.public_opponent_model import public_state_from_one_step_reservoir
from test_initial_public_reservoir import environment,RecordingReservoir


def rng_equal(a,b):
    assert a[0]==b[0] and a[2:]==b[2:]
    np.testing.assert_array_equal(a[1],b[1])


def module(path,name):
    if not Path(path).exists():pytest.skip('Explicit immutable reference source required')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def test_private_constructor_and_public_clones_do_not_advance_global_rng():
    before=np.random.get_state();preview=SuperAutoPetsEnv(manual_battles=True,initialization_seed=0)
    preview._encode_state(agent_idx=1)
    rng_equal(before,np.random.get_state())
    env=environment();before=np.random.get_state()
    public,_=public_state_from_one_step_reservoir(env.core,0,RecordingReservoir())
    clone=public.clone();rng_equal(before,np.random.get_state())
    for key,value in public.observation().items():np.testing.assert_array_equal(value,clone.observation()[key])


def test_default_environment_matches_untouched_reference_observation_and_rng():
    old=module('/reference-env.py','sapai_gym1._reference_env_rng')
    def construct(cls):
        np.random.seed(718001);env=cls(manual_battles=True);obs,_=env.reset(seed=718002)
        return env,obs,np.random.get_state()
    a,obs_a,rng_a=construct(old.SuperAutoPetsEnv);b,obs_b,rng_b=construct(SuperAutoPetsEnv)
    np.testing.assert_array_equal(obs_a,obs_b);rng_equal(rng_a,rng_b)
    encoder=EntityObservationEncoder();roll=FlatActionCodec.encode(AutoregressiveAction(ActionType.ROLL))
    for _ in range(4):
        for side in (0,1):a.resolve_action(side,roll);b.resolve_action(side,roll)
        for side in (0,1):
            left,right=encoder.encode(a,side),encoder.encode(b,side)
            for key in left:np.testing.assert_array_equal(left[key],right[key])
    rng_equal(a.agent.shop.rs.get_state(),b.agent.shop.rs.get_state())
    rng_equal(a.opponent.shop.rs.get_state(),b.opponent.shop.rs.get_state())


def test_sampled_public_world_matches_previous_constructor_path():
    old=module('/reference-rollout.py','sapai_gym1.rl._reference_rollout_rng')
    env=environment();new,_=public_state_from_one_step_reservoir(env.core,0,RecordingReservoir())
    reference=old.PublicSearchState.from_core(env.core,0)
    reference.set_opponent_trajectories(new.opponent_trajectories,new.opponent_trajectory_ids)
    new.opponent_transition_model=None  # Fixed identical hypotheses isolate constructor behavior.
    reference.redeterminize(718003);new.redeterminize(718003)
    end=AutoregressiveAction(ActionType.END_TURN)
    a=reference.step(end,np.random.default_rng(718004));b=new.step(end,np.random.default_rng(718004))
    assert asdict(a)==asdict(b)
    for _ in range(4):
        left,right=reference.observation(),new.observation()
        for key in left:np.testing.assert_array_equal(left[key],right[key])
        roll=AutoregressiveAction(ActionType.ROLL)
        assert asdict(reference.step(roll,np.random.default_rng(1)))==asdict(new.step(roll,np.random.default_rng(1)))
