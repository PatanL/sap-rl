import copy
import gc
import weakref

import numpy as np
import pytest
from sapai import Player, Shop, Team
from sapai_gym1.rl.action_schema import ActionType, AutoregressiveAction, FlatActionCodec
from sapai_gym1.rl.identity_lifetime import IdentityLifetime
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from sapai_gym1.rl.rollout_search import PublicSearchState
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv


class Item:
    pass


def test_retired_identity_cannot_be_reused_and_is_released_on_clear():
    owner = IdentityLifetime()
    item = Item()
    ref = weakref.ref(item)
    identity = owner.remember(item)
    assert owner.remember(item) == identity and len(owner) == 1
    del item
    gc.collect()
    assert ref() is not None
    newcomers = [Item() for _ in range(2000)]
    assert all(id(new) != identity for new in newcomers)
    owner.clear()
    gc.collect()
    assert ref() is None and len(owner) == 0


def test_deepcopy_rekeys_owners_and_handles_cycles():
    owner = IdentityLifetime()
    item = Item()
    item.owner = owner
    original_id = owner.remember(item)
    clone = copy.deepcopy(owner)
    copied = next(iter(clone._objects.values()))
    assert copied is not item and copied.owner is clone
    assert set(clone._objects) == {id(copied)}
    assert original_id not in clone._objects
    assert clone.remember(copied) == id(copied) and len(clone) == 1
    clone.clear()
    assert len(owner) == 1


def fixture(side):
    env = ParallelSelfPlayEnv()
    env.reset(seed=74410001)
    player = Player(shop=Shop(['dolphin', 'cow', 'apple'], turn=7),
                    team=Team(['dolphin', 'ant']), turn=7, gold=10,
                    record_actions=False)
    if side == 0:
        env.core.agent = player
        env.core.shop_agent, env.core.team_agent = player.shop, player.team
    else:
        env.core.opponent = player
        env.core.shop_opponent, env.core.team_opponent = player.shop, player.team
    player._connect_objects()
    env.core.turn = 7
    env.core.agent.turn = env.core.opponent.turn = 7
    env.core.previous_agent_state = Team(['fish'])
    env.core.previous_opponent_state = Team(['beaver'])
    env.core._agent_actions_cache = env.core._opponent_actions_cache = None
    return env, player


def act(env, side, kind, source=10, target=5):
    env.core.resolve_action(side, FlatActionCodec.encode(
        AutoregressiveAction(kind, source, target)))


def equal_observations(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


@pytest.mark.parametrize('side', [0, 1])
def test_combined_unfrozen_dolphin_cannot_lock_fresh_cow_milk(side):
    env, player = fixture(side)
    player.shop[0].freeze()
    old = player.shop[0].obj
    ref = weakref.ref(old)
    old_id = id(old)
    act(env, side, ActionType.UNFREEZE, 0)
    act(env, side, ActionType.BUY_PET, 0, 0)
    del old
    gc.collect()
    assert ref() is not None  # This is the direct pre-fix lifetime failure.
    act(env, side, ActionType.BUY_PET, 0, 2)
    toggled = env.core._freeze_toggled_agent if side == 0 else env.core._freeze_toggled_opponent
    assert old_id in toggled
    milk = [i for i, slot in enumerate(player.shop) if slot.obj.name == 'food-milk']
    assert len(milk) == 2
    obs = env.observations()[side]
    for i in milk:
        assert id(player.shop[i].obj) != old_id
        assert obs['numeric'][6+i, 8] == 0
        assert obs['target_mask'][ActionType.FREEZE, i, 5]
    clone = env.clone()
    equal_observations(obs, clone.observations()[side])
    public = PublicSearchState.from_core(env.core, side)
    equal_observations(obs, public.observation())
    equal_observations(public.observation(), public.clone().observation())


@pytest.mark.parametrize('side', [0, 1])
def test_visited_formation_retains_sold_pet_and_clears_at_reset(side):
    env, player = fixture(side)
    original = player.team[1].pet
    ref = weakref.ref(original)
    key = env.core._team_order_key(player)
    act(env, side, ActionType.SELL, 1)
    del original
    gc.collect()
    assert ref() is not None
    assert key[1] in env.core._legality_identities._objects
    env.reset(seed=74410002)
    assert len(env.core._legality_identities) == 0


def test_parallel_turn_boundary_releases_previous_history():
    env = ParallelSelfPlayEnv()
    env.reset(seed=74410003)
    owner = env.core._legality_identities
    old = Item()
    ref = weakref.ref(old)
    identity = owner.remember(old)
    del old
    for player in (env.core.agent, env.core.opponent):
        player.team = Team(['fish'])
        player.gold = 0
        player._connect_objects()
    env.core._agent_actions_cache = env.core._opponent_actions_cache = None
    result = step_decisions(env, [JointAction(AutoregressiveAction(ActionType.END_TURN), 0)] * 2).result
    assert result.battle_resolved and not result.terminated
    gc.collect()
    assert ref() is None
    assert identity not in env.core._legality_identities._objects
    assert len(env.core._legality_identities) == 2


def test_public_turn_boundary_releases_previous_history():
    env, player = fixture(0)
    public = PublicSearchState.from_core(env.core, 0)
    public.core.agent.gold = 0
    item = Item()
    ref = weakref.ref(item)
    identity = public.core._legality_identities.remember(item)
    del item
    transition = public.step(AutoregressiveAction(ActionType.END_TURN), np.random.default_rng(74410021))
    assert not public.terminated
    gc.collect()
    assert ref() is None
    assert identity not in public.core._legality_identities._objects


@pytest.mark.parametrize('seed', [74410101, 74410102, 74410103])
def test_long_real_traces_bound_ownership_and_preserve_clone_masks(seed):
    env = ParallelSelfPlayEnv()
    observations = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for step in range(2500):
        decisions = []
        for side, active in enumerate(env.active_players()):
            if not active:
                decisions.append(None)
                continue
            legal = np.argwhere(observations[side]['target_mask'])
            t, s, u = map(int, legal[int(rng.integers(len(legal)))])
            decisions.append(JointAction(AutoregressiveAction(t, s, u), 0))
        result = step_decisions(env, decisions).result
        observations = result.observations
        count = env.core.shop_action_count_agent + env.core.shop_action_count_opponent
        assert len(env.core._legality_identities) <= 10 + 5 * count
        if step % 19 == 0:
            cloned = env.clone()
            for side in (0, 1):
                equal_observations(observations[side], cloned.observations()[side])
            assert all(key == id(obj) for key, obj in cloned.core._legality_identities._objects.items())
        if result.terminated:
            break
    else:
        pytest.fail('Complete trace did not terminate within the existing safety bound')
