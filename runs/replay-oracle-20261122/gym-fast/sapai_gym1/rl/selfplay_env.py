"""Explicit two-player environment for batched league self-play actors."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from sapai import Battle, Team
from .action_schema import AutoregressiveAction, FlatActionCodec
from .entity_observation import EntityObservationEncoder
from .battle_equity import BattleEquityConfig,BattleEquityEstimator
from .identity_remap import (
    remap_shop_identities as remap_identity_set,
    remap_team_orders as remap_orders,
)


@dataclass
class SelfPlayStep:
    observations: tuple[dict, dict]
    rewards: tuple[float, float]
    battle_resolved: bool
    terminated: bool
    battle_result: int | None = None


class ParallelSelfPlayEnv:
    """Drive both shop phases explicitly instead of hiding an opponent policy."""

    def __init__(self, fast_battles: bool = True, causal_context_schema=None,
                 battle_equity_config=None,battle_equity_owner='single'):
        from sapai_gym1 import SuperAutoPetsEnv

        self.core = SuperAutoPetsEnv(
            opponent_generator=None,
            valid_actions_only=True,
            manual_battles=True,
        )
        self.encoder = EntityObservationEncoder(causal_context_schema=causal_context_schema)
        self.core.causal_context_schema = causal_context_schema
        self.fast_battles = bool(fast_battles)
        self.battle_equity_config=BattleEquityConfig.parse(battle_equity_config)
        if self.battle_equity_config.samples>1 and not self.fast_battles:
            raise ValueError('M2 runtime requires the validated fast battle path')
        self._battle_equity=(BattleEquityEstimator(self.battle_equity_config,battle_equity_owner)
                             if self.battle_equity_config.samples>1 else None)

    def reset(self, seed: int | None = None, *, paired_tape: bool = False) -> tuple[dict, dict]:
        if type(paired_tape) is not bool:
            raise ValueError('Explicit paired-world mode required')
        if paired_tape and (type(seed) is not int or not 0 <= seed < 2**63 or self._battle_equity is not None):
            raise ValueError('Paired world requires explicit seed and standard single-battle outcome')
        self.core.reset(seed=(seed % (2**32) if paired_tape else seed))
        self._battle_tape = None
        self._paired_battle_seeds = []
        if paired_tape:
            from sapai import Shop
            from sapai.paired_random_tape import RandomTape
            self._battle_tape = RandomTape(seed)
            for side, player in enumerate((self.core.agent, self.core.opponent)):
                shop = Shop(turn=1, fixed_rules=True)
                shop.enable_paired_tape(seed, side)
                shop.fixed_rules = False
                shop.update_shop_rules(roll_only_empty=True)
                player.shop = shop
                player._connect_objects()
            self.core.shop_agent = self.core.agent.shop
            self.core.shop_opponent = self.core.opponent.shop
            self.core._agent_actions_cache = self.core._opponent_actions_cache = None
        if self._battle_equity is not None:self._battle_equity.reset()
        from .causal_transport import refresh_context
        refresh_context(self.core)
        return self.observations()

    def clone(self):
        """Clone the exact hidden state for untouched interactive evaluation."""

        cloned = copy.deepcopy(self)
        cloned.core.agent._connect_objects()
        cloned.core.opponent._connect_objects()

        cloned.core._freeze_toggled_agent = remap_identity_set(
            self.core.agent,
            cloned.core.agent,
            self.core._freeze_toggled_agent,
        )
        cloned.core._freeze_toggled_opponent = remap_identity_set(
            self.core.opponent,
            cloned.core.opponent,
            self.core._freeze_toggled_opponent,
        )
        cloned.core._seen_team_orders_agent = remap_orders(
            self.core.agent,
            cloned.core.agent,
            self.core._seen_team_orders_agent,
        )
        cloned.core._seen_team_orders_opponent = remap_orders(
            self.core.opponent,
            cloned.core.opponent,
            self.core._seen_team_orders_opponent,
        )
        # A clone must not invent a visited formation. In particular, a board
        # changed after exhausting its move budget may not yet be in this set.
        # Normal legality processing registers it when appropriate, just as it
        # would on the original environment.
        cloned.core._agent_actions_cache = None
        cloned.core._opponent_actions_cache = None
        return cloned

    def redeterminize(self, seed):
        """Resample only future RNG while retaining both current information sets."""

        seeds = np.random.RandomState(int(seed) % (2**32))
        for seat, player in enumerate((self.core.agent, self.core.opponent)):
            random_state = np.random.RandomState(
                int(seeds.randint(0, 2**32 - 1))
            )
            player.shop.rs = random_state
            player.shop.seed_state = random_state.get_state()
            if getattr(player.shop, "_paired_tape", None) is not None:
                player.shop.enable_paired_tape(int(seed) % (2**63), seat)
            player._connect_objects()
        self.core.rs_random = np.random.RandomState(
            int(seeds.randint(0, 2**32 - 1))
        )
        if getattr(self, '_battle_tape', None) is not None:
            from sapai.paired_random_tape import RandomTape
            self._battle_tape = RandomTape(int(seed) % (2**63))
        if self._battle_equity is not None:self._battle_equity.redeterminize(int(seed))

    def _next_battle_seed(self):
        if getattr(self, '_battle_tape', None) is None:
            return int(self.core.rs_random.randint(0, 2**32 - 1))
        seed = int(self._battle_tape.uniform(f'battle/turn/{self.core.turn}') * (2**32 - 1))
        self._paired_battle_seeds.append((int(self.core.turn), seed))
        return seed

    def observations(self) -> tuple[dict, dict]:
        return self.encoder.encode(self.core, 0), self.encoder.encode(self.core, 1)

    def active_players(self) -> tuple[bool, bool]:
        return not self.core.agent.turn_ended, not self.core.opponent.turn_ended

    def step(
        self,
        player_zero_action: AutoregressiveAction | np.ndarray | None,
        player_one_action: AutoregressiveAction | np.ndarray | None,
    ) -> SelfPlayStep:
        actions = (player_zero_action, player_one_action)
        for player_idx, action in enumerate(actions):
            player = self.core.agent if player_idx == 0 else self.core.opponent
            if player.turn_ended:
                if action is not None:
                    raise ValueError(f"Player {player_idx} already ended the turn")
                continue
            if action is None:
                raise ValueError(f"Active player {player_idx} requires an action")
            autoregressive = self._coerce_and_validate(player_idx, action)
            self.core.resolve_action(
                agent_idx=player_idx,
                action=FlatActionCodec.encode(autoregressive),
            )

        if not (self.core.agent.turn_ended and self.core.opponent.turn_ended):
            return SelfPlayStep(self.observations(), (0.0, 0.0), False, False)

        self.core.previous_previous_opponent_state = (
            self.core.previous_opponent_state
        )
        self.core.previous_previous_agent_state = self.core.previous_agent_state
        self.core.previous_opponent_state = Team.from_state(
            self.core.opponent.team.state
        )
        self.core.previous_agent_state = Team.from_state(self.core.agent.team.state)
        equity_prepared=(self._battle_equity.prepare(self.core) if self._battle_equity is not None else None)
        battle_result = Battle(
            self.core.agent.team,
            self.core.opponent.team,
            seed=self._next_battle_seed(),
            fast_training=self.fast_battles,
        ).battle()
        lives_before = (self.core.agent.lives, self.core.opponent.lives)
        self.core._player_fight_outcome(battle_result)
        life_deltas = (
            self.core.agent.lives - lives_before[0],
            self.core.opponent.lives - lives_before[1],
        )
        self.core.turn += 1
        terminated = self.core.is_done()

        life_swing = life_deltas[0] - life_deltas[1]
        battle_component=0.1*life_swing
        if self._battle_equity is not None:
            battle_component=self._battle_equity.estimate(equity_prepared,battle_result,battle_component)
        rewards = [battle_component, -battle_component]
        if self.core.opponent.lives <= 0:
            rewards[0] += 2.0
            rewards[1] -= 2.0
        if self.core.agent.lives <= 0:
            rewards[0] -= 2.0
            rewards[1] += 2.0
        if (
            terminated
            and self.core.agent.lives > 0
            and self.core.opponent.lives > 0
        ):
            player_score = (self.core.agent.lives, self.core.agent.wins)
            opponent_score = (
                self.core.opponent.lives,
                self.core.opponent.wins,
            )
            if player_score > opponent_score:
                rewards[0] += 2.0
                rewards[1] -= 2.0
            elif opponent_score > player_score:
                rewards[0] -= 2.0
                rewards[1] += 2.0

        if not terminated:
            self.core.agent.start_turn(winner=self.core.agent.lf_winner)
            self.core.opponent.start_turn(
                winner=self.core.opponent.lf_winner
            )
            self.core.reorder_count_agent = 0
            self.core.reorder_count_opponent = 0
            self.core.shop_action_count_agent = 0
            self.core.shop_action_count_opponent = 0
            self.core._freeze_toggled_agent.clear()
            self.core._freeze_toggled_opponent.clear()
            self.core._legality_identities.clear()
            self.core._seen_team_orders_agent = {
                self.core._team_order_key(self.core.agent)
            }
            self.core._seen_team_orders_opponent = {
                self.core._team_order_key(self.core.opponent)
            }

        from .causal_transport import refresh_context
        refresh_context(self.core)
        return SelfPlayStep(
            observations=self.observations(),
            rewards=(float(rewards[0]), float(rewards[1])),
            battle_resolved=True,
            terminated=terminated,
            battle_result=battle_result,
        )

    def _coerce_and_validate(self, player_idx, action) -> AutoregressiveAction:
        if isinstance(action, AutoregressiveAction):
            autoregressive = action
        else:
            action = np.asarray(action, dtype=np.int64)
            if action.shape != (3,):
                raise ValueError(f"Expected action shape (3,), got {action.shape}")
            autoregressive = AutoregressiveAction(*map(int, action))
        flat_action = FlatActionCodec.encode(autoregressive)
        if flat_action not in self.core._avail_actions(player_idx):
            raise ValueError(
                f"Illegal action {autoregressive.as_tuple()} for player {player_idx}"
            )
        return autoregressive
