"""Construct public search inputs from either observed history or an opening prior."""
from collections import Counter
from numbers import Integral

from .rollout_search import PublicSearchState
from .public_battle_board import sanitize
from .opponent_reservoir import OpponentReservoirSupportMissing


ONE_STEP_PUBLIC_BELIEF_VERSION = 'public-initial-one-step-reservoir-v1-20260907'


class PublicOpponentSupportMissing(ValueError):
    """Expected lack of current-step evidence, not permission to omit an eval state."""


def public_state_from_one_step_reservoir(core, side, reservoir, *, limit=32, opening_seed=None):
    """Initial CURRENT-turn hypotheses, followed by existing one-step updates.

    The policy observation remains past observed history. Never place a
    predicted board into previous_opponent_state before its simulated reveal.
    No horizon argument is offered: requiring five recorded future turns would
    condition initial retrieval on survival the acting agent cannot know.
    """
    if not isinstance(side, Integral) or isinstance(side, bool) or side not in (0, 1):
        raise ValueError('side must be zero or one')
    if not isinstance(limit, Integral) or isinstance(limit, bool) or limit <= 0:
        raise ValueError('A positive integer candidate limit is required')
    latest = core.previous_opponent_state if side == 0 else core.previous_agent_state
    if latest is None:
        if not isinstance(opening_seed, Integral) or isinstance(opening_seed, bool) or opening_seed < 0:
            raise ValueError('A no-history state requires an explicit nonnegative opening seed')
        try:
            matches = reservoir.opening_prior(seed=int(opening_seed), horizon=1, limit=int(limit))
        except OpponentReservoirSupportMissing as error:
            raise PublicOpponentSupportMissing('No public opening support; use an explicit accounted raw fallback') from error
        kind = 'opening_population'
    else:
        if opening_seed is not None:
            raise ValueError('Opening priors must not replace observed history')
        # Sanitize before the retrieval component receives anything, even
        # though its current distance function already uses public attributes.
        own_previous = core.previous_agent_state if side == 0 else core.previous_opponent_state
        older = core.previous_previous_opponent_state if side == 0 else core.previous_previous_agent_state
        own_older = core.previous_previous_agent_state if side == 0 else core.previous_previous_opponent_state
        matches = reservoir.match(sanitize(latest), observed_enemy=sanitize(own_previous),
            older_observed=sanitize(older), older_observed_enemy=sanitize(own_older),
            current_turn=int(core.turn), horizon=1, limit=int(limit))
        kind = 'matched_public_history'
    if not matches:
        raise PublicOpponentSupportMissing('No CURRENT-turn public reservoir support; use an explicit accounted raw fallback')
    if any(len(match.teams) != 1 for match in matches):
        raise ValueError('One-step query returned a mismatched trajectory horizon')
    ids = [str(match.trajectory_id) for match in matches]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate reservoir trajectory identities')
    hypotheses = [tuple(sanitize(team) for team in match.teams) for match in matches]
    state = PublicSearchState.from_core(core, side, opening_prior=hypotheses if latest is None else None)
    state.set_opponent_trajectories(hypotheses, ids)
    state.set_opponent_transition_model(reservoir, candidate_limit=int(limit))
    return state, {'version': ONE_STEP_PUBLIC_BELIEF_VERSION, 'kind': kind,
        'openingSeed': opening_seed, 'retrievalHorizon': 1, 'targetTurn': int(core.turn),
        'candidateIds': ids, 'candidateCount': len(ids), 'initialRetrievalInstalled': True,
        'sourceCounts': dict(sorted(Counter(str(m.source_index) for m in matches).items())),
        'futureSurvivalRequiredBeyondCurrentTurn': False}


def public_state_from_reservoir(core, side, reservoir, *, horizon=5, limit=32, opening_seed=None):
    if side not in (0, 1):
        raise ValueError('side must be zero or one')
    latest = core.previous_opponent_state if side == 0 else core.previous_agent_state
    if latest is None:
        if opening_seed is None:
            raise ValueError('A no-history search requires a declared opening-prior seed')
        matches = reservoir.opening_prior(seed=opening_seed, horizon=horizon, limit=limit)
        state = PublicSearchState.from_core(core, side, opening_prior=[match.teams for match in matches])
        kind = 'opening_population'
    else:
        if opening_seed is not None:
            raise ValueError('Opening priors must not replace observed history')
        own_previous = core.previous_agent_state if side == 0 else core.previous_opponent_state
        older = core.previous_previous_opponent_state if side == 0 else core.previous_previous_agent_state
        own_older = core.previous_previous_agent_state if side == 0 else core.previous_previous_opponent_state
        matches = reservoir.match(latest, observed_enemy=own_previous, older_observed=older,
                                  older_observed_enemy=own_older, current_turn=core.turn,
                                  horizon=horizon, limit=limit)
        if not matches:
            raise ValueError('No public opponent trajectories match this state')
        state = PublicSearchState.from_core(core, side)
        kind = 'matched_public_history'
    templates = state.opponent_trajectories if kind == 'opening_population' else [match.teams for match in matches]
    state.set_opponent_trajectories(templates,
                                    [match.trajectory_id for match in matches])
    state.set_opponent_transition_model(reservoir, candidate_limit=limit)
    return state, {
        'kind': kind, 'openingSeed': opening_seed,
        'candidateIds': [match.trajectory_id for match in matches],
        'sourceCounts': dict(sorted(Counter(str(match.source_index) for match in matches).items())),
    }
