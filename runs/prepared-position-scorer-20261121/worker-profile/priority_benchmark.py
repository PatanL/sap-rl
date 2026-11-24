"""Isolated whole-battle test of grouped attack priority on the released engine."""
import hashlib
import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from compact_scorer import compact_board
from plan_common import install
from prepared_scorer import score_prepared


def grouped_priority(t0, t1, rng=None):
    groups = defaultdict(list)
    for side, team in enumerate((t0, t1)):
        for index, slot in enumerate(team.slots):
            pet = slot.pet
            if pet.name != 'pet-none':
                groups[float(pet.attack)].append((side, index))
    ordered = []
    for attack in sorted(groups, reverse=True):
        group = groups[attack]
        if len(group) > 1:
            order = (np.random if rng is None else rng).permutation(len(group))
            ordered.extend(group[i] for i in order)
        else:
            ordered.extend(group)
    return ordered


def run():
    install(cuda=False)
    from sapai import Battle, Team
    from check_compact_battles import terminal

    library = Path('/workspace/runs/claude-end-positioner-20261112-v1/out/library.pkl')
    with library.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != '8f13394cb9081a528c42afc12659a4a742c073039831a304b792edc7b64da27c':
            raise ValueError('Unexpected board library')
        stream.seek(0)
        library_boards = pickle.load(stream)
    rng = random.Random(202611212)
    tasks = []
    for i in range(96):
        teams = [compact_board(row['board']) for row in rng.sample(library_boards[1 + i % 16], 3)]
        tasks.append((teams[0], teams[1:], [rng.getrandbits(31) for _ in range(4)]))

    # Include maximum equal-attack groups even when uncommon in rollout games.
    # Validate order and random stream after every successive priority query.
    reference_rng, candidate_rng = (np.random.RandomState(25772) for _ in range(2))
    for size in range(11):
        teams = [Team(['ant'] * min(size, 5)), Team(['ant'] * max(size - 5, 0))]
        for _ in range(32):
            if Battle.update_pet_priority(*teams, reference_rng) != grouped_priority(*teams, candidate_rng):
                raise ValueError('Priority order changed')
            if reference_rng.bytes(32) != candidate_rng.bytes(32):
                raise ValueError('Priority RNG changed')

    # Separate state validation from timing. Use all 768 individual battles.
    baseline = Battle.update_pet_priority
    def fingerprint(priority):
        h = hashlib.sha256()
        with patch.object(Battle, 'update_pet_priority', staticmethod(priority)):
            for left, opponents, seeds in tasks:
                team = Team.from_state(left)
                for right in opponents:
                    opponent = Team.from_state(right)
                    for seed in seeds:
                        outcome, state = terminal(team, opponent, seed)
                        h.update(bytes([outcome]))
                        h.update(hashlib.sha256(state.encode()).digest())
        return h.hexdigest()

    expected = fingerprint(baseline)
    if fingerprint(grouped_priority) != expected:
        raise ValueError('Battle outcome, terminal teams, or final RNG changed')
    durations = []
    scores = None
    for name in ('baseline', 'grouped', 'grouped', 'baseline'):
        fn = baseline if name == 'baseline' else grouped_priority
        with patch.object(Battle, 'update_pet_priority', staticmethod(fn)):
            start = time.process_time()
            measured = [score_prepared(task) for task in tasks]
            elapsed = time.process_time() - start
        if scores is not None and scores != measured:
            raise ValueError('A measured battle score changed')
        scores = measured
        durations.append(dict(method=name, cpuSeconds=elapsed))
    totals = {name:sum(x['cpuSeconds'] for x in durations if x['method']==name)
              for name in ('baseline','grouped')}
    return dict(battlesPerPass=768, librarySHA256=digest, terminalDigest=expected,
                tieChecks=352, timings=durations, speedRatio=totals['baseline']/totals['grouped'])


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
