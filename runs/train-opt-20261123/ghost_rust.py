"""Ranked players' boards as a third opponent type on the Rust vector environment.

Same rules as runs/train-ghost-20261123/ghost_worker.py (the Python-environment version, audited there):
  * a ghost seat never shops: its gold is set to 0 and it sends END;
  * on the step in which the live seat ends its turn the ghost seat's team is replaced by a ranked player's board
    for that turn (one player's own sequence for the whole match; another player's board at that turn once that
    player's game has ended);
  * a ghost seat produces no learner trajectory and needs no network.
Every ghost battle is checked: the team the battle engine was given (battle_events) must equal the intended board.
"""
import hashlib, json, time
import numpy as np, torch
from opponents import Opponents, LIVE, PARENT
from actor_wave import WaveActor

GHOST = -3
POOL_PATH = '/sap/runs/train-ghost-20261123/ghost-train-pool.json'
POOL_SHA256 = '55f6219eca9a173f7e288a837c0669aded85a4822fa9f66c0e0a054731a5b702'


class GhostOpponents(Opponents):
    """Per new game: ghost with probability ghost_share, else a frozen network with probability frozen_share of
    all games (parent, or half the time a recent snapshot), else both seats live. With ghost_share 0 and
    frozen_share .5 the random draws are the ones the original class makes."""

    def __init__(self, *args, ghost_share=0., frozen_share=.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not (0 <= ghost_share <= 1 and 0 <= frozen_share <= 1 and ghost_share + frozen_share <= 1):
            raise ValueError('shares must be probabilities that sum to at most 1')
        self.ghost_share = ghost_share; self.frozen_share = frozen_share

    def frozen(self, path, info=None):
        # As the base class, without its requirement that a historical opponent be of the 14.35M arm: the small
        # network's frozen opponents are small networks. The engine stamp is still checked.
        import copy as _copy
        ck = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        if info and ck.get('engine_payload_sha256') != info['payload_sha256']: raise ValueError('Mismatched historical opponent')
        model = _copy.deepcopy(self.actor); model.load_state_dict(ck['model'], strict=True)
        return model.eval().requires_grad_(False)

    def reset_games(self, indices):
        for g in indices:
            self.owners[2 * g:2 * g + 2] = LIVE
            key = 'mirror'
            u = self.rng.random()
            if u < self.ghost_share:
                seat = int(self.rng.integers(2)); self.owners[2 * g + seat] = GHOST
                key = f'ghost:seat{seat}'
            elif u < self.ghost_share + self.frozen_share:
                frozen = PARENT
                if self.recent and self.rng.random() < .5:
                    frozen = self.recent[int(self.rng.integers(len(self.recent)))][0]
                seat = int(self.rng.integers(2)); self.owners[2 * g + seat] = frozen
                key = f'parent:seat{seat}' if frozen == PARENT else f'snapshot-{frozen}:seat{seat}'
            self.assignments[key] += 1
        self.retire()


class GhostWaveActor(WaveActor):
    """WaveActor.act with ghost seats answered on the host: END, no freeze, no network call."""

    @torch.no_grad()
    def act(self, o):
        begin = time.perf_counter(); events = []
        ids = np.flatnonzero(o['active'].ravel()); owners = self.pool.owners[ids]
        active_obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in o['obs'].items()}
        result = torch.zeros((len(ids), 5), device='cuda', dtype=torch.float32)
        for owner in np.unique(owners):
            if owner == GHOST:
                continue
            local = np.flatnonzero(owners == owner); slots = ids[local]
            policy = self.model if owner == LIVE else self.pool.models[owner]
            inp = {k: torch.from_numpy(v[slots]).to('cuda', non_blocking=True) for k, v in active_obs.items()}
            idx = torch.as_tensor(slots, device='cuda')
            before = torch.cuda.Event(enable_timing=True); after = torch.cuda.Event(enable_timing=True); before.record()
            with torch.autocast('cuda', dtype=torch.bfloat16): out = policy.act(inp, memory=self.memory.index_select(0, idx))
            after.record(); events.append((before, after))
            self.memory.index_copy_(0, idx, out.memory.float())
            values = torch.cat((out.actions, out.end_freeze_mask[:, None], out.log_prob[:, None]), dim=1)
            result.index_copy_(0, torch.as_tensor(local, device='cuda'), values)
            self.batches.append(len(slots))
        result = result.cpu().numpy()
        ghost = owners == GHOST
        result[ghost] = (0, 10, 5, 0, 0)
        self.masks.fill(0); self.actions.reshape(-1, 3)[ids] = result[:, :3]; self.masks.ravel()[ids] = result[:, 3]; self.live_log[ids] = result[:, 4]
        self.timing['policy_stream_s'] += sum(a.elapsed_time(b) for a, b in events) / 1000
        self.timing['service_s'] += time.perf_counter() - begin
        return ids, ids[owners == LIVE]


class Ghosts:
    def __init__(self, native, pool, n, seed):
        self.native = native; self.pool = pool; self.n = n
        raw = open(POOL_PATH, 'rb').read()
        if hashlib.sha256(raw).hexdigest() != POOL_SHA256: raise ValueError('ghost pool file changed')
        data = json.loads(raw)
        if data['split'] != 'train': raise ValueError('ghost pool is not the training side')
        self.sequences = []; self.by_turn = {}
        for row in data['sequences']:
            if int(hashlib.sha256(row['match'].encode()).hexdigest(), 16) % 5 < 2: raise ValueError('yardstick-side match in the training pool')
            boards = {int(t): native.encode_board(b) for t, b in row['boards'].items()}
            self.sequences.append(boards)
            for t, b in boards.items(): self.by_turn.setdefault(t, []).append(b)
        self.last = max(t for t, b in self.by_turn.items() if len(b) >= 50)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 20261123, 7]))
        self.turn = np.ones(n, 'i4'); self.sequence = [None] * n; self.expected = {}
        self.stats = dict(battles=0, own_sequence=0, team_checked=0, by_turn={}, set_gold=0)

    def validate_all(self):
        """Every pool board must be accepted by set_team (it refuses values the Python make_team would accept)."""
        from sap_env_rs import SapVecEnv
        from rust_adapter import PIN
        scratch = SapVecEnv(1, threads=1, library=str(PIN / 'libsap_engine.so')); scratch.reset(np.array([1], dtype=np.int64))
        count = 0
        for boards in self.sequences:
            for board in boards.values():
                scratch.set_team(0, 1, board); count += 1
        scratch.close(); return count

    def assign(self, games):
        owners = self.pool.owners.reshape(self.n, 2)
        for g in games:
            self.turn[g] = 1; self.expected.pop(int(g), None)
            self.sequence[g] = self.sequences[int(self.rng.integers(len(self.sequences)))] if (owners[g] == GHOST).any() else None

    def before_step(self, o, actor):
        """Zero the gold of ghost seats that end now; swap the board in where the live seat ends its turn."""
        owners = self.pool.owners.reshape(self.n, 2); active = o['active']
        ghost_seat = np.where(owners[:, 0] == GHOST, 0, np.where(owners[:, 1] == GHOST, 1, -1))
        games = np.flatnonzero(ghost_seat >= 0)
        if not len(games): return
        seats = ghost_seat[games]
        for g in games[active[games, seats]]:
            self.native.set_gold(int(g), int(ghost_seat[g]), 0); self.stats['set_gold'] += 1
        live = 1 - seats
        ending = active[games, live] & (actor.actions[games, live, 0] == 0)
        for g in games[ending]:
            s = int(ghost_seat[g])
            if active[g, s]: raise RuntimeError('ghost seat had not ended its turn before the battle')
            turn = int(self.turn[g]); boards = self.sequence[g]
            own = turn in boards
            if own: board = boards[turn]
            else:
                options = self.by_turn[min(max(turn, 1), self.last)]; board = options[int(self.rng.integers(len(options)))]
            self.native.set_team(int(g), s, board); self.expected[int(g)] = (s, board, own, turn)

    def after_step(self, o):
        resolved = np.flatnonzero(o['battle_resolved'])
        if not len(resolved): return
        owners = self.pool.owners.reshape(self.n, 2)
        ghostly = [int(g) for g in resolved if (owners[g] == GHOST).any()]
        if ghostly:
            turns, seeds, teams = self.native.battle_events()
            for g in ghostly:
                if g not in self.expected: raise RuntimeError('a ghost game fought without a swapped-in board')
                s, board, own, turn = self.expected.pop(g)
                if int(turns[g]) != turn: raise RuntimeError(f'turn counter {turn} disagrees with the engine {int(turns[g])}')
                if not np.array_equal(teams[g, s][:, :6], board[:, :6]): raise RuntimeError(f'ghost battle fought the wrong team in game {g}')
                result = int(o['battle_result'][g]); live = 1 - s
                row = self.stats['by_turn'].setdefault(turn, [0, 0, 0]); row[0 if result == live else 1 if result == 2 else 2] += 1
                self.stats['battles'] += 1; self.stats['own_sequence'] += int(own); self.stats['team_checked'] += 1
        if self.expected.keys() & set(int(g) for g in np.flatnonzero(~o['battle_resolved'])) :
            raise RuntimeError('a board was swapped in but no battle followed')
        self.turn[resolved] += 1
