"""Full matches against ranked players' boards, with optional rollout search for the bot's decisions.

Harness (same idea as runs/player-analysis-20261123/ghost_match.py, on the Rust vector environment): the bot sits in
seat 0 and shops; seat 1 never shops, and just before each battle its team is replaced by a board a ranked player
took into battle on that turn. Those opposing boards come from the YARDSTICK side of the pool and are used for
scoring only.

Search (--cands M --rollouts R): at every decision the bot draws M candidate moves from its own policy (the first
is the greedy one). For each candidate, R copies of the game are made, each with its future shop rolls re-randomised
(the searcher cannot see real future rolls), the candidate is played, and the policy finishes the turn by sampling.
The board each copy would take into battle is fought against B boards from the TRAINING side of the ranked pool for
that turn (never the yardstick side), and the candidate with the best mean score is played. With --cands 1 the bot
just plays its policy (greedy unless --sampled), which must reproduce ghost_match.py.

Usage: search_match.py <checkpoint> <out.json> --matches 256 [--roots 64] [--cands 4] [--rollouts 3] [--boards 16] [--samples 2]
"""
import argparse, hashlib, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from evaluate import load
from runtime import verify_engine

ENV_DIR = '/sap/engine-rs-pinned/v31g-env2'
sys.path.insert(0, ENV_DIR)
from sap_env_rs import SapVecEnv  # noqa: E402

TRAIN_POOL = '/sap/runs/train-ghost-20261123/ghost-train-pool.json'
YARD_POOL = '/sap/runs/search-20261123/yardstick-pool.json'
YARD_PATH = [YARD_POOL]
END = (0, 10, 5)


def pools(env):
    train = json.load(open(TRAIN_POOL)); assert train['split'] == 'train'
    by_turn = defaultdict(list)
    for row in train['sequences']:
        assert int(hashlib.sha256(row['match'].encode()).hexdigest(), 16) % 5 >= 2
        for t, b in row['boards'].items(): by_turn[int(t)].append(env.encode_board(b))
    yard = json.load(open(YARD_PATH[0])); assert yard['split'] == 'yardstick'
    yard_turn = {int(t): [env.encode_board(b) for b in boards] for t, boards in yard['byTurn'].items()}
    last = lambda d: max(t for t, b in d.items() if len(b) >= 30)
    return {t: np.stack(b) for t, b in by_turn.items()}, last(by_turn), yard_turn, last(yard_turn)


@torch.inference_mode()
def main(a):
    verify_engine(); torch.set_num_threads(1)
    model, meta = load(a.checkpoint, 'cuda')
    G, M, R = a.roots, a.cands, a.rollouts
    per_root = M * R if M > 1 else 0
    N = G * (1 + per_root)
    env = SapVecEnv(N, threads=a.threads, library=ENV_DIR + '/libsap_engine.so')
    train_turn, train_last, yard_turn, yard_last = pools(env)
    rng = np.random.default_rng(a.seed)
    width = model.config.memory_dim
    results, per_turn = [], defaultdict(Counter)
    count = Counter(); clock = time.perf_counter; begin = clock()

    rival = None
    if a.versus:
        rival = model if a.versus.resolve() == a.checkpoint.resolve() else load(a.versus, 'cuda')[0]

    def act(slots, memory, deterministic, seat=0, policy=None):
        """Policy decision for one seat of the given game slots."""
        obs = {k: torch.from_numpy(v[slots, seat]).cuda() for k, v in env.obs.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = (policy or model).act(obs, memory=memory, deterministic=deterministic)
        count['network_rows'] += len(slots)
        return out.actions.cpu().numpy(), out.end_freeze_mask.cpu().numpy(), out.memory.float()

    seeds_all = [(a.seed * 1000003 + i * 7919) % 2 ** 31 for i in range(a.matches)]
    for start in range(0, a.matches, G):
        group = seeds_all[start:start + G]; g = len(group); roots = np.arange(g)
        which = np.zeros(N, bool); which[:g] = True
        seeds = np.zeros(N, np.int64); seeds[:g] = group
        env.reset(seeds, which=which)
        memory = model.initial_memory(g, 'cuda'); turn = np.ones(g, np.int32); done = np.zeros(g, bool)
        memory1 = rival.initial_memory(g, 'cuda') if rival is not None else None
        def step_roots(acting, rival_ids, rival_a, rival_m):
            mem = memory.index_select(0, torch.as_tensor(acting, device='cuda'))
            first_actions, first_masks, mem_next = act(acting, mem, deterministic=not a.sampled)
            chosen_a, chosen_m = first_actions.copy(), first_masks.copy()
            if M > 1:
                k = len(acting)
                cand_a = np.empty((k, M, 3), np.int64); cand_m = np.empty((k, M), np.int64)
                cand_a[:, 0], cand_m[:, 0] = first_actions, first_masks
                for m in range(1, M):
                    cand_a[:, m], cand_m[:, m], _ = act(acting, mem, deterministic=False)
                # Roots whose candidates are all the same move need no search.
                same = (cand_a == cand_a[:, :1]).all((1, 2)) & (cand_m == cand_m[:, :1]).all(1)
                todo = np.flatnonzero(~same); count['searched_decisions'] += len(todo); count['unanimous_decisions'] += int(same.sum())
                def playouts():
                    src = np.repeat(acting[todo], per_root)
                    child = (G + (acting[todo, None] * per_root) + np.arange(per_root)[None]).ravel()
                    env.copy(src, child)
                    if a.crn:
                        # Common random numbers: playout j of every candidate of a root sees the same future shop rolls.
                        base = rng.integers(2 ** 62, size=(len(todo), 1, R))
                        env.rekey(child, np.broadcast_to(base, (len(todo), M, R)).reshape(-1))
                    else:
                        env.rekey(child, rng.integers(2 ** 62, size=len(child)))
                    count['copies'] += len(child)
                    if rival is not None:
                        # An active seat must act in every step, so in a copy the other seat ends its turn at once
                        # (zero gold makes END legal). The copy is never taken into a battle.
                        for c in child[env.active[child, 1].astype(bool)]: env.set_gold(int(c), 1, 0)
                    c_action = np.repeat(cand_a[todo], R, axis=1).reshape(-1, 3); c_mask = np.repeat(cand_m[todo], R, axis=1).ravel()
                    c_mem = mem_next.index_select(0, torch.as_tensor(np.repeat(todo, per_root), device='cuda')).clone()
                    live = c_action[:, 0] != 0      # a child whose move is END is evaluated as it stands
                    actions = np.full((N, 2, 3), -1, np.int64); actions[:, :, 1] = 10; actions[:, :, 2] = 5
                    masks = np.zeros((N, 2), np.int64)
                    for _ in range(60):
                        if not live.any(): break
                        ids = child[live]
                        sel = np.zeros(N, bool); sel[ids] = True
                        actions[ids, 0] = c_action[live]; masks[ids, 0] = c_mask[live]
                        ghost_active = env.active[ids, 1].astype(bool)
                        actions[ids, 1] = np.where(ghost_active[:, None], np.array(END), np.array((-1, 10, 5)))
                        env.step(actions, masks, which=sel)
                        if (env.status[ids] < 0).any():
                            bad = int(np.flatnonzero(env.status[ids] < 0)[0]); raise RuntimeError(env.error(int(ids[bad])))
                        if env.battle_resolved[ids].any(): raise RuntimeError('a rollout resolved a battle')
                        live_index = np.flatnonzero(live)
                        na, nm, nmem = act(ids, c_mem.index_select(0, torch.as_tensor(live_index, device='cuda')), deterministic=a.greedy_playouts)
                        c_mem.index_copy_(0, torch.as_tensor(live_index, device='cuda'), nmem)
                        c_action[live_index] = na; c_mask[live_index] = nm
                        live[live_index[na[:, 0] == 0]] = False
                    else:
                        raise RuntimeError('rollout did not end its turn')
                    # Common random numbers within a root: same opposing boards and battle seeds for every candidate.
                    boards = np.empty((len(child), a.boards, 5, 7), np.int32); eval_seeds = np.empty(len(child), np.int64)
                    for j, r in enumerate(acting[todo]):
                        options = train_turn[min(int(turn[r]), train_last)]
                        boards[j * per_root:(j + 1) * per_root] = options[rng.integers(len(options), size=a.boards)]
                        eval_seeds[j * per_root:(j + 1) * per_root] = int(rng.integers(2 ** 62))
                    try:
                        wdl = env.eval_vs_batch(child, np.zeros(len(child), np.int32), boards, a.samples, eval_seeds)
                    except ValueError:
                        # A fight failed inside the engine: find it, keep the evidence, score that copy as all draws.
                        wdl = np.zeros((len(child), a.boards, 3), np.int32)
                        for j, c in enumerate(child):
                            try: wdl[j] = env.eval_vs(int(c), 0, boards[j], a.samples, int(eval_seeds[j]))
                            except ValueError as err:
                                wdl[j, :, 1] = a.samples; count['eval_failures'] += 1
                                if count['eval_failures'] <= 20:
                                    bad = []
                                    for b in range(a.boards):
                                        try: env.eval_vs(int(c), 0, boards[j, b:b + 1], a.samples, int(eval_seeds[j]))
                                        except ValueError: bad.append(b)
                                    json.dump(dict(error=str(err), state=env.state(int(c)), samples=a.samples, seed=int(eval_seeds[j]),
                                                   failing_boards=[env.decode_board(boards[j, b]) for b in bad], failing_raw=[boards[j, b].tolist() for b in bad]),
                                              open(f'/out/results/eval-failure-{int(time.time() * 1000) % 10 ** 9}-{count["eval_failures"]}.json', 'w'), indent=1)
                    count['eval_fights'] += len(child) * a.boards * a.samples
                    score = ((wdl[:, :, 0] + .5 * wdl[:, :, 1]).sum(1) / (a.boards * a.samples)).reshape(len(todo), M, R).mean(2)
                    return score
                if len(todo):
                    score = playouts()
                    if a.audit:
                        # Label noise: a second, independent set of playouts for the same candidates. The move the first
                        # set picks is re-scored by the second, which it did not see.
                        fresh = playouts(); pick = score.argmax(1); rows = np.arange(len(todo))
                        claimed = score[rows, pick] - score[:, 0]; true = fresh[rows, pick] - fresh[:, 0]
                        changed = pick != 0
                        count['audit_decisions'] += len(todo); count['audit_changed'] += int(changed.sum())
                        count['audit_same_pick'] += int((fresh.argmax(1) == pick).sum())
                        count['audit_claimed_sum'] += float(claimed[changed].sum()); count['audit_true_sum'] += float(true[changed].sum())
                        for margin in (0.02, 0.05, 0.10, 0.15):
                            keep = changed & (claimed >= margin)
                            count[f'audit_n_{margin}'] += int(keep.sum()); count[f'audit_true_{margin}'] += float(true[keep].sum())
                            count[f'audit_worse_{margin}'] += int((true[keep] < 0).sum())
                    score[:, 0] += 1e-9                       # ties go to the greedy move
                    best = score.argmax(1)
                    chosen_a[todo] = cand_a[todo, best]; chosen_m[todo] = cand_m[todo, best]
                    count['changed_decisions'] += int((best != 0).sum())
                    count['gain_sum'] += float((score.max(1) - score[:, 0]).sum())
            memory.index_copy_(0, torch.as_tensor(acting, device='cuda'), mem_next)
            # Play the chosen move on the roots; where the bot ends its turn the ranked board goes in first.
            actions = np.full((N, 2, 3), -1, np.int64); actions[:, :, 1] = 10; actions[:, :, 2] = 5
            masks = np.zeros((N, 2), np.int64); sel = np.zeros(N, bool); sel[acting] = True
            actions[acting, 0] = chosen_a; masks[acting, 0] = chosen_m
            if rival is not None:
                if len(rival_ids):
                    sel[rival_ids] = True; actions[rival_ids, 1] = rival_a; masks[rival_ids, 1] = rival_m
                stepped = np.flatnonzero(sel[:g])
                env.step(actions, masks, which=sel)
                if (env.status[stepped] < 0).any(): raise RuntimeError(env.error(int(stepped[np.flatnonzero(env.status[stepped] < 0)[0]])))
                count['root_decisions'] += len(acting); return
            ghost_active = env.active[acting, 1].astype(bool)
            actions[acting[ghost_active], 1] = END
            for r in acting[chosen_a[:, 0] == 0]:
                if env.active[r, 1]: raise RuntimeError('ghost seat had not ended')
                options = yard_turn[min(int(turn[r]), yard_last)]
                # The opposing board depends only on the match seed and the turn, so runs with and without search meet the same boards.
                env.set_team(int(r), 1, options[(int(group[r]) * 1000003 + int(turn[r]) * 7919) % len(options)])
            env.step(actions, masks, which=sel)
            if (env.status[acting] < 0).any():
                bad = int(np.flatnonzero(env.status[acting] < 0)[0]); raise RuntimeError(env.error(int(acting[bad])))
            count['root_decisions'] += len(acting)
        def settle(stepped):
            for r in stepped[env.battle_resolved[stepped].astype(bool)]:
                result = int(env.battle_result[r]); per_turn[int(turn[r])]['w' if result == 0 else 'd' if result == 2 else 'l'] += 1
                turn[r] += 1
                if env.terminated[r]:
                    done[r] = True
                    results.append(dict(seed=int(group[r]), winner=int(env.winner[r]), lastTurn=int(turn[r]) - 1,
                                        lives=[int(x) for x in env.lives[r]], wins=[int(x) for x in env.wins[r]]))
        while not done.all():
            acting = roots[~done & env.active[:g, 0].astype(bool)]
            rival_a = rival_m = None; rival_ids = np.zeros(0, np.int64)
            if rival is not None:
                # The other seat is a live policy playing greedily, without search.
                rival_ids = roots[~done & env.active[:g, 1].astype(bool)]
                if len(rival_ids):
                    index = torch.as_tensor(rival_ids, device='cuda')
                    rival_a, rival_m, next1 = act(rival_ids, memory1.index_select(0, index), True, seat=1, policy=rival)
                    memory1.index_copy_(0, index, next1)
                if not len(acting):
                    actions = np.full((N, 2, 3), -1, np.int64); actions[:, :, 1] = 10; actions[:, :, 2] = 5
                    masks = np.zeros((N, 2), np.int64); sel = np.zeros(N, bool); sel[rival_ids] = True
                    actions[rival_ids, 1] = rival_a; masks[rival_ids, 1] = rival_m
                    env.step(actions, masks, which=sel)
                    if (env.status[rival_ids] < 0).any(): raise RuntimeError(env.error(int(rival_ids[np.flatnonzero(env.status[rival_ids] < 0)[0]])))
                    acting = rival_ids[:0]
            elif not len(acting): raise RuntimeError('no root can act')
            if rival is None:
                # The ghost seat ends at once with no gold; do that on the root before any copy is made.
                for r in acting[env.active[acting, 1].astype(bool)]: env.set_gold(int(r), 1, 0)
            if len(acting):
              step_roots(acting, rival_ids, rival_a, rival_m)
            settle(np.union1d(acting, rival_ids))
        n = len(results); w = sum(x['winner'] == 0 for x in results)
        print(json.dumps(dict(matches=n, seat0Wins=w, score=w / n, seconds=round(clock() - begin, 1), **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in count.items()})), flush=True)
    seconds = clock() - begin; n = len(results); w = sum(x['winner'] == 0 for x in results); d = sum(x['winner'] == 2 for x in results); score = (w + .5 * d) / n
    turns = {t: dict(battles=sum(c.values()), score=(c['w'] + .5 * c['d']) / sum(c.values())) for t, c in sorted(per_turn.items())}
    json.dump(dict(checkpoint=meta, matches=n, seat0Wins=w, draws=d, score=score, se=(score * (1 - score) / n) ** .5,
                   meanLastTurn=sum(x['lastTurn'] for x in results) / n, perTurn=turns, args={k: str(v) for k, v in vars(a).items()},
                   seconds=seconds, counts=dict(count), root_decisions_per_s=count['root_decisions'] / seconds,
                   network_rows_per_s=count['network_rows'] / seconds, results=results), open(a.out, 'w'), indent=1)
    print('per turn:', {t: round(v['score'], 3) for t, v in turns.items() if v['battles'] >= 20})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('checkpoint', type=Path); p.add_argument('out', type=Path)
    p.add_argument('--matches', type=int, default=256); p.add_argument('--roots', type=int, default=64)
    p.add_argument('--cands', type=int, default=1); p.add_argument('--rollouts', type=int, default=3)
    p.add_argument('--boards', type=int, default=16); p.add_argument('--samples', type=int, default=2)
    p.add_argument('--threads', type=int, default=5); p.add_argument('--seed', type=int, default=20261123); p.add_argument('--sampled', action='store_true'); p.add_argument('--versus', type=Path); p.add_argument('--audit', action='store_true'); p.add_argument('--crn', action='store_true'); p.add_argument('--greedy-playouts', action='store_true')
    p.add_argument('--yard', default=YARD_POOL)
    args = p.parse_args(); YARD_PATH[0] = args.yard; main(args)
