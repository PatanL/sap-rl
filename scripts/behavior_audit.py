#!/usr/bin/env python3
"""Passive SAP behavior telemetry. Counts describe play; they are not labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import weakref

import numpy as np
import torch
import sapai_gym1

from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import ActionType, FlatActionCodec, ParallelSelfPlayEnv

sys.path.insert(0, str(Path(sapai_gym1.__file__).resolve().parents[1] / 'scripts'))
import evaluate_paired_policy as evaluation


class BehaviorTracker:
    def __init__(self):
        self.counts = [Counter(), Counter()]
        self.turn_counts = [defaultdict(Counter), defaultdict(Counter)]
        self.pending_sales = [0, 0]
        self.early_cakes = [{}, {}]
        self.offers = [{}, {}]

    def add(self, side, turn, key, count=1):
        self.counts[side][key] += int(count)
        self.turn_counts[side][int(turn)][key] += int(count)

    def before(self, core, side, flat_action):
        player = core.agent if side == 0 else core.opponent
        action = FlatActionCodec.decode(flat_action)
        turn, kind = int(core.turn), ActionType(action.action_type)
        self.add(side, turn, f'action:{kind.name}')
        pets = [slot.pet for slot in player.team if not slot.empty]
        context = {'turn': turn, 'gold': player.gold, 'level_target': None, 'old_level': None}
        legal = [FlatActionCodec.decode(index) for index in core._avail_actions(side)]
        buy_targets = defaultdict(set)
        for option in legal:
            if option.action_type in (ActionType.BUY_PET, ActionType.BUY_FOOD):
                buy_targets[(option.action_type, option.source)].add(option.target)
        for index, slot in enumerate(player.shop):
            if slot.empty:
                continue
            identity = (turn, id(slot.obj))
            previous = self.offers[side].get(identity)
            if previous is None or previous['ref']() is not slot.obj:
                previous = {'ref': weakref.ref(slot.obj), 'tags': set()}
                self.offers[side][identity] = previous
            tags = set()
            if player.gold >= slot.cost:
                tags.add(f'affordable_offer_turn:{slot.obj.name}')
            item_kind = ActionType.BUY_PET if slot.slot_type in ('pet', 'levelup') else ActionType.BUY_FOOD
            targets = buy_targets[(item_kind, index)]
            if targets:
                tags.add(f'directly_buyable_offer_turn:{slot.obj.name}')
            if slot.obj.name == 'food-sleeping-pill' and any(
                target < 5 and not player.team[target].empty
                and player.team[target].pet.ability.get('trigger') == 'Faint'
                for target in targets
            ):
                tags.add('directly_buyable_pill_with_faint_pet_offer_turns')
            for tag in tags - previous['tags']:
                self.add(side, turn, tag)
            previous['tags'].update(tags)

        if kind == ActionType.END_TURN:
            self.add(side, turn, 'shop_ends')
            self.add(side, turn, 'zero_gold_ends', int(player.gold == 0))
            count = core.shop_action_count_agent if side == 0 else core.shop_action_count_opponent
            self.add(side, turn, 'action_cap_ends', int(count >= 50))
            # Count inventory coverage, not "missed good actions". A pet copy
            # may be undesirable; earlier toggle restrictions can also matter.
            level_one = Counter(p.name for p in pets if p.level == 1)
            equity = Counter()
            for pet in pets:
                if pet.level == 1:
                    equity[pet.name] += 1 + int(pet.experience)
            matches = [slot for slot in player.shop if not slot.empty and slot.obj.name in level_one]
            chocolates = [slot for slot in player.shop if not slot.empty and slot.obj.name == 'food-chocolate']
            if player.gold == 0 and count < 50:
                self.add(side, turn, 'zero_gold_noncap_ends')
                all_pet_slots = [slot for slot in player.shop if not slot.empty and slot.slot_type in ('pet', 'levelup')]
                owned_names = {pet.name for pet in pets}
                unowned = [slot for slot in all_pet_slots if slot.obj.name not in owned_names]
                foods = [slot for slot in player.shop if not slot.empty and slot.slot_type == 'food']
                self.add(side, turn, 'end_pet_slots', len(all_pet_slots))
                self.add(side, turn, 'end_frozen_pet_slots', sum(slot.frozen for slot in all_pet_slots))
                self.add(side, turn, 'end_unowned_pet_slots', len(unowned))
                self.add(side, turn, 'end_frozen_unowned_pet_slots', sum(slot.frozen for slot in unowned))
                self.add(side, turn, 'end_food_slots', len(foods))
                self.add(side, turn, 'end_frozen_food_slots', sum(slot.frozen for slot in foods))
                self.add(side, turn, 'ends_with_level1_match', int(bool(matches)))
                self.add(side, turn, 'level1_matching_shop_slots', len(matches))
                self.add(side, turn, 'frozen_level1_matching_shop_slots', sum(slot.frozen for slot in matches))
                immediate = [slot for slot in matches if equity[slot.obj.name] >= 2]
                self.add(side, turn, 'ends_with_level2_completing_copy', int(bool(immediate)))
                self.add(side, turn, 'level2_completing_copy_slots', len(immediate))
                self.add(side, turn, 'frozen_level2_completing_copy_slots', sum(slot.frozen for slot in immediate))
                if any(p.level < 3 for p in pets):
                    self.add(side, turn, 'ends_with_usable_chocolate', int(bool(chocolates)))
                    self.add(side, turn, 'usable_chocolate_slots', len(chocolates))
                    self.add(side, turn, 'frozen_usable_chocolate_slots', sum(slot.frozen for slot in chocolates))
            self.add(side, turn, 'low_gold_sells_without_later_purchase', self.pending_sales[side])
            self.pending_sales[side] = 0

        if kind == ActionType.MOVE:
            self.add(side, turn, 'moves_with_gold', int(player.gold > 0))
        if kind == ActionType.SELL and player.gold < 3:
            self.add(side, turn, 'low_gold_sells')
            self.pending_sales[side] += 1
        if kind in (ActionType.BUY_PET, ActionType.BUY_FOOD):
            self.add(side, turn, 'low_gold_sells_followed_by_purchase', self.pending_sales[side])
            self.pending_sales[side] = 0
            item = player.shop[action.source].obj
            target = player.team[action.target].pet if action.target < 5 and not player.team[action.target].empty else None
            self.add(side, turn, f'buy:{item.name}')
            if kind == ActionType.BUY_PET:
                existing_copy = any(p.name == item.name for p in pets)
                if existing_copy:
                    self.add(side, turn, 'matching_pet_buy_stacked' if target else 'matching_pet_buy_separate')
                if target is not None:
                    context.update(level_target=target, old_level=max(target.level, item.level))
            else:
                if item.name == 'food-sleeping-pill' and target is not None:
                    self.add(side, turn, f'pill_target:{target.name}')
                if item.name == 'food-chocolate' and target is not None:
                    context.update(level_target=target, old_level=target.level)
                if item.name == 'food-cake' and turn in (5, 6) and target is not None:
                    self.add(side, turn, 'early_cake_purchases')
                    self.add(side, turn, 'early_cake_distinct_targets', int(id(target) not in self.early_cakes[side]))
                    # Holding the reference avoids Python ID reuse. Merging a
                    # recorded source into a different pet is treated as lost
                    # identity, not silently credited to an inferred successor.
                    self.early_cakes[side].setdefault(id(target), {'pet': target, 'turn9_seen': False, 'sold': False})
        if kind == ActionType.COMBINE:
            target, source = player.team[action.target].pet, player.team[action.source].pet
            context.update(level_target=target, old_level=max(target.level, source.level))
        if kind == ActionType.SELL:
            sold = player.team[action.source].pet
            record = self.early_cakes[side].get(id(sold))
            if record is not None:
                context['cake_sale'] = record
        return context

    def after(self, core, side, context):
        player = core.agent if side == 0 else core.opponent
        target, old = context['level_target'], context['old_level']
        turn = context['turn']
        if target is not None and target.level > old:
            self.add(side, turn, f'levelup:{target.name}:L{target.level}')
        if 'cake_sale' in context:
            context['cake_sale']['sold'] = True
            if turn >= 9:
                self.add(side, turn, 'early_cake_pet_sold_turn9_plus')
                self.add(side, turn, 'early_cake_pet_sale_gold_turn9_plus', int(player.gold - context['gold']))

    def battle(self, core):
        turn = int(core.turn) - 1
        for side, team in enumerate((core.previous_agent_state, core.previous_opponent_state)):
            pets = [slot.pet for slot in team if not slot.empty]
            names = [pet.name for pet in pets]
            adjacent = set(zip(names, names[1:]))
            hurt = ('pet-elephant', 'pet-camel') in adjacent or ('pet-elephant', 'pet-blowfish') in adjacent
            lines = {
                'elephant_hurt': hurt,
                'elephant_hurt_wolverine': hurt and 'pet-wolverine' in names,
                'shark_fly': 'pet-shark' in names and 'pet-fly' in names,
                'parrot_deer': ('pet-deer', 'pet-parrot') in adjacent,
                'mushroom_deer_shark': 'pet-shark' in names and any(p.name == 'pet-deer' and p.status == 'status-extra-life' for p in pets),
                'mushroom_mammoth_tiger': any(p.name == 'pet-mammoth' and p.status == 'status-extra-life' and q.name == 'pet-tiger'
                                               for p, q in zip(pets, pets[1:])),
                'level2_plus_dragon': any(p.name == 'pet-dragon' and p.level >= 2 for p in pets),
                'level2_plus_monkey': any(p.name == 'pet-monkey' and p.level >= 2 for p in pets),
            }
            self.add(side, turn, 'battle_boards')
            for line, present in lines.items():
                self.add(side, turn, f'line:{line}', int(present))
            if turn == 9:
                player = core.agent if side == 0 else core.opponent
                current = {id(slot.pet): slot.pet for slot in player.team if not slot.empty}
                for identity, record in self.early_cakes[side].items():
                    if identity in current and not record['turn9_seen']:
                        record['turn9_seen'] = True
                        self.add(side, turn, 'early_cake_same_pet_present_turn9')
                        self.add(side, turn, 'early_cake_equipped_turn9', int(current[identity].status == 'status-cake'))

    def side_report(self, side):
        return {'counts': dict(sorted(self.counts[side].items())),
                'byTurn': {str(turn): dict(sorted(counts.items())) for turn, counts in sorted(self.turn_counts[side].items())}}


class ObservedSelfPlayEnv(ParallelSelfPlayEnv):
    def __init__(self):
        super().__init__()
        self.tracker = BehaviorTracker()
        resolve = self.core.resolve_action
        def observed_resolve(agent_idx, action):
            before = self.tracker.before(self.core, agent_idx, action)
            result = resolve(agent_idx, action)
            self.tracker.after(self.core, agent_idx, before)
            return result
        self.core.resolve_action = observed_resolve

    def reset(self, seed=None):
        self.tracker = BehaviorTracker()
        self.reset_seed = seed
        return super().reset(seed)

    def clone(self):
        raise ValueError('Passive telemetry environments are not search clones')

    def step(self, *actions):
        result = super().step(*actions)
        if result.battle_resolved:
            self.tracker.battle(self.core)
        return result


def merge_reports(reports):
    counts, turns = Counter(), defaultdict(Counter)
    for report in reports:
        counts.update(report['counts'])
        for turn, values in report['byTurn'].items():
            turns[turn].update(values)
    return {'counts': dict(sorted(counts.items())), 'byTurn': dict(sorted(turns.items(), key=lambda item: int(item[0])))}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--opponent-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=4400801)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    source_digest = file_sha256(Path(__file__))
    torch.set_num_threads(1)
    loaded = [evaluation.load_policy(path, torch.device('cpu')) for path in (args.checkpoint, args.opponent_checkpoint)]
    policies = [item[0] for item in loaded]
    seeds = np.random.default_rng(args.seed).integers(0, 2**31 - 1, size=args.pairs)
    envs = []
    def factory():
        env = ObservedSelfPlayEnv()
        envs.append(env)
        return env
    previous_factory = evaluation.ParallelSelfPlayEnv
    evaluation.ParallelSelfPlayEnv = factory
    records = []
    start = time.perf_counter()
    try:
        evaluation.run_batched_checkpoint_evaluation(*policies, seeds, torch.device('cpu'), batch_size=args.batch_size,
                                                    progress_every=8, game_records=records)
    finally:
        evaluation.ParallelSelfPlayEnv = previous_factory
    games = []
    if len(envs) != args.pairs * 2 or len(records) != args.pairs * 2:
        raise ValueError('Incomplete paired telemetry population')
    for index, env in enumerate(envs):
        pair, candidate_side = divmod(index, 2)
        if env.reset_seed != int(seeds[pair]) or not env.core.is_done():
            raise ValueError('Telemetry game ownership or completion mismatch')
        games.append({'pair': pair, 'seed': int(seeds[pair]), 'candidateSide': candidate_side,
                      'candidate': env.tracker.side_report(candidate_side),
                      'opponent': env.tracker.side_report(1 - candidate_side)})
    payload = {
        'version': 'neutral-behavior-audit-v3-20260905', 'checkpoint': str(args.checkpoint),
        'opponentCheckpoint': str(args.opponent_checkpoint), 'runtimeSimulatorRevision': SIMULATOR_REVISION,
        'checkpointSha256': file_sha256(args.checkpoint), 'opponentCheckpointSha256': file_sha256(args.opponent_checkpoint),
        'checkpointUpdate': loaded[0][1].get('update'), 'opponentCheckpointUpdate': loaded[1][1].get('update'),
        'checkpointTrainingRevisions': [item[1].get('simulator_revision') for item in loaded],
        'auditScriptSha256': source_digest,
        'seed': args.seed, 'pairs': args.pairs, 'games': games, 'matchRecords': records,
        'candidate': merge_reports([game['candidate'] for game in games]),
        'opponent': merge_reports([game['opponent'] for game in games]), 'elapsedSeconds': time.perf_counter() - start,
        'usedForTraining': False,
        'notes': [
            'Counts describe behavior, not whether an action was strategically correct.',
            'Frozen-copy coverage excludes positive-gold and hard-cap endings; counts slots, not independent samples.',
            'Composition presence does not establish that its abilities triggered or helped win.',
            'Offer-turn counts deduplicate each physical item within a turn; affordability alone does not mean a free slot exists.',
            'Pill opportunity means a legal target with a Faint ability, not proof that pilling it would be good.',
            'Early Cake retention follows the same physical pet only; merge transfers are not guessed.',
            'Gold from a later early-Cake pet sale is total proceeds, not a causal estimate of Cake-only value.',
            'CPU diagnostic games are not the primary GPU promotion evaluation.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'games': len(games), 'seconds': payload['elapsedSeconds'], 'candidateCounts': payload['candidate']['counts']}, indent=2))


if __name__ == '__main__':
    main()
