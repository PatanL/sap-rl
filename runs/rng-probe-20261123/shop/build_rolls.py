#!/usr/bin/env python3
"""Build the "known rolls" dataset from Super Auto Pets replays. Stdlib only.

For every perspective and turn this replays the id counter (`Uni`) through the turn:
  Type 4 (start turn) and Type 5 (roll) each create (pet slots - frozen pets) pets followed by
  (food slots - frozen foods) foods, with consecutive `Uni` ids starting at the counter.
A turn is used only if the counter replayed that way lands exactly on the end-of-turn snapshot's
`UNC` (so nothing else consumed ids: no level-up reward, Pigeon crumbs, summons...). Species of
each generated id comes from: the leftover shop in the end-of-turn snapshot, a team pet with the
same id in any snapshot, the target of a buy-and-stack, or the buy-food response (`Spell.Enu`).

Output: rolls.jsonl (one generation event per line) and build_stats.json.
Usage: build_rolls.py <out dir> <raw dir>...
"""
import glob, json, os, sys
from collections import Counter

out_dir = sys.argv[1]
stats = Counter()
rows = []
genesis_rows = []


def J(s):
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


def seed_of(a):
    d = (J(a.get('Response')).get('Data') or {})
    return d.get('Seed'), d.get('Hash')


def shop_items(bor):
    pets = [(m['Id']['Uni'], m.get('Enu') or 0, bool(m.get('Fro')), bool(m.get('Rwrd'))) for m in (bor.get('MiSh') or []) if m]
    foods = [(m['Id']['Uni'], m.get('Enu') or 0, bool(m.get('Fro'))) for m in (bor.get('SpSh') or []) if m]
    return pets, foods


def team_items(bor):
    res = []
    for i, m in enumerate((bor.get('Mins') or {}).get('Items') or []):
        if m:
            res.append((m['Id']['Uni'], m.get('Enu') or 0, (m.get('Poi') or {}).get('x', 0)))
    return res


for d in sys.argv[2:]:
    dname = os.path.basename(d.rstrip('/'))
    for f in sorted(glob.glob(d + '/*.json')):
        j = json.load(open(f))
        for per in j['perspectives']['perspectives']:
            raw = per['raw']
            stats['perspectives'] += 1
            acts = sorted(raw['Actions'], key=lambda a: a['BuildChangeCount'] or 0)
            gen = (raw.get('GenesisBuildModel') or {}).get('Bor')
            if not gen:
                stats['no genesis'] += 1
                continue
            # global species map from every snapshot (ids are unique within a perspective)
            pet_enu, food_enu = {}, {}
            snaps = {0: gen}
            for a in acts:
                if a['Type'] == 0 and a.get('Build'):
                    bor = J(a['Build']).get('Bor')
                    if bor:
                        snaps[a['Turn']] = bor
            for bor in snaps.values():
                p, fd = shop_items(bor)
                for u, e, _, _ in p:
                    pet_enu[u] = e
                for u, e, _ in fd:
                    food_enu[u] = e
                for u, e, _ in team_items(bor):
                    pet_enu[u] = e
            by_turn = {}
            for a in acts:
                by_turn.setdefault(a['Turn'], []).append(a)
            # genesis shop
            gp, gf = shop_items(gen)
            t1 = [a for a in by_turn.get(1, []) if a['Type'] == 4]
            genesis_rows.append({'dir': dname, 'match': j['meta']['id'], 'pid': per['participationId'],
                                 'pets': [e for u, e, _, _ in sorted(gp)], 'foods': [e for u, e, _ in sorted(gf)],
                                 'list_pets': [e for u, e, _, _ in gp],
                                 'seed_after': seed_of(t1[0])[0] if t1 else None, 'hash_after': seed_of(t1[0])[1] if t1 else None,
                                 'unc': gen.get('UNC')})
            prev_seed = prev_hash = None
            for turn in sorted(by_turn):
                tacts = by_turn[turn]
                stats['turns'] += 1
                s_prev, s_end = snaps.get(turn - 1), snaps.get(turn)
                n4 = sum(a['Type'] == 4 for a in tacts)
                n5 = sum(a['Type'] == 5 for a in tacts)
                ok = True
                if s_prev is None or s_end is None or n4 != 1 or (s_end.get('Tur') or 0) != turn:
                    stats['turn skipped: no snapshot pair / odd type-4 count'] += 1
                    ok = False
                if ok:
                    stats['turns with snapshots'] += 1
                    if (s_end.get('Rold') or 0) == n5:
                        stats['Rold == number of Type 5'] += 1
                    else:
                        stats['Rold != number of Type 5'] += 1
                    misc, spsc = s_end.get('MiSC') or 0, s_end.get('SpSC') or 0
                    cur = s_prev.get('UNC') or 0
                    pp, pf = shop_items(s_prev)
                    kind = {}
                    frozen = set()
                    shop = set()
                    carry_rwrd = False
                    if turn > 1:
                        for u, e, fro, rw in pp:
                            if fro:
                                kind[u] = 'P'; frozen.add(u); shop.add(u); carry_rwrd |= rw
                        for u, e, fro in pf:
                            if fro:
                                kind[u] = 'F'; frozen.add(u); shop.add(u)
                    else:
                        for u, e, fro, rw in pp:
                            kind[u] = 'P'; shop.add(u)
                        for u, e, fro in pf:
                            kind[u] = 'F'; shop.add(u)
                    pos = {x: u for u, e, x in team_items(s_prev)}      # position -> uni
                    merged_into = {}                                     # bought uni -> target uni
                    food_bought = {}
                    events = []
                    turn_seed = None
                    rolls_before = 0
                    bad = None
                    stray_buy = 0
                    for a in tacts:
                        t = a['Type']
                        if t in (0, 1, 2, 3, 12):
                            continue
                        rq = J(a.get('Request'))
                        data = rq.get('Data') or {}
                        for fz in data.get('BoardFreezes') or []:
                            u = (fz.get('ItemId') or {}).get('Uni', 0)
                            if fz.get('Freeze'):
                                frozen.add(u)
                            else:
                                frozen.discard(u)
                        bo = data.get('BoardOrders')
                        if bo:
                            listed = {(o.get('MinionId') or {}).get('Uni', 0): (o.get('Point') or {}).get('x', 0) for o in bo}
                            pos = {x: u for x, u in pos.items() if u not in listed}
                            for u, x in listed.items():
                                pos[x] = u
                        seed, hsh = seed_of(a)
                        if t == 4:
                            turn_seed = seed
                        if t == 4 or t == 5:
                            if t == 4 and turn == 1:
                                pass  # genesis shop already present; nothing generated
                            else:
                                keep = {u for u in shop if u in frozen}
                                if any(u not in kind for u in keep):
                                    bad = 'frozen item of unknown kind'
                                    break
                                fp = sum(kind[u] == 'P' for u in keep)
                                ff = sum(kind[u] == 'F' for u in keep)
                                npet, nfood = max(0, misc - fp), max(0, spsc - ff)
                                ids_p = list(range(cur + 1, cur + 1 + npet))
                                ids_f = list(range(cur + 1 + npet, cur + 1 + npet + nfood))
                                cur += npet + nfood
                                for u in ids_p:
                                    kind[u] = 'P'
                                for u in ids_f:
                                    kind[u] = 'F'
                                shop = keep | set(ids_p) | set(ids_f)
                                events.append({'type': t, 'bcc': a['BuildChangeCount'], 'seed': seed, 'hash': hsh, 'turn_seed': turn_seed, 'rolls_before': rolls_before,
                                               'prev_seed': prev_seed, 'prev_hash': prev_hash, 'ids_p': ids_p, 'ids_f': ids_f,
                                               'frozen_pets': fp, 'frozen_foods': ff, 'bought': 0, 'time': a.get('CreatedOn')})
                            if t == 5:
                                rolls_before += 1
                        elif t == 6:
                            u = (rq.get('MinionId') or {}).get('Uni', 0)
                            x = (rq.get('Point') or {}).get('x', 0)
                            if u not in shop:
                                stray_buy += 1
                            shop.discard(u); frozen.discard(u)
                            if x in pos:
                                merged_into[u] = pos[x]
                            else:
                                pos[x] = u
                            if events and (u in events[-1]['ids_p']):
                                events[-1]['bought'] += 1
                        elif t == 8:
                            u = (rq.get('SpellId') or {}).get('Uni', 0)
                            ev = J(a.get('Response')).get('Event') or {}
                            ev = ev.get('Event', ev)
                            food_bought[u] = (ev.get('Spell') or {}).get('Enu') or 0
                            if u not in shop:
                                stray_buy += 1
                            shop.discard(u); frozen.discard(u)
                            if events and (u in events[-1]['ids_f']):
                                events[-1]['bought'] += 1
                        elif t == 7:
                            su = (rq.get('SourceMinionId') or {}).get('Uni', 0)
                            tu = (rq.get('TargetMinionId') or {}).get('Uni', 0)
                            pos = {x: u for x, u in pos.items() if u != su}
                            merged_into[su] = tu
                        elif t == 9:
                            u = (rq.get('Minion') or {}).get('Uni', 0)
                            pos = {x: uu for x, uu in pos.items() if uu != u}
                        if seed is not None:
                            prev_seed, prev_hash = seed, hsh
                    if bad:
                        stats['turn skipped: ' + bad] += 1
                    else:
                        extra = (s_end.get('UNC') or 0) - cur
                        stats['turns replayed'] += 1
                        if extra != 0:
                            stats['turn rejected: id counter off (extra ids consumed)'] += 1
                        else:
                            stats['turns with exact id count'] += 1
                            if stray_buy:
                                stats['exact turns with a buy of an id not in the modelled shop'] += 1
                            # verify leftover shop against the model
                            ep, ef = shop_items(s_end)
                            left = {u for u, *_ in ep} | {u for u, *_ in ef}
                            wrongkind = sum(kind.get(u) != 'P' for u, *_ in ep) + sum(kind.get(u) != 'F' for u, *_ in ef)
                            if left != shop or wrongkind or stray_buy:
                                stats['exact-count turns failing the leftover/kind/buy check (dropped)'] += 1
                            else:
                                stats['turns verified'] += 1

                                def species(u, depth=0):
                                    if u in pet_enu:
                                        return pet_enu[u]
                                    if u in merged_into and depth < 6:
                                        return species(merged_into[u], depth + 1)
                                    return None
                                list_order = [u for u, *_ in ep]
                                for k, ev in enumerate(events):
                                    pets = [species(u) for u in ev['ids_p']]
                                    foods = [food_enu.get(u, food_bought.get(u)) for u in ev['ids_f']]
                                    complete = all(p is not None for p in pets) and all(x is not None for x in foods)
                                    row = {'dir': dname, 'match': j['meta']['id'], 'pid': per['participationId'], 'turn': turn,
                                           'tier': s_end.get('Ti'), 'type': ev['type'], 'k': k, 'n_events': len(events), 'bcc': ev['bcc'],
                                           'seed': ev['seed'], 'prev_seed': ev['prev_seed'], 'hash': ev['hash'], 'prev_hash': ev['prev_hash'],
                                           'turn_seed': ev['turn_seed'], 'rolls_before': ev['rolls_before'],
                                           'pet_slots': misc, 'food_slots': spsc, 'frozen_pets': ev['frozen_pets'], 'frozen_foods': ev['frozen_foods'],
                                           'first_uni': (ev['ids_p'] + ev['ids_f'] + [None])[0], 'pets': pets, 'foods': foods, 'complete': complete,
                                           'bought': ev['bought'], 'time': ev['time']}
                                    if k == len(events) - 1 and ev['bought'] == 0 and ev['frozen_pets'] == 0:
                                        # untouched last shop: record display order of the pets (list order in the snapshot)
                                        if set(list_order) == set(ev['ids_p']) and not any(fr for _, _, fr, _ in ep):
                                            row['display_rank'] = [list_order.index(u) for u in ev['ids_p']]
                                    rows.append(row)
                                    stats['generation events'] += 1
                                    stats['complete events'] += complete
                # keep prev_seed in step even when the turn was not replayed
                if not ok or True:
                    for a in tacts:
                        s, h = seed_of(a)
                        if s is not None:
                            prev_seed, prev_hash = s, h

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'rolls.jsonl'), 'w') as fh:
    for r in rows:
        fh.write(json.dumps(r, separators=(',', ':')) + '\n')
with open(os.path.join(out_dir, 'genesis.jsonl'), 'w') as fh:
    for r in genesis_rows:
        fh.write(json.dumps(r, separators=(',', ':')) + '\n')
by = Counter()
for r in rows:
    by[(r['turn'] if r['turn'] < 12 else 12, r['type'], 'complete' if r['complete'] else 'partial')] += 1
stats_out = {'stats': dict(stats), 'by_turn_type': {f'turn {t} type {ty} {c}': n for (t, ty, c), n in sorted(by.items())}}
json.dump(stats_out, open(os.path.join(out_dir, 'build_stats.json'), 'w'), indent=1)
for k, v in sorted(stats.items()):
    print(f'{v:8d}  {k}')
print('events by turn (turn, type): complete / partial')
for t in range(1, 13):
    for ty in (4, 5):
        print(f'  turn {t:2d}{"+" if t == 12 else " "} type {ty}: {by[(t, ty, "complete")]:6d} / {by[(t, ty, "partial")]:6d}')
