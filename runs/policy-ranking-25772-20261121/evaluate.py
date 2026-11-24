"""Read-only early-shop probe on the archived 20806 terminal tournament."""
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from plan_common import install, load_actor


def observation(packed):
    return {k: torch.tensor(np.array(v["data"], dtype=v["dtype"])).unsqueeze(0)
            for k, v in packed.items()}


def compare_rungs(folder, earlier, later):
    def means(rung):
        return json.loads((folder / f"RUNG{rung}.json").read_text())["summary"]["means"]
    a, b = means(earlier), means(later)
    keys = sorted(b)
    assert set(keys) <= a.keys()
    # Compare the same actions on new worlds. Ties contribute no direction.
    signs = [int(np.sign(a[x] - a[y]) * np.sign(b[x] - b[y]))
             for i, x in enumerate(keys) for y in keys[:i]]
    return {key: signs.count(s) for key, s in [('agree', 1), ('tie', 0), ('reverse', -1)]}


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    install(cuda=False)
    policy = load_actor(device="cpu").model
    folders = sorted(Path('/panel').glob('root-*'))
    results = []
    for folder in folders:
        root = json.loads((folder / 'ROOT.json').read_text())
        memory = None
        for previous in root['publicHistory']:
            memory = policy.encode_decision(observation(previous), memory, forecast=False).memory
        features = policy.encode_decision(observation(root['publicObservation']), memory, forecast=False)
        choice = policy.sample_joint_action(features, deterministic=True)
        key = (choice.actions[0].tolist(), choice.end_freeze_mask.item())
        candidates = [(c['decision']['primitive'], c['decision']['end_freeze_mask']) for c in root['candidates']]
        rung = json.loads((folder / 'RUNG1.json').read_text())['summary']
        raw = rung['raw']
        assert candidates[raw] == (root['rawDecision']['primitive'],root['rawDecision']['end_freeze_mask'])
        idx = candidates.index(key) if key in candidates else None
        effect = None if idx is None else rung['means'][str(idx)] - rung['means'][str(raw)]
        results.append(dict(origin=root['origin'], root=folder.name, raw=raw, latest=idx,
                            effect=effect, rank12=compare_rungs(folder,1,2),
                            rank23=compare_rungs(folder,2,3)))
    assert len({r['origin'] for r in results}) == len(results) == 32
    scored = [r for r in results if r['effect'] is not None]
    effects = np.array([r['effect'] for r in scored])
    rng = np.random.default_rng(20261121)
    interval = np.quantile(rng.choice(effects,(10000,len(effects))).mean(axis=1),[.025,.975])
    summary = dict(origins=len(results), covered=len(scored), switches=sum(r['latest']!=r['raw'] for r in scored),
                   first_action_effect=statistics.mean(effects), origin_bootstrap95=interval.tolist())
    for pair in ('rank12','rank23'):
        summary[pair] = {k:sum(r[pair][k] for r in results) for k in ('agree','tie','reverse')}
    print(json.dumps(summary))
    Path('/out/results.json').write_text(json.dumps(dict(summary=summary, roots=results),indent=2))


if __name__ == '__main__':
    main()
