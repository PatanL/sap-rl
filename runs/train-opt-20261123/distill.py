"""Distil a 14.35M teacher into the 1.35M network ("fresh" arm): the student plays, the teacher labels.

Games run on the Rust vector environment with the STUDENT choosing the moves (early on a share of moves is the
teacher's, decaying to zero, so the first states are not random play). On every decision the teacher, whose memory
follows the same observations, gives its full move distribution, its END-freeze slot probabilities, its value and its
battle forecast, and the student is trained to match them. No game result and no human data is used.

Losses per decision: KL(teacher moves || student moves) over the type x source x target table of primitive moves; where END is available at zero
gold, the teacher's END probability times the cross-entropy of the per-slot freeze toggles; 0.5 * squared value error;
0.1 * KL of the three-way battle forecast. Gradients run through `--window` consecutive waves of the recurrent memory.

Checkpoints are written in the collector's format (arm "fresh"), so `collect_v2.py --arm fresh --resume` continues
from them and `evaluate_vec.py` can score them.
Usage: distill.py <teacher.pt> <out dir> [--games 1024] [--steps 4000] [--window 8] [--lr 3e-4] [--teacher-moves 300] [--save-every 500]
"""
import argparse, json, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from evaluate import load
from runtime import verify_engine, SIMULATOR_REVISION, OBSERVATION_REVISION, PARENT_SHA, SOURCE_OF_TRUTH
from learner import prepare, optimizer_for
from rust_adapter import RustVector, LIB_SHA
from sapai_gym1.rl.action_schema import FlatActionCodec as C, ActionType, MAX_TEAM_SLOTS, MAX_SHOP_SLOTS
from sapai_gym1.rl.causal_transport import checkpoint_context_metadata

END = int(ActionType.END_TURN)
START = 1 + MAX_TEAM_SLOTS


def heads(policy, obs, memory):
    """Everything the loss needs from one network on one batch of decisions."""
    f = policy.encode_decision(obs, memory, forecast=False)
    logp = policy.primitive_log_probabilities(f).float().flatten(1)     # [B, 9 types * 11 sources * 6 targets]
    head = policy.end_freeze_head
    z = head.scorer(torch.cat([f.context[:, None].expand(-1, MAX_SHOP_SLOTS, -1), f.tokens[:, START:START + MAX_SHOP_SLOTS]], dim=-1)).squeeze(-1).float()
    return f, logp, z, policy.value_head(f.context).squeeze(-1).float(), policy.battle_head(f.context).float()


def main(a):
    info = verify_engine(); torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    teacher, teacher_meta = load(a.teacher, 'cuda')
    student, _, _, sem, _, _, _ = prepare(Path('/sap/checkpoints/update-025772.pt'), a.resume, 'fresh', info, a.seed)
    start = 0
    if a.resume is not None:
        start = int(torch.load(a.resume, map_location='cpu', weights_only=True, mmap=True)['update'])
    student = student.cuda().train()
    rates = dict(base=a.lr, opponent_history=a.lr, opponent_adapter=a.lr, end_freeze=a.lr)
    opt = optimizer_for(student, None, rates, 1)
    for field, group in (('opponent_module_learning_rate', 'opponent_history'), ('opponent_adapter_learning_rate', 'opponent_adapter'), ('end_freeze_learning_rate', 'end_freeze')):
        sem[field] = rates[group]
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'args.json').write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    n = a.games; env = RustVector(n, a.threads); o = env.output; rng = np.random.default_rng([a.seed, start])
    env.reset(np.arange(n), rng.integers(2 ** 62, size=n), o)
    s_mem = student.initial_memory(2 * n, 'cuda'); t_mem = teacher.initial_memory(2 * n, 'cuda')
    moves = np.zeros((n, 2, 3), 'i8'); freeze = np.zeros((n, 2), 'i8')
    loss_sum = torch.zeros((), device='cuda'); rows = 0; stats = dict(kl=0., value=0., agree=0., freeze=0., rows=0)
    step, wave, decisions, matches, begin = start, 0, 0, 0, time.perf_counter()

    def save(path):
        record = dict(model=student.state_dict(), optimizer=opt.state_dict(), config=asdict(student.config), update=step,
                      transitions=decisions, learner_primitive_actions=decisions, simulator_revision=SIMULATOR_REVISION,
                      engine_payload_sha256=info['payload_sha256'], source_of_truth=SOURCE_OF_TRUTH, observation_revision=OBSERVATION_REVISION,
                      causal_context=checkpoint_context_metadata(student.config), training_semantics=sem, wide_arm='fresh',
                      parent_checkpoint_sha256=PARENT_SHA, rust_environment_sha256=LIB_SHA, collector='distill-student-acts',
                      teacher_sha256=teacher_meta['sha256'], distillation=dict(games=n, window=a.window, lr=a.lr, teacher_moves=a.teacher_moves))
        tmp = path.with_suffix('.pt.tmp'); torch.save(record, tmp); tmp.replace(path)

    while step < start + a.steps and time.perf_counter() - begin < a.seconds:
        ids = np.flatnonzero(o['active'].ravel()); ix = torch.from_numpy(ids).cuda()
        obs = {k: torch.from_numpy(v.reshape(-1, *v.shape[2:])[ids]).cuda() for k, v in o['obs'].items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                tf, t_logp, t_z, t_value, t_battle = heads(teacher, obs, t_mem.index_select(0, ix))
            sf, s_logp, s_z, s_value, s_battle = heads(student, obs, s_mem.index_select(0, ix))
        p = t_logp.exp()
        kl = (p * (t_logp.clamp_min(-1e4) - s_logp.clamp_min(-1e4))).sum(1)      # illegal moves: p = 0 on both sides
        # END-freeze toggles: only where END is available at zero gold, on slots that may be toggled.
        mask = obs['target_mask'].bool()
        eligible = mask[:, ActionType.FREEZE, :MAX_SHOP_SLOTS, -1] | mask[:, ActionType.UNFREEZE, :MAX_SHOP_SLOTS, -1]
        active = (obs['numeric'][:, 0, 11].abs() < 1e-8) & obs['type_mask'][:, END].bool()
        slot = F.binary_cross_entropy_with_logits(s_z, torch.sigmoid(t_z), reduction='none') - F.binary_cross_entropy_with_logits(t_z, torch.sigmoid(t_z), reduction='none')
        freeze_loss = (slot * (eligible & active[:, None])).sum(1) * p.view(-1, 9, 66)[:, END].sum(1)
        value_loss = .5 * (s_value - t_value).square()
        battle = (F.softmax(t_battle, -1) * (F.log_softmax(t_battle, -1) - F.log_softmax(s_battle, -1))).sum(1)
        loss_sum = loss_sum + (kl + freeze_loss + value_loss + .1 * battle).sum(); rows += len(ids)
        with torch.no_grad():
            stats['kl'] += float(kl.sum()); stats['value'] += float(value_loss.sum()); stats['freeze'] += float(freeze_loss.sum())
            stats['agree'] += float((s_logp.argmax(1) == t_logp.argmax(1)).sum()); stats['rows'] += len(ids)
            beta = max(0., 1. - (step - start) / a.teacher_moves) if a.teacher_moves > 0 and a.resume is None else 0.
            use_teacher = torch.rand(len(ids), device='cuda') < beta
            flat = torch.where(use_teacher, Categorical(logits=t_logp).sample(), Categorical(logits=s_logp).sample())
            action = torch.stack((flat // 66, (flat // 6) % 11, flat % 6), 1)
            z = torch.where(use_teacher[:, None], t_z, s_z)
            dist, _ = student.end_freeze_head.distribution(z, obs, action[:, 0])
            subset = dist.sample()
        s_mem = s_mem.index_copy(0, ix, sf.memory.float()); t_mem.index_copy_(0, ix, tf.memory.float())
        moves.reshape(-1, 3)[ids] = action.cpu().numpy(); freeze.ravel()[ids] = subset.cpu().numpy()
        env.step(moves, freeze, o); decisions += len(ids); wave += 1
        done = np.flatnonzero(o['terminated'])
        if len(done):
            matches += len(done); env.reset(done, rng.integers(2 ** 62, size=len(done)), o)
            seats = torch.as_tensor(np.stack((done * 2, done * 2 + 1), 1).ravel(), device='cuda')
            keep = torch.ones(2 * n, 1, device='cuda'); keep[seats] = 0.
            s_mem = s_mem * keep; t_mem.index_fill_(0, seats, 0.)
        if wave % a.window == 0:
            opt.zero_grad(set_to_none=True); (loss_sum / rows).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 10.); opt.step()
            s_mem = s_mem.detach(); loss_sum = torch.zeros((), device='cuda'); rows = 0; step += 1
            if step % 50 == 0:
                r = stats['rows']; elapsed = time.perf_counter() - begin
                line = dict(step=step, seconds=round(elapsed, 1), decisions=decisions, decisions_per_s=round(decisions / elapsed), matches=matches,
                            kl=round(stats['kl'] / r, 4), value=round(stats['value'] / r, 5), freeze=round(stats['freeze'] / r, 5), agree=round(stats['agree'] / r, 4), teacher_moves=round(beta, 2))
                print(json.dumps(line), flush=True)
                with open(a.out / 'progress.jsonl', 'a') as fp: fp.write(json.dumps(line) + '\n')
                stats = dict(kl=0., value=0., agree=0., freeze=0., rows=0)
            if step % a.save_every == 0: save(a.out / f'update-{step:06d}.pt')
    if step % a.save_every: save(a.out / f'update-{step:06d}.pt')
    env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('teacher', type=Path); p.add_argument('out', type=Path)
    p.add_argument('--games', type=int, default=1024); p.add_argument('--steps', type=int, default=4000); p.add_argument('--window', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--teacher-moves', type=int, default=300); p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--threads', type=int, default=5); p.add_argument('--seed', type=int, default=20261123); p.add_argument('--seconds', type=float, default=1e9)
    p.add_argument('--resume', type=Path)
    main(p.parse_args())
