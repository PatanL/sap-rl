"""The weight update in its own process.

One interpreter runs acting, the other learning, so they no longer take turns on one interpreter lock (measured: the
learner reaches 5,500 decisions/s beside a busy actor, against 4,900 to 5,000 learned/s with both in one process).
Nothing about the update changes: the child builds the same policy and optimizer from the same checkpoint with the
same `prepare`, and calls the same `update_fast` on the same segments.

Transport: a batch is sent as the constructor arguments of OwnedTrajectoryBatch (contiguous CPU tensors in shared
memory, a few dozen handles per batch) and rebuilt into segments in the child. After every update the child copies
the new weights into a fixed set of shared CPU tensors, then reports; the parent reads them only after that report and
sends the next batch only after reading, so the two never touch them at the same time.
"""
import os, time, traceback
import torch
import torch.multiprocessing as mp


def pack(segments):
    cat = torch.cat
    first = segments[0]
    return dict(
        observations={k: cat([s.observations[k] for s in segments]) for k in first.observations},
        actions=cat([s.actions for s in segments]), behavior_log_probs=cat([s.behavior_log_probs for s in segments]),
        behavior_versions=cat([s.behavior_versions for s in segments]), rewards=cat([s.rewards for s in segments]),
        discounts=cat([s.discounts for s in segments]), freeze_masks=cat([s.freeze_masks for s in segments]),
        lengths=tuple(int(s.length) for s in segments), initial_memories=torch.stack([s.initial_memory for s in segments]),
        bootstrap_observations={k: torch.stack([s.bootstrap_observation[k] for s in segments]) for k in first.bootstrap_observation},
        terminals=torch.tensor([bool(s.terminal) for s in segments]),
        battle_outcomes=torch.tensor([-1 if s.battle_outcome is None else int(s.battle_outcome) for s in segments]),
        slot_ids=tuple(int(s.slot_id) for s in segments))


def unpack(payload):
    from sapai_gym1.rl.dense_trajectory import OwnedTrajectoryBatch
    return OwnedTrajectoryBatch(**payload).segments()


def child(pipe, a, env_overrides):
    try:
        os.environ.update(env_overrides)
        from dataclasses import asdict
        from pathlib import Path
        from runtime import verify_engine, SIMULATOR_REVISION, OBSERVATION_REVISION, PARENT_SHA, SOURCE_OF_TRUTH
        from learner import prepare, optimizer_for
        from learner_fast import update_fast
        from learner_ops import enable_learner
        from sapai_gym1.rl.causal_transport import checkpoint_context_metadata
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.backends.mha.set_fastpath_enabled(False); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        info = verify_engine()
        _, total = torch.cuda.mem_get_info(); torch.cuda.set_per_process_memory_fraction(float(a['gib']) * 2 ** 30 / total)
        model, state, rates, sem, start, rows, primitives = prepare(Path('/sap/checkpoints/update-025772.pt'), a['resume'], a['arm'], info, a['seed'])
        model.cuda(); opt = optimizer_for(model, state, rates, 1); del state
        enable_learner(model)
        if a['compile']:
            import fast_encoder; fast_encoder.enable(model)
        torch.cuda.manual_seed(a['seed'] + start)
        shared = {k: v.detach().cpu().clone().share_memory_() for k, v in model.state_dict().items()}
        pipe.send(('ready', shared, start))
        version, learned = start, 0

        def save(path):
            record = dict(model=model.state_dict(), optimizer=opt.state_dict(), config=asdict(model.config), update=version,
                          transitions=rows + learned, learner_primitive_actions=primitives, simulator_revision=SIMULATOR_REVISION,
                          engine_payload_sha256=info['payload_sha256'], source_of_truth=SOURCE_OF_TRUTH, observation_revision=OBSERVATION_REVISION,
                          causal_context=checkpoint_context_metadata(model.config), training_semantics=sem, wide_arm=a['arm'],
                          parent_checkpoint_sha256=PARENT_SHA, **a['record'])
            path = Path(path); tmp = path.with_suffix('.pt.tmp'); torch.save(record, tmp); tmp.replace(path)

        while True:
            message = pipe.recv()
            if message is None: break
            kind, payload, path = message
            if kind == 'save':
                save(path); pipe.send(('saved', None, None)); continue
            t = time.perf_counter()
            segments = unpack(payload); del payload
            stats = update_fast(model, opt, segments, sem, 'cuda', a['cap'], compact=a['compact'], retry_oom=True)
            torch.cuda.synchronize()
            for k, v in model.state_dict().items(): shared[k].copy_(v)
            version += 1; learned += stats['decisions']; primitives += stats['primitive_actions']
            elapsed = time.perf_counter() - t
            if path is not None: save(path)
            pipe.send(('done', stats, elapsed))
    except BaseException:
        try: pipe.send(('error', traceback.format_exc(), None))
        except Exception: pass
        raise
    finally:
        pipe.close()


class Future:
    def __init__(self, owner): self.owner = owner
    def done(self): return self.owner.pipe.poll()
    def result(self):
        kind, stats, elapsed = self.owner.pipe.recv()
        if kind == 'error': raise RuntimeError('learner process failed:\n' + stats)
        return stats, elapsed


class RemoteLearner:
    def __init__(self, a, record):
        ctx = mp.get_context('spawn'); self.pipe, other = ctx.Pipe()
        keys = ('SPEED_GEOMETRY', 'SPEED_FUSED_LOSS', 'OPT_CUDNN_GRU')
        args = dict(resume=a.resume, arm=a.arm, seed=a.seed, cap=a.cap, compact=a.compact, compile=a.compile, gib=a.learner_gib, record=record)
        self.process = ctx.Process(target=child, args=(other, args, {k: os.environ[k] for k in keys if k in os.environ}), daemon=True)
        self.process.start(); other.close()
        kind, shared, start = self.pipe.recv()
        if kind != 'ready': raise RuntimeError('learner process failed to start:\n' + str(shared))
        self.shared = shared; self.start = start

    def submit(self, segments, save_path=None):
        self.pipe.send(('update', pack(segments), None if save_path is None else str(save_path)))
        return Future(self)

    def save(self, path):
        self.pipe.send(('save', None, str(path)))
        kind, text, _ = self.pipe.recv()
        if kind != 'saved': raise RuntimeError('learner process failed while saving:\n' + str(text))

    def close(self):
        try: self.pipe.send(None)
        except Exception: pass
        self.process.join(20)
        if self.process.is_alive(): self.process.terminate()
