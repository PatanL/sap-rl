"""Run the entity transformer through torch.compile without touching the module tree.

`policy.transformer` stays the plain module, so state_dict keys, checkpoints, deepcopy and load_state_dict are
unchanged. The compiled callable is looked up by the transformer's identity, so a deep-copied policy (frozen
opponents, the acting copy) gets its own wrapper around its own weights and never a stale one.
"""
import torch

import threading

_COMPILED = {}
MODE = {}
import os as _os
if _os.environ.get('OPT_ENCODER_MODE'):
    MODE['mode'] = _os.environ['OPT_ENCODER_MODE']   # e.g. max-autotune-no-cudagraphs: picks the faster of two ways to run each biased linear layer
# One thread at a time may enter a compiled callable. Entering it evaluates guards and may compile; doing that in
# the learner thread while the acting thread runs compiled code deadlocked the process (every thread asleep, one
# holding the interpreter lock, so even a watchdog thread could not run). The GPU work itself is asynchronous, so
# the lock is held only while kernels are issued.
LOCK = threading.Lock()


def fast(transformer):
    key = id(transformer)
    hit = _COMPILED.get(key)
    if hit is None:
        hit = _COMPILED[key] = (transformer, torch.compile(transformer, dynamic=True, **MODE))
    return hit[1]


def enable(policy):
    """Swap the instance's class for one whose encode_entities calls the compiled transformer."""
    base = type(policy)
    if getattr(base, '_fast_encoder', False):
        return policy

    class Fast(base):
        _fast_encoder = True

        def encode_entities(self, observation):
            from sapai_gym1.rl.causal_transport import validate_context
            validate_context(observation, required=self.config.causal_context_schema is not None)
            token_input = (
                self.item_embedding(observation["item_ids"].long())
                + self.kind_embedding(observation["kind_ids"].long())
                + self.status_embedding(observation["status_ids"].long())
                + self.position_embedding(observation["positions"].long())
                + self.numeric_projection(observation["numeric"].float())
            )
            legality = torch.cat(
                [observation["type_mask"].float().flatten(start_dim=1),
                 observation["source_mask"].float().flatten(start_dim=1),
                 observation["target_mask"].float().flatten(start_dim=1)], dim=1)
            token_input[:, 0] = token_input[:, 0] + self.legality_projection(legality)
            token_input = self.input_norm(token_input)
            padding_mask = ~observation["entity_mask"].bool()
            if token_input.shape[0] < 2:
                # The compiled graph is specialised to two or more rows; a single row goes through the plain module
                # (compiling a one-row variant fails inside the compiler and must never be triggered at run time).
                return self.transformer(token_input, src_key_padding_mask=padding_mask)
            with LOCK:
                return fast(self.transformer)(token_input, src_key_padding_mask=padding_mask)

    Fast.__name__ = 'Fast' + base.__name__
    policy.__class__ = Fast
    return policy
