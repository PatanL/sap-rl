"""Versioned, public-only causal context wire contract for actors and learners."""
import hashlib
import json
from functools import lru_cache

import numpy as np
import torch

from sapai.data import data
from .opponent_context_data import PublicOpponentContext
from .opponent_forecast_data import PublicContextBatch, collate_contexts

CONTEXT_SCHEMA = 'public-before-shop-causal-context-v1-20260906'
WIRE_VERSION = 1
PREFIX = 'causal_context_'
LAYOUT = {
    PREFIX+'version': ((1,), np.dtype('int64'), torch.int64),
    PREFIX+'vocab': ((32,), np.dtype('uint8'), torch.uint8),
    PREFIX+'species': ((15,), np.dtype('int64'), torch.int64),
    PREFIX+'perks': ((15,), np.dtype('int64'), torch.int64),
    PREFIX+'numbers': ((15, 7), np.dtype('float32'), torch.float32),
    PREFIX+'visible': ((15,), np.dtype('int8'), torch.int8),
    PREFIX+'globals': ((8,), np.dtype('float32'), torch.float32),
}


@lru_cache(maxsize=1)
def vocabulary():
    species = ('none', *sorted(k for k in data['pets'] if k != 'pet-none'))
    perks = tuple(dict.fromkeys(['none', 'status-bread', 'status-cake', *sorted(data.get('statuses', {}))]))
    digest = hashlib.sha256(json.dumps({'species': species, 'perks': perks}, separators=(',', ':')).encode()).hexdigest()
    return species, perks, digest


def require_schema(schema):
    if schema not in (None, CONTEXT_SCHEMA):
        raise ValueError('Unsupported causal opponent-context schema')


def capture_context(core, side):
    context = PublicOpponentContext.capture(core, side).features()
    species, perks, digest = vocabulary()
    batch = collate_contexts([context], {s:i for i,s in enumerate(species)}, {s:i for i,s in enumerate(perks)})
    fields = {PREFIX+'version': np.array([WIRE_VERSION], dtype=np.int64),
              PREFIX+'vocab': np.frombuffer(bytes.fromhex(digest), dtype=np.uint8).copy()}
    for key in ('species', 'perks', 'numbers', 'visible', 'globals'):
        fields[PREFIX+key] = getattr(batch, key)[0].numpy().astype(LAYOUT[PREFIX+key][1], copy=True)
    validate_context(fields, required=True)
    return fields


def refresh_context(core):
    """Called only on reset or public battle reveal/next-shop boundary."""
    if getattr(core, 'causal_context_schema', None) is not None:
        require_schema(core.causal_context_schema)
        core._causal_public_context = (capture_context(core, 0), capture_context(core, 1))


def context_for_observation(core, side, schema):
    require_schema(schema)
    if schema is None:
        return {}
    if getattr(core, 'causal_context_schema', None) != schema:
        raise ValueError('Environment did not declare the requested causal context schema')
    cache = getattr(core, '_causal_public_context', None)
    if cache is None or len(cache) != 2:
        raise ValueError('Missing public-before-shop capture; do not infer it from private current state')
    return {key: value.copy() for key, value in cache[side].items()}


def validate_context(observation, *, required=False):
    keys = {key for key in observation if key.startswith(PREFIX)}
    if not keys:
        if required:
            raise ValueError('Causal context fields are required by this checkpoint')
        return False
    if keys != set(LAYOUT):
        raise ValueError('Partial/unknown causal context fields')
    tensors = {}; leading = None
    for key, (shape, np_dtype, dtype) in LAYOUT.items():
        original = observation[key]
        if not torch.is_tensor(original) and np.asarray(original).dtype != np_dtype:
            raise ValueError(f'Causal context dtype mismatch: {key}')
        value = torch.as_tensor(original)
        if value.dtype != dtype or value.ndim < len(shape) or tuple(value.shape[-len(shape):]) != shape:
            raise ValueError(f'Causal context shape/dtype mismatch: {key}')
        batch_shape = tuple(value.shape[:-len(shape)])
        if leading is not None and batch_shape != leading:
            raise ValueError('Causal context batch dimensions disagree')
        leading = batch_shape; tensors[key.removeprefix(PREFIX)] = value
    if not (tensors['version'] == WIRE_VERSION).all():
        raise ValueError('Causal context wire version mismatch')
    species, perks, digest = vocabulary()
    expected = torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8, device=tensors['vocab'].device)
    if not (tensors['vocab'] == expected).all():
        raise ValueError('Causal context vocabulary mismatch')
    ids, statuses, values = tensors['species'], tensors['perks'], tensors['numbers']
    visible, globals = tensors['visible'], tensors['globals']
    if (not ((ids >= 0) & (ids < len(species))).all()
            or not ((statuses >= 0) & (statuses < len(perks))).all()
            or not torch.isfinite(values).all() or not torch.isfinite(globals).all()
            or not ((values >= 0) & (values <= 1)).all()
            or not ((globals >= 0) & (globals <= 2)).all()
            or not ((visible == 0) | (visible == 1)).all()):
        raise ValueError('Invalid causal context values')
    occupied = ids != 0
    if not (values[..., 6] == occupied).all() or not ((values[..., 4:6] == 0) | (values[..., 4:6] == 1)).all():
        raise ValueError('Invalid causal occupancy or temporary flags')
    if ((~occupied) & ((statuses != 0) | (values.abs().sum(-1) != 0))).any():
        raise ValueError('Empty causal slots carry hidden/nonempty fields')
    if (occupied & ((values[..., 1] <= 0) | (values[..., 2] <= 0) | (visible == 0))).any():
        raise ValueError('Invalid occupied causal slot')
    board_visible = visible.reshape(*leading, 3, 5)
    if not (board_visible == board_visible[..., :1]).all():
        raise ValueError('Partial public-history board visibility')
    if not (globals[..., 5:8] == (board_visible[..., 0] == 0)).all():
        raise ValueError('Causal missing-history flags disagree')
    return True


def context_batch(observation, *, required=True):
    if not validate_context(observation, required=required):
        return None
    values = [observation[PREFIX+k] for k in ('species', 'perks', 'numbers', 'visible', 'globals')]
    values = [torch.as_tensor(v) for v in values]
    values[3] = values[3].bool()
    return PublicContextBatch(*values)


def validate_transport(expected, observation):
    if set(expected) != set(observation):
        raise ValueError('Actor/trajectory observation schema mismatch; fields cannot be dropped')
    validate_context(observation, required=PREFIX+'version' in expected)


def checkpoint_context_metadata(config):
    schema = getattr(config, 'causal_context_schema', None)
    return {'schema': schema, 'adapterMode': getattr(config, 'causal_adapter_mode', 'disabled'),
            'vocabularySHA256': vocabulary()[2] if schema is not None else None}


def validate_checkpoint_context(checkpoint, *, allow_observation_transfer=False):
    from .observation_contract import validate_checkpoint_observation
    validate_checkpoint_observation(checkpoint,allow_transfer=allow_observation_transfer)
    config = checkpoint['config']
    schema = config.get('causal_context_schema')
    require_schema(schema)
    mode = config.get('causal_adapter_mode', 'disabled')
    if schema is not None and mode not in ('control', 'residual'):
        raise ValueError('Invalid causal checkpoint adapter mode')
    expected = {'schema': schema, 'adapterMode': mode, 'vocabularySHA256': vocabulary()[2] if schema else None}
    declared = checkpoint.get('causal_context')
    if schema is not None and declared != expected:
        raise ValueError('Context checkpoint is missing/mismatches its explicit schema/adapter metadata')
    if schema is None and (mode != 'disabled' or declared not in (None, expected)):
        raise ValueError('Disabled legacy checkpoint declares incompatible context metadata')


def initialize_causal_from_parent(policy, parent_state, forecast_state=None):
    """Explicit model-only upgrade, never an implicit permissive resume."""
    if policy.config.causal_context_schema != CONTEXT_SCHEMA:
        raise ValueError('Explicit causal upgrade requires an enabled schema')
    current = policy.state_dict()
    additions = {k:v for k,v in current.items() if k.startswith(('causal_forecast.', 'causal_projection.'))}
    if set(parent_state) != set(current)-set(additions):
        raise ValueError('Parent schema mismatch during causal upgrade')
    policy.load_state_dict({**parent_state, **additions}, strict=True)
    if forecast_state is not None:
        policy.causal_forecast.load_state_dict(forecast_state, strict=True)


def validate_forecast_review(review_path, checkpoint_path, payload):
    from pathlib import Path
    from sapai.turtle_pack import SIMULATOR_REVISION
    review = json.loads(Path(review_path).read_text())
    with Path(checkpoint_path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    species, perks, _ = vocabulary()
    dataset_hash = payload.get('datasetSHA256', '')
    if (review.get('schema') != 'reviewed-causal-forecast-improvement-v1'
            or review.get('reviewed') is not True or review.get('forecastQualityImproved') is not True
            or review.get('forecastCheckpointSHA256') != digest
            or review.get('datasetSHA256') != payload.get('datasetSHA256')
            or not isinstance(dataset_hash, str) or len(dataset_hash) != 64
            or any(c not in '0123456789abcdef' for c in dataset_hash)
            or review.get('simulatorRevision') != SIMULATOR_REVISION
            or payload.get('variant') not in ('causal_marginal', 'causal_autoregressive')
            or tuple(payload['config']['species']) != species or tuple(payload['config']['perks']) != perks):
        raise ValueError('New policy context requires reviewed independent forecast improvement and matching vocabulary')
