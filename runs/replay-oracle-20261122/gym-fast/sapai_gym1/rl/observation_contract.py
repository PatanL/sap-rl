"""Versioned conservative public observation semantics; tensor shape unchanged."""
LEGACY_OBSERVATION_REVISION='entity-legacy-exact-opponent-sell-value'
PREVIOUS_PUBLIC_OBSERVATION_REVISION='entity-public-opponent-base-sell-v1-20260907'
PREVIOUS_CANONICAL_OBSERVATION_REVISION='entity-public-canonical-order-v2-20260907'
OBSERVATION_REVISION='entity-public-reorder-budget-v3-20260911'

def checkpoint_revision(checkpoint):
    return checkpoint.get('observation_revision',LEGACY_OBSERVATION_REVISION)

def validate_checkpoint_observation(checkpoint,*,allow_transfer=False):
    revision=checkpoint_revision(checkpoint)
    if revision==OBSERVATION_REVISION:return False
    if allow_transfer and revision in (LEGACY_OBSERVATION_REVISION,PREVIOUS_PUBLIC_OBSERVATION_REVISION,PREVIOUS_CANONICAL_OBSERVATION_REVISION):return True
    raise ValueError(f'Observation contract mismatch: {revision!r} -> {OBSERVATION_REVISION!r}; old weights require explicit observation transfer')

def require_record_observation(record):
    if record.get('observationRevision',LEGACY_OBSERVATION_REVISION)!=OBSERVATION_REVISION:
        raise ValueError('Observation record revision mismatch; do not reuse legacy inputs/memories silently')
