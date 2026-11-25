"""Rule switches for the v32 battle rules (real-client findings R1-R16).

The environment variable ``SAP_RULES`` selects the rule set, with the same
grammar as ``engine-rs/src/engine.rs``:

    v31            every v32 rule off (the engine behaves like engine-v31)
    v32            the accepted v32 set (``V32`` below)
    R13,R14        exactly these rules
    v32,-R1        a set with one rule taken out; ``v32,R1`` adds one

Unset means ``v32`` here and ``v31`` in the Rust engine. An unknown token is
an error, so a typo cannot silently run another rule set. ``V32`` must stay
identical to ``RULES_V32`` in the Rust engine.
"""
import os

ALL = (
    "R1", "R2", "R4", "R5", "R5b", "R8", "R9", "R11", "R12", "R13", "R14", "R16",
)

# Rules that are on under "v32". A rule that broke more real fights than it
# fixed would stay out; none did (see engine-rs/PROGRESS.md, "Rules v32 from
# the real client").
V32 = frozenset(ALL)


def parse(text):
    active = set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if token == "v31":
            active.clear()
        elif token == "v32":
            active |= V32
        elif token in ALL:
            active.add(token)
        elif token.startswith("-") and token[1:] in ALL:
            active.discard(token[1:])
        else:
            raise ValueError(f"SAP_RULES: unknown token {token!r}")
    return frozenset(active)


ACTIVE = parse(os.environ.get("SAP_RULES", "v32"))


def on(name):
    return name in ACTIVE
