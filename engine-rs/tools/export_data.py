#!/usr/bin/env python3
"""Dump the reference engine's Turtle Pack tables for the Rust engine.

Usage (inside the training image): export_data.py <engine dir> <out.json>
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from sapai.data import data  # noqa: E402
from sapai.turtle_pack import TURTLE_PET_STATS  # noqa: E402

pets = {}
for name, entry in data["pets"].items():
    packs = entry.get("packs") or []
    if name in TURTLE_PET_STATS or "StandardPack" in packs or entry.get("tier") in ("Summoned", "Token") or name in (
            "pet-bus", "pet-ram", "pet-zombie-cricket", "pet-zombie-fly", "pet-chick", "pet-bee",
            "pet-dirty-rat", "pet-butterfly", "pet-sloth", "pet-none"):
        pets[name] = {key: entry.get(key) for key in (
            "name", "tier", "baseAttack", "baseHealth", "status",
            "level1Ability", "level2Ability", "level3Ability")}
source = Path(sys.argv[1]) / "sapai"
digest = hashlib.sha256()
for path in sorted(source.glob("*.py")):
    digest.update(path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).hexdigest().encode() + b"\n")
out = {"engineSha256": digest.hexdigest(), "rollable": sorted(TURTLE_PET_STATS),
       "pets": pets, "statuses": data["statuses"],
       "foods": {k: v for k, v in data["foods"].items()}}
json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True, default=str)
print(len(pets), "pets;", "engine", out["engineSha256"][:16])
