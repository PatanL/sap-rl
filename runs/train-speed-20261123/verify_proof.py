"""Check the numerical gate, data ledger, parity and evaluation inputs for the reported proof."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

def read(name):
    return json.loads((RESULTS / name).read_text())

def main():
    settings = read("proof/args.json")
    result = read("proof/result.json")
    updates = [json.loads(line) for line in (RESULTS / "proof/updates.jsonl").read_text().splitlines()]
    expected = range(25773, 25873)
    assert [row["version"] for row in updates] == list(expected)
    assert settings["cap"] == read("final-choice.json")["qualifier"]
    assert settings["max_lag"] == 1
    for step in updates:
        rollout_version = step["version"] - 1
        assert rollout_version - settings["max_lag"] <= step["behavior_min"] <= step["behavior_max"] <= rollout_version
    assert sum(row["decisions"] for row in updates) == result["learned"]
    assert result["delivered"] == result["learned"] + result["stale"] + result["queued"]
    assert result["live"] == result["pending"] + result["delivered"]
    assert result["seat"] == result["live"] + result["frozen"]
    assert result["elapsed"] < 870
    assert result["peak_bytes"] < 8 * 2**30
    numerical = read("final-exactness.json")
    assert {(row["rows"], row["prec"]) for row in numerical} == {(n, p) for n in (8193, 8194) for p in ("fp32", "bf16")}
    assert all(row["passed"] and row["cap"] == settings["cap"] and row["layout"] == settings["geometry"] and int(row["fused"]) == int(settings["fused_loss"]) for row in numerical)
    parity = read("proof/battle-parity.json")
    assert parity["battles"] == parity["matches"] and parity["battles"] > 0
    checkpoint = RESULTS / "proof" / f"update-{result['version']:06d}.pt"
    with checkpoint.open("rb") as source:
        checkpoint_sha = hashlib.file_digest(source, "sha256").hexdigest()
    seeds_by_mode = []
    for name, pairs, opponent in [("proof-vs-parent.json", 64, 25772), ("proof-vs-old.json", 256, 26022)]:
        score = read(name)
        assert score["pairs"] == pairs and score["opponent"]["update"] == opponent
        assert score["candidate"]["sha256"] == checkpoint_sha
        assert score["engine"] == result["engine"]
        worlds = {(row["seed"], row["seat"]) for row in score["records"]}
        seeds = {seed for seed, _ in worlds}
        assert len(seeds) == pairs and worlds == {(seed, seat) for seed in seeds for seat in (0, 1)}
        seeds_by_mode.append(seeds)
    assert seeds_by_mode[0].isdisjoint(seeds_by_mode[1])
    document = (ROOT / "REPORT.md").read_text()
    for target in re.findall(r'\]\(([^)]+)\)', document):
        assert (ROOT / target).exists(), target
    print(f"Verified {len(updates)} updates, {result['learned']:,} learned decisions, four numerical checks, "
          f"{parity['matches']:,} audited fights and {sum(map(len,seeds_by_mode))} paired evaluation seeds.")
    print("Checkpoint SHA-256", checkpoint_sha)

if __name__ == "__main__":
    main()
