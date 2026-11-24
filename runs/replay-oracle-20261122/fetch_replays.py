#!/usr/bin/env python3
"""Politely download ranked Turtle-mirror replays from the SAP Library JSON API.

One search page per 50 results, then one `perspectives` request per replay
(both seats, every action, and every battle with its seed, boards, outcome and
surviving board). Sequential, with a fixed pause between requests. Resumable:
replays already on disk are skipped.

Usage: fetch_replays.py <out dir> <max new replays> [pause seconds] [start page] [dirs to skip ...]
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://www.sap-library.com/api"
OUT = Path(sys.argv[1])
LIMIT = int(sys.argv[2])
PAUSE = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
START_PAGE = int(sys.argv[4]) if len(sys.argv) > 4 else 1
SKIP_DIRS = [Path(p) for p in sys.argv[5:]]  # replays already held elsewhere
# Optional: only games of players whose name contains this text (SAP_PLAYER env var).
import os
import urllib.parse
PLAYER = "&player=" + urllib.parse.quote(os.environ["SAP_PLAYER"]) if os.environ.get("SAP_PLAYER") else ""
HEADERS = {"User-Agent": "Mozilla/5.0 (replay-fixture-fetch)", "Accept": "application/json"}


def get(url):
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    have = {path.stem for path in OUT.glob("*.json")}
    for other in SKIP_DIRS:
        have |= {path.stem for path in other.glob("*.json")}
    seen_pairs, fetched, page, errors = set(), 0, START_PAGE, 0
    while fetched < LIMIT:
        listing = get(f"{BASE}/search?packA=Turtle&packB=Turtle&version=current&matchType=ranked"
                      f"&page={page}&pageSize=50&sort=created_at&order=desc{PLAYER}")
        results = listing.get("results") or []
        if not results:
            break
        time.sleep(PAUSE)
        for meta in results:
            if fetched >= LIMIT:
                break
            # Both seats of one match are listed separately; one download covers both.
            pair = frozenset({(meta.get("player_id"), meta.get("player_rank")),
                              (meta.get("opponent_id"), meta.get("opponent_rank"))})
            if meta["id"] in have or pair in seen_pairs:
                seen_pairs.add(pair)
                continue
            seen_pairs.add(pair)
            try:
                perspectives = get(f"{BASE}/replays/{meta['id']}/perspectives")
            except Exception as error:
                errors += 1
                print("error", meta["id"], repr(error)[:120], flush=True)
                if errors > 10:
                    raise SystemExit("too many errors; stopping")
                time.sleep(PAUSE * 4)
                continue
            (OUT / f"{meta['id']}.json").write_text(json.dumps({"meta": meta, "perspectives": perspectives}))
            fetched += 1
            if fetched % 25 == 0:
                print(f"fetched {fetched} (page {page}, total listed {listing.get('total')})", flush=True)
            time.sleep(PAUSE)
        page += 1
    print(f"done: fetched {fetched}, errors {errors}, pages {page - 1}", flush=True)


if __name__ == "__main__":
    main()
