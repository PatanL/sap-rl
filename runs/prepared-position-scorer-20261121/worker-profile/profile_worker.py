"""Profile compact battle scoring on fixed synthetic self-play teams and seeds."""
import cProfile
import hashlib
import json
import pickle
import pstats
import random
import time
from pathlib import Path

from compact_scorer import compact_board
from plan_common import install
from prepared_scorer import score_prepared

install(cuda=False)
library = Path('/workspace/runs/claude-end-positioner-20261112-v1/out/library.pkl')
with library.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != '8f13394cb9081a528c42afc12659a4a742c073039831a304b792edc7b64da27c':
        raise ValueError('Unexpected board library')
    stream.seek(0)
    boards = pickle.load(stream)

rng = random.Random(202611212)
tasks = []
for index in range(96):
    turn = 1 + index % 16
    teams = [compact_board(row['board']) for row in rng.sample(boards[turn], 3)]
    tasks.append((teams[0], teams[1:], [rng.getrandbits(31) for _ in range(4)]))

def execute():
    return [score_prepared(task) for task in tasks]

start = time.process_time()
reference = execute()
baseline = time.process_time() - start
profiler = cProfile.Profile(timer=time.process_time)
actual = profiler.runcall(execute)
if actual != reference:
    raise ValueError('Instrumented scorer differs on the same tasks')
profile = pstats.Stats(profiler)

def top(column):
    rows = []
    for (path, line, name), (_, count, own, cumulative, _) in profile.stats.items():
        rows.append(dict(path=path, line=line, name=name, calls=count, own=own, cumulative=cumulative))
    return sorted(rows, key=lambda row: row[column], reverse=True)[:35]

print(json.dumps(dict(librarySHA256=digest, tasks=len(tasks), battles=768, baselineCPUSeconds=baseline,
                      profiledCPUSeconds=profile.total_tt, cumulative=top('cumulative'), own=top('own'))), flush=True)
