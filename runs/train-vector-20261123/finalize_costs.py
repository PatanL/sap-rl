"""Attach active service cost to locally generated records (separate from time queued for benchmarks)."""
from pathlib import Path
import json
from board_metric import measured_cost
for file in Path('receipts').glob('board-*/board.json'):
    value=measured_cost(json.loads(file.read_text()))
    tmp=file.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(file)
for file in Path('results').glob('*progression.json'):
    value=json.loads(file.read_text())
    if 'board_baseline' in value:measured_cost(value['board_baseline'])
    for row in value.get('checkpoints',[]):measured_cost(row['board_strength'])
    tmp=file.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(file)
