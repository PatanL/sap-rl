#!/bin/bash
# Usage: ./run_one.sh /path/to/battle.json descriptive-name
set -eu
cd "$(dirname "$0")"
py=/home/dzliu/pzliu/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
taskset -c 10-15 "$py" prepare_fixture.py "$1" --label "$2"
taskset -c 10-15 "$py" client.py --replay-manifest "$2"
taskset -c 10-15 "$py" rendered_boards.py "$2" 120
