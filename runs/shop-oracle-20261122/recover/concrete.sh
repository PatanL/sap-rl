#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export SAP_ENGINE_ROOT="$PWD/pinned-recover"
./run.sh -c 'import os,runpy,sys;os.environ["CONCRETE"]="1";sys.path.insert(0,"/audit/recover");runpy.run_path("/audit/recover/search.py",run_name="__main__")' --tier 1 --out recover/concrete.jsonl >recover/concrete.log 2>&1
