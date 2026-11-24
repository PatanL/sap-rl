#!/bin/bash
# Side-by-side evidence for one fight: calculator log, then engine trace.
# Usage: diag.sh <battle id prefix> [engine dir name, default engine-v31] [fixtures file]
HERE="$(cd "$(dirname "$0")" && pwd)"; ID="$1"; ENGINE="${2:-engine-v31}"; FX="${3:-fixtures-340.json}"
python3 - "$HERE" "$ID" "$FX" <<'PY'
import json, subprocess, sys
here, prefix, fx = sys.argv[1:4]
sys.argv = ["triage.py", f"{here}/{fx}", "/dev/null", f"{here}/../../SAP-Calculator", "/dev/null", "1"]
sys.path.insert(0, here)
import triage
fixture = next(f for f in json.load(open(f"{here}/{fx}"))["fixtures"] if f["battleId"].startswith(prefix))
out = triage.run_calculator([triage.calc_config(fixture)], log=True)[0]
print("== calculator"); print("\n".join(out["log"]))
PY
echo "== engine ($ENGINE)"
docker run --rm --cpus=1 --network none -v "$HERE:/w/out:ro" sap-rl-shared:research-25772 python /w/out/trace_sapai.py "/w/out/$ENGINE" "/w/out/$FX" "$ID" 2>&1 | grep -E "^(init|start|atk|final|real|Traceback|  File|\w+Error)"
