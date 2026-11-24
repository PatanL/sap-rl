#!/usr/bin/env bash
# Runner revision 2: no deadline/exit-code-only inference of natural completion.
set -euo pipefail
if [[ -z "${SAP_BENCH_REQUEST:-}" || -z "${SAP_BENCH_COMPLETION:-}" ]]; then
  echo 'Invocation/checkpoint-bound completion evidence is absent; benchmark blocked.' >&2
  exit 1
fi
exec /bin/bash /home/dzliu/pzliu/sap-rl/scripts/benchmark_abc_boundary.sh
