#!/usr/bin/env bash
# Never interrupt the active stage. An explicit validated marker authorizes
# isolated GPU benchmarking only after that stage and its evaluations finish.
set -euo pipefail
project=/home/dzliu/pzliu/sap-rl
review="${project}/runs/performance-review-20260906/abc-benchmark-v1"
while systemctl --user is-active --quiet sap-v45-corrected.service; do sleep 20; done
if systemctl --user is-failed --quiet sap-v45-corrected.service; then
  echo 'Predecessor failed; no benchmark or continuation launched.' >&2
  exit 1
fi
if [[ -f "${review}/READY.json" && ! -f "${review}/ATTEMPTED" ]]; then
  if /bin/bash "${project}/scripts/benchmark_abc_boundary.sh"; then
    echo 'Isolated A-C benchmark finished. Diagnostic weights will not be used.'
  else
    echo 'A-C benchmark did not pass/complete. Preserve validated v45 runtime.' >&2
  fi
fi
exec /bin/bash "${project}/scripts/continue_v45_overnight.sh"
