#!/usr/bin/env bash
# Runner revision 2. Original retained in runner-v1-archive/.
set -euo pipefail
project=/home/dzliu/pzliu/sap-rl
require_inactive_success() {
  local unit=$1 state result load
  if ! state=$(systemctl --user show "${unit}" -p ActiveState --value); then
    echo "Cannot query ${unit}; blocking GPU launch" >&2; return 1
  fi
  if ! result=$(systemctl --user show "${unit}" -p Result --value); then
    echo "Cannot query ${unit} result; blocking GPU launch" >&2; return 1
  fi
  if ! load=$(systemctl --user show "${unit}" -p LoadState --value); then
    echo "Cannot query ${unit} load state; blocking GPU launch" >&2; return 1
  fi
  if [[ "${load}" != loaded || "${state}" != inactive || "${result}" != success ]]; then
    echo "${unit} is not a clean inactive service; blocking GPU launch" >&2; return 1
  fi
}
require_inactive_success sap-v45-corrected.service
require_inactive_success sap-v45-overnight.service
if [[ -z "${SAP_BENCH_REQUEST:-}" || -z "${SAP_BENCH_COMPLETION:-}" ]]; then
  echo 'Invocation/checkpoint-bound completion evidence required; time or idle GPU is insufficient.' >&2
  exit 1
fi
python3 "${project}/scripts/benchmark_guards.py" "${SAP_BENCH_REQUEST}" "${SAP_BENCH_COMPLETION}"
exec python3 "${project}/scripts/abc_validation_runner_v2.py" --request "${SAP_BENCH_REQUEST}" --completion "${SAP_BENCH_COMPLETION}"
