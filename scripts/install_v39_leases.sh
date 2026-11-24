#!/usr/bin/env bash
set -euo pipefail
run=/home/dzliu/pzliu/sap-rl/runs/turtle-league-v39-unrestricted-from-v38-250
marker="${run}/leases-installed"
# PathExists would otherwise retrigger after this one-shot finishes.
systemctl --user disable --now sap-v39-install-leases.path
if [[ -f "${marker}" ]]; then exit 0; fi
[[ -f "${run}/update-000500.pt" ]]
# The checkpoint is renamed before its league entry is saved. Wait for both,
# rather than interrupting the old runtime halfway through league.json.
deadline=$((SECONDS + 30))
until jq -e 'any(.members[]; .member_id == "historical-000500")' "${run}/league.json" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo 'Checkpoint 500 has no durable matching league entry; leaving training running' >&2
    exit 1
  fi
  sleep 1
done
# Respect a manual user pause: installation must not bring back a stopped run.
state="$(systemctl --user show sap-v39-unrestricted.service -p ActiveState --value)"
if [[ "${state}" != activating && "${state}" != active ]]; then
  echo 'v39 is stopped; preserving the pause' >&2
  exit 0
fi
systemctl --user stop sap-v39-unrestricted.service
systemctl --user start --no-block sap-v39-unrestricted.service
touch "${marker}"
