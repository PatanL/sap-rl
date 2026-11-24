#!/usr/bin/env bash
set -euo pipefail
run=/home/dzliu/pzliu/sap-rl/runs/turtle-league-v45-corrected-from-v44-head-3250
while true; do
  state=$(systemctl --user show sap-v45-corrected.service -p ActiveState --value)
  [[ "${state}" == active ]] || exit 0
  if [[ -f "${run}/update-004900.pt" ]] && python3 -c '
import json,sys
p=json.load(open(sys.argv[1]))
raise SystemExit(0 if any(m["member_id"]=="historical-004900" for m in p["members"]) else 1)
' "${run}/league.json"; then
    systemctl --user stop sap-v45-corrected.service
    echo 'Paused v45 after checkpoint 4900 and durable league publication; benchmark may now use the GPU.'
    exit 0
  fi
  sleep 2
done
