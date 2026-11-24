#!/bin/bash
# send one command to the interactive client and wait for its DONE line
cd /home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/
n=$(grep -c '^DONE' session3/client-interactive.log)
echo "$*" >> session3/cmd.txt
for i in $(seq 1 600); do
  m=$(grep -c '^DONE' session3/client-interactive.log)
  if [ "$m" -gt "$n" ]; then break; fi
  sleep 0.2
done
tail -6 session3/client-interactive.log | grep -v versus/datetime
