#!/bin/bash
cd /home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/
session3/chain.sh c3b2a c3b2c SAP_SHOT_INTERVAL_MS=120
session3/chain.sh c3b2c s3real SAP_SHOT_INTERVAL_MS=120
session3/chain.sh s3real c3b2b SAP_SHOT_INTERVAL_MS=120
