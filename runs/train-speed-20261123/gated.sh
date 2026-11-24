#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
while [[ ! -f ../train-live-20261123/STOPPED ]];do sleep 3;done
./stage1.sh
