#!/usr/bin/env python3
"""Select the current v45 stage without changing either training process."""
import os
import subprocess

active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'sap-v45-corrected.service']).returncode == 0
service, container = ('sap-v45-corrected.service', 'sap-rl-v45-corrected') if active else (
    'sap-v45-overnight.service', 'sap-rl-v45-overnight')
os.execv('/usr/bin/python3', [
    '/usr/bin/python3', '/home/dzliu/pzliu/sap-rl/scripts/monitor_v39.py',
    '--run', '/home/dzliu/pzliu/sap-rl/runs/turtle-league-v45-corrected-from-v44-head-3250',
    '--service', service, '--training-container', container,
    '--evaluation-container', 'sap-rl-v45-evaluation',
])
