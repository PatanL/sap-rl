#!/usr/bin/env python3
"""Isolate A+B after C's unchanged BF16 gate fails, then continue the campaign."""
import json
from pathlib import Path
import subprocess
import sys

from early_abc_handoff import PROJECT

root=Path(sys.argv[1])
result=subprocess.run(['python3',str(PROJECT/'scripts/abc_validation_runner_v2.py'),
    '--request',str(root/'request.json'),'--completion',str(root/'completion.json')])
if not (root/'decision.json').exists():
    raise RuntimeError('Missing explicit benchmark decision; no rollout')
# Installer independently checks completion, ownership and rollout authority.
subprocess.run(['python3',str(PROJECT/'scripts/install_validated_campaign_v2.py'),
    str(root/'request.json'),str(root/'completion.json'),str(root/'decision.json')],check=True)
