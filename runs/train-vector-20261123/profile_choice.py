"""Detailed time breakdown for a changed winning cohort shape."""
from pathlib import Path
import time,json,os,subprocess
R=Path('results')
while not (R/'chosen.json').exists():time.sleep(5)
c=json.load(open(R/'chosen.json'));k=c['winner']
if k.startswith('cohort'):
    with open(R/'final-profile.log','w') as f:
        subprocess.run(['./run.sh','collector_g.py','--concurrent','--priority-inference','--workers','12','--games',k.split('x')[1],
                        '--target','8192','--cap','1536','--backlog','.5','--battle-backend','rust','--profile','--updates','14','--out','results/final-profile'],
                        env={**os.environ,'SPEED_CPUS':'5-19'},stdout=f,stderr=f,check=True)
