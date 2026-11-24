"""Finalize the throughput-only choice before any strength evaluation."""
import json,os,subprocess
from pathlib import Path
from summarize import summary
R=Path('results')
def run(name,args,env):
    with (R/(name+'.log')).open('w') as log:
        subprocess.run(['./run.sh','collector_async.py','--out','results/'+name,*map(str,args)],env=env,stdout=log,stderr=log,check=True)
    return summary(name)
def main():
    c=json.load(open(R/'final-choice.json'));previous=c.copy()
    (R/'earlier-choice.json').write_text(json.dumps(previous,indent=2))
    env={**os.environ,'SPEED_FUSED_LOSS':str(int(c['fused'])),'SPEED_GEOMETRY':c['geometry']}
    best=summary(c['measured_winner']);half=summary('backpressure')
    if half['steady_s']>=.98*best['steady_s'] and half['steady_stale']<best['steady_stale']:
        c['backlog']=.5;c['measured_winner']='backpressure';best=half
    args=c['args']+['--backlog',c['backlog']]
    upper=R/'upper-bounds.json'
    if c['geometry']=='block-both' and c['fused'] and upper.exists():
        ok=[r for r in json.load(open(upper)) if r['passed']]
        if ok:
            cap=max(ok,key=lambda r:r['rate'])['cap']
            trial=run('larger-cap',args+['--cap',cap,'--updates',16],env)
            if trial['steady_s']>best['steady_s']*1.02:
                c['qualifier']=cap;c['measured_winner']='larger-cap';best=trial
    c['steady']=best['steady_s'];c['args']=args+['--cap',c['qualifier']]
    (R/'final-choice.json').write_text(json.dumps(c,indent=2))
    if (c['backlog'],c['qualifier'])!=(previous['backlog'],previous['qualifier']):
        run('profile-final',c['args']+['--profile','--updates',14],env)
    if c['qualifier']!=previous['qualifier']:
        with (R/'final-exactness.log').open('a') as log:subprocess.run(['./run.sh','exact_final.py'],env=env,stdout=log,stderr=log,check=True)
if __name__=='__main__':main()
