"""Finalize the bounded schedule using throughput only. Never select by strength."""
import json,os,subprocess
from pathlib import Path
from summarize import summary
P=Path('results')
def run(label,args,env):
    with open(P/(label+'.log'),'w') as f:
        subprocess.run(['./run.sh','collector_async.py','--out','results/'+label,*map(str,args),'--updates','16'],env=env,stdout=f,stderr=f,check=True)
    return summary(label)['steady_s']
def main():
    selected=json.load(open(P/'selected.json'));args=selected['args'];env=os.environ.copy();name=selected['reference']
    env['SPEED_GEOMETRY']=selected['geometry'];env['SPEED_FUSED_LOSS']='0'
    base=summary(name)['steady_s']
    checks=list(P.glob('fused-*-*.json'))
    use_fused=False
    if len(checks)==4 and all(json.load(open(p)).get('passed') for p in checks) and selected['geometry']=='short' and selected['qualifier']==512:
        env['SPEED_FUSED_LOSS']='1'
        perf=run('fused-final',args,env)
        if perf>base*1.02:base=perf;name='fused-final';use_fused=True
        else:env['SPEED_FUSED_LOSS']='0'
    tile_file=P/'tiles.json'
    if tile_file.exists():
        qualified=[r for r in json.load(open(tile_file)) if r.get('rate') and len(r['checks'])==4]
        if qualified:
            tile=max(qualified,key=lambda r:r['rate'])
            trial_env={**env,'SPEED_FUSED_LOSS':'1','SPEED_GEOMETRY':tile['layout']}
            alt=run('tiled-final',args+['--cap',tile['cap']],trial_env)
            if alt>base*1.02:
                base=alt;name='tiled-final';use_fused=True;env=trial_env
                selected['geometry']=tile['layout'];selected['qualifier']=tile['cap'];args=args+['--cap',tile['cap']]
                # Recheck the other near-tied small layout with the faster learner.
                other=24 if selected['games']!=24 else 48
                perf_alt=run('tiled-other',args+['--games',other],env)
                if perf_alt>base*1.02:
                    base=perf_alt;name='tiled-other';selected['games']=other;args=args+['--games',other]
    perf=run('backpressure',args+['--backlog','.5'],env)
    backlog=.5 if perf>base*1.02 else 1.
    if backlog==.5:name='backpressure';base=perf
    w,g=selected['workers'],selected['games']
    before='original'
    if (w,g)!=(8,96):
        before='matched-original'
        with open(P/(before+'.log'),'w') as f:
            subprocess.run(['./run.sh','/sap/runs/train-eval-20261123/collector_join.py','train','--arm','parent','--parent','/sap/checkpoints/update-025772.pt',
                '--workers',str(w),'--games',str(g),'--transitions','8192','--updates','16','--max-padded-rows','128','--frozen-share','.5',
                '--output','/job/results/'+before],env=env,stdout=f,stderr=f,check=True)
    selected['args']=args
    (P/'final-choice.json').write_text(json.dumps(dict(**selected,measured_winner=name,steady=base,fused=use_fused,backlog=backlog,matched_original=before),indent=2))
if __name__=='__main__':main()
