from pathlib import Path
import json,os,subprocess
from summarize import summary
P=Path(__file__).resolve().parent
def trial(name,*args):
    if (P/'results'/name/'result.json').exists():return
    subprocess.run([str(P/'trial.sh'),name,*map(str,args)],check=True)
def main():
    q=json.load(open('results/qualified.json'));os.environ['SPEED_GEOMETRY']=q['layout']
    params=['--concurrent','--cap',q['cap']]
    if q['compact']:params+=['--compact']
    checked=summary('actor96')['steady_s']>summary('async96')['steady_s']*1.02
    if checked:params+=['--fast-actor']
    base='actor96' if checked else 'async96'
    layouts=[(base,8,96),('g48',8,48),('g24',8,24),('w6g48',6,48)]
    trial('g48',*params,'--workers',8,'--games',48,'--target',8192,'--updates',16,'--capture-at',3)
    if not Path('results/verified-config.json').exists():
      with open('results/crossqual.log','w') as log:
        subprocess.run([str(P/'run.sh'),'crossqual.py'],stdout=log,stderr=subprocess.STDOUT,check=True)
    verified=json.load(open('results/verified-config.json'))
    if any(verified[k]!=q[k] for k in ('cap','layout','compact')):
        os.environ['SPEED_GEOMETRY']=verified['layout'];params=['--concurrent','--cap',verified['cap']]
        if verified['compact']:params+=['--compact']
        if checked:params+=['--fast-actor']
        layouts[:2]=[('verified96',8,96),('verified48',8,48)]
        for name,w,g in layouts[:2]:trial(name,*params,'--workers',w,'--games',g,'--target',8192,'--updates',16)
    for name,w,g in layouts[2:]:
        trial(name,*params,'--workers',w,'--games',g,'--target',8192,'--updates',16)
    best=max(layouts,key=lambda a:summary(a[0])['steady_s']);_,w,g=best
    for target,updates in [(4096,22),(16384,12)]:
        trial(f'target{target}',*params,'--workers',w,'--games',g,'--target',target,'--updates',updates)
    # Compare a real port only after fixed-tape parity passed.
    parity=json.load(open('results/boundary-parity.json'));assert parity['battles']==762
    trial('rust96',*params,'--battle-backend','rust','--workers',8,'--games',96,'--target',8192,'--updates',18)
    configs=[(name,workers,games,'python') for name,workers,games in layouts]+[('rust96',8,96,'rust')]
    for _,wr,ga in sorted(layouts,key=lambda a:-summary(a[0])['steady_s']):
        if (wr,ga)==(8,96):continue
        label=f'rust-w{wr}g{ga}';trial(label,*params,'--battle-backend','rust','--workers',wr,'--games',ga,'--target',8192,'--updates',16)
        configs.append((label,wr,ga,'rust'))
        if len(configs)>=len(layouts)+3:break
    name,w,g,backend=max(configs,key=lambda a:summary(a[0])['steady_s'])
    final=[*params,'--workers',w,'--games',g,'--target',8192,'--battle-backend',backend]
    trial('priority',*final,'--updates',16,'--priority-inference')
    if summary('priority')['steady_s']>summary(name)['steady_s']*1.02:
        name='priority';final+=['--priority-inference']
    Path('results/selected.json').write_text(json.dumps(dict(reference=name,workers=w,games=g,target=8192,backend=backend,fast_actor=checked,qualifier=verified['cap'],geometry=verified['layout'],args=final),indent=2))
    trial('profile',*final,'--profile','--updates',14)
    trial('proof',*final,'--updates',100,'--seconds',810,'--save',*(['--audit-battles'] if backend=='rust' else []))
if __name__=='__main__':main()
