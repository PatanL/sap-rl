"""Select by production learned throughput alone. Score one additional proof only if throughput warrants it."""
import json,os,subprocess,sys,time
from pathlib import Path
from measure_diff import receipt
from board_metric import run as measure_board
P=Path(__file__).resolve().parent
R=P/'results'
def read(path):return json.loads(path.read_text())
def until(path,log=None):
    while not path.exists():
        if log and log.exists() and any(k in log.read_text() for k in ('Traceback (most recent call last)', 'CUDA out of memory', 'Killed')):
            raise RuntimeError(f'Failed prerequisite. Inspect {log}')
        time.sleep(4)
def cmd(args,label,env=None):
    with (R/(label+'.log')).open('w') as out:subprocess.run(args,stdout=out,stderr=out,env=env,check=True)
def main():
    os.chdir(P)
    names=['v-python-loose','v-python-train','v-python-train-128','v-python-train-256','v-python-train-512','cohort12x16','cohort12x24','cohort12x40']
    for k in names:until(R/k/'result.json',R/(k+'.log'))
    base=receipt(R/'proof')['steady_learned_s'];values={}
    for k in names:
        metrics=read(R/k/'result.json')
        if k.startswith('cohort'):value=receipt(R/k)['steady_learned_s']
        else:
            ev=metrics['events'];head=ev[3];tail=ev[-1];value=sum(e['decisions'] for e in ev[4:])/(tail['seconds']-head['seconds'])
        values[k]=value
    winner=max(names,key=values.get);selected=winner if values[winner]>=base*1.03 else 'proof';v=selected.startswith('v-python')
    choice=dict(winner=selected,proof='proof' if selected=='proof' else 'final-proof',rates=values,baseline=base,
                ledger='progression.json' if selected=='proof' else 'final-progression.json',
                confirmation='proof-confirm.json' if selected=='proof' else 'final-confirm.json',
                board_test='paired-board-check.json' if selected=='proof' else 'final-board-check.json')
    (R/'chosen.json').write_text(json.dumps(choice,indent=2))
    if selected=='proof':return
    folder=R/choice['proof'];relative=str(folder.relative_to(P))
    env={**os.environ,'SPEED_CPUS':'5-19'}
    if not (folder/'result.json').exists():
        if v:
            g=read(R/selected/'result.json')['games']
            cmd(['./run.sh','vector_collect.py','--backend','python','--workers','12','--games',str(g),'--backlog',str(read(R/selected/'result.json').get('backlog',.5)),'--seconds','810','--updates','100','--train','--save','--out',relative],'final-run',env)
        else:
            g=selected.split('x')[1]
            cmd(['./run.sh','collector_g.py','--concurrent','--priority-inference','--workers','12','--games',g,'--target','8192','--cap','1536','--backlog','.5','--battle-backend','rust','--updates','100','--save','--out',relative],'final-run',env)
    ck=next(folder.glob('update-*.pt'))
    if not (R/choice['confirmation']).exists():
        cmd(['./run.sh','evaluate.py','/job/'+str(ck.relative_to(P)),'/sap/checkpoints/update-025772.pt','results/'+choice['confirmation'],'--mode','confirm','--pairs','256'],'final-confirm')
    cmd([sys.executable,'watch_g.py','--once','--run',str(folder.relative_to(P.parents[1])),'--progression',str(R/choice['ledger'])],'final-watch')
    l=read(R/choice['ledger'])
    files=[str((Path(m['receipt']).parent/'seats-battles.jsonl').relative_to(P)) for m in (l['board_baseline'],l['checkpoints'][-1]['board_strength'])]
    cmd(['./run.sh','paired_boards.py',*files,'--out','results/'+choice['board_test']],'final-board-check',{**os.environ,'CPU_ONLY':'1','SPEED_CPUS':'5-9'})
if __name__=='__main__':main()
