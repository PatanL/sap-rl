"""Idempotent host-side board-strength measurement with fixed matches and the ranked yardstick split."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from config import PAYLOAD,REV
ROOT=Path(__file__).resolve().parents[2]
JOB=Path(__file__).resolve().parent
def measured_cost(result):
    path=Path(result['receipt'])
    ghost=json.loads(path.read_text())
    ts=[json.loads(line)['active_record_command_s'] for line in (path.parent/'record.log').read_text().splitlines() if line.startswith('{"active_record_command_s"')]
    result['record_active_s']=ts[-1] if ts else result['recording_seconds']
    result['ghost_active_s']=ghost.get('active_s',result['ghost_wall_s'])
    result['measured_service_s']=result['record_active_s']+result['ghost_active_s']
    result['includes_recorder_loading']=bool(ts)
    return result

def run(checkpoint,seed=20261123):
    ck=Path(checkpoint).resolve()
    with ck.open('rb') as fp:sha=hashlib.file_digest(fp,'sha256').hexdigest()
    base=JOB/'receipts'/f'board-{sha[:16]}-{PAYLOAD[:12]}-{seed}'
    base.mkdir(exist_ok=True);summary=base/'board.json'
    if summary.exists():
        r=json.loads(summary.read_text());assert r['sha256']==sha and r['engine']==PAYLOAD;return measured_cost(r)
    import fcntl
    with (base/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if summary.exists():return measured_cost(json.loads(summary.read_text()))
        rec=base/'seats.jsonl';fight=base/'seats-battles.jsonl';relative=str(rec.relative_to(JOB))
        started=time.monotonic()
        with (base/'record.log').open('w') as f:subprocess.run(['./run.sh','record_boards.py','/sap/'+str(ck.relative_to(ROOT)),relative,'1000','--greedy','--seed',str(seed),'--batch','128'],cwd=JOB,stdout=f,stderr=f,check=True)
        mark=time.monotonic()
        assert sum(1 for _ in rec.open())==2000
        fixtures=json.loads((JOB/'results/fixtures.json').read_text())
        oracle='/sap/runs/replay-oracle-20261122/'
        ghost=base/'ghost.json'
        with (base/'ghost.log').open('w') as f:subprocess.run(['./run.sh','ghost_g.py',str(fight.relative_to(JOB)),str(ghost.relative_to(JOB)),'maxwu','4000',*[oracle+x for x in fixtures],'--',*[oracle+x for x in ('raw','raw-fresh','raw-fresh2','raw-fresh3','raw-players')]],cwd=JOB,check=True,stdout=f,stderr=f,env={**os.environ,'CPU_ONLY':'1','SPEED_CPUS':'5-9'})
        g=json.loads(ghost.read_text())
        assert g['split']=='yardstick' and g['boards']['field']==json.loads((ROOT/'runs/player-analysis-20261123/yardstick-baseline.json').read_text())['fieldBoardsPerTurn']
        sel={k:g['comparisons']['bot v field'][k]['score'] for k in ('3','5','7','10','12')}
        result=dict(matches=1000,sha256=sha,engine=PAYLOAD,revision=REV,seed=seed,split=g['split'],scores=sel,index=sum(sel.values())/5,record_command_wall_s=mark-started,ghost_wall_s=time.monotonic()-mark,receipt=str(ghost),fixtures=fixtures,
            recording_seconds=json.loads([line for line in (base/'record.log').read_text().splitlines() if line.startswith('{"matches"')][-1])['seconds'])
        crumbs=[json.loads(line) for line in (base/'record.log').read_text().splitlines() if line.startswith('{"active_record_command_s"')]
        if crumbs:result['active_record_command_s']=crumbs[-1]['active_record_command_s']
        measured_cost(result)
        tmp=summary.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(summary);return result
if __name__=='__main__':
    import sys
    print(json.dumps(run(sys.argv[1]),indent=2))
