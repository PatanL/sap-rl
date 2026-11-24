"""Use an independent legacy recorder for the exact board index, and a separate Rust recording rehearsal."""
import json,os,subprocess,sys,time
from pathlib import Path
from hashlib import file_digest

BASE='/sap/runs/train-vector-20261123'
ORACLE='/sap/runs/replay-oracle-20261122/'
def execute(argv,log,cpu=False):
    with Path(log).open('w') as out:
        subprocess.run(['./run.sh',*argv],stdout=out,stderr=out,check=True,
                       env={**os.environ,'CPU_ONLY':str(int(cpu)),'AFTER_RELEASE':'1'})
def main():
    checkpoint=Path(sys.argv[1]);ck='/job/'+str(checkpoint);seats='results/boards.jsonl'
    execute([BASE+'/record_boards.py',ck,seats,'1000','--greedy','--batch','128'],'results/boards.log')
    assert sum(1 for _ in Path(seats).open())==2000
    fights='results/boards-battles.jsonl'
    execute(['board_rehearsal.py',ck,fights,'results/board-rehearsal.json'],'results/rehearsal.log')
    fixtures=json.loads(Path('../train-vector-20261123/results/fixtures.json').read_text())
    execute([BASE+'/ghost_g.py',fights,'results/ghost.json','maxwu','4000',*[ORACLE+f for f in fixtures],
             '--',*[ORACLE+d for d in ('raw','raw-fresh','raw-fresh2','raw-fresh3','raw-players')]],'results/ghost.log',True)
    ghost=json.loads(Path('results/ghost.json').read_text());assert ghost['split']=='yardstick'
    scores={t:ghost['comparisons']['bot v field'][t]['score'] for t in ['3','5','7','10','12']}
    record=[json.loads(x) for x in Path('results/boards.log').read_text().splitlines() if x.startswith('{')]
    with checkpoint.open('rb') as f:sha=file_digest(f,'sha256').hexdigest()
    Path('results/board.json').write_text(json.dumps(dict(sha256=sha,split=ghost['split'],matches=1000,
        batch=128,record_s=next(x['active_record_command_s'] for x in record if 'active_record_command_s' in x),
        rounds_s=next(x['seconds'] for x in record if x.get('matches')==1000),ghost_s=ghost['active_s'],scores=scores,
        index=sum(scores.values())/5),indent=2))
if __name__=='__main__':main()
