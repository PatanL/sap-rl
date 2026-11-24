"""Extract effective-rate evidence and opponent sampling lines from saved runs, without a training job."""
import json,re
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
def read(f):return json.loads(f.read_text())
def phase(path):
    cfg=read(path/'config.json')['arguments']
    log=path/'training.log'
    last=None
    if log.exists():
        with log.open() as fp:
            for l in fp:
                if l.startswith('opponent_sampling='):last=l.strip()
    return dict(source=str(path.relative_to(R)),learning_rate=cfg.get('learning_rate'),resume_rate=cfg.get('resume_learning_rate'),history_rate=cfg.get('opponent_module_learning_rate'),
                adapter_rate=cfg.get('opponent_adapter_learning_rate'),freeze_rate=cfg.get('end_freeze_learning_rate'),
                last_logged_mix=last,residents=cfg.get('resident_checkpoint'),historical_limit=cfg.get('max_historical_residents'),
                decisions_per_update=cfg.get('transitions_per_update'),init=cfg.get('init_checkpoint'),resume=cfg.get('resume_checkpoint'))
stages=['turtle-league-v13-selected-pa99-diverse-to2000','corrected-observer-continuation-20260907-v1/run-v1/training',
 'search-transfer-24h-20260916-v1/outputs/training','search-transfer-24h-20260916-v1/outputs/iteration-v3/training',
 'search-iterated-campaign-20260917-v1/outputs/cycle-01/learn2','direct-opening-exploration-20260918-v1/outputs/cycle-02/learn2',
 'ordinary-rl-shared-20260924-v2/outputs/chunk-01','ordinary-rl-shared-20260924-v2/outputs/chunk-20','population-league-25772-v1/training']
evals=['search-transfer-24h-20260916-v1/outputs/evaluation/REPORT.json','search-iterated-campaign-20260917-v1/outputs/cycle-01/evaluation/REPORT.json','direct-opening-exploration-20260918-v1/outputs/cycle-02/evaluation/REPORT.json']
evals.extend(str(p.relative_to(R)) for p in sorted((R/'ordinary-rl-shared-20260924-v2/outputs').glob('eval-*/REPORT.json')))
evals.append('population-league-25772-v1/result.json')
data=dict(phases=[phase(R/f) for f in stages],evals={f:read(R/f) for f in evals})
(P/'historical-evidence.json').write_text(json.dumps(data,indent=2))
