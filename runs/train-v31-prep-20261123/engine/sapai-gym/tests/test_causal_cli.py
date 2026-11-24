from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.causal_transport import CONTEXT_SCHEMA, vocabulary
from sapai_gym1.rl.opponent_distribution import AutoregressiveOpponentBoard, OpponentDistributionConfig
from test_causal_transport import setup
from test_causal_checkpoint_live import save


def test_real_cli_ab_transport_upgrade_save_resume_and_mode_guard(tmp_path):
    _, base, _, _ = setup()
    parent = tmp_path/'parent.pt'; save(parent, base)
    species, perks, _ = vocabulary()
    forecast = AutoregressiveOpponentBoard(OpponentDistributionConfig(species, perks, width=32))
    forecast_path = tmp_path/'test-only-forecast.pt'
    torch.save({'state_dict':forecast.state_dict(),'config':asdict(forecast.config),
                'variant':'causal_autoregressive','datasetSHA256':'a'*64}, forecast_path)
    review = tmp_path/'test-only-review.json'
    review.write_text(json.dumps({'schema':'reviewed-causal-forecast-improvement-v1','reviewed':True,
        'forecastQualityImproved':True,'forecastCheckpointSHA256':hashlib.sha256(forecast_path.read_bytes()).hexdigest(),
        'datasetSHA256':'a'*64,'simulatorRevision':SIMULATOR_REVISION,'testOnly':True}))
    output = tmp_path/'context-control'
    script = str(Path(__file__).parents[1]/'scripts/train_entity_selfplay.py')
    shared = ['--device','cpu','--distributed','--num-envs','1','--games-per-actor','1','--segments-per-update','2',
        '--transitions-per-update','16','--trajectory-queue-size','8','--discount-mode','per_turn',
        '--lean-policy-execution','--lean-bootstrap','--dense-trajectories','--checkpoint-every','1','--seed','812',
        '--checkpoint-dir',str(output)]
    command = [sys.executable,script,*shared,'--init-checkpoint',str(parent),'--updates','2',
        '--causal-context-schema',CONTEXT_SCHEMA,'--causal-adapter-mode','control',
        '--causal-forecast-checkpoint',str(forecast_path),'--causal-forecast-review',str(review)]
    first = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert first.returncode == 0, first.stdout+first.stderr
    assert 'finished updates=2 ' in first.stdout
    checkpoint = output/'update-000002.pt'
    data = torch.load(checkpoint, weights_only=False)
    assert data['causal_context']['schema'] == CONTEXT_SCHEMA and data['causal_context']['adapterMode'] == 'control'
    configuration = json.loads((output/'config.json').read_text())['arguments']
    assert configuration['lean_policy_execution'] and configuration['lean_bootstrap'] and configuration['dense_trajectories']
    assert configuration['batched_learner'] is False
    resume = subprocess.run([sys.executable,script,*shared,'--resume-checkpoint',str(checkpoint),'--updates','3'],
                            capture_output=True, text=True, timeout=60)
    assert resume.returncode == 0, resume.stdout+resume.stderr
    assert 'finished updates=3 ' in resume.stdout
    bad = subprocess.run([sys.executable,script,*shared,'--resume-checkpoint',str(output/'update-000003.pt'),'--updates','4',
                          '--causal-adapter-mode','residual'], capture_output=True, text=True, timeout=30)
    assert bad.returncode != 0 and 'silently change architecture' in bad.stderr+bad.stdout
