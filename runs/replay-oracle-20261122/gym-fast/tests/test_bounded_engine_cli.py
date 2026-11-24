"""Actual diagnostic CLI with a tiny on-policy terminal-turn fixture."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from sapai import Team
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.decision_manifest import decision_record,restore_decision
from sapai_gym1.rl.observation_contract import OBSERVATION_REVISION
from sapai_gym1.rl.joint_action import JointAction,step_decisions
from sapai_gym1.rl.action_schema import AutoregressiveAction,ActionType
from sapai_gym1.rl.bounded_engine import EngineLimits
from sapai_gym1.rl.puct_search import PUCTConfig
from sapai_gym1.rl.training import stack_observations
from test_end_freeze_policy import policies

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from train_entity_selfplay import save_checkpoint
from evaluate_bounded_engine import validate_spec


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path):
    env,_,policy=policies();torch.set_num_threads(1)
    with torch.no_grad():
        policy.type_head.weight.zero_();policy.type_head.bias.fill_(-100)
        policy.type_head.bias[ActionType.END_TURN]=100;policy.type_head.bias[ActionType.ROLL]=90
        policy.end_freeze_head.scorer.weight.zero_();policy.end_freeze_head.scorer.bias.zero_()
    checkpoint=tmp_path/'tiny.pt';opt=torch.optim.AdamW(policy.parameters())
    save_checkpoint(checkpoint,policy,opt,policy.config,0,0,discount_mode='per_turn',discount=.99,
        battle_aux_cost=.05,opponent_aux_cost=.1,opponent_module_learning_rate=.001,opponent_adapter_learning_rate=.001)
    observations=env.reset(seed=716001);memories=[policy.initial_memory(1)[0] for _ in range(2)];replay=[]
    # Genuine decisions/memories from these exact frozen tiny weights. No
    # mutation of terminal rules or imported foreign recurrent state.
    with torch.no_grad():
        while env.core.turn<24 or env.core.agent.gold>0:
            out=policy.act(stack_observations(observations,'cpu'),torch.stack(memories),deterministic=True)
            decisions=[JointAction(AutoregressiveAction(*a.tolist()),int(m)) for a,m in zip(out.actions,out.end_freeze_mask)]
            step=step_decisions(env,decisions).result
            assert not step.terminated
            memories=[v.clone() for v in out.memory];observations=step.observations
            replay.append([d.to_dict() for d in decisions])
    record=decision_record(env,memories,0,('terminal-turn-smoke',),{'resetSeed':716001,'actions':replay},
        search_seed=1,validation_seed=2,truth_seeds=(101,102))
    record.update(state=0,checkpointSha256=sha(checkpoint),policySha256BySeat=[sha(checkpoint)]*2)
    restore_decision(record)
    panel=tmp_path/'panel';panel.mkdir();(panel/'state-0000.json').write_text(json.dumps(record))
    digest=hashlib.sha256(json.dumps([record],sort_keys=True).encode()).hexdigest()
    (panel/'manifest.json').write_text(json.dumps({'complete':True,'states':1,'checkpointSha256':sha(checkpoint),
        'observationRevision':OBSERVATION_REVISION,'panelSHA256':digest}))
    reservoir=tmp_path/'reservoir.json';reservoir.write_text(json.dumps({'schemaVersion':2,
        'metadata':{'simulatorRevision':SIMULATOR_REVISION},'trajectories':[{'id':'tiny-test-only',
        'boards':[{'turn':23,'team':Team().state},{'turn':24,'team':Team().state}]}]}))
    limits=EngineLimits(max_searches=1,searches_per_turn=1,total_decisions=1000,total_neural_states=1000,
        total_seconds=10.,decisions_per_search=1000,neural_states_per_search=1000,seconds_per_search=10.,minimum_start_seconds=.001)
    engine_limits={**asdict(limits),'max_searches':2,'searches_per_turn':2,'decisions_per_search':500,
        'neural_states_per_search':500,'seconds_per_search':5.}
    planner=PUCTConfig(max_simulations=2,max_simulator_transitions=10,max_rollout_steps=10,horizon_battles=1,
        tree_leaf_rollout_steps=0,root_screening_top_k=2,root_screening_min_survivors=1,root_screening_rollouts=2,
        root_screening_rounds=1,validation_rollouts=2,max_validation_rollouts=2,audit_rollouts=0,continuation_mode='tree_plan')
    spec={'schema':'bounded-engine-diagnostic-v1','checkpointSHA256':sha(checkpoint),'reservoirSHA256':sha(reservoir),
        'panelSHA256':digest,'planner':asdict(planner),'rootLimits':asdict(limits),'engineLimits':engine_limits,
        'selector':'screened','futures':2,'truthBatchSize':4,'truthMaxSteps':100,'truthHorizonBattles':25,
        'openingSeed':716002,'candidateLimit':4}
    path=tmp_path/'spec.json';path.write_text(json.dumps(spec))
    return checkpoint,panel,reservoir,path


def test_actual_cpu_cli_completes_raw_root_and_engine_without_authorizing_training(tmp_path):
    checkpoint,panel,reservoir,spec=fixture(tmp_path);out=tmp_path/'result'
    result=subprocess.run([sys.executable,str(SCRIPTS/'evaluate_bounded_engine.py'),str(checkpoint),
        '--panel',str(panel),'--reservoir',str(reservoir),'--spec',str(spec),'--output',str(out),
        '--device','cpu','--diagnostic'],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    summary=json.loads((out/'summary.json').read_text());row=json.loads((out/'state-0000.json').read_text())
    assert summary['complete'] and summary['allTerminal'] and not summary['approvedForDistillation']
    assert summary['playingStrengthEstablished'] is False
    assert set(row['modes'])=={'raw','root','engine'}
    for mode in row['modes'].values():
        assert len(mode['outcomes'])==2 and all(v['terminated'] and v['score'] in (0,.5,1) and v['executedRoot'] for v in mode['outcomes'])
    assert all(v['counts']['searchAttempts']>=1 for v in row['modes']['root']['execution'])


@pytest.mark.parametrize('mutation',['unknown','budget','root_attempts'])
def test_spec_rejects_unmatched_or_implicit_experiments(mutation):
    limits=asdict(EngineLimits(1,1,10,10,1.,10,10,1.,.01))
    spec={'schema':'bounded-engine-diagnostic-v1','checkpointSHA256':'a'*64,'reservoirSHA256':'b'*64,
        'panelSHA256':'c'*64,'planner':{},'rootLimits':limits,'engineLimits':dict(limits),
        'selector':'screened','futures':2,'truthBatchSize':4,'truthMaxSteps':100,'truthHorizonBattles':25,
        'openingSeed':1,'candidateLimit':4}
    if mutation=='unknown':spec['secret_override']=True
    if mutation=='budget':spec['engineLimits']['total_decisions']=11
    if mutation=='root_attempts':spec['rootLimits']['max_searches']=2
    with pytest.raises(ValueError):validate_spec(spec)
