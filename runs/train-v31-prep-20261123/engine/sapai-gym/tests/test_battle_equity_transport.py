import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from sapai_gym1.rl.battle_equity import BattleEquityConfig,resolve_config
from sapai_gym1.rl.training import BatchedSelfPlayCollector,learner_update
from sapai_gym1.rl.distributed import DistributedSelfPlayRuntime
from test_end_freeze_policy import policies
from test_dense_trajectory import convert_through_ring


@pytest.mark.parametrize('resident',[False,True])
def test_real_collector_mean_rewards_and_factual_aux_labels_keep_actions_memory_rng(resident):
    torch.manual_seed(71958001);_,_,policy=policies()
    opponent=copy.deepcopy(policy) if resident else None
    common=dict(num_envs=2,device='cpu',seed=71958002,use_bfloat16=False,opponent_policy=opponent,
                discount_mode='per_turn',learn_from_resident_sides=False,lean_policy_execution=True)
    a=BatchedSelfPlayCollector(policy,**common)
    b=BatchedSelfPlayCollector(policy,**common,battle_equity_config=BattleEquityConfig(2,71958003))
    events=[]
    for env in b.envs:
        original=env.step
        def tracked(*args,original=original,env=env):
            result=original(*args)
            if result.battle_resolved:
                assert env._battle_equity.last is not None
                for side in ((0,) if resident else (0,1)):
                    events.append((result.rewards[side],2 if result.battle_result==2 else 0 if result.battle_result==side else 1))
            return result
        env.step=tracked
    torch.manual_seed(71958004);left=a.collect(16);rng=torch.get_rng_state().clone()
    torch.manual_seed(71958004);right=b.collect(16)
    assert torch.equal(torch.get_rng_state(),rng) and len(left)==len(right)==len(events)
    for first,second,event in zip(left,right,events):
        assert first.terminal==second.terminal and first.battle_outcome==second.battle_outcome==event[1]
        assert second.transitions[-1].reward==event[0]
        torch.testing.assert_close(first.initial_memory,second.initial_memory,atol=0,rtol=0)
        for key in first.bootstrap_observation:assert np.array_equal(first.bootstrap_observation[key],second.bootstrap_observation[key])
        assert len(first.transitions)==len(second.transitions)
        for x,y in zip(first.transitions,second.transitions):
            assert np.array_equal(x.action,y.action) and x.behavior_log_prob==y.behavior_log_prob
            assert x.end_freeze_mask==y.end_freeze_mask and x.discount==y.discount
            for key in x.observation:assert np.array_equal(x.observation[key],y.observation[key])


def test_m2_dense_ring_and_full_optimizer_step_equal_legacy_packing():
    torch.manual_seed(71958011);_,_,policy=policies()
    data=BatchedSelfPlayCollector(policy,2,'cpu',seed=71958012,discount_mode='per_turn',lean_policy_execution=True,
                                  battle_equity_config=BattleEquityConfig(2,71958013)).collect(18)
    dense=convert_through_ring(data)
    for segment,owned in zip(data,dense):
        torch.testing.assert_close(owned.rewards,torch.tensor([r.reward for r in segment.transitions]),atol=0,rtol=0)
        assert owned.battle_outcome==segment.battle_outcome
    models=[copy.deepcopy(policy),copy.deepcopy(policy)]
    optimizers=[torch.optim.AdamW(m.parameters()) for m in models]
    a=learner_update(models[0],optimizers[0],data,'cpu',use_bfloat16=False,lean_bootstrap=True)
    b=learner_update(models[1],optimizers[1],dense,'cpu',use_bfloat16=False,lean_bootstrap=True)
    assert vars(a)==vars(b)
    for x,y in zip(models[0].parameters(),models[1].parameters()):
        torch.testing.assert_close(x,y,atol=0,rtol=0)
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,atol=0,rtol=0)
    for x,y in zip(optimizers[0].state.values(),optimizers[1].state.values()):
        for key in x:torch.testing.assert_close(x[key],y[key],atol=0,rtol=0)


def test_spawned_m2_actor_to_dense_ring_and_learner():
    torch.manual_seed(71958021);env,_,policy=policies()
    runtime=DistributedSelfPlayRuntime(policy,1,env.observations()[0],'cpu',seed=71958022,
        gamma=.99,queue_size=16,games_per_actor=2,learn_from_resident_sides=False,
        discount_mode='per_turn',lean_policy_execution=True,battle_equity_config=BattleEquityConfig(2,71958023))
    assert runtime.battle_equity_config.samples==2
    runtime.start()
    try:
        segments=runtime.collect_segments(8,timeout=60,dense=True)
        assert all(s.length>0 and torch.isfinite(s.rewards).all() for s in segments)
        optimizer=torch.optim.AdamW(policy.parameters())
        metrics=learner_update(policy,optimizer,segments,'cpu',use_bfloat16=False,lean_bootstrap=True)
        assert np.isfinite(metrics.total_loss)
    finally:runtime.stop()


def test_actual_cli_m2_save_resume_and_fail_closed_mode_change(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/train_entity_selfplay.py'
    common=[sys.executable,str(script),'--device','cpu','--smoke','--end-freeze','--num-envs','2',
        '--lean-policy-execution','--lean-bootstrap','--discount-mode','per_turn','--disable-expert-anchor']
    directory=tmp_path/'run'
    first=subprocess.run(common+['--updates','1','--checkpoint-dir',str(directory),
        '--battle-equity-samples','2','--battle-equity-seed','71958031'],capture_output=True,text=True,timeout=60)
    assert first.returncode==0,first.stdout+first.stderr
    path=directory/'update-000001.pt';one=torch.load(path,map_location='cpu',weights_only=False)
    assert resolve_config(one)==BattleEquityConfig(2,71958031)
    second=subprocess.run(common+['--updates','2','--checkpoint-dir',str(directory),'--resume-checkpoint',str(path)],
        capture_output=True,text=True,timeout=60)
    assert second.returncode==0,second.stdout+second.stderr
    two=torch.load(directory/'update-000002.pt',map_location='cpu',weights_only=False)
    assert resolve_config(two)==resolve_config(one) and two['update']==2
    assert json.loads((directory/'config.json').read_text())['reward_estimator']==resolve_config(two).to_dict()
    failed=subprocess.run(common+['--updates','2','--checkpoint-dir',str(tmp_path/'forbidden'),
        '--resume-checkpoint',str(path),'--battle-equity-samples','1'],capture_output=True,text=True,timeout=60)
    assert failed.returncode!=0 and 'transfer' in failed.stderr
