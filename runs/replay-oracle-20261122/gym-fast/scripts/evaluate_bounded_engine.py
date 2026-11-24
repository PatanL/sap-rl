"""Diagnostic CLI: raw vs one planned root vs budgeted actual replanning.

No production defaults, promotion or teaching authorization. Real quality
experiments require a separately frozen fresh-state protocol after RL.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from generate_search_demonstrations import load_policy
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.observation_contract import OBSERVATION_REVISION
from sapai_gym1.rl.decision_manifest import restore_decision
from sapai_gym1.rl.rollout_search import NeuralSearchEvaluator
from sapai_gym1.rl.puct_search import PUCTConfig
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.bounded_engine import BoundedPublicEngine,BoundedEngineEvaluator,EngineLimits,action_record
from sapai_gym1.rl.fixed_interactive_evaluation import FixedBatchInteractiveEvaluator
from sapai_gym1.rl.joint_demonstrations import runtime_fingerprint


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def validate_spec(spec):
    required={'schema','checkpointSHA256','reservoirSHA256','panelSHA256','planner','rootLimits','engineLimits',
        'selector','futures','truthBatchSize','truthMaxSteps','truthHorizonBattles','openingSeed','candidateLimit'}
    if set(spec)!=required or spec['schema']!='bounded-engine-diagnostic-v1':raise ValueError('Unknown/missing diagnostic specification fields')
    root,engine=EngineLimits(**spec['rootLimits']),EngineLimits(**spec['engineLimits'])
    if root.max_searches!=1 or engine.max_searches<1:raise ValueError('Root arm searches once; engine must declare its attempts')
    for field in ('total_decisions','total_neural_states','total_seconds'):
        if getattr(root,field)!=getattr(engine,field):raise ValueError('Matched maximum total budgets required; actual use is reported separately')
    for field in ('futures','truthBatchSize','truthMaxSteps','truthHorizonBattles','candidateLimit'):
        if type(spec[field]) is not int or spec[field]<1:raise ValueError(f'Invalid {field}')
    if spec['futures']<2:raise ValueError('At least two declared terminal futures required')
    if type(spec['openingSeed']) is not int or spec['openingSeed']<0:raise ValueError('Explicit opening seed required')
    if spec['selector'] not in ('gated','screened','tree'):raise ValueError('Explicit selector required')
    return PUCTConfig(**spec['planner']),root,engine


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--reservoir',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--gpu-boundary-approved',action='store_true')
    p.add_argument('--allow-observation-transfer',action='store_true');a=p.parse_args()
    if not a.diagnostic:raise ValueError('This CLI is diagnostic-only, not a production quality/distillation gate')
    if a.device=='cuda' and not a.gpu_boundary_approved:raise ValueError('A separately allocated GPU boundary is required')
    a.output.mkdir(parents=True,exist_ok=False);summary={'complete':False,'diagnosticOnly':True,'approvedForDistillation':False}
    started=time.perf_counter()
    try:
        torch.set_num_threads(1);device=torch.device(a.device)
        if device.type=='cuda':
            if not torch.cuda.is_available():raise ValueError('Requested CUDA is unavailable; no silent CPU fallback')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        spec=json.loads(a.spec.read_text());planner,root_limits,engine_limits=validate_spec(spec)
        checkpoint_hash=sha(a.checkpoint)
        if checkpoint_hash!=spec['checkpointSHA256'] or sha(a.reservoir)!=spec['reservoirSHA256']:raise ValueError('Checkpoint/reservoir content hash mismatch')
        panel=json.loads((a.panel/'manifest.json').read_text())
        if (panel['complete'] is not True or type(panel['states']) is not int or panel['states']<1
                or panel['checkpointSha256']!=checkpoint_hash or panel['observationRevision']!=OBSERVATION_REVISION):
            raise ValueError('Panel parent/observer/completion mismatch')
        records=[json.loads((a.panel/f'state-{i:04d}.json').read_text()) for i in range(panel['states'])]
        digest=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
        if digest!=panel['panelSHA256'] or digest!=spec['panelSHA256']:raise ValueError('Panel content hash mismatch')
        if any(len(r['truthSeeds'])<spec['futures'] or len(set(r['truthSeeds'][:spec['futures']]))!=spec['futures']
               or r['policySha256BySeat']!=[checkpoint_hash]*2 for r in records):
            raise ValueError('Insufficient seeds or foreign recurrent-state ownership')
        policy,checkpoint=load_policy(a.checkpoint,device,False,allow_observation_transfer=a.allow_observation_transfer)
        if getattr(policy.config,'causal_context_schema',None) is not None:raise ValueError('This first engine diagnostic holds the existing opponent architecture fixed')
        reservoir=OpponentTrajectoryReservoir.load(a.reservoir)
        if reservoir.metadata.get('simulatorRevision')!=SIMULATOR_REVISION:raise ValueError('Reservoir simulator revision mismatch')
        evaluator=NeuralSearchEvaluator(policy,device=device,chunk_size=64)
        def factory(limits):
            return lambda:BoundedPublicEngine(evaluator,planner,limits,checkpoint_sha256=checkpoint_hash,
                memory_dim=policy.config.memory_dim,reservoir=reservoir,selector=spec['selector'],
                candidate_limit=spec['candidateLimit'],opening_seed=spec['openingSeed'])
        kwargs={'device':device,'batch_size':spec['truthBatchSize'],'max_steps':spec['truthMaxSteps'],
                'horizon_battles':spec['truthHorizonBattles']}
        modes={'raw':FixedBatchInteractiveEvaluator(policy,**kwargs),
               'root':BoundedEngineEvaluator(policy,engine_factory=factory(root_limits),**kwargs),
               'engine':BoundedEngineEvaluator(policy,engine_factory=factory(engine_limits),**kwargs)}
        (a.output/'manifest.json').write_text(json.dumps({'spec':spec,'specSHA256':sha(a.spec),'runnerSHA256':sha(__file__),
            'runtimeFingerprint':runtime_fingerprint(),'checkpointSHA256':checkpoint_hash,
            'simulatorRevision':SIMULATOR_REVISION,'observationRevision':OBSERVATION_REVISION,
            'policyNumerics':'FP32 fixed padded batches; no GPU BF16 change',
            'matchedMaximumBudgetNotEqualActualUse':True,'diagnosticOnly':True},indent=2)+'\n')
        rows=[]
        for index,record in enumerate(records):
            row={'state':index,'modes':{}};mode=None;name=None
            env,memories,side,_=restore_decision(record)
            raw=modes['raw']._choose_many([(0,side,env.observations()[side],memories[side])])[0][0]
            seeds=record['truthSeeds'][:spec['futures']];row['requestedComparatorRoot']=action_record(raw)
            for name,mode in modes.items():
                before=time.perf_counter();outcomes=mode.evaluate(env,memories,side,(raw,),seeds)
                if len(outcomes)!=len(seeds) or [r.seed for r in outcomes]!=seeds:raise ValueError('Terminal future identity/count mismatch')
                row['modes'][name]={'wallSeconds':time.perf_counter()-before,'outcomes':[
                    {'seed':r.seed,'score':r.match_score,'terminated':r.terminated,'truncated':r.truncated,
                     'executedRoot':action_record(r.executed_root_action)} for r in outcomes],
                    'execution':[] if name=='raw' else mode.execution_reports()}
                if any(not r.terminated or r.truncated or r.match_score is None for r in outcomes):
                    with (a.output/f'state-{index:04d}.json').open('x') as f:json.dump(row,f,indent=2)
                    raise ValueError('Incomplete terminal continuations; never substitute draws or omit cases')
            baseline=np.asarray([r['score'] for r in row['modes']['raw']['outcomes']])
            row['terminalScoreDifference']={name:float((np.asarray([r['score'] for r in row['modes'][name]['outcomes']])-baseline).mean()) for name in ('root','engine')}
            with (a.output/f'state-{index:04d}.json').open('x') as f:json.dump(row,f,indent=2);f.write('\n')
            rows.append(row);print(json.dumps({'state':index,'terminalScoreDifference':row['terminalScoreDifference']}),flush=True)
        summary.update(complete=True,states=len(rows),futuresPerState=spec['futures'],allTerminal=True,
            meanStateScoreDifferences={name:float(np.mean([r['terminalScoreDifference'][name] for r in rows])) for name in ('root','engine')},
            wholeEvaluationWallSeconds=time.perf_counter()-started,playingStrengthEstablished=False,
            inference='Diagnostic point estimates only; independent predeclared state-clustered confirmation still required.')
    except BaseException as error:
        summary['error']=repr(error)
        if 'row' in locals():
            partial={**row,'failure':repr(error),'failedMode':locals().get('name')}
            if 'mode' in locals() and hasattr(mode,'execution_reports'):
                partial['partialExecution']=mode.execution_reports()
            with (a.output/'failed-state.json').open('x') as f:json.dump(partial,f,indent=2);f.write('\n')
        raise
    finally:
        with (a.output/'summary.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')


if __name__=='__main__':main()
