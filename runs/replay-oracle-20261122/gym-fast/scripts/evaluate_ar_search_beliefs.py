"""Three-way root-decision screen on independent terminal continuations."""
import argparse,hashlib,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np,torch
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.model import EntityTransformerConfig,EntityTransformerPolicy
from sapai_gym1.rl.causal_transport import validate_checkpoint_context
from sapai_gym1.rl.observation_contract import OBSERVATION_REVISION, checkpoint_revision
from sapai_gym1.rl.decision_manifest import restore_decision
from sapai_gym1.rl.rollout_search import PublicSearchState,NeuralSearchEvaluator
from sapai_gym1.rl.puct_search import StochasticPUCTSearch,PUCTConfig
from sapai_gym1.rl.opponent_belief import OpponentBeliefMix
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.ar_search_belief import ARBeliefProvider
from sapai_gym1.rl.search_work_budget import SearchWorkBudget,SearchBudgetExceeded
from sapai_gym1.rl.fixed_interactive_evaluation import FixedBatchInteractiveEvaluator
from sapai_gym1.rl.joint_demonstrations import serialize_action
from sapai_gym1.rl.opponent_forecast_quality import paired_match_interval

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def public_state(env,side,reservoir):
    prior=None
    if (env.core.previous_opponent_state if side==0 else env.core.previous_agent_state) is None:
        prior=[m.teams for m in reservoir.opening_prior(horizon=1,limit=32,seed=609770)]
    state=PublicSearchState.from_core(env.core,side,opening_prior=prior)
    state.set_opponent_transition_model(reservoir,candidate_limit=32);state.enable_learned_opponent_belief(OpponentBeliefMix())
    return state

def main():
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path);parser.add_argument('--forecast',type=Path,required=True)
    parser.add_argument('--uniforms',type=Path,required=True);parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--development',action='store_true')
    parser.add_argument('--allow-observation-transfer',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    if args.device=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(exist_ok=False,parents=True)
    meta=json.loads((args.panel/'manifest.json').read_text());assert meta['complete'] and meta['checkpointSha256']==sha(args.checkpoint)
    assert (meta['stage']=='development')==args.development
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    validate_checkpoint_context(payload,allow_observation_transfer=args.allow_observation_transfer)
    if meta.get('observationRevision') != OBSERVATION_REVISION:
        raise ValueError('Evaluation panel observation contract mismatch')
    assert payload['simulator_revision']==SIMULATOR_REVISION
    policy=EntityTransformerPolicy(EntityTransformerConfig(**payload['config'])).to(args.device).eval();policy.load_state_dict(payload['model'])
    reservoir=OpponentTrajectoryReservoir.load(args.panel/'reservoir.json')
    assert sha(args.panel/'reservoir.json')==meta['reservoir']['sha256']
    provider=ARBeliefProvider.load(args.forecast,args.uniforms,device=args.device)
    evaluators={name:NeuralSearchEvaluator(policy,device=args.device,chunk_size=64,opponent_belief_provider=p)
                for name,p in [('existing',None),('ar',provider)]}
    config=PUCTConfig(max_simulations=128,max_simulator_transitions=2000,horizon_battles=5,validation_rollouts=8,
        max_validation_rollouts=8,audit_rollouts=0,root_screening_rollouts=4,root_screening_rounds=2,tree_leaf_rollout_steps=12)
    truth=FixedBatchInteractiveEvaluator(policy,device=args.device,batch_size=64,horizon_battles=25,max_steps=5000)
    specification={'policySHA256':sha(args.checkpoint),'forecastSHA256':sha(args.forecast),'uniformsSHA256':sha(args.uniforms),
        'observationRevision':OBSERVATION_REVISION,'checkpointTrainingObservationRevision':checkpoint_revision(payload),
        'explicitObservationTransfer':args.allow_observation_transfer,
        'panelSHA256':meta['panelSHA256'],'reservoirSHA256':meta['reservoir']['sha256'],'planner':asdict(config),
        'totalDecisionCap':20000,'deadlineSeconds':30,'deadlineIsCooperative':True,'development':args.development,
        'truthFutures':4 if args.development else 32,'publicRepresentation':'effective totals plus flags; no hidden amounts/RNG',
        'distillationApproved':False,'truthPolicyNumerics':'FP32 fixed padded batch64, greedy',
        'rules':SIMULATOR_REVISION,'runnerSHA256':sha(__file__)}
    (args.output/'specification.json').write_text(json.dumps(specification,indent=2)+'\n')
    rows=[];clusters=[];gains={'existing':[],'ar':[],'ar_vs_existing':[]}
    for index in range(meta['states']):
        record=json.loads((args.panel/f'state-{index:04d}.json').read_text())
        assert record['policySha256BySeat']==[sha(args.checkpoint)]*2
        env,memories,side,_=restore_decision(record)
        fallback_eval=NeuralSearchEvaluator(policy,device=args.device)
        raw_state=public_state(env,side,reservoir);raw_state.opponent_belief_mix=None
        real_observation=env.observations()[side]
        for key,value in raw_state.observation().items():
            np.testing.assert_array_equal(value,real_observation[key],err_msg='Public policy-input parity: '+key)
        raw=truth._choose_many([(0,side,real_observation,memories[side])])[0][0]
        decisions={'raw':raw};planning={}
        for name in ('existing','ar'):
            state=public_state(env,side,reservoir);budget=SearchWorkBudget();state.work_budget=budget
            evaluator=evaluators[name];before=evaluator.metrics();start=time.perf_counter()
            try:
                result=StochasticPUCTSearch(evaluator,config).search(state,memories[side],seed=record['searchSeed'],validation_seed=record['validationSeed'])
                budget.check();decisions[name]=result.selected_action if result.accepted else raw
                planning[name]={'fallback':result.selected_action==raw,'budgetExhausted':False,
                    'omittedMaskMass':result.omitted_end_mask_probability_mass,'internalAccepted':result.accepted,
                    'plannerRawDiffersFromFixedBatchRaw':result.policy_action!=raw}
                planning[name]['fallback']=decisions[name]==raw
            except SearchBudgetExceeded as error:
                decisions[name]=raw;planning[name]={'fallback':True,'budgetExhausted':True,'reason':str(error)}
            planning[name].update(wallSeconds=time.perf_counter()-start,budget=budget.report(),neuralBefore=before,neuralAfter=evaluator.metrics())
        actions=tuple(dict.fromkeys(decisions.values()));seeds=record['truthSeeds'][:4 if args.development else 32]
        started=time.perf_counter();outcomes=truth.evaluate(env,memories,side,actions,seeds)
        assert all(r.terminated and r.match_score is not None for r in outcomes),'Incomplete terminal panel; no selective exclusions'
        scores={(r.root_action,r.seed):r.match_score for r in outcomes}
        composition_check=None
        if args.development and index<2:
            alone=truth.evaluate(env,memories,side,(raw,),seeds)
            assert all(r.match_score==scores[raw,r.seed] for r in alone),'Raw terminal score depends on alternative batch composition'
            alternatives=fallback_eval.evaluate(raw_state,memories[side]).priors
            alternate=next((a for a in alternatives if a!=raw),None)
            if alternate is not None:
                mixed=truth.evaluate(env,memories,side,(raw,alternate),seeds)
                assert all(r.match_score==scores[raw,r.seed] for r in mixed if r.root_action==raw),'Forced alternative changed raw terminal outcomes'
            composition_check={'passed':True,'distinctAlternativeUsed':alternate is not None}
        values={name:np.asarray([scores[action,s] for s in seeds]) for name,action in decisions.items()}
        delta={name:float((values[name]-values['raw']).mean()) for name in ('existing','ar')}
        delta['ar_vs_existing']=float((values['ar']-values['existing']).mean())
        for name in gains:gains[name].append(delta[name])
        clusters.append({'metadata':{'collection_id':meta['panelSHA256'],'match_id':str(record['replay']['resetSeed'])}})
        row={'state':index,'decisions':{n:serialize_action(a) for n,a in decisions.items()},'planning':planning,
            'terminalScoreGains':delta,'truthSeconds':time.perf_counter()-started,'truthSeeds':seeds,
            'terminalScores':{n:v.tolist() for n,v in values.items()},'provider':provider.metrics(),
            'rawBatchCompositionCheck':composition_check}
        with (args.output/f'state-{index:04d}.json').open('x') as f:json.dump(row,f,indent=2);f.write('\n')
        rows.append(row);print(json.dumps({'state':index,'gains':delta,'planningSeconds':{n:v['wallSeconds'] for n,v in planning.items()}}),flush=True)
    comparisons={}
    for name,v in gains.items():
        estimate=paired_match_interval(clusters,v,[0.]*len(v))
        comparisons[name]={'originatingMatches':estimate['matches'],'meanTerminalScoreGain':estimate['meanMatchNllImprovement'],
            'pairedOriginatingMatch95CI':estimate['bootstrapMatch95CI'],'bootstrapReplicates':estimate['bootstrapReplicates']}
    summary={'comparisons':comparisons,
        'note':'Units are TERMINAL MATCH SCORE, not internal shaped return or forecast NLL.',
        'states':len(rows),'futuresPerState':len(seeds),'allTerminal':True,'provider':provider.metrics(),
        'fallbackFractions':{name:sum(r['planning'][name]['fallback'] for r in rows)/len(rows) for name in ('existing','ar')},
        'development':args.development,'distillationApproved':False,'superhumanClaim':False}
    with (args.output/'summary.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
if __name__=='__main__':main()
