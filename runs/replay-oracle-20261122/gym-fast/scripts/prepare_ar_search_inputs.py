"""Fresh current-rule public state panels and TRAIN-only public reservoir."""
import argparse,hashlib,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np,torch
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.model import EntityTransformerConfig,EntityTransformerPolicy
from sapai_gym1.rl.causal_transport import validate_checkpoint_context
from sapai_gym1.rl.observation_contract import OBSERVATION_REVISION, checkpoint_revision
from sapai_gym1.rl.decision_manifest import decision_record,restore_decision,MANIFEST_VERSION
from sapai_gym1.rl.opponent_forecast_data import _match_error
from sapai_gym1.rl.public_battle_board import from_public_records,to_state
from search_state_mining import mine_match_opportunity

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def make_reservoir(data_path,path):
    data=json.loads(data_path.read_text());assert data['testExcluded'] and 'test' not in data
    grouped=defaultdict(list)
    for row in data['train']:grouped[row['metadata']['collection_id'],row['metadata']['match_id']].append(row)
    trajectories=[];sources=Counter()
    for key,rows in sorted(grouped.items()):
        _match_error(rows);indexed={(r['metadata']['turn'],r['metadata']['side']):r for r in rows}
        turns=sorted({t for t,s in indexed})
        for owner in (0,1):
            boards=[];source=rows[0]['metadata']['checkpoint_populations'][owner];sources[source]+=1
            for turn in turns:
                own=indexed[turn,1-owner]['target'];enemy=indexed[turn,owner]['target']
                boards.append({'turn':turn,'team':to_state(from_public_records(own)),
                               'opponentTeam':to_state(from_public_records(enemy))})
            trajectories.append({'id':f'{key[0]}:{key[1]}:{owner}','sourceIndex':source,'boards':boards})
    payload={'schemaVersion':2,'metadata':{'simulatorRevision':SIMULATOR_REVISION,'dataSHA256':sha(data_path),
        'inputSplit':'TRAIN ONLY','matches':len(grouped),'bothPerspectives':True,'publicTotalsAndFlagsOnly':True,
        'futureSurvivalFiltering':False,'components':[{'sourceIndex':s,'games':n} for s,n in sorted(sources.items())]},'trajectories':trajectories}
    with path.open('x') as f:json.dump(payload,f,separators=(',',':'));f.write('\n')
    return {'sha256':sha(path),'matches':len(grouped),'trajectories':len(trajectories),'populations':dict(sources)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stage',choices=('development','screen'),required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--allow-observation-transfer',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);device=torch.device(args.device);args.output.mkdir(exist_ok=False,parents=True)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    validate_checkpoint_context(ckpt,allow_observation_transfer=args.allow_observation_transfer)
    assert ckpt['simulator_revision']==SIMULATOR_REVISION and ckpt['config'].get('entropy_mode','legacy_path')=='legacy_path'
    policy=EntityTransformerPolicy(EntityTransformerConfig(**ckpt['config'])).to(args.device).eval();policy.load_state_dict(ckpt['model'])
    checkpoint_hash=sha(args.checkpoint)
    reservoir=make_reservoir(args.data,args.output/'reservoir.json')
    windows=([(1,4)]*2+[(5,8)]*3+[(9,25)]*3 if args.stage=='development' else [(1,4)]*8+[(5,8)]*12+[(9,25)]*12)
    base_seed=60977000 if args.stage=='development' else 60978000
    limit=64 if args.stage=='development' else 256;attempts=0;reset_seeds=[];records=[]
    for index,(minimum,maximum) in enumerate(windows):
        torch.manual_seed(base_seed+index);rng=np.random.default_rng(base_seed+index)
        chosen=None
        while chosen is None and attempts<limit:
            attempts+=1
            chosen=mine_match_opportunity(policy,rng,device,minimum,maximum,band='any',with_replay=True,
                acting_side=index%2,opportunity_filter=lambda core,side:('uniform-multiple-legal',) if len(core._avail_actions(side))>1 else ())
            if chosen is not None and chosen[-1]['resetSeed'] in reset_seeds:chosen=None
        if chosen is None:raise RuntimeError('Predeclared collection attempt budget exhausted; do not silently change panel')
        env,memories,side,reasons,replay=chosen;reset_seeds.append(replay['resetSeed'])
        sr=np.random.default_rng(base_seed+10000+index)
        record=decision_record(env,memories,side,reasons,replay,search_seed=int(sr.integers(0,2**63-1)),
            validation_seed=int(sr.integers(0,2**63-1)),truth_seeds=sr.integers(0,2**63-1,size=32))
        record.update({'state':index,'checkpointSha256':checkpoint_hash,'policySha256BySeat':[checkpoint_hash]*2,
                       'turnWindow':[minimum,maximum],'panelStage':args.stage})
        restore_decision(record)
        with (args.output/f'state-{index:04d}.json').open('x') as f:json.dump(record,f,separators=(',',':'));f.write('\n')
        records.append(record);print(f'state={index} turn={record["turn"]} gold={record["gold"]} seat={side}',flush=True)
    manifest={'version':MANIFEST_VERSION,'checkpointSha256':checkpoint_hash,'simulatorRevision':SIMULATOR_REVISION,
        'observationRevision':OBSERVATION_REVISION,'checkpointTrainingObservationRevision':checkpoint_revision(ckpt),
        'explicitObservationTransfer':args.allow_observation_transfer,
        'states':len(records),'truthRollouts':32,'stage':args.stage,'generatorSHA256':sha(__file__),
        'reservoir':reservoir,'attemptedMatches':attempts,'uniqueOriginatingMatches':len(set(reset_seeds)),
        'panelSHA256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
        'purpose':'independent terminal evaluation only; not training labels','complete':True}
    with (args.output/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
if __name__=='__main__':main()
