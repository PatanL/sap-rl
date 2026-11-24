import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from launch_post_entropy_forecast import (AB_FLAGS, DE_FLAGS, IMAGE, SELECTION_SCHEMA, PLAN_SCHEMA, DATASET_SCHEMA,
                                         LaunchBlocked, object_hash, sha, validate_selection, resources_clean)


class FakeCommands:
    def __init__(self, state='inactive', failure=None, gpu='', collector=False):
        self.state, self.failure, self.gpu, self.collector = state, failure, gpu, collector
        self.calls = []
    def __call__(self, command, **kwargs):
        self.calls.append(command)
        assert 'run' not in command and 'start' not in command and 'stop' not in command
        if command[0] == self.failure: return SimpleNamespace(returncode=1, stdout='', stderr='failed')
        if command[0] == 'systemctl':
            output = f'LoadState=loaded\nActiveState={self.state}\nSubState={"dead" if self.state == "inactive" else "running"}\nMainPID={0 if self.state == "inactive" else 123}\nResult=success\nExecMainCode=1\nExecMainStatus=0'
        elif command[0] == 'nvidia-smi': output = self.gpu
        elif command[1] == 'ps': output = 'sap-one-step-cpu-corpus' if self.collector else ''
        elif command[1] == 'inspect': output = 'false'
        else: output = IMAGE
        return SimpleNamespace(returncode=0, stdout=output, stderr='')


def fixture(tmp_path):
    project = tmp_path/'project'; runs = project/'runs'; runs.mkdir(parents=True)
    source = runs/'source'; source.mkdir(); (source/'model.py').write_text('test source')
    (source/'guide.md').write_text('test asset')
    manifest = runs/'manifest.json'; assets_path = runs/'assets.json'
    files = {'model.py': sha(source/'model.py')}; assets = {'guide.md': sha(source/'guide.md')}
    inventory = {'sha256': object_hash(files), 'files': files}
    manifest.write_text(json.dumps(inventory)); assets_path.write_text(json.dumps(assets))
    campaign = runs/'entropy'; campaign.mkdir(); (campaign/'evaluations').mkdir()
    parent = runs/'update-009050.pt'; parent.write_bytes(b'real-parent')
    final = {}
    for mode in ('legacy_path', 'exact_joint'):
        folder = campaign/mode; folder.mkdir(); path = folder/'update-010050.pt'; path.write_bytes(mode.encode()); final[mode] = str(path)
        (campaign/f'train-{mode}.log').write_text('finished updates=10050 checkpoint=selected\n')
        (campaign/f'train-{mode}-wall.json').write_text('{}')
    state = {'phase':'entropy_results_ready','parent':str(parent),'checkpoints':final}
    (campaign/'campaign-state.json').write_text(json.dumps(state))
    evidence = {}
    for name, seed in [('exact-vs-legacy',62909050),('legacy-vs-parent',62919050),('exact-vs-parent',62919050)]:
        records = [{'pairIndex':i,'seed':i+1,'candidateSide':side,'candidateResult':'win','terminationReason':'lives',
                    'turnLimit':25,'rearrangementLimit':2} for i in range(100) for side in (0,1)]
        report = {'pairs':100,'games':200,'seed':seed,'pairSeeds':list(range(1,101)), 'gameResults':records,
                  'outcomes':{'win':200},'pairedMeanScore':1.,'pairScores':[1.]*100,'pairedScore95CI':[1.,1.],
                  'candidateActionSchema':'primitive-plus-end-freeze-v1-20260905','opponentActionSchema':'primitive-plus-end-freeze-v1-20260905',
                  'runtimeSimulatorRevision':'turtle-current-frozen-shop-cap-v28-20260905',
                  'checkpoint':final['exact_joint' if name.startswith('exact') else 'legacy_path'],
                  'opponentCheckpoint': final['legacy_path'] if name=='exact-vs-legacy' else str(parent)}
        path = campaign/'evaluations'/f'{name}.json'; path.write_text(json.dumps(report)); evidence[name] = sha(path)
    plan = {'schema':PLAN_SCHEMA,'collection':{'totalGames':280,'pairs':[{'games':40,'index':i,'seed':100+i*100,
            'policies':['parent','heldout' if i >= 5 else 'parent'],'populationHeldOut':i >= 5,'allowHistoricalTrainingRevisionMismatch':False} for i in range(7)],'maxCpuCores':1,
            'bothPerspectives':True,'completeMatchesOnly':True,'terminalTurnsRetained':True,'futureSurvivalFiltering':False},
            'populationProvenance':{'parent':{'path':str(parent),'sha256':sha(parent)},
                                    'heldout':{'path':final['legacy_path'],'sha256':sha(Path(final['legacy_path']))}},
            'splits':{'heldoutPopulationHashes':[sha(Path(final['legacy_path']))],'archiveAndPilotRowsInFinalTest':False,'validationFraction':.2,'randomTestFraction':0.,'testPopulationGames':80},
            'quality':{'variants':['history_marginal','causal_marginal','causal_autoregressive'],'width':128,'epochs':30,
                       'batchSize':128,'samplesPerContext':128,'learningRate':.0003}}
    plan_path = runs/'plan.json'; plan_path.write_text(json.dumps(plan))
    runtime = {'flags':AB_FLAGS,'packageC':False,'nativeSimulation':False,'DEFlags':{k:False for k in DE_FLAGS}}
    selection = {'schema':SELECTION_SCHEMA,'reviewed':True,'entropyMode':'legacy_path','checkpoint':final['legacy_path'],
        'checkpointSHA256':sha(Path(final['legacy_path'])),'source':str(source),'sourceManifest':str(manifest),
        'sourceManifestSHA256':sha(manifest),'assetsManifest':str(assets_path),'assetsManifestSHA256':sha(assets_path),
        'sourceBundleSHA256':hashlib.sha256(json.dumps({'code':inventory['sha256'],'assets':assets},sort_keys=True).encode()).hexdigest(),
        'image':IMAGE,'runtime':runtime,'runtimeSHA256':object_hash(runtime),'entropyCampaign':str(campaign),'evaluations':evidence,
        'forecastPlan':str(plan_path),'forecastPlanSHA256':sha(plan_path),'forecastPlanVersion':PLAN_SCHEMA,'forecastDatasetVersion':DATASET_SCHEMA}
    selection_path = runs/'selection.json'; selection_path.write_text(json.dumps(selection))
    return project, selection_path, selection, campaign


def test_reviewed_selection_validates_without_starting_jobs_or_choosing_entropy(tmp_path):
    project, path, selection, _ = fixture(tmp_path); commands = FakeCommands()
    selected, plan = validate_selection(path, project, run=commands)
    assert selected['entropyMode'] == 'legacy_path'  # no automatic exact selection
    assert plan['collection']['totalGames'] == 280 and commands.calls


@pytest.mark.parametrize('state', ['active','activating','deactivating','reloading','failed'])
def test_active_or_transitional_service_blocks_even_with_no_gpu_pid(tmp_path, state):
    project, path, _, _ = fixture(tmp_path)
    with pytest.raises(LaunchBlocked, match='service'):
        validate_selection(path, project, run=FakeCommands(state=state))


@pytest.mark.parametrize('failure', ['systemctl','nvidia-smi','docker'])
def test_query_failures_are_fail_closed(tmp_path, failure):
    project, path, _, _ = fixture(tmp_path)
    with pytest.raises(LaunchBlocked, match='query failed'):
        validate_selection(path, project, run=FakeCommands(failure=failure))


@pytest.mark.parametrize('change', ['unreviewed','runtime','parent','source','plan','manual_stop','failed_campaign','wrong_opponent','missing_completion'])
def test_semantic_provenance_and_completion_mismatches_block(tmp_path, change):
    project, path, selection, campaign = fixture(tmp_path)
    if change == 'unreviewed': selection['reviewed'] = False
    if change == 'runtime': selection['runtime']['packageC'] = True
    if change == 'parent': selection['checkpointSHA256'] = '0'*64
    if change == 'source': (Path(selection['source'])/'model.py').write_text('changed')
    if change == 'plan': selection['forecastPlanSHA256'] = '0'*64
    if change in ('manual_stop','failed_campaign'):
        state = json.loads((campaign/'campaign-state.json').read_text()); state['phase'] = 'train-exact_joint' if change=='manual_stop' else 'failed'
        (campaign/'campaign-state.json').write_text(json.dumps(state))
    if change == 'wrong_opponent':
        report_path = campaign/'evaluations/exact-vs-parent.json'; report = json.loads(report_path.read_text())
        report['opponentCheckpoint'] = selection['checkpoint']; report_path.write_text(json.dumps(report)); selection['evaluations']['exact-vs-parent'] = sha(report_path)
    if change == 'missing_completion': (campaign/'train-exact_joint.log').write_text('manually stopped, no finished marker')
    path.write_text(json.dumps(selection))
    with pytest.raises(LaunchBlocked): validate_selection(path, project, run=FakeCommands())


@pytest.mark.parametrize('resource', ['gpu','cpu_collector'])
def test_unrelated_resources_are_never_evicted(tmp_path, resource):
    project, path, _, _ = fixture(tmp_path)
    commands = FakeCommands(gpu='999' if resource=='gpu' else '', collector=resource=='cpu_collector')
    with pytest.raises(LaunchBlocked): validate_selection(path, project, run=commands)
