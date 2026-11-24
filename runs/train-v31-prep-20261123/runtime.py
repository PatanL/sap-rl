"""Pin public observation, Python battle implementation, and checkpoint identity."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(os.environ['SAP_ENGINE_ROOT']).resolve()
sys.path[:0]=[str(ROOT/'sapai'),str(ROOT/'sapai-gym')]
from sapai.turtle_pack import SIMULATOR_REVISION,SOURCE_OF_TRUTH
from sapai.battle import Battle
from sapai_gym1.rl import selfplay_env
from sapai_gym1.rl.observation_contract import OBSERVATION_REVISION
from pack_engine import manifest,digest,sha
PARENT_SHA='0669109e9b6ee2d4f0309a3fec83039fc8a7d74f74e3c9f629d2f7026e515d49'

def verify_engine():
    info=json.loads((ROOT/'MANIFEST.json').read_text())
    files=manifest(ROOT);files.pop('MANIFEST.json')
    if files!=info['files'] or digest(files)!=info['payload_sha256']:raise RuntimeError('Engine payload differs from its manifest')
    if info['simulator_revision']!=SIMULATOR_REVISION or not SIMULATOR_REVISION.startswith('turtle-v31-replay-'):raise RuntimeError('Revision mismatch')
    if not Path(sys.modules['sapai'].__file__).resolve().is_relative_to(ROOT/'sapai') or not Path(sys.modules['sapai_gym1'].__file__).resolve().is_relative_to(ROOT/'sapai-gym'):raise RuntimeError('Unpinned imports')
    if selfplay_env.Battle is not Battle or info['native_rust_scorer'] or info['battle_backend']!='sapai.battle.Battle':raise RuntimeError('Unexpected battle backend')
    if any(x in sys.modules for x in ('native_scorer','native_battle','fast_public_planner')):raise RuntimeError('External native planner loaded')
    if 'libsap_battle.so' in Path('/proc/self/maps').read_text():raise RuntimeError('v30 Rust scorer is mapped')
    if os.environ.get('SAP_BATTLE_LIBRARY') and Path(os.environ['SAP_BATTLE_LIBRARY']).exists():raise RuntimeError('External scorer library configured')
    return info
