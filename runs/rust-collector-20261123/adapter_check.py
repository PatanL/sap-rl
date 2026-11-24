import json
from pathlib import Path
import numpy as np
from rust_adapter import RustVector

def main():
    env=RustVector(3,1);o=env.output
    try:
        env.reset(np.arange(3),[4,6,8],o)
        prev={k:v[[0,2]].copy() for k,v in o['obs'].items()}
        env.reset([1],[22],o)
        for k,v in prev.items():np.testing.assert_array_equal(v,o['obs'][k][[0,2]])
        assert o['active'].dtype==np.bool_ and o['terminated'].dtype==np.bool_ and o['rewards'].dtype==np.float32
        assert all(np.shares_memory(v,env.native.obs[k]) for k,v in o['obs'].items())
        assert np.shares_memory(o['active'],env.native.active)
        try:env.step(np.full((3,2,3),-50,'i4'),np.zeros((3,2),'i4'),o)
        except RuntimeError as err:
            assert 'status -1' in str(err);illegal_message=str(err)
        else:raise AssertionError('Illegal Rust input silently accepted')
        Path('results/adapter-check.json').write_text(json.dumps(dict(partial_reset=True,zero_copy_public_obs=True,illegal_message=illegal_message)))
    finally:env.close()
if __name__=='__main__':main()
