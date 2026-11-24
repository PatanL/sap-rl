"""Run inside a validation container; preserve the gate and add first-tensor detail."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,'/workspace/sapai-gym/scripts')
import benchmark_abc_fixed_v2 as benchmark

original=benchmark.assert_state


def detailed(first,second,opt_first,opt_second,*,atol,rtol):
    for (name,a),(_,b) in zip(first.named_parameters(),second.named_parameters()):
        limit=min(atol,1e-6)+min(rtol,1e-5)*b.detach().abs()
        difference=(a.detach()-b.detach()).abs()
        failed=difference>limit
        if failed.any():
            indices=failed.nonzero().cpu().tolist()
            records=[]
            for index in indices[:16]:
                key=tuple(index)
                row={'index':index,'old':float(a[key]),'new':float(b[key]),
                     'difference':float(difference[key]),'allowed':float(limit[key]),
                     'oldGradient':None if a.grad is None else float(a.grad[key]),
                     'newGradient':None if b.grad is None else float(b.grad[key])}
                if name.endswith('in_proj_bias') and len(index)==1:
                    row['qkvBlock']=('q','k','v')[index[0]//(a.numel()//3)]
                for label,opt,param in [('old',opt_first,a),('new',opt_second,b)]:
                    row[label+'Adam']={k:float(v if v.ndim==0 else v[key]) for k,v in opt.state[param].items() if torch.is_tensor(v)}
                records.append(row)
            report={'firstDivergentQuantity':'parameter','tensor':name,'shape':list(a.shape),
                    'violations':len(indices),'maxAbsoluteDifference':float(difference.max()),
                    'records':records,'tolerancesUnchanged':True}
            out=Path(sys.argv[sys.argv.index('--output')+1]).with_suffix('.detail.json')
            out.write_text(json.dumps(report,indent=2)+'\n')
            break
    return original(first,second,opt_first,opt_second,atol=atol,rtol=rtol)


benchmark.assert_state=detailed
benchmark.main()
