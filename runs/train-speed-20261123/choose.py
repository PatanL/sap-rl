"""Pick by measured speed with allocation headroom for the actor and frozen pool."""
import json,sys
from pathlib import Path
def best(path):
    sweep=json.loads(Path(path).read_text())
    ok=[(v['rate'],int(k)) for k,v in sweep.items() if k!='ref' and not v.get('oom') and v['peak']<=7*1024**3]
    if not ok:raise RuntimeError('No benchmark size leaves a GiB for collector and allocator headroom')
    # Within two percent, prefer the lighter packing.
    top=max(x for x,k in ok);return min(k for x,k in ok if x>=.98*top)
if __name__=='__main__':print(best(sys.argv[1]))
