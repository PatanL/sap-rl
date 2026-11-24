"""Zombie-fly spawn order per capture: first movie frame at which each player slot shows the zombie stats."""
import json,sys,glob
for pat in sys.argv[2:]:
    for d in sorted(glob.glob('/home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/session2/captures/'+pat)):
        try:rows=json.load(open(d+'/timeline.json'))
        except Exception:continue
        z=sys.argv[1];first={};started=False
        for r in rows:
            if any(t[2].endswith('/0') or '/-' in t[2] for t in r['P']):started=True
            if not started:continue
            for t in r['P']:
                if t[2]==z and t[0]>=0 and t[0] not in first:first[t[0]]=r['f']
        last=rows[-1];fin=' | '.join(f"s{t[0]} {t[2]}" for t in last['P'])
        order=sorted(first,key=lambda s:first[s])
        print(d.split('/')[-1].ljust(40),'spawn order',[(f's{s}',first[s]) for s in order],' final:',fin)
