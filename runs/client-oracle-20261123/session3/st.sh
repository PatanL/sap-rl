#!/bin/bash
# usage: st.sh <prefix> [minrun] : settled readings only (n>=minrun, default 2) for finished captures matching prefix
cd /home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/
for d in session2/captures/$1*; do
  [ -f $d/run.json ] || continue
  [ -f $d/timeline.json ] || taskset -c 14 /home/dzliu/pzliu/.venv/bin/python timeline.py $d > /dev/null
  echo "=== $(basename $d)"
  /home/dzliu/pzliu/.venv/bin/python - "$d" "${2:-2}" <<'P'
import json,sys
rows=json.load(open(sys.argv[1]+'/timeline.json'));m=int(sys.argv[2]);prev=None
for i,r in enumerate(rows):
    if not (r['n']>=m or r['dmg'] or i==len(rows)-1):continue
    def s(side):return ' | '.join((f"s{t[0]} " if t[0]>=0 else '~')+t[2] for t in r[side])
    txt=f"P: {s('P')} || O: {s('O')}"
    if txt==prev or (not r['P'] and not r['O']):continue
    prev=txt;print(f"f={r['f']:04d} n={r['n']:2d} {'DMG' if r['dmg'] else '   '} {txt}")
P
done
