"""usage: startrow.py <glob> -> last settled board before the first exchange, per capture"""
import json,sys,glob,os,subprocess
B='/home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/'
for d in sorted(glob.glob(B+'session2/captures/'+sys.argv[1])):
    if not os.path.exists(d+'/run.json'):continue
    if not os.path.exists(d+'/timeline.json'):subprocess.run(['/home/dzliu/pzliu/.venv/bin/python',B+'timeline.py',d],capture_output=True,cwd=B)
    rows=json.load(open(d+'/timeline.json'));i=next((i for i,r in enumerate(rows) if r['dmg']),len(rows))
    c=[r for r in rows[:i] if r['n']>=2 and r['O']];r=c[-1]
    print(d.split('/')[-1][3:].ljust(44),'P',' | '.join(t[2] for t in r['P']),'|| O',' | '.join(t[2] for t in r['O']),'  ',r['frame'])
