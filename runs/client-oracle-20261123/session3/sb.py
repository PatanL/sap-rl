"""usage: sb.py <glob> [k] -> for each capture: the distinct settled boards (first k, default 5) and the terminal board"""
import json,sys,glob,subprocess
k=int(sys.argv[2]) if len(sys.argv)>2 else 5
for d in sorted(glob.glob('/home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/session2/captures/'+sys.argv[1])):
    import os
    if not os.path.exists(d+'/run.json'):continue
    if not os.path.exists(d+'/timeline.json'):subprocess.run(['/home/dzliu/pzliu/.venv/bin/python','/home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/timeline.py',d],capture_output=True,cwd='/home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123')
    rows=json.load(open(d+'/timeline.json'));u=[]
    for r in rows:
        if r['n']<2 or not (r['P'] or r['O']):continue
        if any('?' in t[2] or '-' in t[2] or t[2].endswith('/0') or t[0]<0 for t in r['P']+r['O']):continue
        s=' | '.join(f"s{t[0]} {t[2]}" for t in r['P'])+' || '+' | '.join(f"s{t[0]} {t[2]}" for t in r['O'])
        if not u or u[-1]!=s:u.append(s)
    print(d.split('/')[-1][3:]);[print('    ',x) for x in u[:k]]
    if len(u)>k:print('     ... final:',u[-1])
