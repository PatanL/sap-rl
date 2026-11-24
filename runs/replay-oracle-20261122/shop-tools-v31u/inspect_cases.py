import json,argparse
from case import load,details,compact
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('prefix');p.add_argument('turn',type=int)
 a=p.parse_args()
 for line in open(a.result):
  r=json.loads(line);m,s,n=r['key']
  if n!=a.turn or not m.startswith(a.prefix):continue
  row,n=load(r['key']);d=details(row,n,r.get('seed'));compact(d)
