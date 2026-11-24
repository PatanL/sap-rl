"""Replay a saved concrete candidate by match, perspective, turn, and zero-based witness index."""
import argparse,gzip
from infer import *
if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('match');cli.add_argument('seat');cli.add_argument('turn',type=int);cli.add_argument('--index',type=int,default=0)
    args=cli.parse_args();key=[args.match,args.seat,args.turn]
    rec=next(x for x in map(json.loads,gzip.open(P/'assignments.jsonl.gz','rt')) if x['key']==key)
    values,si,path=rec['witnesses'][args.index]
    row,n,_=next((r,n,o) for r,n,o in jobs() if o['key']==key)
    p=prep(row,n);install();hit,why,out,ctx=trial(p,values[:len(p['b'])],si,path)
    assert hit and [ctx.assigned[i] for i in p['e']]==values[len(p['b']):]
    print(json.dumps(dict(key=key,identities=dict(zip(rec['buy']+rec['endogenous'],values)),seedIndex=si,
        seed=(p['seed']+si*2654435761)%2**32,conditional_choices=path,state=out['state']),indent=2))
