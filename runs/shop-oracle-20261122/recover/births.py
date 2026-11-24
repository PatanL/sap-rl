from search import *
if __name__=='__main__':
 os.environ['BIRTH_IDS']='1'
 cap=int(sys.argv[1]);target=sys.argv[2]
 js=[(r,n,o,cap,None) for r,n,o in jobs() if any(a['t']==8 and a['spell']['name']=='Sleeping Pill' for a in r['actions'][str(n)])]
 if len(sys.argv)>3:
    subset={tuple(x['key']) for x in map(json.loads,open(sys.argv[3])) if x.get('n',0)==0}
    js=[x for x in js if tuple(x[2]['key']) in subset]
 with mp.Pool(6,initializer=install) as pool,open(P/(target+'.jsonl'),'w',buffering=1) as fp:
    for x in pool.imap_unordered(run_case,js):fp.write(json.dumps(x,separators=(',',':'))+'\n')
