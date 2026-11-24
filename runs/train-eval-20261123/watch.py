"""Local scoring ledger. Checkpoint inputs are read-only; its champion is an evaluation reference."""
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time,random
from pathlib import Path
from config import ENGINE,PAYLOAD,REV,LIVE,PARENT,PARENT_HASH,PANEL
JOB=Path(__file__).resolve().parent
ROOT=JOB.parent.parent
PROTOCOL='paired-greedy-v31-v1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(path.read_text())
def atomic(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');temp.replace(path)
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ledger(path,run):
    if path.exists():
        d=read(path)
        if (d['run'],d['protocol'],d['engine'])!=(str(run),PROTOCOL,PAYLOAD):raise RuntimeError('Different run/protocol already owns progression.json')
        return d
    return dict(run=str(run),protocol=PROTOCOL,engine=PAYLOAD,revision=REV,champion=PARENT,
                champion_sha256=PARENT_HASH,pair_counts=dict(screen=64,confirm=128,panel=32),baseline_panel={},checkpoints=[])
def scoring(a,b,mode,pairs,tag):
    ah=digest(ROOT/a);bh=digest(ROOT/b)
    file=JOB/'receipts'/f'{tag}.json'
    if not file.exists():
        with (JOB/'gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not file.exists():
                with (JOB/'scoring.log').open('a') as log:
                    subprocess.run([str(JOB/'run.sh'),'evaluate.py','/sap/'+a,'/sap/'+b,'receipts/'+file.name,'--mode',mode,'--pairs',str(pairs)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
    r=read(file)
    if (r['candidate']['sha256'],r['opponent']['sha256'],r['mode'],r['pairs'],r['engine'],r['protocol'])!=(ah,bh,mode,pairs,PAYLOAD,PROTOCOL):
        raise RuntimeError(f'Incompatible scoring receipt {file}')
    return {**{k:r[k] for k in ('score','ci95','pairs','games','wdl','seconds')},'receipt':str(file),'opponent':b,'opponent_sha256':bh,'candidate_update':r['candidate']['update']}
def panel(a,prefix):
    res={}
    for name,(b,bh) in PANEL.items():
        if not (ROOT/b).is_file():res[name]=dict(unavailable='checkpoint file absent',expected=b,sha256=bh);continue
        if digest(ROOT/b)!=bh:raise ValueError('Historical policy digest changed')
        res[name]=scoring(a,b,'panel',32,prefix+'-'+name)
    return res
def panel_changes(scores,baselines):
    for name,entry in scores.items():
        base=baselines.get(name,{})
        if 'receipt' not in entry or 'receipt' not in base:continue
        def pairs(record):
            points={}
            for g in read(Path(record['receipt']))['records']:points.setdefault(g['seed'],{})[g['seat']]=g['score']
            if not all(set(s)=={0,1} for s in points.values()):raise ValueError('Incomplete pair')
            return {k:(s[0]+s[1])/2 for k,s in points.items()}
        curr,prev=pairs(entry),pairs(base)
        if curr.keys()!=prev.keys():raise ValueError('Unpaired panel change')
        ds=[curr[k]-prev[k] for k in sorted(curr)];n=len(ds);rng=random.Random(20261123)
        means=sorted(sum(rng.choices(ds,k=n))/n for _ in range(20000))
        entry.update(delta_from_25772=sum(ds)/n,delta_ci95=[means[500],means[19499]])
    return scores
def observe(run,path):
    d=ledger(path,run)
    if not d['baseline_panel']:
        d['baseline_panel']=panel(PARENT,'baseline');atomic(path,d)
    known={x['checkpoint']:x for x in d['checkpoints']}
    for ck in sorted(run.glob('update-*.pt')):
        a=str(ck.relative_to(ROOT))
        if a in known:continue
        version=int(ck.stem.split('-')[-1]);ch=d['champion'];prefix=run.name+'-'+str(version)
        screen=scoring(a,ch,'screen',64,prefix+'-champ-'+d['champion_sha256'][:12])
        if screen['candidate_update']!=version:raise ValueError('Serialized update disagrees with filename')
        base=screen if ch==PARENT else scoring(a,PARENT,'screen',64,prefix+'-parent')
        confirmation=None
        if screen['ci95'][0]>.5 or screen['ci95'][1]<.5:
            confirmation=scoring(a,ch,'confirm',128,prefix+'-confirm-'+d['champion_sha256'][:12])
            used=set()
            for prev in d['checkpoints']:
                if prev['confirmation']:used.update(x['seed'] for x in read(Path(prev['confirmation']['receipt']))['records'])
            if used.intersection(x['seed'] for x in read(Path(confirmation['receipt']))['records']):raise RuntimeError('Reused confirmation worlds')
        promoted=screen['ci95'][0]>.5 and confirmation is not None and confirmation['ci95'][0]>.5
        row=dict(version=version,checkpoint=a,sha256=digest(ck),finished_utc=now(),against_champion=screen,against_25772=base,confirmation=confirmation,
                 panel=panel_changes(panel(a,prefix+'-panel'),d['baseline_panel']),confirmed_champion=promoted,
                 confirmed_regression=screen['ci95'][1]<.5 and confirmation is not None and confirmation['ci95'][1]<.5)
        d['checkpoints'].append(row)
        if promoted:d['champion']=a;d['champion_sha256']=row['sha256']
        atomic(path,d);print(now(),'scored',version,base['score'],base['ci95'],flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=LIVE);p.add_argument('--once',action='store_true');p.add_argument('--poll',type=int,default=15);p.add_argument('--progression',type=Path)
    args=p.parse_args();run=(ROOT/args.run).resolve();path=args.progression or (run.parent/'progression.json')
    if not run.is_dir():raise FileNotFoundError(run)
    lock=(JOB/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=False
    def finish(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,finish);signal.signal(signal.SIGINT,finish)
    (JOB/'watch.pid').write_text(str(os.getpid()))
    try:
        while not stop:
            try:observe(run,path)
            except Exception as exc:
                print(now(),repr(exc),flush=True)
                if args.once:raise
            if args.once:break
            for _ in range(args.poll):
                if stop:break
                time.sleep(1)
    finally:(JOB/'watch.pid').unlink(missing_ok=True)
if __name__=='__main__':main()
