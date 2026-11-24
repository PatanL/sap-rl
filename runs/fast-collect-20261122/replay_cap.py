"""Upper throughput from replaying the collector's actual actions, with native legality and battles."""
import argparse,multiprocessing as mp,time
from common import *
from collector import allocate_games
from sapai_gym1.rl.end_freeze import step_with_end_freeze
from sapai_gym1.rl.distributed import _completed_match_result
clock=time.perf_counter
def one(w,g,sizes,offset,tape,warm,barrier,conn,noobs):
    if noobs:ParallelSelfPlayEnv.observations=lambda self:(None,None)
    envs,obs,owners,reset=allocate_games(w,g,1)
    events=np.zeros((len(sizes),5),dtype=np.int64);start=None
    for tick in range(len(sizes)):
        if tick==warm:
            barrier.wait(timeout=150);start=clock()
        idx=offset[tick]+int(sizes[tick,:w].sum());begin=idx
        for i,env in enumerate(envs):
            actions=[None,None]; masks=[0,0]
            for s,active in enumerate(env.active_players()):
                if active:
                    a=tape[idx];idx+=1
                    actions[s]=AutoregressiveAction(int(a[0]),int(a[1]),int(a[2]));masks[s]=int(a[3])
                    events[tick,1]+=int(s==0 or owners[i]==0)
            step=step_with_end_freeze(env,*actions,mask_ids=masks).result
            if step.battle_resolved:events[tick,3]+=1
            if step.terminated:
                events[tick,2]+=1;events[tick,4]+=(i+1)*(1+_completed_match_result(env.core));reset(i)
        events[tick,0]=idx-begin
        assert idx-begin==sizes[tick,w]
    end=clock();conn.send((start,end,events));conn.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--noobs',action='store_true');args=p.parse_args()
    with np.load(args.name+'-tape.npz',allow_pickle=False) as a:
        actions=a['actions'];sizes=a['sizes'];warm=int(a['warm']);w=int(a['workers']);g=int(a['games']);expected=a['events']
    off=np.r_[0,np.cumsum(sizes.sum(1))[:-1]];ctx=mp.get_context('fork');barrier=ctx.Barrier(w)
    pipes=[];workers=[]
    for i in range(w):
        recv,send=ctx.Pipe(False);pr=ctx.Process(target=one,args=(i,g,sizes,off,actions,warm,barrier,send,args.noobs))
        pr.start();send.close();workers.append(pr);pipes.append(recv)
    try:
        rows=[c.recv() for c in pipes]
    finally:
        for pr in workers:
            pr.join(2)
            if pr.is_alive():pr.terminate();pr.join()
    arrays=sum(r[2] for r in rows)
    if not np.array_equal(arrays,expected):
        ix=np.argwhere(arrays!=expected)[0];raise ValueError((ix,arrays[ix[0]],expected[ix[0]]))
    a=arrays[warm:].sum(0);secs=max(r[1] for r in rows)-min(r[0] for r in rows)
    out=dict(seconds=secs,workers=w,gamesPerActor=g,observationEncoding=not args.noobs,seat=int(a[0]),eligible=int(a[1]),matches=int(a[2]),battles=int(a[3]),
             seat_s=a[0]/secs,eligible_s=a[1]/secs,matches_s=a[2]/secs,battles_s=a[3]/secs,verifiedTicks=len(sizes),warm=warm)
    save(args.name+('-core-cap' if args.noobs else '-env-cap')+'.json',out);print(out)
if __name__=='__main__':main()
