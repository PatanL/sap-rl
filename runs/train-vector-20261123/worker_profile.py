"""Optional worker-side timers. Consecutive labeled spans are disjoint, including all nested calls."""
from functools import wraps
from contextlib import contextmanager
from collections import Counter
import json,os,time,traceback
import numpy as np
import torch
import frozen_wide as wide
from runtime import selfplay_env
from sapai.player import Player
from sapai_gym1.rl.entity_observation import EntityObservationEncoder
clock=time.perf_counter
class Timeline:
    def __init__(self):
        self.kind='other';self.start=clock();self.spans=[];self.totals=Counter()
    def switch(self,kind):
        now=clock();old=self.kind
        self.spans.append((self.start,now,old));self.totals[old]+=now-self.start
        self.start=now;self.kind=kind
    def wrap(self,obj,name,kind):
        old=getattr(obj,name)
        @wraps(old)
        def call(*a,**kw):
            prev=self.kind
            if prev==kind:return old(*a,**kw)
            self.switch(kind)
            try:return old(*a,**kw)
            finally:self.switch(prev)
        setattr(obj,name,call)
class Pipe:
    def __init__(self,conn,timeline,counter):
        self.conn=conn;self.tm=timeline;self.first=True;self.count=counter
    def send(self,msg):
        if msg[0]=='error':self.conn.send(msg);return
        self.tm.switch('other')
        self.conn.send((*msg, [] if self.first else self.tm.spans, {} if self.first else dict(self.tm.totals),dict(self.count)))
        self.first=False;self.tm.spans=[];self.tm.totals.clear();self.count.clear();self.tm.start=clock()
    def recv(self):
        self.tm.switch('actor_wait')
        try:return self.conn.recv()
        finally:self.tm.switch('other')
    def close(self):self.conn.close()
def profiled_worker(w,games,width,page,pipe,seed,start_version,profiling,backend,audit_dir):
    bridge=None;audit=None
    if backend=='rust':
        from rust_g import Bridge
        audit=open(audit_dir/f'battles-{w}.jsonl','w',buffering=1) if audit_dir else None
        bridge=Bridge(audit)
        selfplay_env.Battle=bridge.battle_class()
    def execute(conn):
        try:return wide.worker(w,games,width,page,conn,seed,start_version)
        finally:
            if audit is not None:
                audit.close()
                (audit_dir/f'battles-{w}-counts.json').write_text(json.dumps(dict(rust=bridge.calls,python_fallback=bridge.fallbacks)))
    if not profiling:return execute(pipe)
    tm=Timeline();counts=Counter()
    tm.wrap(wide,'step_with_end_freeze','environment')
    tm.wrap(selfplay_env.ParallelSelfPlayEnv,'reset','environment')
    for s in ('start_turn','end_turn','buy_pet','sell','buy_food','roll','freeze','unfreeze','combine','buy_combine','swap'):
        tm.wrap(Player,s,'shop_api')
    tm.wrap(EntityObservationEncoder,'encode','encoding')
    base=selfplay_env.Battle
    class Battle(base):
        def battle(self):
            counts['battles']+=1
            return super().battle()
    tm.wrap(Battle,'__init__','battle');tm.wrap(Battle,'battle','battle')
    selfplay_env.Battle=Battle
    return execute(Pipe(pipe,tm,counts))

@contextmanager
def actors(workers,games,width,seed,start_version,profiling=False,backend="python",audit_dir=None):
    if not profiling and backend=="python":
        with wide.actors(workers,games,width,seed,start_version) as a:yield a
        return
    ctx=wide.mp.get_context('spawn');example=selfplay_env.ParallelSelfPlayEnv().reset(seed=42,paired_tape=True)[0]
    pages=[];pipes=[];procs=[]
    try:
        for i in range(workers):
            page=wide.Page(games,example,width);parent,child=ctx.Pipe()
            pr=ctx.Process(target=profiled_worker,args=(i,games,width,page,child,seed,start_version,profiling,backend,audit_dir))
            pr.start();child.close();procs.append(pr);pipes.append(parent);pages.append(page)
        yield pages,pipes
    finally:
        for p in pipes:
            try:p.send(None)
            except (BrokenPipeError,EOFError):pass
        for pr in procs:
            pr.join(2)
            if pr.is_alive():pr.terminate();pr.join()
        for p in pipes:p.close()
        for page in pages:page.close()
