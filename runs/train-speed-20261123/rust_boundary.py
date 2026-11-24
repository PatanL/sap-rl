"""Outcome-only ABI for ordinary paired shop environments. Unsupported hidden state falls back explicitly."""
import ctypes as C
import json
from functools import lru_cache
from pathlib import Path
from runtime import Battle
from sapai.data import data
class Bridge:
    def __init__(self,audit=None):
        self.audit=audit
        path=Path(__file__).resolve().parent/'rust-snapshot'
        self.lib=C.CDLL(str(path/'libsap_engine.so'))
        for name in ('sap_engine_init','sap_status','sap_kind'):
            func=getattr(self.lib,name);func.argtypes=[C.c_char_p];func.restype=C.c_int32
        ptr=C.POINTER(C.c_int32)
        self.lib.sap_battle.argtypes=[ptr,C.c_int32,ptr,C.c_int32,C.c_uint32];self.lib.sap_battle.restype=C.c_int32
        if self.lib.sap_engine_init(str(path/'data').encode())!=0:raise RuntimeError('Incompatible Rust rules')
        self.boards=[(C.c_int32*35)(),(C.c_int32*35)()];self.calls=self.fallbacks=0
    @lru_cache(None)
    def kind(self,n):return self.lib.sap_kind(n.encode())
    @lru_cache(None)
    def status(self,n):return self.lib.sap_status(n.encode())
    def encode(self,team,buf):
        j=0
        for slot in team:
            if slot.empty:continue
            p=slot.pet;kind=self.kind(p.name);status=self.status(p.status);copy=-1
            if kind<0 or status<0 or p.friends_hurt_counter or p._hurt or p.swallowed_pets:return None
            if p.override_ability:
                copied=p.override_ability_dict.get('_copiedFrom')
                if p.name!='pet-parrot' or not copied:return None
                expected=data['pets'].get(copied,{}).get(f'level{p.level}Ability')
                if expected is None or {k:v for k,v in p.override_ability_dict.items() if k!='_copiedFrom'}!=expected:return None
                copy=self.kind(copied)
                if copy<0:return None
            buf[j:j+7]=(kind,int(p.attack),int(p.health),int(p.total_experience),status,int(p.ability_counter),copy);j+=7
        return j//7
    def battle_class(self):
        api=self
        class RustBattle:
            def __init__(self,t0,t1,seed=None,fast_training=False):
                n=api.encode(t0,api.boards[0]);m=api.encode(t1,api.boards[1])
                if n is None or m is None or seed is None or not fast_training:
                    api.fallbacks+=1;self.value=Battle(t0,t1,seed=seed,fast_training=fast_training).battle()
                else:
                    value=api.lib.sap_battle(api.boards[0],n,api.boards[1],m,int(seed))
                    if value not in (0,1,2):raise RuntimeError(f'Rust battle rejected input: {value}')
                    self.value=value;api.calls+=1
                    if api.audit is not None:
                        def packed(t,b):
                            specs=[];j=0
                            for pos,slot in enumerate(t):
                                if slot.empty:continue
                                p=slot.pet
                                specs.append([pos,p.name,int(p._attack),int(p._health),int(p._until_end_of_battle_attack_buff),int(p._until_end_of_battle_health_buff),
                                              int(p.total_experience),p.status,int(p.ability_counter),p.ability.get('_copiedFrom') if p.override_ability else None])
                                j+=7
                            return specs
                        api.audit.write(json.dumps([int(seed),packed(t0,api.boards[0]),packed(t1,api.boards[1]),value],separators=(',',':'))+'\n')
            def battle(self):return self.value
        return RustBattle
