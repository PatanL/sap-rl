"""Serialize the proposed opt-in post-END event. This module does not infer missing metadata."""
import ctypes
import json
from pathlib import Path
import numpy as np
from rust_adapter import PIN

class BoardCodec:
    def __init__(self):
        lib=ctypes.CDLL(str(PIN/'libsap_engine.so'))
        for key in ('sap_engine_init','sap_kind','sap_status'):
            func=getattr(lib,key);func.argtypes=[ctypes.c_char_p];func.restype=ctypes.c_int32
        if lib.sap_engine_init(str(PIN/'data').encode())!=0:raise RuntimeError('Native kind table unavailable')
        names=json.loads((PIN/'data/turtle_v31_env.json').read_text())
        self.species={};self.perks={}
        for key in names['itemIds']:
            native=lib.sap_kind(key.encode())
            if native>=0:self.species[native]=key
        for key in names['statusIds']:
            native=lib.sap_status(key.encode())
            if native>=0:self.perks[native]=key

    def team(self,cells):
        pets=[]
        for species,attack,health,xp,status,count,copied in cells:
            if species==-1:continue
            level=1+int(xp>=2)+int(xp>=5)
            pet=dict(id=self.species[species],attack=int(attack),health=int(health),level=level,
                     experience=(0,2,5)[level-1],status=self.perks[status],triggersConsumed=int(count))
            if pet['id']=='pet-parrot':
                pet['parrotCopies']=None if copied in (-2,-1) else self.species[copied]
            pets.append(pet)
        return pets

def test():
    codec=BoardCodec();kind={v:k for k,v in codec.species.items()};perk={v:k for k,v in codec.perks.items()}
    path=Path('/sap/runs/train-vector-20261123/receipts/board-0669109e9b6ee2d4-d1a450928296-20261123/seats-battles.jsonl')
    count=0
    for line in path.open():
        record=json.loads(line)
        for side in record['teams']:
            cells=np.full((5,7),-1,'i4')
            for i,p in enumerate(side):
                copied=-2 if 'parrotCopies' not in p else (-1 if p['parrotCopies'] is None else kind[p['parrotCopies']])
                cells[i]=[kind[p['id']],p['attack'],p['health'],p['experience'],perk[p['status']],p['triggersConsumed'],copied]
            assert codec.team(cells)==side
            count+=1
    Path('results/codec.json').write_text(json.dumps(dict(archived_teams_round_tripped=count,
        provenance='Historical boards -> proposed ABI representation -> historical board JSON. No native event API assumed.')))
if __name__=='__main__':test()
