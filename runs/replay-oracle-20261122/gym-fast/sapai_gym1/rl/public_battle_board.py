"""Search-only effective opponent stats and separately retained PUBLIC flags.

No live hidden permanent/temporary decomposition, counters or RNG is copied.
This is an approximate battle-effective hypothesis, never a hidden shop state.
"""
from sapai import Pet,Team
import copy

FLAG_KEY='publicTemporaryFlags'

def flags(team):
    if team is None:return None
    saved=getattr(team,'_public_temporary_flags',None)
    if saved is not None:return tuple(tuple(map(bool,p)) for p in saved)
    return tuple((False,False) if s.empty else (bool(s.pet._until_end_of_battle_attack_buff>0),
        bool(s.pet._until_end_of_battle_health_buff>0)) for s in team)

def public_records(team):
    if team is None:return None
    ff=flags(team)
    return tuple(None if s.empty else {'species':s.pet.name,'attack':int(s.pet.attack),'health':int(s.pet.health),
        'level':int(s.pet.level),'experience':int(s.pet.experience),'perk':s.pet.status,
        'temporaryAttack':ff[i][0],'temporaryHealth':ff[i][1]} for i,s in enumerate(team))

def from_public_records(board):
    if board is None:return None
    if len(board)!=5:raise ValueError('Five public board slots required')
    team=Team();ff=[]
    for i,p in enumerate(board):
        if p is None:ff.append((False,False));continue
        pet=Pet(p['species']);pet._attack=int(p['attack']);pet._health=int(p['health'])
        pet.level=int(p['level']);pet.experience=int(p['experience']);pet.sell_value=pet.level;pet.status=p['perk']
        pet._until_end_of_battle_attack_buff=pet._until_end_of_battle_health_buff=0
        team[i]=pet;ff.append((bool(p['temporaryAttack']),bool(p['temporaryHealth'])))
    team._public_temporary_flags=tuple(ff)
    return prepare_battle_copies(team)

def prepare_battle_copies(team):
    """Derive Parrot's submitted-board ability under the pinned rules.

    This mirrors TransferAbility's native-source-at-Parrot-level rule, not
    a full End Turn replay (which would apply already-included scalers twice).
    No hidden override from a live opposing Player is imported.
    """
    ahead=None
    for slot in team:
        if slot.empty:continue
        pet=slot.pet
        if pet.name=='pet-parrot':
            pet.override_ability=False;pet.override_ability_dict={}
            if ahead is not None:
                ability=copy.deepcopy(ahead.fd.get(f'level{pet.level}Ability',ahead.ability))
                ability['_copiedFrom']=ahead.name;pet.set_ability(ability)
        ahead=pet
    return team

def sanitize(team):return from_public_records(public_records(team))

def copy_team(team):
    if team is None:return None
    clone=Team.from_state(team.state)
    if hasattr(team,'_public_temporary_flags'):clone._public_temporary_flags=flags(team)
    return clone

def to_state(team):
    value=team.state
    if hasattr(team,'_public_temporary_flags'):value[FLAG_KEY]=flags(team)
    return value

def from_state(value):
    team=Team.from_state(value)
    if FLAG_KEY in value:team._public_temporary_flags=tuple(tuple(p) for p in value[FLAG_KEY])
    return team
