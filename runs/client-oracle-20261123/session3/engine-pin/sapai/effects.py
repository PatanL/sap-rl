import copy
import sys, inspect
import itertools
import numpy as np
import sapai
from sapai import status as sap_status

from sapai.data import data
from sapai.tiers import pet_tier_lookup, pet_tier_lookup_std
from sapai.foods import Food


"""
This module implements all effects in the game. API description is as follows:

    Arguments
    ---------
    apet: Pet
        Activating Pet. Must include the attached shop or player when necessary.
    apet_idx: list
        List of two indices for the [team_list_idx, team_idx] from which the 
        activating pet was called from
    teams: list
        List of Team objects. Either one or two teams. 
    te: Pet
        Triggering pet. If a pet has trigger the action. For example, if a horse
        is on the team and then an animal is summoned. The summoned animal would
        be the triggering pet. 
    te_idx: list
        List of two indices for the [team_list_idx, team_idx] from which the 
        triggering entity exists. This is only used for SummonPet in the case
        of a Fly summoning in the position of the fainted pet, which is the 
        triggering entity. In all other cases, te_idx should not be used as it
        is unnecessary. 
    fixed_targets: list of pets
        If a fixed target is desired for the effect. For common use, this is not
        required. It is useful in the case that certain outcomees are being 
        tested. It is also useful in the case that all outcomes due to 
        randomness are exactly exactly considered rather than having to rely on bootstrapped probabilities. This leads to significantly improved 
        efficiency for training by database purposes. 

    Returns
    -------
    targets: list
        List of pets that have been targeted by the effect of the ability
    possible: list of lists
        List of lists of all possible targets that could also be targeted. 
        If there is an element of randomness in the outcome, all possible 
        targets, and potentially all possible combinations of targets, is also
        returned. 

"""


def get_effect_function(effect_kind):
    if isinstance(effect_kind, sapai.Pet):
        effect_kind = effect_kind.ability["effect"]["kind"]
    elif isinstance(effect_kind, sapai.TeamSlot):
        effect_kind = effect_kind.pet.ability["effect"]["kind"]
    elif type(effect_kind) == str:
        pass
    else:
        raise Exception(f"Unrecognized input {effect_kind}")
    if effect_kind not in func_dict:
        raise Exception(
            f"Input effect_kind {effect_kind} not found in {list(func_dict.keys())}"
        )
    return func_dict[effect_kind]


def get_pet(pet_idx, teams, fainted_pet=None, te=None):
    """Helper function with error catching"""
    team_idx = pet_idx[0]
    team_pet_idx = pet_idx[1]
    if len(teams) > 2:
        raise Exception("Cannot input more than 2 teams")
    if team_idx >= len(teams):
        raise Exception("Team idx greater than provided number of teams")
    if team_pet_idx >= 5:
        raise Exception("Team pet idx greater than 5")
    if fainted_pet is None:
        pet = teams[team_idx][team_pet_idx].pet
    else:
        pet = fainted_pet
    if te is not None:
        pet = te
    return pet


def get_teams(pet_idx, teams):
    if len(teams) == 1:
        return teams[0], []
    if pet_idx[0] == 0:
        fteam = teams[0]
        oteam = teams[1]
    elif pet_idx[0] == 1:
        fteam = teams[1]
        oteam = teams[0]
    else:
        raise Exception("That's impossible")
    return fteam, oteam


def get_target(
    apet,
    apet_idx,
    teams,
    te=None,
    fixed_targets=None,
    get_from=False,
    test_kind="",
):
    """
    Returns the targets for a given effect. Targets are returned as a list of
    pets.

    Arguments
    ---------
    apet: Pet or Food
        Data to use for ability/effect/etc
    apet_idx: list
        List of two indices that provide the team index and the pet index
        that has requested to obtain target pets
    teams: list
        List of two teams
    fainted_pet: Pet
        If the target has been requested due to fainting, the fainted pet should
        be input.
    get_from: bool
        For correting some database inconsistencies
    te: Pet
        Triggering entity
    """
    fixed_targets = fixed_targets or []

    if isinstance(apet, sapai.Pet):
        effect = apet.ability["effect"]
    elif isinstance(apet, Food):
        effect = apet.effect

    if len(teams) == 1:
        teams = [teams[0], []]

    ### Logic switch because data-dictionary is not consistent
    if "target" not in effect:
        if "to" in effect:
            target = effect["to"]
        else:
            print(apet, apet_idx, teams, te, fixed_targets, get_from)
            raise Exception("Target not found")
    else:
        target = effect["target"]

    if get_from:
        if "from" not in effect:
            raise Exception("from not found in effect")
        else:
            target = effect["from"]

    if type(target) != dict:
        raise Exception("This should not be possible")
    kind = target["kind"]
    if "n" in target:
        n = target["n"]
    else:
        n = 1

    if len(test_kind) != 0:
        kind = test_kind

    if apet_idx[0] == 0:
        fteam = teams[0]
        oteam = teams[1]
    elif apet_idx[0] == 1:
        fteam = teams[1]
        oteam = teams[0]
    else:
        raise Exception("That's impossible")

    ### Get possible indices for each team
    fidx = []
    for iter_idx, temp_slot in enumerate(fteam):
        if not temp_slot.empty:
            ### Skiped if health is less than 0
            if temp_slot.pet.health > 0:
                fidx.append(iter_idx)
    oidx = []
    for iter_idx, temp_slot in enumerate(oteam):
        if not temp_slot.empty:
            ### Skiped if health is less than 0
            if temp_slot.pet.health > 0:
                oidx.append(iter_idx)

    if kind == "AdjacentAnimals":
        # Turtle Badger targets the nearest *living* pet on each side. In the
        # shop there may be physical gaps, and in battle another queued death
        # may occupy the neighboring slot until synchronization. Both must be
        # skipped. The concatenated orientation makes the opposing front pet
        # the next pet ahead when no friendly pet remains ahead.
        all_pets = []
        fpet_slot_idx = []
        ### First add opponent backward
        for temp_slot in oteam[::-1]:
            all_pets.append(temp_slot)
        ### Then friendly
        for temp_slot in fteam:
            fpet_slot_idx.append(len(all_pets))
            all_pets.append(temp_slot)
        apet_in_all = fpet_slot_idx[apet_idx[1]]
        left_slot = None
        for candidate_idx in range(apet_in_all - 1, -1, -1):
            candidate = all_pets[candidate_idx]
            if not candidate.empty and candidate.pet.health > 0:
                left_slot = candidate
                break

        right_slot = None
        for candidate_idx in range(apet_in_all + 1, len(all_pets)):
            candidate = all_pets[candidate_idx]
            if not candidate.empty and candidate.pet.health > 0:
                right_slot = candidate
                break

        ret_pets = []
        for temp_slot in [left_slot, right_slot]:
            if temp_slot is None:
                continue
            if temp_slot.empty or temp_slot.pet.health <= 0:
                continue
            else:
                ret_pets.append(temp_slot.pet)
        return ret_pets, [ret_pets]

    elif kind == "AdjacentFriends":
        all_pets = []
        fpet_slot_idx = []
        for temp_slot in fteam:
            fpet_slot_idx.append(len(all_pets))
            all_pets.append(temp_slot)
        apet_in_all = fpet_slot_idx[apet_idx[1]]
        if (apet_in_all - 1) >= 0:
            left_slot = all_pets[apet_in_all - 1]
        else:
            left_slot = None

        if (apet_in_all + 1) < len(all_pets):
            right_slot = all_pets[apet_in_all + 1]
        else:
            right_slot = None

        ret_pets = []
        for temp_slot in [left_slot, right_slot]:
            if temp_slot is None:
                continue
            if temp_slot.empty:
                continue
            else:
                ret_pets.append(temp_slot.pet)
        return ret_pets, [ret_pets]

    elif kind == "All":
        ret_pets = []
        for temp_idx in fidx:
            ret_pets.append(fteam[temp_idx].pet)
        for temp_idx in oidx:
            ret_pets.append(oteam[temp_idx].pet)
        return ret_pets, [ret_pets]

    elif kind == "AllOtherFriends":
        # v31: "other" means other than the ability's owner. Dragon buffs
        # every friend except itself, including the tier-1 pet just bought
        # (44 real shop turns: Dragon unchanged, bought pet +1/+1).
        ret_pets = [
            fteam[index].pet
            for index in fidx
            if fteam[index].pet is not apet
        ]
        return ret_pets, [ret_pets]

    elif kind == "DifferentTierAnimals":
        pet_tier_lookup = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        for temp_idx in fidx:
            temp_tier = fteam[temp_idx].pet.tier
            pet_tier_lookup[temp_tier].append(temp_idx)

        ### Build lookup of all possible pets
        idx_list = []
        for key, value in pet_tier_lookup.items():
            if len(value) > 0:
                idx_list.append(value)
        grid = np.meshgrid(*idx_list)
        ravel_grid = [x.ravel() for x in grid]
        all_idx = np.array(ravel_grid).T
        all_possible = []
        for temp_idx in all_idx:
            temp_chosen = [fteam[x].pet for x in temp_idx]
            all_possible.append(temp_chosen)
        if len(all_possible) == 0:
            return [], []
        ### Choose one to return for current
        choice_idx_range = np.arange(0, len(all_possible))
        choice_idx = apet.rs.choice(choice_idx_range, (1,))[0]
        ### Update internal state of random generator
        apet.seed_state = apet.rs.get_state()
        ret_pets = all_possible[choice_idx]
        return ret_pets, all_possible

    elif kind == "EachEnemy":
        ret_pets = []
        for temp_idx in oidx:
            ret_pets.append(oteam[temp_idx].pet)
        return ret_pets, [ret_pets]

    elif kind == "EachFriend":
        ret_pets = []
        for temp_idx in fidx:
            if temp_idx != apet_idx[1]:
                ret_pets.append(fteam[temp_idx].pet)
        return ret_pets, [ret_pets]

    elif kind == "EachShopAnimal":
        shop = apet.shop
        if shop is None:
            return [], []
        else:
            return shop.pets, [shop.pets]

    elif kind == "FirstEnemy":
        if len(oidx) > 0:
            return [oteam[oidx[0]].pet], [[oteam[oidx[0]].pet]]
        else:
            return [], []

    elif kind == "FriendAhead":
        chosen_idx = []
        for temp_idx in fidx:
            if temp_idx < apet_idx[1]:
                chosen_idx.append(temp_idx)
        ret_pets = []
        for temp_idx in chosen_idx[::-1]:
            ret_pets.append(fteam[temp_idx].pet)
            if len(ret_pets) >= n:
                break
        return ret_pets, [ret_pets]

    # elif kind == "FriendBehind":
    #     chosen_idx = []
    #     for temp_idx in fidx:
    #         if temp_idx > apet_idx[1]:
    #             chosen_idx.append(temp_idx)
    #     ret_pets = []
    #     for temp_idx in chosen_idx:
    #         ret_pets.append(fteam[temp_idx].pet)
    #         if len(ret_pets) >= n:
    #             break
    #     return ret_pets, [ret_pets]
    elif kind == "FriendBehind":
        chosen_idx = []
        for temp_idx in fidx:
            if temp_idx > apet_idx[1]:
                chosen_idx.append(temp_idx)
        ret_pets = []
        exclude_status = target.get("excludeStatus")
        for temp_idx in chosen_idx:
            temp_pet = fteam[temp_idx].pet
            if exclude_status:
                if temp_pet.status != exclude_status:
                    ret_pets.append(temp_pet)
                    if len(ret_pets) >= n:
                        break
            else:
                ret_pets.append(temp_pet)
                if len(ret_pets) >= n:
                    break

        return ret_pets, [ret_pets]

    
    # elif kind == "":
    #     pass

    elif kind == "HighestHealthEnemy":
        health_list = []
        for temp_idx in oidx:
            health_list.append(oteam[temp_idx].pet.health)
        if len(health_list) > 0:
            max_health = np.max(health_list)
            choice_idx_range = np.where(np.array(health_list) == max_health)[0]
            choice_idx = apet.rs.choice(choice_idx_range, (1,), replace=False)[0]
            all_possible = [[oteam[oidx[x]].pet] for x in choice_idx_range]
            ### Update internal state of random generator
            apet.seed_state = apet.rs.get_state()
            ### Dereference max_idx
            return [oteam[oidx[choice_idx]].pet], all_possible
        else:
            return [], []

    elif kind == "LastEnemy":
        if len(oidx) > 0:
            return [oteam[np.max(oidx)].pet], [[oteam[np.max(oidx)].pet]]
        else:
            return [], []

    elif kind == "LeftMostFriend":
        max_idx = np.max(fidx)
        return [fteam[max_idx].pet], [[fteam[max_idx].pet]]

    elif kind == "Level2And3Friends":
        level_list = []
        for temp_idx in fidx:
            level_list.append(
                0
                if temp_idx == apet_idx[1]
                else fteam[temp_idx].pet.level
            )
        if len(level_list) > 0:
            keep_idx = np.where(np.array(level_list) > 1)[0]
            ret_pets = []
            for temp_idx in keep_idx:
                ### Dereference idx
                temp_idx = fidx[temp_idx]
                ret_pets.append(fteam[temp_idx].pet)
            
            # Get the 'n' value from the target dictionary
            n = target.get("n", len(ret_pets))  # Default to all if 'n' is not specified
            
            # Randomly select 'n' pets if there are more than 'n' eligible pets
            if len(ret_pets) > n:
                choice = apet.rs.choice(
                    np.arange(len(ret_pets)), size=n, replace=False
                )
                apet.seed_state = apet.rs.get_state()
                ret_pets = [ret_pets[idx] for idx in choice]
            
            return ret_pets, [ret_pets]
        else:
            return [], []


    elif kind == "LowestHealthEnemy":
        health_list = []
        for temp_idx in oidx:
            health_list.append(oteam[temp_idx].pet.health)
        if len(health_list) > 0:
            min_health = np.min(health_list)
            choice_idx_range = np.where(np.array(health_list) == min_health)[0]
            choice_idx = apet.rs.choice(choice_idx_range, (1,), replace=False)[0]
            all_possible = [[oteam[oidx[x]].pet] for x in choice_idx_range]
            ### Update internal state of random generator
            apet.seed_state = apet.rs.get_state()
            ### Dereference max_idx
            return [oteam[oidx[choice_idx]].pet], all_possible
        else:
            return [], []

    elif kind == "RandomEnemy":
        ret_pets = []
        all_possible = []
        if len(oidx) > 0:
            if len(oidx) < n:
                n = len(oidx)
            all_idx = [x for x in itertools.combinations(oidx, n)]
            all_possible = []
            for temp_idx in all_idx:
                temp_chosen = [oteam[x].pet for x in temp_idx]
                all_possible.append(temp_chosen)
            if len(all_possible) == 0:
                return [], []
            crange = np.arange(0, len(all_possible))
            cidx = apet.rs.choice(crange, (1,), replace=False)[0]
            ### Update internal state of random generator
            apet.seed_state = apet.rs.get_state()
            ret_pets = all_possible[cidx]
        return ret_pets, all_possible

    elif kind == "RandomFriend":
        ret_pets = []
        all_possible = []
        if len(fidx) > 0:
            # Exclude the activating pet before clamping n. Seal is the
            # important current Turtle case: on a small team it must buff all
            # available *other* friends rather than fail to form a size-3
            # combination.
            fidx = [index for index in fidx if index != apet_idx[1]]
            n = min(n, len(fidx))
            if n == 0:
                return [], []
            all_idx = [x for x in itertools.combinations(fidx, n)]
            all_possible = []
            for temp_idx in all_idx:
                temp_chosen = [fteam[x].pet for x in temp_idx]
                all_possible.append(temp_chosen)
            if len(all_possible) == 0:
                return [], []
            crange = np.arange(0, len(all_possible))
            cidx = apet.rs.choice(crange, (1,), replace=False)[0]
            ### Update internal state of random generator
            apet.seed_state = apet.rs.get_state()
            ret_pets = all_possible[cidx]
        return ret_pets, all_possible

    elif kind == "RightMostFriend":
        return [fteam[fidx[0]].pet], [[fteam[fidx[0]].pet]]

    elif kind == "FrontMostFriend":
        if not fidx:
            return [], []
        return [fteam[fidx[0]].pet], [[fteam[fidx[0]].pet]]

    elif kind == "Self":
        return [apet], [[apet]]

    elif kind == "StrongestFriend":
        stat_list = []
        for temp_idx in fidx:
            temp_stats = fteam[temp_idx].pet.attack + fteam[temp_idx].pet.health
            stat_list.append(temp_stats)
        stat_list = np.array(stat_list)
        max_stats = np.max(stat_list)
        max_idx = np.where(stat_list == max_stats)[0]
        all_possible = []
        for temp_idx in max_idx:
            all_possible.append(fteam[temp_idx].pet)
        if len(all_possible) == 0:
            return [], []
        choice = apet.rs.choice(np.arange(len(all_possible)), (1,), replace=False)[0]
        ### Update internal state of random generator
        apet.seed_state = apet.rs.get_state()
        ret_pets = [all_possible[choice]]
        return ret_pets, all_possible

    elif kind == "HighestHealthFriend":
        health_list = []
        for temp_idx in fidx:
            health_list.append(fteam[temp_idx].pet.health)
        if len(health_list) == 0:
            return [], []
        max_health = np.max(health_list)
        max_idx = np.where(health_list == max_health)[0]
        all_possible = []
        for temp_idx in max_idx:
            all_possible.append(fteam[fidx[temp_idx]].pet)
        choice = apet.rs.choice(max_idx, (1,), replace=False)[0]
        ### Update internal state of random generator
        apet.seed_state = apet.rs.get_state()
        ret_pets = [fteam[fidx[choice]].pet]
        return ret_pets, all_possible

    elif kind == "TriggeringEntity":
        if te is not None:
            return [te], [[te]]
        else:
            return [], []

    elif kind == "NonWeakEnemy":
        possible = []
        for temp_idx in oidx:
            temp_pet = oteam[temp_idx].pet
            if temp_pet.status == "status-weak":
                continue
            else:
                possible.append([temp_pet])
        if len(possible) == 0:
            return [], []
        idx_range = np.arange(0, len(possible))
        chosen_idx = apet.rs.choice(idx_range, (1,), replace=False)[0]
        ### Update internal state of random generator
        apet.seed_state = apet.rs.get_state()
        return possible[chosen_idx], possible

    elif kind == "none":
        ### No targets
        return [], []

    else:
        raise Exception(f"Target {kind} impelementation not found")


def AllOf(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    """AllOf will return list of lists"""
    fixed_targets = fixed_targets or []

    original_effect = apet.ability["effect"]
    effects = apet.ability["effect"]["effects"]
    target = []
    possible_targets = []
    for iter_idx, temp_effect in enumerate(effects):
        effect_kind = temp_effect["kind"]
        func = get_effect_function(effect_kind)
        apet.ability["effect"] = temp_effect
        temp_target, temp_possible = func(apet, apet_idx, teams, te, fixed_targets)
        target.append(temp_target)
        possible_targets.append(temp_possible)
    apet.ability["effect"] = original_effect
    return target, possible_targets


# def ApplyStatus(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
#     fixed_targets = fixed_targets or []

#     if len(fixed_targets) == 0:
#         target, possible = get_target(apet, apet_idx, teams, te=te)
#     else:
#         target = fixed_targets
#         possible = [fixed_targets]
#     status = apet.ability["effect"]["status"]
#     for target_pet in target:
#         target_pet.status = status
#     return target, possible
def ApplyStatus(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    # print(f"ApplyStatus called for {apet.name}")
    fixed_targets = fixed_targets or []
    if len(fixed_targets) == 0:
        target, possible = get_target(apet, apet_idx, teams, te=te)
        # print(f"Targets from get_target: {[t.name for t in target]}")
    else:
        target = fixed_targets
        possible = [fixed_targets]
        # print(f"Using fixed targets: {[t.name for t in target]}")
    
    status = apet.ability["effect"]["status"]
    target_spec = apet.ability["effect"].get(
        "to", apet.ability["effect"].get("target", {})
    )
    exclude_status = target_spec.get("excludeStatus")
    # print(f"Status to apply: {status}, Exclude status: {exclude_status}")
    
    applied_targets = []
    for target_pet in target:
        # print(f"Checking target: {target_pet.name}, Current status: {target_pet.status}")
        if exclude_status and target_pet.status == exclude_status:
            # print(f"Skipping {target_pet.name} due to existing status")
            continue
        target_pet.status = status
        applied_targets.append(target_pet)
        if (
            status in sap_status.PERK_STATUSES
            and target_pet.player is not None
            and target_pet.team is not None
            and not target_pet.team.battle
        ):
            target_pet.player._notify_friend_gained_perk(target_pet)
        elif (
            status in sap_status.PERK_STATUSES
            and target_pet.team is not None
            and target_pet.team.battle
        ):
            # v31: a perk gained during battle (Turtle, Ox) triggers Seal
            # (the gaining pet) and Rabbit (a friend observer), as in the
            # shop. Per-turn limits carry over from the shop phase through
            # ability_counter, so a Rabbit that already used its three
            # triggers this turn does nothing. Evidence: real replays (TrCo
            # field) and SAP-Calculator's givePetEquipment food triggers.
            # A Rabbit that is fainting in the same resolution still fires
            # (11 real fights fixed, none broken), so health is not checked.
            for slot in list(target_pet.team):
                if slot.empty:
                    continue
                if slot.pet.ability["trigger"] in {
                    "GainsPerk",
                    "EatsShopFoodOrGainsPerk",
                }:
                    slot.pet.gains_perk_trigger(target_pet)
        # print(f"Applied {status} to {target_pet.name}")
    
    # print(f"Final applied targets: {[t.name for t in applied_targets]}")
    return applied_targets, possible



def DealDamage(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    fixed_targets = fixed_targets or []

    if len(fixed_targets) == 0:
        target, possible = get_target(apet, apet_idx, teams, te=te)
    else:
        target = fixed_targets
        possible = [fixed_targets]
    base_health_amount = apet.ability["effect"]["amount"]
    if type(base_health_amount) == dict:
        if "attackDamagePercent" in base_health_amount:
            base_health_amount = int(
                apet.attack * base_health_amount["attackDamagePercent"] * 0.01
            )
        else:
            raise Exception()
    for target_pet in target:
        health_amount = base_health_amount
        if (
            apet.ability["effect"].get("doubleAgainstTier")
            == target_pet.tier
        ):
            health_amount *= 2
        if target_pet.status == "status-melon-armor":
            health_amount = max(0, health_amount - 20)
            target_pet.status = "none"
        elif target_pet.status == "status-garlic-armor":
            # v31: same rule as attack damage; a hit of 1 or 2 is unchanged.
            if health_amount > 2:
                health_amount = max(2, health_amount - 2)
        elif target_pet.status == "status-coconut-shield":
            health_amount = 0
            target_pet.status = "none"
        elif target_pet.status == "status-weak":
            health_amount += 3

        target_pet.hurt(health_amount)
    return target, possible


def GainHealthPercentHighestFriend(
    apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None
):
    """Gain a percentage of the highest-health *other* friendly pet.

    This is the current Turtle Pack Crab behaviour.  It is intentionally a
    gain rather than the legacy Crab's stat-copy operation.
    """

    fteam, _ = get_teams(apet_idx, teams)
    friends = [
        slot.pet
        for slot in fteam
        if not slot.empty and slot.pet is not apet and slot.pet.health > 0
    ]
    if not friends:
        return [], []
    highest = max(friend.health for friend in friends)
    candidates = [friend for friend in friends if friend.health == highest]
    chosen_idx = apet.rs.choice(np.arange(len(candidates)), size=(1,))[0]
    apet.seed_state = apet.rs.get_state()
    source = candidates[chosen_idx]
    percentage = apet.ability["effect"]["percentage"] * 0.01
    gain = int(np.ceil(source.health * percentage))
    apet._health = min(50, apet._health + gain)
    return [apet], [[apet]]


def StockFood(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    """Stock configured food in the activating pet's shop."""

    if apet.shop is None:
        return [], []
    effect = apet.ability["effect"]
    requested = int(effect.get("n", 1))
    free_slots = max(0, apet.shop.max_slots - len(apet.shop.slots))
    removable = [
        index
        for index in range(len(apet.shop.slots) - 1, -1, -1)
        if not apet.shop.slots[index].frozen
    ]
    generated_count = min(requested, free_slots + len(removable))
    overflow = max(
        0, len(apet.shop.slots) + generated_count - apet.shop.max_slots
    )
    remove_indices = set(removable[:overflow])
    if remove_indices:
        surviving_slots = [
            slot
            for index, slot in enumerate(apet.shop.slots)
            if index not in remove_indices
        ]
        apet.shop._nslots = len(surviving_slots)
        apet.shop.slots = surviving_slots

    targets = []
    for _ in range(generated_count):
        food = Food(effect["food"], shop=apet.shop)
        if "attackAmount" in effect:
            food.attack = int(effect["attackAmount"])
            food.base_attack = food.attack
        if "healthAmount" in effect:
            food.health = int(effect["healthAmount"])
            food.base_health = food.health
        if "cost" in effect:
            food.cost = int(effect["cost"])
        slot = sapai.ShopSlot(
            food,
            pack=apet.shop.pack,
            turn=apet.shop.turn,
            cost=food.cost,
            seed_state=apet.shop.seed_state,
        )
        apet.shop.slots.append(slot)
        apet.shop._nslots = len(apet.shop.slots)
        targets.append(food)
    return targets, [targets]


def GainExperience(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    fixed_targets = fixed_targets or []

    if len(fixed_targets) == 0:
        target, possible = get_target(apet, apet_idx, teams, te=te)
    else:
        target = fixed_targets
        possible = [fixed_targets]
    for target_pet in target:
        amount = apet.ability["effect"]["amount"]
        previous_level = target_pet.level
        level_ups = target_pet.gain_experience(amount=amount)
        if level_ups:
            target_pet.trigger_levelup_abilities(previous_level)
            player = apet.player
            if player is not None:
                apet.player.shop.levelup(level_ups)
    return target, possible


def GainGold(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    fixed_targets = fixed_targets or []

    amount = apet.ability["effect"]["amount"]
    player = apet.player
    if player is not None:
        apet.player.gold += amount
    return player, [player]


def Evolve(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    fixed_targets = fixed_targets or []

    if len(fixed_targets) == 0:
        target, possible = get_target(apet, apet_idx, teams, te=te)
    else:
        target = fixed_targets
        possible = [fixed_targets]
    fteam, oteam = get_teams(apet_idx, teams)
    spet = sapai.Pet(target[0].ability["effect"]["into"])
    try:
        fteam.remove(target[0])
    except Exception:
        # tiger behind, just does nothing
        return target, possible
    fteam[apet_idx[1]] = spet
    return target, possible


def FoodMultiplier(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    fixed_targets = fixed_targets or []

    if te is None:
        raise Exception("Must input purchased food to FoodMultiplier")

    mult = int(apet.ability["effect"]["amount"])
    food_list = [te]
    for food in food_list:
        ### Multiplier is not strict multiplier of current value, but additive
        ###   multiplier of base attack and health
        mult = mult - 1
        food.attack += food.base_attack * mult
        food.health += food.base_health * mult
    return food_list, [food_list]


def ModifyStats(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    # print("CALLED ModifyStats")
    # print("----------------")
    # print(apet)
    # print(apet_idx)
    # print(teams)
    # print(te)
    fixed_targets = fixed_targets or []

    if len(fixed_targets) == 0:
        target, possible = get_target(apet, apet_idx, teams, te=te)
    else:
        target = fixed_targets
        possible = [fixed_targets]

    attack_amount = 0
    health_amount = 0
    if "attackAmount" in apet.ability["effect"]:
        attack_amount = apet.ability["effect"]["attackAmount"]
    if "healthAmount" in apet.ability["effect"]:
        health_amount = apet.ability["effect"]["healthAmount"]
    if "amount" in apet.ability["effect"]:
        if type(apet.ability["effect"]["amount"]) == dict:
            if "attackPercent" in apet.ability["effect"]["amount"]:
                attack_amount = int(
                    apet.attack
                    * apet.ability["effect"]["amount"]["attackPercent"]
                    * 0.01
                )
            else:
                raise Exception()
    for target_pet in target:
        if target_pet._health <= 0:
            # v31: a pet already reduced to 0 health cannot be buffed back to
            # life before its faint resolves (real replays; SAP-Calculator's
            # increaseAttack/increaseHealth ignore pets that are not alive).
            continue
        if (
            "untilEndOfBattle" in apet.ability["effect"]
            and apet.ability["effect"]["untilEndOfBattle"] is True
        ):
            target_pet._until_end_of_battle_attack_buff = min(
                target_pet._until_end_of_battle_attack_buff + attack_amount,
                max(0, 50 - target_pet._attack),
            )
            target_pet._until_end_of_battle_health_buff = min(
                target_pet._until_end_of_battle_health_buff + health_amount,
                max(0, 50 - target_pet._health),
            )
        else:
            target_pet._attack += attack_amount
            if health_amount < 0:
                # v31: stat removal cannot faint a pet. The real game floors
                # health at 1 (Wolverine); SAP-Calculator's increaseHealth
                # clamps the same way.
                if target_pet._health > 0:
                    target_pet._health = max(1, target_pet._health + health_amount)
            else:
                target_pet._health += health_amount
        if (
            "includingFuture" in apet.ability["effect"]["target"]
            and apet.ability["effect"]["target"]["includingFuture"] is True
        ):
            if target_pet.shop is not None:
                target_pet.shop.shop_attack += attack_amount
                target_pet.shop.shop_health += health_amount
        target_pet._attack = min([target_pet._attack, 50])
        target_pet._health = min([target_pet._health, 50])

    return target, possible


def OneOf(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    """
    Dog is only one with OneOf anyways
    However, OneOf current not returning possible correctly because I haven't
      decided exactly how possible will be used...
    """
    # if len(fixed_targets) == 0:
    #     target,possible = get_target(apet, apet_idx, teams, te=te)
    # else:
    #     target = fixed_targets
    #     possible = [fixed_targets]
    fixed_targets = fixed_targets or []

    possible = [[apet]]
    original_effect = apet.ability["effect"]
    effects = apet.ability["effect"]["effects"]
    chosen_idx = apet.rs.choice(np.arange(0, len(effects)), size=(1,))[0]
    ### Update internal state of random generator
    apet.seed_state = apet.rs.get_state()
    effect = effects[chosen_idx]
    effect_kind = effect["kind"]
    apet.ability["effect"] = effect
    func = get_effect_function(effect_kind)
    target = func(apet, apet_idx, teams, te, fixed_targets)[0]
    apet.ability["effect"] = original_effect
    return target, possible


def ReduceHealth(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    fixed_targets = fixed_targets or []

    if len(fixed_targets) == 0:
        target, possible = get_target(apet, apet_idx, teams, te=te)
    else:
        target = fixed_targets
        possible = [fixed_targets]
    per = apet.ability["effect"]["percentage"] * 0.01
    for temp_pet in target:
        if temp_pet.health > 0:
            temp_pet._health = max(
                1, int(np.floor(temp_pet.health * (1.0 - per)))
            )
    return target, possible


def RefillShops(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    """Replace the food shop with two free level-scaled Milks (Cow)."""
    fixed_targets = fixed_targets or []

    shop = apet.shop
    if shop is None:
        return [], []
    level = apet.level
    effect = apet.ability["effect"]
    food_name = effect.get("food", "food-milk")
    count = int(effect.get("n", 2))
    surviving_slots = [slot for slot in shop if slot.slot_type != "food"]
    missing_capacity = max(0, len(surviving_slots) + count - shop.max_slots)
    removable = [
        index
        for index in range(len(surviving_slots) - 1, -1, -1)
        if not surviving_slots[index].frozen
    ]
    remove_indices = set(removable[:missing_capacity])
    surviving_slots = [
        slot
        for index, slot in enumerate(surviving_slots)
        if index not in remove_indices
    ]
    milk_slots = []
    for _ in range(min(count, shop.max_slots - len(surviving_slots))):
        food = Food(food_name, shop=shop)
        food.attack = food.base_attack * level
        food.health = food.base_health * level
        food.cost = 0
        milk_slots.append(
            sapai.ShopSlot(
                food,
                pack=shop.pack,
                turn=shop.turn,
                cost=0,
                seed_state=shop.seed_state,
            )
        )
    rebuilt = surviving_slots + milk_slots
    shop._nslots = len(rebuilt)
    shop.slots = rebuilt
    targets = [slot.obj for slot in milk_slots]
    return targets, [targets]

def RefillShopsWorm(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    """
    Refills shop with apples, scaling with the activating pet's level.
    """
    fixed_targets = fixed_targets or []

    if apet.name != "pet-worm":
        raise Exception("Only worm implemented for RefillShops")
    
    shop = apet.shop
    level = apet.level
    targets = []
    
    for slot in shop:
        if slot.slot_type == "food":
            temp_food = Food("apple")
            temp_food.attack *= level
            temp_food.health *= level
            
            # Add the new food to the slot instead of replacing
            if hasattr(slot, 'items'):
                slot.items.append(temp_food)
            else:
                # If the slot doesn't support multiple items, create a list
                slot.items = [slot.item, temp_food] if slot.item else [temp_food]
            
            targets.append(slot)
    
    return targets, [targets]



def RepeatAbility(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    """
    Tiger implementation.

    """
    fixed_targets = fixed_targets or []

    ### First, modify te by the level of the tiger
    original_level = te.level
    te.level = apet.level

    ### Get te ability
    func = get_effect_function(te)

    ### Call will te as apet
    if len(fixed_targets) == 0:
        targets, possible = func(te, apet_idx, teams, te=None, fixed_targets=None)
    else:
        ### Use fixed_targets if the repeated function is supposed to target
        ### exactly the same pets again
        targets, possible = func(te, apet_idx, teams, te=fixed_targets)

    te.level = original_level
    return targets, possible


def RespawnPet(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    """
    Only for Mushroom food at the moment

    """
    te_idx = te_idx or []
    fixed_targets = fixed_targets or []

    if len(te_idx) == 0:
        raise Exception("Indices of triggering entity must be provided as te_idx")

    fteam, _ = get_teams(apet_idx, teams)
    spet_name = apet.name

    if len(fixed_targets) > 0:
        raise NotImplementedError

    target_team = fteam
    #### First, determine how many pets should be infront
    nahead = len(fteam.get_ahead(te_idx[1], n=5))
    npets = len(fteam)
    ### Then move team as far backward as possible
    fteam.move_backward()
    ### Move nahead pets forward which should be infront of the triggering pet
    end_idx = (5 - npets) + nahead
    fteam.move_forward(start_idx=0, end_idx=end_idx)

    target = []
    ### Check for furthest back open position
    empty_idx = []
    for iter_idx, temp_slot in enumerate(target_team):
        if temp_slot.empty:
            empty_idx.append(iter_idx)
    if len(empty_idx) == 0:
        ### Can safely return, cannot summon
        return target, [target]

    target_slot_idx = np.max(empty_idx)
    target_team[target_slot_idx] = spet_name
    spet = target_team[target_slot_idx].pet

    if "baseAttack" in apet.ability["effect"]:
        spet._attack = apet.ability["effect"]["baseAttack"]
    if "baseHealth" in apet.ability["effect"]:
        spet._health = apet.ability["effect"]["baseHealth"]
    spet.level = apet.level
    target.append(spet)

    ### Move back forward
    target_team.move_forward()
    for temp_slot in target_team:
        ### Make sure team is assigned correctly to all pets
        temp_slot.pet.team = target_team

    return target, [target]


def SummonPet(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    """ """
    # print("CALLED SUMMON")
    # print("----------------")
    # print(apet_idx)
    # print(teams)
    # print(apet)
    # print(te)
    te_idx = te_idx or []
    fixed_targets = fixed_targets or []

    if len(te_idx) == 0:
        raise Exception("Indices of triggering entity must be provided as te_idx")

    fteam, oteam = get_teams(apet_idx, teams)
    spet_name = apet.ability["effect"]["pet"]
    team = apet.ability["effect"]["team"]

    if len(fixed_targets) > 0:
        raise NotImplementedError

    if team == "Friendly":
        target_team = fteam
        #### First, determine how many pets should be infront
        nahead = len(fteam.get_ahead(te_idx[1], n=5))
        npets = len(fteam)
        ### Then move team as far backward as possible
        fteam.move_backward()
        ### Move nahead pets forward which should be infront of the triggering pet
        end_idx = (5 - npets) + nahead
        fteam.move_forward(start_idx=0, end_idx=end_idx)
    elif team == "Enemy":
        target_team = oteam
        if type(target_team).__name__ != "Team":
            ### Assume that oteam was not input, for example if Rat was pilled
            ###   during the shop phase
            return [], []
        target_team.move_backward()
    else:
        raise Exception(apet.ability["effect"]["team"])

    n = apet.ability["effect"].get("n", 1)
    # Preserve legacy data which encoded repeat count by summoned token. The
    # activating pet is not necessarily the native ability owner: Parrot can
    # copy Rooster and Tiger can repeat that copied ability.  Keying this on
    # ``apet.name == "pet-rooster"`` therefore left copied Chicks with the
    # data placeholder ``"?"`` for attack and summoned only one Chick.
    if "n" not in apet.ability["effect"]:
        # v31: keyed on the summoned token, not the owner's name, so a
        # Parrot copying Sheep also summons two Rams (real replays;
        # SAP-Calculator "Parrot's Sheep spawned Ram" twice).
        if spet_name == "pet-ram":
            n = 2
        elif spet_name == "pet-chick":
            n = apet.level

    target = []
    for _ in range(n):
        ### Check for furthest back open position
        empty_idx = []
        for iter_idx, temp_slot in enumerate(target_team):
            if temp_slot.empty:
                empty_idx.append(iter_idx)
        if len(empty_idx) == 0:
            ### Can safely return, cannot summon
            return target, [target]

        target_slot_idx = np.max(empty_idx)
        target_team[target_slot_idx] = spet_name
        spet = target_team[target_slot_idx].pet

        if "withAttack" in apet.ability["effect"]:
            spet._attack = apet.ability["effect"]["withAttack"]
        if "withHealth" in apet.ability["effect"]:
            spet._health = apet.ability["effect"]["withHealth"]
        if "withLevel" in apet.ability["effect"]:
            spet.level = apet.ability["effect"]["withLevel"]
        # v31: a summoned token sells for its summoner's level. Real shop
        # turns: two Rams from a pilled level-2 Sheep sold for 4 gold in
        # total; the engine paid 2. A Honey Bee comes from the perk, not
        # from the pet's ability, and sells for 1 whatever the pet's level
        # (three real shop turns).
        if team == "Friendly" and spet_name != "pet-bee":
            spet.sell_value = max(
                int(spet.sell_value), int(spet.level), int(apet.level)
            )
        if spet_name == "pet-chick":
            spet._attack = int(apet.attack * 0.5)

        if not isinstance(spet._attack, (int, float, np.integer, np.floating)):
            raise TypeError(
                "SummonPet produced an unresolved attack stat: "
                f"source={apet.name} summoned={spet_name} "
                f"level={apet.level} attack={spet._attack!r} "
                f"effect={apet.ability['effect']!r}"
            )
        if not isinstance(spet._health, (int, float, np.integer, np.floating)):
            raise TypeError(
                "SummonPet produced an unresolved health stat: "
                f"source={apet.name} summoned={spet_name} "
                f"level={apet.level} health={spet._health!r} "
                f"effect={apet.ability['effect']!r}"
            )

        target.append(spet)

    ### Move back forward
    target_team.move_forward()
    for temp_slot in target_team:
        ### Make sure team is assigned correctly to all pets
        temp_slot.pet.team = target_team

    return target, [target]


def SummonRandomPet(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    """ """
    te_idx = te_idx or []
    fixed_targets = fixed_targets or []

    fteam, oteam = get_teams(apet_idx, teams)

    if len(fixed_targets) > 0:
        chosen = fixed_targets[0]
        if len(fixed_targets) > 1:
            raise Exception("Only 1 fixed_targets input allowed for SummonRandomPet")
    else:
        tier = apet.ability["effect"]["tier"]
        if fteam.pack == "StandardPack":
            possible = pet_tier_lookup_std[tier]
        elif fteam.pack == "ExpansionPack1":
            possible = pet_tier_lookup[tier]
        else:
            raise Exception()
        chosen = apet.rs.choice(possible, (1,))[0]
        ### Update internal state of random generator
        apet.seed_state = apet.rs.get_state()

    #### Perform team movement to ensure that the pet is summoned in the
    ####   correct position
    #### First, determine how many pets should be infront
    nahead = len(fteam.get_ahead(te_idx[1], n=5))
    npets = len(fteam)
    ### Then move team as far backward as possible
    fteam.move_backward()
    ### Move nahead pets forward which should be infront of the triggering pet
    end_idx = (5 - npets) + nahead
    fteam.move_forward(start_idx=0, end_idx=end_idx)

    ### Check for furthest back open position
    empty_idx = []
    for iter_idx, temp_slot in enumerate(fteam):
        if temp_slot.empty:
            empty_idx.append(iter_idx)
    if len(empty_idx) == 0:
        ### Can safely return, cannot summon
        return [], []

    target_slot_idx = np.max(empty_idx)
    fteam[target_slot_idx] = str(chosen)
    spet = fteam[target_slot_idx].pet
    if "baseAttack" in apet.ability["effect"]:
        sattack = apet.ability["effect"]["baseAttack"]
    else:
        sattack = data["pets"][spet.name]["baseAttack"]
    if "baseHealth" in apet.ability["effect"]:
        shealth = apet.ability["effect"]["baseHealth"]
    else:
        shealth = data["pets"][spet.name]["baseHealth"]
    if "level" in apet.ability["effect"]:
        spet.level = apet.ability["effect"]["level"]
        # v31: sell value follows the summoned level (real replays: a
        # level-2 Spider summon sells for 2, the engine paid 1).
        spet.sell_value = max(spet.sell_value, spet.level)
    if "statsModifier" in apet.ability["effect"]:
        sattack *= apet.ability["effect"]["statsModifier"]
        shealth *= apet.ability["effect"]["statsModifier"]

    spet._attack = sattack
    spet._health = shealth
    fteam.move_forward()
    for temp_slot in fteam:
        temp_slot.pet.team = fteam

    return [spet], [[x] for x in possible]


# def Swallow(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
#     fixed_targets = fixed_targets or []

#     fteam, oteam = get_teams(apet_idx, teams)
#     if len(fixed_targets) == 0:
#         target, possible = get_target(apet, apet_idx, teams, te=te)
#     else:
#         target = fixed_targets
#         possible = [fixed_targets]

#     if len(target) == 0:
#         return target, possible
#     output_level = apet.level
#     if output_level == 1:
#         level_attack = 0
#         level_health = 0
#     elif output_level == 2:
#         level_attack = 2
#         level_health = 2
#     elif output_level == 3:
#         level_attack = 5
#         level_health = 5
#     else:
#         raise Exception()

#     # if apet.name != "pet-whale":
#     #     raise Exception("Swallow only done by whale")

#     ### Remove target from team and shop this pet as the given level as a
#     ### Summon ability
#     for temp_target in target:
#         if data["pets"][temp_target.name]["baseAttack"] != "?":
#             base_attack = data["pets"][temp_target.name]["baseAttack"]
#         else:
#             base_attack = temp_target.attack
#         if data["pets"][temp_target.name]["baseHealth"] != "?":
#             base_health = data["pets"][temp_target.name]["baseHealth"]
#         else:
#             base_health = temp_target.health

#         summon_dict = {
#             "description": f"Swallowed and summon level {temp_target.name} {output_level}",
#             "trigger": "Faint",
#             "triggeredBy": {"kind": "Self"},
#             "effect": {
#                 "kind": "SummonPet",
#                 "pet": temp_target.name,
#                 "withAttack": base_attack + level_attack,
#                 "withHealth": base_health + level_health,
#                 "withLevel": output_level,
#                 "team": "Friendly",
#             },
#         }
#         apet.set_ability(summon_dict)
#         temp_target._health = -1

#     fteam.move_forward()
#     return target, possible
def Swallow(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    fixed_targets = fixed_targets or []

    fteam, oteam = get_teams(apet_idx, teams)
    if len(fixed_targets) == 0:
        target, possible = get_target(apet, apet_idx, teams, te=te)
    else:
        target = fixed_targets
        possible = [fixed_targets]

    if len(target) == 0:
        return target, possible

    output_level = apet.level

    for temp_target in target:
        # Current Whale preserves the swallowed pet's attack and health while
        # releasing it at the Whale's level. Equipment is intentionally not
        # copied by the generated summon.
        summon_attack = temp_target.attack
        summon_health = temp_target.health

        apet.swallowed_pets.append(
            {
                "pet": temp_target.name,
                "withAttack": summon_attack,
                "withHealth": summon_health,
                "withLevel": output_level,
                "team": "Friendly",
            }
        )
        apet.set_ability(
            {
                "description": "Release swallowed pets after fainting.",
                "trigger": "Faint",
                "triggeredBy": {"kind": "Self"},
                "effect": {"kind": "SummonStoredPets"},
            }
        )
        temp_target._health = -1

    fteam.move_forward()

    return target, possible


def SummonStoredPets(
    apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None
):
    """Release every pet stored by Whale/Parrot, without Tiger repeats."""

    te_idx = te_idx or []
    fteam, _ = get_teams(apet_idx, teams)
    if fteam.check_friend(apet):
        fteam.remove(apet)
    stored = list(apet.swallowed_pets)
    original_override = apet.override_ability
    original_dict = apet.override_ability_dict
    targets = []
    try:
        for descriptor in stored:
            apet.override_ability = True
            apet.override_ability_dict = {
                "description": "Release swallowed pet.",
                "trigger": "Faint",
                "triggeredBy": {"kind": "Self"},
                "effect": {"kind": "SummonPet", **descriptor},
            }
            summoned, _ = SummonPet(
                apet, apet_idx, teams, te=te, te_idx=te_idx
            )
            targets.extend(summoned)
    finally:
        apet.override_ability = original_override
        apet.override_ability_dict = original_dict
        apet.swallowed_pets = []
    return targets, [targets]



def TransferAbility(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    fixed_targets = fixed_targets or []

    fteam, oteam = get_teams(apet_idx, teams)
    if len(fixed_targets) == 0:
        target, possible = get_target(apet, apet_idx, teams, te=te, get_from=True)
    else:
        target = fixed_targets
        possible = [fixed_targets]

    if len(target) > 0:
        source = target[0]
        origin = source.name
        native = source.fd
        # v31: a Parrot behind a Parrot that has already copied this turn
        # copies what that Parrot holds, at its own level. End-of-turn copies
        # resolve in attack order, so a back Parrot with higher attack acts
        # first and copies nothing useful. Real replays record each Parrot's
        # ability: Sheep, Parrot 9/5, Parrot 5/3 both hold Sheep's; Deer,
        # Parrot 7/3, Parrot 11/11 leaves the back one with its own.
        if source.override_ability and source.ability.get("_copiedFrom"):
            origin = source.ability["_copiedFrom"]
            native = data["pets"][origin]
        ability = native.get(f"level{apet.level}Ability", source.ability)
        ability = copy.deepcopy(ability)
        ability["_copiedFrom"] = origin
        apet.set_ability(ability)

    return target, possible


def TransferStats(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    fixed_targets = fixed_targets or []

    fteam, oteam = get_teams(apet_idx, teams)
    if len(fixed_targets) == 0:
        target, possible = get_target(apet, apet_idx, teams, te=te)
    else:
        target = fixed_targets
        possible = [fixed_targets]

    effect = apet.ability["effect"]
    copy_attack = effect.get("copyAttack", False)
    copy_health = effect.get("copyHealth", False)
    percentage = 1
    if "percentage" in effect:
        percentage = effect["percentage"] * 0.01
    from_self = effect["from"]["kind"] == "Self"

    for entry in target:
        if from_self:
            #### dodo is only from_self and it is additive, not copy, unlike
            #### what the database says
            if copy_attack:
                entry._attack = min(
                    50,
                    entry._attack
                    + max(int(apet.attack * percentage), 1),
                )
            if copy_health:
                raise Exception("This should not be possible")
        else:
            temp_from = get_target(apet, apet_idx, teams, te=te, get_from=True)
            ### Randomness not needed as outcome will be the same for all pets
            ###   that have this ability
            if not temp_from or not temp_from[0]:
                continue
            temp_from = temp_from[0][0]
            if copy_attack:
                apet._attack = int(temp_from.attack * percentage)
            if copy_health:
                apet._health = int(temp_from.health * percentage)

    return target, possible


def DiscountFood(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    fixed_targets = fixed_targets or []

    shop = apet.shop
    if shop is None and apet.team is not None:
        shop = apet.team.shop
        apet.shop = shop
    if shop is None and apet.player is not None:
        shop = apet.player.shop
        apet.shop = shop
    if shop is None:
        raise Exception(
            "No shop found to discount food: "
            f"pet={apet.name} team={apet.team is not None} "
            f"player={apet.player is not None} ability={apet.ability!r}"
        )
    amount = apet.ability["effect"]["amount"]
    targets = []
    for slot in shop:
        if slot.slot_type == "food":
            slot.cost = max((slot.cost - amount), 0)
            targets.append(slot)
    return targets, [targets]


def GainAbility(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    """
    Only Octopus has GainAbility. Also, within programming framework,
    GainAbility is not necessary because the ability is automatically
    updated with levelup.
    """
    fixed_targets = fixed_targets or []

    return [apet], [[apet]]


def none(apet, apet_idx, teams, te=None, te_idx=None, fixed_targets=None):
    fixed_targets = fixed_targets or []

    return [], []


curr = sys.modules[__name__]
mem = inspect.getmembers(curr, inspect.isfunction)
func_dict = {}
for temp_name, func in mem:
    func_dict[temp_name] = func
