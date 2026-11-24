import numpy as np

from sapai.data import data
from sapai.effects import (
    get_effect_function,
    get_teams,
    RespawnPet,
    SummonPet,
    SummonRandomPet,
    SummonStoredPets,
)
from sapai import status
from collections import defaultdict


class _PhaseLog(list):
    """Phase events with a cheap semantic-only mode for RL battles."""

    def __init__(self, fast_training=False):
        super().__init__()
        self.fast_training = bool(fast_training)


class Battle:
    """
    Performs a battle.

    Most important thing here to implement is the action queue including the
    logic for when actions should be removed from the action queue upon death.

    Note that effects are performed in the order of highest attack to lowest
    attack. If there is a tie, then health values are compared. If there is a
    tie then a random animal is chosen first. This is tracked by the
    pet_priority which is updated before every turn of the battle.

    Any effect which is in the queue for a given turn is executed, even if the
    animal dies due to preceeding effect, as the game entails.

    A Battle goes as follows:
        1. execute start-of-turn abilities according to pet priority
        2. perform hurt and faint abilities according to pet priority
                2.1 Execute 2 until there are no new fainted animals
        3. before-attack abilities according to pet priority
        4. perform fainted pet abilities via pet priority
                4.1 Execute 4 until there are no new fainted animals
        5. attack phase
            5.0 perform before_attack abilities
            5.1. perform hurt and fainted abilities according to pet priority
                   5.1.1 Execute 5.1 until there are no new fainted animals
            5.2 perform attack damage
            5.3 perform after attack abilities
            5.4 perform hurt and fainted abilities according to pet priority
                   5.4.1 Execute 5.4 until there are no new fainted animals
            5.5. check if knock-out abilities should be performed
                    5.5.1 if knock-out ability activated jump to 5.5
            5.6. if battle has not ended, jump to 5.0

    """

    def __init__(self, t0, t1, seed=None, fast_training=False):
        """
        Performs the battle between the input teams t1 and t2.

        """
        ### Make copy each team to cary out the battle so that the original
        ### pets are not modified in any way after the battle
        self.fast_training = bool(fast_training)
        self.t0 = t0.copy()
        self.t0._battle = True
        self.t1 = t1.copy()
        self.t1._battle = True
        self.rs = np.random.RandomState()
        if seed is not None:
            self.rs.seed(int(seed))
        elif self.t0.seed_state is not None:
            self.rs.set_state(self.t0.seed_state)
        self._bind_battle_rng(self.t0)
        self._bind_battle_rng(self.t1)

        ### Internal storage
        self.pet_priority = []
        self.battle_history = {}
        self._pending_post_removal = []

        ### Build initial effect queue order
        self.pet_priority = self.update_pet_priority(self.t0, self.t1, self.rs)

    def _bind_battle_rng(self, team):
        """Attach every current/future battle pet to one sequential stream."""

        team.battle_rng = self.rs
        for slot in team:
            slot.pet.rs = self.rs
            slot.pet.seed_state = self.rs.get_state()

    def battle(self):
        ### Perform all effects that occur at the start of the battle
        # print(f"\nBefore Battle:")
        # print("Team 0:", [str(pet) for pet in self.t0 if not pet.empty])
        # print("Team 1:", [str(pet) for pet in self.t1 if not pet.empty])
        self.start()

        battle_iter = 0
        # print(f"\nStart of battle:")
        # print("Team 0:", [str(pet) for pet in self.t0 if not pet.empty])
        # print("Team 1:", [str(pet) for pet in self.t1 if not pet.empty])
        while True:
            ### First update effect order
            self.pet_priority = self.update_pet_priority(self.t0, self.t1, self.rs)
            ### Then attack
            result = self.attack(battle_iter)
            # print(f"\nAfter attack {battle_iter}:")
            # print("Team 0:", [str(pet) for pet in self.t0 if not pet.empty])
            # print("Team 1:", [str(pet) for pet in self.t1 if not pet.empty])
            battle_iter += 1
            if not result:
                break

        ### Check winner and return 0 for t0 win, 1 for t1 win, 2 for draw
        return self.check_battle_result()

    def start(self):
        """
        Perform all start of battle effects

        """
        ### First move the teams forward
        t0 = self.t0
        t1 = self.t1
        teams = [t0, t1]
        self._pending_post_removal = []

        ### Phase of start
        phase_dict = {
            "init": (
                []
                if self.fast_training
                else [[str(x) for x in t0], [str(x) for x in t1]]
            ),
            "start": {
                "phase_move_start": _PhaseLog(self.fast_training),
                "phase_start": _PhaseLog(self.fast_training),
                "phase_hurt_and_faint": _PhaseLog(self.fast_training),
                "phase_move_end": _PhaseLog(self.fast_training),
            },
        }

        for temp_phase in phase_dict["start"]:
            battle_phase(
                self, temp_phase, teams, self.pet_priority, phase_dict["start"]
            )

            if not self.fast_training:
                self.battle_history.update(phase_dict)

            ### If animals have moved or fainted then effect order must be updated
            if temp_phase.startswith("phase_move"):
                self.pet_priority = self.update_pet_priority(t0, t1, self.rs)

    def attack(self, battle_iter):
        """
        Perform and attack and then check for new pet triggers

        Returns whether or not another attack should occur. This depends on
        if all animals of one team have a health of 0 already.

        Order of operations for an attack are:
            - Pets in the front of each team attack
            - Apply effects related to this attack
            - Apply effects related to pet deaths
            - Summon phase
            - Check if battle is over

        """
        t0 = self.t0
        t1 = self.t1
        self._pending_post_removal = []

        attack_str = f"attack {battle_iter}"
        phase_dict = {
            attack_str: {
                "phase_move_start": _PhaseLog(self.fast_training),
                "phase_attack_before": _PhaseLog(self.fast_training),
                "phase_hurt_and_faint_ab": _PhaseLog(self.fast_training),
                "phase_attack": _PhaseLog(self.fast_training),
                "phase_attack_after": _PhaseLog(self.fast_training),
                "phase_hurt_and_faint_aa": _PhaseLog(self.fast_training),
                "phase_knockout": _PhaseLog(self.fast_training),
                "phase_hurt_and_faint_k": _PhaseLog(self.fast_training),
                "phase_move_end": _PhaseLog(self.fast_training),
            }
        }

        ### Check exit condition, if one team has no animals, return False
        found0 = False
        for temp_slot in t0:
            if not temp_slot.empty:
                found0 = True
                break
        found1 = False
        for temp_slot in t1:
            if not temp_slot.empty:
                found1 = True
                break
        if not found0:
            return False
        if not found1:
            return False

        teams = [t0, t1]
        for temp_phase in phase_dict[attack_str]:
            if temp_phase == "phase_hurt_and_faint_k":
                ### This is checked in phase_knockout for recursive Rhino behavior
                continue

            battle_phase(
                self, temp_phase, teams, self.pet_priority, phase_dict[attack_str]
            )

            if not self.fast_training:
                self.battle_history.update(phase_dict)

        ### Check if battle is over
        status = self.check_battle_result()
        if status < 0:
            return True
        else:
            ### End battle
            return False

    def check_battle_result(self):
        t0 = self.t0
        t1 = self.t1
        found0 = False
        for temp_slot in t0:
            if not temp_slot.empty:
                if temp_slot.pet.health > 0:
                    found0 = True
                    break
        found1 = False
        for temp_slot in t1:
            if not temp_slot.empty:
                if temp_slot.pet.health > 0:
                    found1 = True
                    break
        if found0 and found1:
            ### Fight not over
            return -1
        if found0:
            ### t0 won
            return 0
        if found1:
            ### t1 won
            return 1
        ### Must have been draw
        return 2

    @staticmethod
    def update_pet_priority(t0, t1, rng=None):
        """

        Prepares the order that the animals effects should be considered in

        Effects resolve from highest attack to lowest attack. SAP-Calculator
        uses a random tie-breaker for equal attack; health is not a secondary
        priority key.

        """
        rng = rng or np.random
        entries = []
        for team_idx, team in enumerate((t0, t1)):
            for pet_idx, slot in enumerate(team):
                if slot.empty:
                    continue
                entries.append((float(slot.pet.attack), team_idx, pet_idx))

        ordered = []
        for attack in sorted({entry[0] for entry in entries}, reverse=True):
            tied = [entry for entry in entries if entry[0] == attack]
            if len(tied) > 1:
                order = rng.choice(
                    np.arange(len(tied)), size=len(tied), replace=False
                )
                tied = [tied[int(index)] for index in order]
            ordered.extend((team_idx, pet_idx) for _, team_idx, pet_idx in tied)
        return ordered


class RBattle(Battle):
    """
    This class will calculate all possible outcomes of a SAP battle considering
    all paths of random behavior. The advantage is that probabilities of winning
    are evaluated exactly rather than requiring bootstrapped probabilities.

    Disadvantage is that it is possible that huge number of paths must be
    evaluated to determine exact probabilities. Protection against (could) be
    implemented in two ways:
        1. Determining that paths lead to nomial identical results and can
            merge back together improving calculation efficiency
        2. Define a maximum path size and if the number paths detected is larger
            then probabilities are bootstrapped.

    """

    def __init__(self, t0, t1, max_paths=1000):
        """
        Performs the battle between the input teams t1 and t2.

        """
        super(RBattle, self).__init__(t0, t1)
        ### Make copy each team to cary out the battle so that the original
        ### pets are not modified in any way after the battle
        self.t0 = t0.copy()
        self.t0._battle = True
        self.t1 = t1.copy()
        self.t1._battle = True

        ### Internal storage
        self.battle_list = []

        ### Build initial effect queue order
        self.pet_priority = self.update_pet_priority(self.t0, self.t1, self.rs)

        raise NotImplementedError


def battle_phase(battle_obj, phase, teams, pet_priority, phase_dict):
    """
    Definition for performing all effects and actions throughout the battle.
    Implemented as function instead of class method to save an extra
    indentation.
    s
    """

    ##### Trigger logic for starting battle
    if phase.startswith("phase_move"):
        if not battle_obj.fast_training:
            start_order = [
                [str(x) for x in teams[0]],
                [str(x) for x in teams[1]],
            ]
        teams[0].move_forward()
        teams[1].move_forward()
        if not battle_obj.fast_training:
            end_order = [
                [str(x) for x in teams[0]],
                [str(x) for x in teams[1]],
            ]
            phase_dict[phase] = [start_order, end_order]

    elif phase == "phase_start":
        battle_phase_start(battle_obj, phase, teams, pet_priority, phase_dict)

    ##### Trigger logic for an attack
    elif phase == "phase_attack_before":
        battle_phase_attack_before(battle_obj, phase, teams, pet_priority, phase_dict)

    elif phase == "phase_attack":
        ### Check if fainted and performed fainted triggers
        battle_phase_attack(battle_obj, phase, teams, pet_priority, phase_dict)

    elif phase == "phase_attack_after":
        battle_phase_attack_after(battle_obj, phase, teams, pet_priority, phase_dict)

    elif "phase_hurt_and_faint" in phase:
        battle_phase_hurt_and_faint(battle_obj, phase, teams, pet_priority, phase_dict)

    elif phase == "phase_knockout":
        battle_phase_knockout(battle_obj, phase, teams, pet_priority, phase_dict)

    else:
        raise Exception(f"Phase {phase} not found")


def append_phase_list(phase_list, p, team_idx, pet_idx, activated, targets, possible):
    if activated:
        tiger = False
        if len(targets) > 0:
            if type(targets[0]) == list:
                tiger = True
        if getattr(phase_list, "fast_training", False):
            # Some battle control flow (notably Rhino recursion) uses only the
            # number of emitted events. Preserve that semantic signal without
            # constructing effect names or pet/target string snapshots.
            phase_list.extend([None] * (len(targets) if tiger else 1))
            return
        func = get_effect_function(p)

        if not tiger:
            phase_list.append(
                (
                    func.__name__,
                    (team_idx, pet_idx),
                    (p.__repr__()),
                    [str(x) for x in targets],
                )
            )
        else:
            for temp_target in targets:
                phase_list.append(
                    (
                        func.__name__,
                        (team_idx, pet_idx),
                        (p.__repr__()),
                        [str(x) for x in temp_target],
                    )
                )


def check_summon_triggers(
    phase_list, p, team_idx, pet_idx, teams, activated, targets, possible
):
    if not activated:
        return 0

    func = get_effect_function(p)
    if func not in [RespawnPet, SummonPet, SummonRandomPet, SummonStoredPets]:
        return 0

    if "team" in p.ability["effect"]:
        team = p.ability["effect"]["team"]
        target_team_idx = team_idx if team == "Friendly" else 1 - team_idx
    else:
        target_team_idx = team_idx
    target_team = teams[target_team_idx]

    ### Otherwise, summon triggers need to be checked for each Pet in targets
    if len(targets) > 0:
        if type(targets[0]) == list:
            temp_all_targets = []
            for entry in targets:
                temp_all_targets += entry
            targets = temp_all_targets

    for temp_te in targets:
        # Generated units such as Deer's Bus enter with a built-in perk; that
        # is not a perk-gain event. Scorpion is the exception because its
        # actual Summoned ability is "gain Peanut" even though sapai models it
        # as an initial status.
        if (
            not target_team.battle
            and target_team.player is not None
            and temp_te.name == "pet-scorpion"
            and temp_te.status == "status-poison-attack"
        ):
            target_team.player._notify_friend_gained_perk(temp_te)
        for observer_idx, temp_slot in enumerate(target_team):
            temp_pet = temp_slot.pet
            tempa, tempt, tempp = temp_pet.friend_summoned_trigger(temp_te)
            append_phase_list(
                phase_list,
                temp_pet,
                target_team_idx,
                observer_idx,
                tempa,
                tempt,
                tempp,
            )

    return len(targets)


def check_self_summoned_triggers(teams, pet_priority, phase_dict):
    """
    Currently only butterfly

    """

    phase_list = phase_dict["phase_start"]
    pp = pet_priority
    for team_idx, pet_idx in pp:
        p = teams[team_idx][pet_idx].pet
        if p.health <= 0:
            continue
        if p.ability["trigger"] != "Summoned":
            continue
        if p.ability["triggeredBy"]["kind"] != "Self":
            continue

        func = get_effect_function(p)
        target = func(p, [0, pet_idx], teams, te=p)
        append_phase_list(phase_list, p, team_idx, pet_idx, True, target, [target])


def check_status_triggers(phase_list, p, team_idx, pet_idx, teams):
    if p.status not in ["status-honey-bee", "status-extra-life"]:
        return

    ability = data["statuses"][p.status]["ability"]
    p.set_ability(ability)
    te_idx = [team_idx, pet_idx]
    activated, targets, possible = p.faint_trigger(p, te_idx)
    append_phase_list(phase_list, p, team_idx, pet_idx, activated, targets, possible)
    check_summon_triggers(
        phase_list, p, team_idx, pet_idx, teams, activated, targets, possible
    )


def battle_phase_start(battle_obj, phase, teams, pet_priority, phase_dict):
    phase_list = phase_dict["phase_start"]
    pp = pet_priority
    for team_idx, pet_idx in pp:
        p = teams[team_idx][pet_idx].pet
        fteam, oteam = get_teams([team_idx, pet_idx], teams)
        activated, targets, possible = p.sob_trigger(oteam)
        append_phase_list(
            phase_list, p, team_idx, pet_idx, activated, targets, possible
        )

    check_self_summoned_triggers(teams, pet_priority, phase_dict)

    return phase_list


# def battle_phase_hurt_and_faint(battle_obj, phase, teams, pet_priority, phase_dict):
#     phase_list = phase_dict[phase]
#     pp = pet_priority
#     status_list = []
#     while True:
#         ### Get a list of fainted pets
#         fainted_list = []
#         for team_idx, pet_idx in pp:
#             p = teams[team_idx][pet_idx].pet
#             if p.name == "pet-none":
#                 continue
#             if p.health <= 0:
#                 fainted_list.append([team_idx, pet_idx])
#                 if p.status != "none":
#                     status_list.append([p, team_idx, pet_idx])

#         ### Check every fainted pet
#         faint_targets_list = []
#         for team_idx, pet_idx in fainted_list:
#             fteam, oteam = get_teams([team_idx, pet_idx], teams)
#             fainted_pet = fteam[pet_idx].pet
#             ### Check for all pets that trigger off this fainted pet (including self)
#             for te_team_idx, te_pet_idx in pp:
#                 other_pet = teams[te_team_idx][te_pet_idx].pet
#                 te_idx = [te_team_idx, te_pet_idx]
#                 activated, targets, possible = other_pet.faint_trigger(
#                     fainted_pet, te_idx, oteam
#                 )
#                 if activated:
#                     faint_targets_list.append(
#                         [
#                             fainted_pet,
#                             te_team_idx,
#                             te_pet_idx,
#                             activated,
#                             targets,
#                             possible,
#                         ]
#                     )
#                 append_phase_list(
#                     phase_list,
#                     other_pet,
#                     te_team_idx,
#                     te_pet_idx,
#                     activated,
#                     targets,
#                     possible,
#                 )

#             ### If no trigger was activated, then the pet was never removed.
#             ###   Check to see if it should be removed now.
#             if teams[team_idx].check_friend(fainted_pet):
#                 teams[team_idx].remove(fainted_pet)
#                 ### Add this info to phase list
#                 phase_list.append(
#                     ("Fainted", (team_idx, pet_idx), (fainted_pet.__repr__()), [""])
#                 )

#         ### If pet was summoned, then need to check for summon triggers
#         for (
#             fainted_pet,
#             team_idx,
#             pet_idx,
#             activated,
#             targets,
#             possible,
#         ) in faint_targets_list:
#             fteam, _ = get_teams([team_idx, pet_idx], teams)
#             check_summon_triggers(
#                 phase_list,
#                 fainted_pet,
#                 team_idx,
#                 pet_idx,
#                 fteam,
#                 activated,
#                 targets,
#                 possible,
#             )

#         ### If pet was hurt, then need to check for hurt triggers
#         hurt_list = []
#         for team_idx, pet_idx in pp:
#             fteam, oteam = get_teams([team_idx, pet_idx], teams)
#             p = fteam[pet_idx].pet
#             while p._hurt > 0:
#                 hurt_list.append([team_idx, pet_idx])
#                 activated, targets, possible = p.hurt_trigger(oteam)
#                 append_phase_list(
#                     phase_list, p, team_idx, pet_idx, activated, targets, possible
#                 )

#         battle_obj.pet_priority = battle_obj.update_pet_priority(
#             battle_obj.t0, battle_obj.t1
#         )
#         pp = battle_obj.pet_priority

#         ### If nothing happend, stop the loop
#         if len(fainted_list) == 0 and len(hurt_list) == 0:
#             break

#     ### Check for status triggers on pet
#     for p, team_idx, pet_idx in status_list:
#         check_status_triggers(phase_list, p, team_idx, pet_idx, teams)

#     return phase_list
def find_pet_team_and_index(pet, teams):
    for team_idx, team in enumerate(teams):
        for pet_idx, slot in enumerate(team):
            if slot.pet == pet:
                return team_idx, pet_idx
    return None, None  # Return None if pet is not found


def check_friend_hurt_triggers(
    phase_list, hurt_pet, team_idx, friendly_team, opponent_team
):
    """Notify Wolverine-style abilities about one friendly hurt event."""

    for slot in list(friendly_team):
        if slot.empty:
            continue
        observer = slot.pet
        activated, targets, possible = observer.friend_hurt_trigger(
            hurt_pet, opponent_team
        )
        if activated:
            observer_idx = friendly_team.get_idx(observer)
            append_phase_list(
                phase_list,
                observer,
                team_idx,
                observer_idx,
                activated,
                targets,
                possible,
            )


_POST_REMOVAL_FAINT_EFFECTS = {
    "SummonPet",
    "SummonRandomPet",
    "SummonStoredPets",
    "RespawnPet",
}


def _is_post_removal_faint(pet):
    if pet.ability.get("trigger") != "Faint":
        return False
    return pet.ability.get("effect", {}).get("kind") in (
        _POST_REMOVAL_FAINT_EFFECTS
    )


def _flatten_pet_targets(value):
    pets = []
    if isinstance(value, (list, tuple)):
        for nested in value:
            pets.extend(_flatten_pet_targets(nested))
    elif type(value).__name__ == "Pet":
        pets.append(value)
    return pets


def _flush_post_removal_events(battle_obj, teams, phase_list):
    """Resolve SAP-Calculator priorities 25 then 26."""

    pending = list(battle_obj._pending_post_removal)
    battle_obj._pending_post_removal = []
    if not pending:
        return

    # Native PostRemovalFaint and faint equipment share priority 25. Pet
    # source resolves before equipment for the same owner.
    for record in pending:
        pet = record["pet"]
        team_idx = record["team_idx"]
        pet_idx = record["pet_idx"]
        _, opponent_team = get_teams([team_idx, pet_idx], teams)
        if record["native_post"]:
            activated, targets, possible = pet.faint_trigger(
                pet,
                [team_idx, pet_idx],
                opponent_team,
                remove_before_effect=True,
                tiger_override=record["tiger"],
            )
            append_phase_list(
                phase_list,
                pet,
                team_idx,
                pet_idx,
                activated,
                targets,
                possible,
            )
            check_summon_triggers(
                phase_list,
                pet,
                team_idx,
                pet_idx,
                teams,
                activated,
                targets,
                possible,
            )
        if record["has_status"]:
            check_status_triggers(
                phase_list, pet, team_idx, pet_idx, teams
            )

    # Shark/Fly are PostRemovalFriendFaints priority 26. Only observers that
    # existed when the faint event was created may observe it.
    for record in pending:
        fainted_pet = record["pet"]
        team_idx = record["team_idx"]
        pet_idx = record["pet_idx"]
        friendly_team, opponent_team = get_teams(
            [team_idx, pet_idx], teams
        )
        for observer in record["post_observers"]:
            if (
                not friendly_team.check_friend(observer)
                or observer.health <= 0
            ):
                continue
            observer_idx = friendly_team.get_idx(observer)
            activated, targets, possible = observer.faint_trigger(
                fainted_pet,
                [team_idx, pet_idx],
                opponent_team,
                remove_before_effect=True,
            )
            append_phase_list(
                phase_list,
                observer,
                team_idx,
                observer_idx,
                activated,
                targets,
                possible,
            )
            check_summon_triggers(
                phase_list,
                observer,
                team_idx,
                observer_idx,
                teams,
                activated,
                targets,
                possible,
            )

def battle_phase_hurt_and_faint(battle_obj, phase, teams, pet_priority, phase_dict):
    phase_list = phase_dict[phase]
    records = []
    recorded_pets = set()

    # SAP-Calculator priorities 3-4 always preempt Faint. Recompute event
    # priority after every activation because Hurt effects can enqueue more
    # Hurt events or create additional deaths.
    while True:
        priority = battle_obj.update_pet_priority(
            battle_obj.t0, battle_obj.t1, battle_obj.rs
        )
        hurt_event = None
        for team_idx, pet_idx in priority:
            pet = teams[team_idx][pet_idx].pet
            if pet.name != "pet-none" and pet._hurt > 0:
                hurt_event = (team_idx, pet_idx, pet)
                break
        if hurt_event is not None:
            team_idx, pet_idx, pet = hurt_event
            friendly_team, opponent_team = get_teams(
                [team_idx, pet_idx], teams
            )
            activated, targets, possible = pet.hurt_trigger(opponent_team)
            append_phase_list(
                phase_list,
                pet,
                team_idx,
                pet_idx,
                activated,
                targets,
                possible,
            )
            check_friend_hurt_triggers(
                phase_list,
                pet,
                team_idx,
                friendly_team,
                opponent_team,
            )
            continue

        new_dead = None
        for team_idx, pet_idx in priority:
            pet = teams[team_idx][pet_idx].pet
            if (
                pet.name != "pet-none"
                and pet.health <= 0
                and pet not in recorded_pets
            ):
                new_dead = (team_idx, pet_idx, pet)
                break
        if new_dead is None:
            break

        team_idx, pet_idx, pet = new_dead
        friendly_team, opponent_team = get_teams(
            [team_idx, pet_idx], teams
        )
        recorded_pets.add(pet)
        listeners = [
            slot.pet
            for slot in friendly_team
            if not slot.empty and slot.pet is not pet
        ]
        record = {
            "pet": pet,
            "team_idx": team_idx,
            "pet_idx": pet_idx,
            # SAP-Calculator snapshots the immediate support pet when the
            # Faint event is queued. Post-removal summon Faints execute only
            # after every simultaneous death has left the board, so looking
            # behind dynamically at execution time loses a valid Tiger.
            "tiger": _capture_tiger_support(pet),
            "support_chain": [
                slot.pet
                for index, slot in enumerate(friendly_team)
                if index > pet_idx and not slot.empty
            ],
            "native_post": _is_post_removal_faint(pet),
            "has_status": pet.status != "none",
            "listeners": listeners,
            "post_observers": [],
        }
        records.append(record)

        if not record["native_post"]:
            activated, targets, possible = pet.faint_trigger(
                pet,
                [team_idx, pet_idx],
                opponent_team,
                remove_before_effect=False,
                tiger_override=record["tiger"],
            )
            append_phase_list(
                phase_list,
                pet,
                team_idx,
                pet_idx,
                activated,
                targets,
                possible,
            )

    # Verified live-game exception to SAP-Calculator's strict event-time
    # snapshot: post-removal summon Faints skip all simultaneously dead
    # intervening pets. Therefore Deer -> Wolverine -> Tiger repeats Deer
    # when Deer and Wolverine die together. If the middle pet is Mammoth, its
    # own queued Faint is Tiger-repeated and Deer also reaches the same Tiger
    # after synchronized removal.
    dead_pets = {record["pet"] for record in records}
    for record in records:
        if not record["native_post"] or record["tiger"] is not None:
            continue
        for support in record["support_chain"]:
            if support.name == "pet-tiger":
                record["tiger"] = support
                break
            if support in dead_pets:
                continue
            break

    # FriendAheadFainted/FriendFaints priorities 10-11 observe the original
    # relation, after every priority-9 own Faint event but before removal.
    for record in records:
        fainted_pet = record["pet"]
        team_idx = record["team_idx"]
        pet_idx = record["pet_idx"]
        _, opponent_team = get_teams([team_idx, pet_idx], teams)
        for observer in record["listeners"]:
            if observer.name in {"pet-shark", "pet-fly"}:
                continue
            if not observer.team.check_friend(observer):
                continue
            observer_idx = observer.team.get_idx(observer)
            activated, targets, possible = observer.faint_trigger(
                fainted_pet,
                [team_idx, pet_idx],
                opponent_team,
                remove_before_effect=False,
            )
            append_phase_list(
                phase_list,
                observer,
                team_idx,
                observer_idx,
                activated,
                targets,
                possible,
            )

    # Synchronize deaths before priorities 25-26.
    for record in records:
        pet = record["pet"]
        team_idx = record["team_idx"]
        pet_idx = record["pet_idx"]
        friendly_team = teams[team_idx]
        if friendly_team.check_friend(pet):
            friendly_team.remove(pet)
            if getattr(phase_list, "fast_training", False):
                phase_list.append(None)
            else:
                phase_list.append(
                    ("Fainted", (team_idx, pet_idx), pet.__repr__(), [""])
                )

    for record in records:
        friendly_team = teams[record["team_idx"]]
        record["post_observers"] = [
            slot.pet
            for slot in friendly_team
            if not slot.empty and slot.pet.name in {"pet-shark", "pet-fly"}
        ]
        battle_obj._pending_post_removal.append(record)

    # Attack knockouts are priority 13 and must run before priorities 25-26.
    defer_post = phase in {"phase_hurt_and_faint_aa", "phase_hurt_and_faint_k"}
    if not defer_post:
        _flush_post_removal_events(battle_obj, teams, phase_list)

    battle_obj.pet_priority = battle_obj.update_pet_priority(
        battle_obj.t0, battle_obj.t1, battle_obj.rs
    )
    return phase_list


# def battle_phase_attack_before(battle_obj, phase, teams, pet_priority, phase_dict):
#     phase_list = phase_dict["phase_attack_before"]
#     aidx, nidx = get_attack_idx(phase, teams, pet_priority, phase_dict)
#     pp = pet_priority
#     if len(aidx) != 2:
#         ### Must be two animals available for attacking to continue with battle
#         return phase_list
#     for team_idx, pet_idx in pp:
#         if aidx[team_idx][1] != pet_idx:
#             ### Effects are only activated for the attacking pet
#             continue
#         p = teams[team_idx][pet_idx].pet
#         fteam, oteam = get_teams([team_idx, pet_idx], teams)
#         activated, targets, possible = p.before_attack_trigger(oteam)
#         append_phase_list(
#             phase_list, p, team_idx, pet_idx, activated, targets, possible
#         )

#     return phase_dict
def battle_phase_attack_before(battle_obj, phase, teams, pet_priority, phase_dict):
    # print("\n--- Starting battle_phase_attack_before ---")
    phase_list = phase_dict["phase_attack_before"]
    aidx, nidx = get_attack_idx(phase, teams, pet_priority, phase_dict)
    pp = pet_priority
    # print(f"Attack index: {aidx}")
    # print(f"Next index: {nidx}")
    if len(aidx) != 2:
        # print("Not enough animals for attacking, returning")
        ### Must be two animals available for attacking to continue with battle
        return phase_list
    for team_idx, pet_idx in pp:
        if aidx[team_idx][1] != pet_idx:
            ### Effects are only activated for the attacking pet
            continue
        p = teams[team_idx][pet_idx].pet
        fteam, oteam = get_teams([team_idx, pet_idx], teams)
        # print(f"Checking before_attack_trigger for {p} from team {team_idx}")
        activated, targets, possible = p.before_attack_trigger(oteam)
        # print(f"Before attack trigger - Activated: {activated}, Targets: {targets}")
        append_phase_list(
            phase_list, p, team_idx, pet_idx, activated, targets, possible
        )

    # print("--- Ending battle_phase_attack_before ---")
    return phase_dict


def get_attack_idx(phase, teams, pet_priority, phase_dict):
    """
    Helper function to get the current animals participating in the attack.
    These are defined as the first animals in each team that have a health above
    zero.
    """
    ### Only check for the first target
    ### Ff there is no target it means the target fainted in the 'before_attack' phase
    if not teams[0][0].empty and teams[0][0].pet.health > 0:
        t0_idx = 0
    else:
        t0_idx = -1

    if not teams[1][0].empty and teams[1][0].pet.health > 0:
        t1_idx = 0
    else:
        t1_idx = -1

    ret_idx = []
    if t0_idx > -1:
        ret_idx.append((0, t0_idx))
    if t1_idx > -1:
        ret_idx.append((1, t1_idx))

    ### Getting next idx at the same time
    t0_next_idx = -1
    for iter_idx, temp_slot in enumerate(teams[0]):
        if not temp_slot.empty:
            if temp_slot.pet.health > 0:
                if t0_idx == iter_idx:
                    continue
                t0_next_idx = iter_idx
                break
    t1_next_idx = -1
    for iter_idx, temp_slot in enumerate(teams[1]):
        if not temp_slot.empty:
            if temp_slot.pet.health > 0:
                if t1_idx == iter_idx:
                    continue
                t1_next_idx = iter_idx
                break
    ret_next_idx = []
    if t0_next_idx > -1:
        ret_next_idx.append((0, t0_next_idx))
    else:
        ret_next_idx.append(())
    if t1_next_idx > -1:
        ret_next_idx.append((1, t1_next_idx))
    else:
        ret_next_idx.append(())

    return ret_idx, ret_next_idx


# def battle_phase_attack_after(battle_obj, phase, teams, pet_priority, phase_dict):
#     phase_list = phase_dict[phase]
#     pp = pet_priority

#     #### Can get the two animals that just previously attacked from the
#     ####   phase_dict
#     attack_history = phase_dict["phase_attack"]
#     if len(attack_history) == 0:
#         return phase_dict

#     t0_pidx = attack_history[0][1][0]
#     t1_pidx = attack_history[0][1][1]

#     for team_idx, pet_idx in pp:
#         ### Check if current pet is directly behind the pet that just attacked
#         test_idx = [t0_pidx, t1_pidx][team_idx] + 1
#         if pet_idx != test_idx:
#             continue

#         ### If it is, then the after_attack ability can be activated
#         p = teams[team_idx][pet_idx].pet
#         fteam, oteam = get_teams([team_idx, pet_idx], teams)
#         activated, targets, possible = p.after_attack_trigger(oteam)
#         append_phase_list(
#             phase_list, p, team_idx, pet_idx, activated, targets, possible
#         )

#     return phase_dict
def battle_phase_attack_after(battle_obj, phase, teams, pet_priority, phase_dict):
    # print("\n--- Starting battle_phase_attack_after ---")
    phase_list = phase_dict[phase]
    pp = pet_priority

    # print(f"Phase: {phase}")
    # print(f"Pet priority: {pp}")

    #### Can get the two animals that just previously attacked from the
    ####   phase_dict
    attack_history = phase_dict["phase_attack"]
    # print(f"Attack history: {attack_history}")
    if len(attack_history) == 0:
        # print("No attack history, returning phase_dict")
        return phase_dict

    t0_pidx = attack_history[0][1][0]
    t1_pidx = attack_history[0][1][1]
    # print(f"Team 0 pet index that attacked: {t0_pidx}")
    # print(f"Team 1 pet index that attacked: {t1_pidx}")

    for team_idx, pet_idx in pp:
        attacker_idx = [t0_pidx, t1_pidx][team_idx]
        if pet_idx not in {attacker_idx, attacker_idx + 1}:
            continue

        p = teams[team_idx][pet_idx].pet
        if p.ability["trigger"] != "AfterAttack":
            continue
        triggered_by = p.ability["triggeredBy"]["kind"]
        if pet_idx == attacker_idx and triggered_by != "Self":
            continue
        if pet_idx == attacker_idx + 1 and triggered_by != "FriendAhead":
            continue
        # print(f"Pet that can activate after_attack: {p}")
        fteam, oteam = get_teams([team_idx, pet_idx], teams)
        # print(f"Friend team: {fteam}")
        # print(f"Opponent team: {oteam}")
        activated, targets, possible = p.after_attack_trigger(oteam)
        # print(f"After attack trigger - Activated: {activated}, Targets: {targets}, Possible: {possible}")
        append_phase_list(
            phase_list, p, team_idx, pet_idx, activated, targets, possible
        )

    # print(f"Final phase list: {phase_list}")
    # print("--- Ending battle_phase_attack_after ---")
    return phase_dict



# def battle_phase_knockout(battle_obj, phase, teams, pet_priority, phase_dict):
#     phase_list = phase_dict[phase]

#     #### Get knockout list from the end of the phase_attack info and remove
#     ####   the knockout list from phase attack
#     attack_history = phase_dict["phase_attack"]
#     if len(attack_history) == 0:
#         return phase_dict
#     knockout_list = attack_history[-1]
#     phase_dict["phase_attack"] = phase_dict["phase_attack"][0:-1]

#     for apet, team_idx in knockout_list:
#         if apet.health > 0:
#             ### Need to loop to handle Rhino
#             fteam, oteam = get_teams([team_idx, 0], teams)
#             current_length = 0
#             while True:
#                 pet_idx = fteam.index(apet)
#                 activated, targets, possible = apet.knockout_trigger(oteam)
#                 append_phase_list(
#                     phase_list, apet, team_idx, pet_idx, activated, targets, possible
#                 )

#                 if not activated:
#                     ### Easy breaking condition
#                     break

#                 battle_phase(
#                     battle_obj,
#                     "phase_hurt_and_faint_k",
#                     teams,
#                     pet_priority,
#                     phase_dict,
#                 )

#                 if len(phase_dict["phase_hurt_and_faint_k"]) == current_length:
#                     ### No more recursion needed because nothing else fainted
#                     break
#                 else:
#                     ### Otherwise, something has been knockedout by Rhino
#                     ### ability and while loop should iterate again
#                     current_length = len(phase_dict["phase_hurt_and_faint_k"])

#     return phase_dict

def battle_phase_knockout(battle_obj, phase, teams, pet_priority, phase_dict):
    phase_list = phase_dict[phase]
    attack_history = phase_dict["phase_attack"]
    if len(attack_history) == 0:
        _flush_post_removal_events(battle_obj, teams, phase_list)
        return phase_dict
    knockout_queue = list(attack_history[-1])
    phase_dict["phase_attack"] = phase_dict["phase_attack"][0:-1]

    while knockout_queue:
        knockout_queue.sort(
            key=lambda event: (event["attack"], event["tie"]),
            reverse=True,
        )
        event = knockout_queue.pop(0)
        apet = event["pet"]
        team_idx = event["team_idx"]
        friendly_team, opponent_team = get_teams([team_idx, 0], teams)
        try:
            pet_idx = friendly_team.index(apet)
        except Exception:
            pet_idx = 0

        activated, targets, possible = apet.knockout_trigger(
            opponent_team, tiger_override=event["tiger"]
        )
        append_phase_list(
            phase_list,
            apet,
            team_idx,
            pet_idx,
            activated,
            targets,
            possible,
        )
        if not activated:
            continue

        killed_targets = []
        for target in _flatten_pet_targets(targets):
            if (
                target.team is opponent_team
                and target.health <= 0
                and target not in killed_targets
            ):
                killed_targets.append(target)

        # Newly queued Hurt/Faint events preempt remaining KnockOut events.
        battle_phase(
            battle_obj,
            "phase_hurt_and_faint_k",
            teams,
            pet_priority,
            phase_dict,
        )
        for _ in killed_targets:
            knockout_queue.append(
                _make_knockout_event(battle_obj, apet, team_idx)
            )

    _flush_post_removal_events(battle_obj, teams, phase_list)
    return phase_dict



def get_attack(p0, p1):
    """Ugly but works"""
    attack_list = [
        p1.get_damage(p0.attack_damage),
        p0.get_damage(p1.attack_damage),
    ]
    if p0.status in status.apply_once:
        p0.status = "none"
    if p1.status in status.apply_once:
        p1.status = "none"
    return attack_list


def get_splash_damage(target, amount=5):
    """Resolve Chili damage without consuming the target's attack perk."""

    damage = target.get_damage(amount)
    if target.status in {"status-coconut-shield", "status-melon-armor"}:
        target.status = "none"
    return damage


def _capture_tiger_support(pet):
    if pet.team is None or not pet.team.check_friend(pet):
        return None
    behind = pet.team.get_behind(pet, n=1)
    if not behind:
        return None
    support = behind[0].pet
    return support if support.name == "pet-tiger" else None


def _make_knockout_event(battle_obj, pet, team_idx):
    return {
        "pet": pet,
        "team_idx": int(team_idx),
        "tiger": _capture_tiger_support(pet),
        "attack": float(pet.attack),
        "tie": float(battle_obj.rs.random_sample()),
    }


def battle_phase_attack(battle_obj, phase, teams, pet_priority, phase_dict):
    phase_list = phase_dict["phase_attack"]
    aidx, nidx = get_attack_idx(phase, teams, pet_priority, phase_dict)
    if len(aidx) != 2:
        ### Must be two animals available for attacking to continue with battle
        return phase_list

    p0 = teams[0][aidx[0][1]].pet
    p1 = teams[1][aidx[1][1]].pet

    #### Implement food
    p0a, p1a = get_attack(p0, p1)

    teams[0][aidx[0][1]].pet.hurt(p1a)
    teams[1][aidx[1][1]].pet.hurt(p0a)
    # Peanut knocks out only a pet that took positive post-defense damage.
    # Melon/Coconut can therefore block a low-attack Peanut hit, matching the
    # pinned calculator rather than treating Peanut as synthetic 1000 attack.
    if p1.status == "status-poison-attack" and p1a > 0:
        p0._health = -p0._until_end_of_battle_health_buff
    if p0.status == "status-poison-attack" and p0a > 0:
        p1._health = -p1._until_end_of_battle_health_buff
    if getattr(phase_list, "fast_training", False):
        # attack-after needs the attacker indices, while the pet snapshots are
        # diagnostic only. Both attackers are in the physical front slot.
        phase_list.append(["Attack", aidx[0]])
    else:
        phase_list.append(["Attack", aidx[0], str(p0), [str(p1)]])

    ### Keep track of knockouts for rhino and hippo by:
    ###   (attacking_pet, team_idx)
    knockout_list = []
    if teams[0][aidx[0][1]].pet.health <= 0:
        knockout_list.append(_make_knockout_event(battle_obj, p1, 1))
    if teams[1][aidx[1][1]].pet.health <= 0:
        knockout_list.append(_make_knockout_event(battle_obj, p0, 0))

    ### Implement chili
    if p0.status == "status-splash-attack":
        original_attack = p0._attack
        original_tmp_attack = p0._until_end_of_battle_attack_buff
        original_status = p0.status
        p0._attack = 5
        p0._until_end_of_battle_attack_buff = 0
        if len(nidx[1]) != 0:
            pn1 = teams[1][nidx[1][1]].pet
            pn1.hurt(get_splash_damage(pn1))
            if not getattr(phase_list, "fast_training", False):
                phase_list.append(
                    ["splash", aidx[0], str(p0), [str(pn1)]]
                )

            if pn1.health <= 0:
                knockout_list.append(_make_knockout_event(battle_obj, p0, 0))

        p0.status = original_status
        p0._attack = original_attack
        p0._until_end_of_battle_attack_buff = original_tmp_attack

    if p1.status == "status-splash-attack":
        original_attack = p1._attack
        original_tmp_attack = p1._until_end_of_battle_attack_buff
        original_status = p1.status
        p1._attack = 5
        p1._until_end_of_battle_attack_buff = 0
        if len(nidx[0]) != 0:
            pn0 = teams[0][nidx[0][1]].pet
            pn0.hurt(get_splash_damage(pn0))
            if not getattr(phase_list, "fast_training", False):
                phase_list.append(
                    ["splash", aidx[1], str(p1), [str(pn0)]]
                )

            if pn0.health <= 0:
                knockout_list.append(_make_knockout_event(battle_obj, p1, 1))

        p1.status = original_status
        p1._attack = original_attack
        p1._until_end_of_battle_attack_buff = original_tmp_attack

    ### Add knockout list to the end of phase_list. This is later removed
    ###   in the knockout phase
    phase_list.append(knockout_list)

    return phase_dict
