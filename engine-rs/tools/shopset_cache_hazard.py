"""Does the reference's legality cache return stale actions after hand edits that reuse an address?"""
import gen_env_golden as G, gen_env_ghost as H, gen_env_shopset as S
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec as C
from sapai_gym1.rl.joint_action import JointAction, step_decisions
hits = tries = 0
for seed in range(40):
    env = ParallelSelfPlayEnv(); env.reset(seed=seed, paired_tape=True)
    player = env.core.agent
    name = player.shop.slots[0].obj.name
    step_decisions(env, [JointAction(C.decode(1 + 0 * 6 + 0), None), JointAction(C.decode(151), None)])   # buy slot 0 onto team 0
    keep = []
    S.place(player, 0, name, False, keep)                    # the same species: combining onto team slot 0 is legal
    before = G.legal_of(env)[0]
    assert 1 in before
    other = "pet-fish" if name != "pet-fish" else "pet-ant"
    S.place(player, 0, other, False, []); S.place(player, 0, other, False, [])     # displaced objects are freed
    stale = G.legal_of(env)[0]
    env.core._agent_actions_cache = None
    fresh = G.legal_of(env)[0]
    tries += 1; hits += stale != fresh
    if stale != fresh and hits == 1:
        print("seed", seed, "shop slot 0 now holds", player.shop.slots[0].obj.name, "team slot 0", player.team[0].pet.name,
              "| cached legal list still has BUY_PET(0 -> 0):", 1 in stale, "| recomputed:", 1 in fresh)
print(f"stale legal lists in {hits} of {tries} tries")
