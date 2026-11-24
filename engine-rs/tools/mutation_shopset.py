"""Mutation test of the shopset-* trace family: does it notice a wrong `set_shop_item`?

A copy of the sources is made in <work dir>, `Seat::set_shop_item` in the copy is patched so that
the environment variable SAP_MUT selects one deliberate error, `envcheck` is built from the copy
(the library in target/ is not touched) and the trace family is replayed once per error. Every
error must be reported; SAP_MUT=0 (no error) must pass. Not part of tools/check_search.sh (one
extra build and thirteen replays, 5 to 10 minutes on cores 0-4).

Usage: python3 tools/mutation_shopset.py <work dir> [trace files; default env-golden/shopset-*.jsonl.gz]
"""
import glob, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.abspath(sys.argv[1])
TRACES = sys.argv[2:] or sorted(glob.glob(os.path.join(ROOT, "env-golden", "shopset-*.jsonl.gz")))
MUTATIONS = {
    0: "none (control: must pass)",
    1: "the placed pet holds a private copy of the shop stream instead of the stream itself",
    2: "the placed food holds a private copy of the shop stream",
    3: "a reward slot keeps its reward group",
    4: "a reward slot stays a `levelup` slot",
    5: "the shop's permanent bonus is not added",
    6: "a food slot keeps its previous cost",
    7: "the cached legal actions are not invalidated",
    8: "the frozen argument is ignored",
    9: "the bonus is not capped at 50",
    10: "every team pet and shop item is rebound to the shop stream (the whole `_connect_objects`)",
    11: "the freeze-toggled set is cleared",
    12: "the action counter is advanced",
}

os.makedirs(WORK, exist_ok=True)
for name in ("src", "data"):
    target = os.path.join(WORK, name)
    if os.path.isdir(target):
        for folder, _, files in os.walk(target):
            for file in files:
                os.remove(os.path.join(folder, file))
    shutil.copytree(os.path.join(ROOT, name), target, dirs_exist_ok=True)
for name in ("Cargo.toml", "Cargo.lock"):
    shutil.copy(os.path.join(ROOT, name), os.path.join(WORK, name))

path = os.path.join(WORK, "src", "env.rs")
source = open(path).read()
start = source.index("    pub fn set_shop_item(&mut self, slot: usize, item: ShopItem, frozen: bool) -> R<()> {")
end = source.index("    /// The seat's share of `ParallelSelfPlayEnv.redeterminize(seed)`")
body = source[start:end]


def patch(old, new):
    global body
    if body.count(old) != 1:
        raise SystemExit("set_shop_item changed; update tools/mutation_shopset.py:\n" + old)
    body = body.replace(old, new)


patch("""                self.touch();
                let p = self.new_pet(kind, 0, Rs::Shared);""", """                if m != 7 { self.touch(); }
                let rs = if m == 1 { Rs::Private(Box::new(self.rs.clone())) } else { Rs::Shared };
                let p = self.new_pet(kind, 0, rs);
                if m == 10 { self.connect_objects(); }""")
patch("""                pet.attack = (pet.attack + sa).min(50);
                pet.health = (pet.health + sh).min(50);""", """                if m != 5 { pet.attack = pet.attack + sa; pet.health = pet.health + sh; }
                if m != 9 { pet.attack = pet.attack.min(50); pet.health = pet.health.min(50); }""")
patch("""                s.ty = SlotType::Pet;
                s.reward_group = None;""", """                if m != 4 { s.ty = SlotType::Pet; }
                if m != 3 { s.reward_group = None; }""")
patch("""                s.frozen = frozen;
                s.obj = Obj::Pet(p);""", """                if m != 8 { s.frozen = frozen; }
                s.obj = Obj::Pet(p);""")
patch("""                self.touch();
                let food = self.new_food(kind, Rs::Shared);""", """                if m != 7 { self.touch(); }
                let food = self.new_food(kind, if m == 2 { Rs::Private(Box::new(self.rs.clone())) } else { Rs::Shared });
                if m == 10 { self.connect_objects(); }""")
patch("""                s.cost = food.cost;
                s.frozen = frozen;""", """                if m != 6 { s.cost = food.cost; }
                if m != 8 { s.frozen = frozen; }""")
patch("""        let t: &'static Tables = self.t;
        let pool = &t.e.turns[11];""", """        let t: &'static Tables = self.t;
        let m: u32 = std::env::var("SAP_MUT").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if m == 11 { self.toggled.clear(); }
        if m == 12 { self.action_count += 1; }
        let pool = &t.e.turns[11];""")
open(path, "w").write(source[:start] + body + source[end:])

build = subprocess.run(["taskset", "-c", "0-4", "cargo", "build", "--release", "-j5", "--offline", "--bin", "envcheck"],
                       cwd=WORK, capture_output=True, text=True)
if build.returncode != 0:
    raise SystemExit(build.stderr[-3000:])
binary = os.path.join(WORK, "target", "release", "envcheck")
failed = False
for number, what in MUTATIONS.items():
    run = subprocess.run(["taskset", "-c", "0-4", binary, "--threads", "5", "--report", "1"] + TRACES,
                         capture_output=True, text=True, cwd=ROOT, env={**os.environ, "SAP_MUT": str(number)})
    total = next((l for l in run.stdout.splitlines() if l.startswith("TOTAL")), "")
    ok, of = total.split()[2].split("/") if total else ("0", "0")
    first = next((l for l in run.stdout.splitlines() if l.startswith("MISMATCH")), "")
    first = first.split(" match ", 1)[1][:110] if first else ""
    good = (run.returncode == 0 and ok == of) if number == 0 else (run.returncode != 0 and ok != of)
    failed |= not good
    print(f"{'ok    ' if good else 'FAILED'} {number:2d} {what}: {int(of) - int(ok)} of {of} matches differ" + (f"; first: {first}" if first else ""))
print("MUTATION TEST FAILED" if failed else "ALL MUTATIONS NOTICED")
sys.exit(1 if failed else 0)
