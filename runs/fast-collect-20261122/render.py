"""Generate the audit report from immutable timing results."""
import json
import statistics
from pathlib import Path

import tables as t


def full(n):
    r = t.cases[n]
    return {
        "matches_s": r["matches_s"],
        "seat_decisions_s": r["seat_s"],
        "accepted_decisions_s": r["accepted_s"],
        "learner_decisions_s": t.L(n),
        "projected_learned_decisions_s": t.rate(n),
        "wall_percent": dict(zip(("engine", "observation", "forward", "wait", "other"), t.split(r))),
    }


def overhead(model):
    first = t.actor(model, 1, "bf16")
    last = t.actor(model, 16, "bf16")
    return (16 * first - last) / 15


chosen = "pipe96-mid"
r = t.cases[chosen]
c = full(chosen)
l = t.actual(chosen)
b1 = t.actor("parent", 1, "bf16")
b16 = t.actor("parent", 16, "bf16")
raw = t.read(chosen + "-update")
summ = {
    "chosen": chosen,
    "platform": "NVIDIA GB10, eight permitted CPU cores",
    "parameters": r["parameters"],
    "parent_effective_fixed_ms": overhead("parent"),
    "actor_comparison_7_to_13": (13 / t.actor("parent", 13, "bf16")) / (7 / t.actor("parent", 7, "bf16")),
    "four_core_batching_collection_ratio": t.cases["4c-wide-parent"]["accepted_s"] / t.cases["4c-narrow-parent"]["accepted_s"],
    "four_core_batching_projected_complete_ratio": t.rate("4c-wide-parent") / t.rate("4c-narrow-parent"),
    "chosen_main_policy_batch": r["policyBatchMeans"][0],
    "outcomes": {n: full(n) for n in t.cases},
    "ceiling_with_public_observations": None,
}
top = (f"**For a fresh run on the GB10, I would start with 1,349,741 parameters and eight actor processes, "
       f"96 paired games each, split into two cohorts of four.** The randomly initialized policy completed "
       f"**{r['matches_s']:.2f} matches/s**, produced **{r['seat_s']:,.0f} seat decisions/s**, and delivered "
       f"**{r['accepted_s']:,.0f} learner-eligible decisions/s**. The native learner separately processed "
       f"**{l[0]:,.0f} decisions/s** on those sampled turns. Together they project to **{t.rate(chosen):,.0f} learned decisions/s**. "
       f"This is a throughput recommendation. Only training on the corrected engine could establish the network capacity needed for playing strength.")
floor = (f"**Small-batch cost.** The 14.35M BF16 actor took {b1:.2f} ms at B=1 and {b16:.2f} ms at B=16. "
         f"A local linear fit to those two points gives an effective fixed cost of **{overhead('parent'):.2f} ms per call**, about {100*overhead('parent')/b16:.0f}% of B=16 latency. "
         f"For 1.35M it was {overhead('mid'):.2f} ms and for 0.39M it was {overhead('small'):.2f} ms. "
         f"These are empirical latency floors covering dispatch, GPU kernels and poor small-batch occupancy. "
         f"The difference between B=1 and B=16 is within run-to-run noise. "
         f"At B=7 and B=13 the parent took {t.actor('parent',7,'bf16'):.2f} and {t.actor('parent',13,'bf16'):.2f} ms, "
         f"or {summ['actor_comparison_7_to_13']:.2f}× as many forward decisions/s. "
         f"The historical August collection comparison of 826 to about 1,000/s is ~1.21× and covers work outside the network.")
pr = t.P / "fixed-profile.json"
if pr.exists():
    prof = json.loads(pr.read_text())
    if prof.get("available"):
        x = [v for v in prof["rows"] if v["model"] == "parent" and v["batch"] == 1]
        active = statistics.median(v["deviceActiveMs"] for v in x)
        elapsed = statistics.median(v["profiledWallMs"] for v in x)
        count = statistics.median(v["deviceEvents"] for v in x)
        floor += (f" In a separate instrumented B=1 run, the union of device kernels and copies occupied {active:.2f} ms "
                  f"of a {elapsed:.2f} ms profiled call (medians of five, {count:.0f} device events). "
                  f"Profiler overhead changes the wall time, so subtracting these figures from the unprofiled 4.64 ms would be invalid.")
cross = (f"On the recommended run an average request wave carried **{r['mean_items_per_wave']:.0f} seats**; "
         f"mean forward batches were **{r['policyBatchMeans'][0]:.0f}** for the main policy and "
         + ", ".join(f"{v:.0f}" for v in r["policyBatchMeans"][1:]) +
         f" for the three residents. The **{int(r['totals']['matches']):,}** matches and **{int(r['totals']['accepted']):,}** delivered rows "
         f"were measured over {r['seconds']:.2f} s. "
         f"The sampled learner batch had {raw[0]['rows']:,} decisions in {raw[0]['turns']} turns, with "
         f"{statistics.median(raw[0]['seconds'])*1000:.1f} ms per standard update. "
         f"Using the optional batched loss on that same batch gave {l[1]:,.0f} learner decisions/s and would project to "
         f"{t.harmonic(r['accepted_s'],l[1]):,.0f} learned/s. The headline keeps the production loss path. "
         f"For a sensitivity check, replacing the measured turn distribution with the uniform length-eight B=1,024 microbenchmark "
         f"gives {t.micro_rate(chosen):,.0f} learner decisions/s and {t.harmonic(r['accepted_s'],t.micro_rate(chosen)):,.0f} projected learned/s.")
decision = []
if "pipe96-small" in t.cases:
    small, medium = full("pipe96-small"), full("pipe96-medium")
    decision.append(f"**How far to shrink.** Under the same 8 × 96 two-cohort layout, the 0.390M, 1.350M and 4.467M designs project to "
                    f"**{small['projected_learned_decisions_s']:,.0f}**, **{t.rate(chosen):,.0f}** and **{medium['projected_learned_decisions_s']:,.0f}** "
                    f"learned decisions/s, respectively. The 4.47M model buys 3.3× the parameters at a "
                    f"{100*(1-t.rate('pipe96-medium')/t.rate(chosen)):.0f}% projected speed cost versus 1.35M. ")
    diff = 100*(t.rate("pipe96-small")/t.rate(chosen)-1)
    decision[-1] += (f"The 390K model has only 29% of the 1.35M model's parameters and its projected rate was {abs(diff):.0f}% "
                     + ("higher" if diff > 0 else "lower") +
                     ". Different random policies produce different turn lengths and simulator workloads, so these gaps combine model and trajectory effects.")
else:
    decision.append("The narrowest variant's larger-layout run was pending at report generation.")
if 'pipe96-parent' in t.cases:
    parent=full('pipe96-parent')
    decision.append(f"**Keeping the checkpoint is also viable for collection.** The 14.35M trained network on the same 8 × 96 layout completed "
                    f"{parent['matches_s']:.2f} matches/s and delivered {parent['accepted_decisions_s']:,.0f} accepted/s. Its sampled turns project to "
                    f"{parent['projected_learned_decisions_s']:,.0f} learned/s, {parent['projected_learned_decisions_s']/315.3:.1f}× the historical rate. "
                    f"The 1.35M model projects to {t.rate(chosen)/parent['projected_learned_decisions_s']:.1f}× that rate, with 9.4% of its parameters. "
                    f"The parent and smaller policy play different game distributions.")
decision.append(f"**Where Python takes over.** On the four-core parent comparison, the mean per-policy batch rose from 6 to 79. "
                f"Policy forward fell from 45.9% to 12.9% of collection wall, while engine mechanics rose from 41.0% to 73.8%. "
                f"On the recommended pipelined layout, mechanics plus public encoding account for "
                f"{sum(t.split(r)[:2]):.1f}% of the critical path against {t.split(r)[2]:.1f}% for forward. "
                f"There is no universal batch-size threshold across layouts or policies.")
caps = t.caps_table()
capfile = t.P / (chosen + "-env-cap.json")
if capfile.exists():
    cap = json.loads(capfile.read_text())
    summ["ceiling_with_public_observations"] = cap
    caps += (f"\n| {chosen} | Encoded | {cap['matches_s']:.2f} | {cap['seat_s']:,.0f} | {cap['eligible_s']:,.0f} | {cap['battles_s']:.1f} |\n\n"
             f"The eight-core ceiling on the recommended policy's **exact action tape** was **{cap['seat_s']:,.0f} seat decisions/s**, "
             f"**{cap['eligible_s']:,.0f} eligible decisions/s**, or **{cap['matches_s']:.2f} matches/s** with public observations. "
             f"The live collector reached {100*r['seat_s']/cap['seat_s']:.0f}% of that seat-decision rate. "
             f"The replay's eligible counter counts decisions when made, while the collector's accepted counter counts complete turns when delivered.")
decision.append("Absolute rates are one timed window per live configuration on a shared GB10. They measure fresh random play and a frozen trained checkpoint. "
                "The training distribution will evolve. Sampling quality, win rate, stability after weight publication and eventual strength remain unmeasured.")
repeat = "pipe96-mid-repeat"
if (t.P/(repeat+"-update.json")).exists():
    rr=t.read(repeat);lr=t.actual(repeat)[0]
    sr=t.harmonic(rr['accepted_s'],lr)
    summ["second_sampling_seed"]={"seed":rr['sampling_seed'],"matches_s":rr['matches_s'],"seat_s":rr['seat_s'],
                                  "accepted_s":rr['accepted_s'],"learner_s":lr,"projected_s":sr}
    cross += (f"\n\n**Second sampling seed, tape recording disabled.** With the same 1.35M weights and layout, seed {rr['sampling_seed']} completed "
              f"{rr['matches_s']:.2f} matches/s, generated {rr['seat_s']:,.0f} seat actions/s, and delivered {rr['accepted_s']:,.0f} accepted/s. "
              f"With {lr:,.0f} timed learner decisions/s, that projects to **{sr:,.0f} learned/s**, against {t.rate(chosen):,.0f} on the primary seed. "
              f"The second window lasted 60 s and omitted action-tape recording. Shared-machine load can also affect these rates.")
    decision[-1]=decision[-1].replace("one timed window per live configuration", "one timed window per configuration and a second sampling seed for the recommended layout")
    stockfile=t.P/(repeat+"-stock-update.json")
    if stockfile.exists():
        stock=json.loads(stockfile.read_text());summ["stock_grouping_update"]=stock
        if stock.get('oom'):
            cross += " The unmodified stock grouping of up to 128 turns exceeded the 1.5-GiB allocator cap on the repeat batch."
        else:
            ul=stock["decisions_s"]
            cross += (f" The unmodified stock learner grouping, up to 128 turns, fit this repeat batch and processed "
                      f"{ul:,.0f} decisions/s, projecting to {t.harmonic(rr['accepted_s'],ul):,.0f} learned/s on that seed.")
    samefile=t.P/(repeat+"-same-turns.json")
    if samefile.exists():
        same=json.loads(samefile.read_text())
        if len(same)==4:
            summ["same_turns_update"]=same
            seq=", ".join(f"{t.names[k]} at {v['decisions_s']:,.0f}/s" if 'decisions_s' in v else f"{t.names[k]} exceeded allocator budget" for k,v in same.items())
            cross += (f"\n\n**Holding learner data fixed.** Four randomly initialized policies processed exactly the same {next(iter(same.values()))['rows']:,} "
                      f"repeat-seed public decision rows and turn lengths with zero initial state of the appropriate width. "
                      f"The standard loss gave {seq}. The parent needed the 128-padded-row fallback, while all smaller models fit the 16-turn grouping. "
                      f"The 390K learner was {100*(same['small']['decisions_s']/same['mid']['decisions_s']-1):.0f}% faster than 1.35M on identical work. "
                      f"That learner-only advantage did not translate into a higher fresh self-play projection in the measured configurations.")
other="pipe96-mid-repeat-taped"
if (t.P/(other+"-update.json")).exists():
    rr=t.read(other);lr=t.actual(other)[0];sr=t.harmonic(rr['accepted_s'],lr)
    summ["matched_taped_repeat"]={"seed":rr['sampling_seed'],"matches_s":rr['matches_s'],"seat_s":rr['seat_s'],
                                  "accepted_s":rr['accepted_s'],"learner_s":lr,"projected_s":sr}
    top += (f" A second sampling seed under the same recording and timing settings measured **{rr['matches_s']:.2f} matches/s** "
            f"and projects to **{sr:,.0f} learned decisions/s**.")
    cross += (f"\n\n**Matched repeat.** Seed {rr['sampling_seed']} also ran with the primary 70-second window and action recording enabled. "
              f"It completed {rr['matches_s']:.2f} matches/s, generated {rr['seat_s']:,.0f} seat decisions/s, and delivered {rr['accepted_s']:,.0f} eligible decisions/s. "
              f"With {lr:,.0f} learner decisions/s on its own sampled turns, it projects to **{sr:,.0f} learned/s**. "
              f"The two matched collection windows cover throughput variability across seeds and shared-machine conditions.")
subs = {"TOP":top,"GPU":t.gpu_tables(),"FLOOR":floor,"COLLECT":t.collect_table(),"CROSS":cross,
        "WALL":t.wall_table(),"CAPS":caps,"DECISION":"\n\n".join(decision),
        "COMMAND":"./run.sh async_collector.py --model mid --workers 8 --games 96 --cohorts 2 --seconds 70 --tape --out repeat-fresh"}
report = (t.P/"report_template.md").read_text()
for key, value in subs.items():
    report = report.replace("@@"+key+"@@", value)
assert "@@" not in report
(t.P/"REPORT.md").write_text(report)
(t.P/"SUMMARY.json").write_text(json.dumps(summ, indent=2) + "\n")
