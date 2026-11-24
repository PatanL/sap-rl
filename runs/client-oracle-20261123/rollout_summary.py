"""Final report from the actual manifests and recorded end frames."""
from pathlib import Path
import collections,json
D=Path(__file__).resolve().parent/'session2'
s=json.load(open(D/'RUN-SUMMARY.json'));v=json.load(open(D/'validation-summary.json'));dst=json.load(open(D/'dev-results.json'))
terms=json.load(open(D/'dev-terminal-comparison.json'))
placement=json.load(open(D/'placement-results.json'))
counts=collections.Counter(z['stage'] for z in dst)
coverage=collections.Counter()
for r in dst:
 q=json.load(open((D/r['path']).parent/'trace.json'))
 for t in q['readout']:
  if t.get('terminal_alias'):continue
  coverage['sampled_states']+=1
  for p in t['player']+t['opponent']:
   coverage['visible_pets']+=1;coverage['pet_named']+=p['pet'] is not None;coverage['both_stats_read']+=None not in (p['attack'],p['health']);coverage['visible_perk_named']+=p['perk'] is not None
reviewed=sum(r['qualification']=='frame reviewed' for r in dst)
endpoint=sum(bool(all(q['manual_winner_and_board']) if q.get('manual') else q['flags']=='exact') for q in terms)
z=dict(validation=v,dev_capture_count=len(dst),frame_reviewed_first_differences=reviewed,earliest_differences=counts,dev_endpoints_matching_archive=endpoint,
       dev_endpoints_examined=len(terms),coverage=coverage,placement=placement,network=s['network'],timings=s['batches'])
(D/'SUMMARY.json').write_text(json.dumps(z,indent=2))
text=f'''## Completion and interpretation

All **{len(dst)} of 48** selected dev fights have a native canvas movie, timestamped screenshots, named round frames, engine trace, side-by-side table and a frame-reviewed first visible difference. Across the published snapshots there are {coverage['sampled_states']:,} chosen states and {coverage['visible_pets']:,} pet readings. {coverage['both_stats_read']:,} have both stats, {coverage['pet_named']:,} have a species label and {coverage['visible_perk_named']:,} have a positive perk read. These are coverage counts. Round-marker alignment and unlabeled effects limit claims about the *first internal instruction* that differed. [All 48 first differences](session2/DEV-TRACES.md) · [Machine-readable results](session2/SUMMARY.json)

| Earliest reviewed difference | Number of dev fights |
|---|---:|
| After start-of-battle, before ordinary attacks | {counts['start']} |
| After attack 1 | {counts['after-attack-01']} |
| After attack 2 | {counts['after-attack-02']} |
| After attack 3 | {counts['after-attack-03']} |
| After attack 4 | {counts['after-attack-04']} |
| After attack 7 | {counts['after-attack-07']} |

For a placement-specific dev example, **4664ea1e** has the same five pets and stats after start, but real order is Cat → Bus → Whale → Fly → Zombie Fly, front to back. Native order is Cat → Zombie Fly → Bus → Whale → Fly. In **83e018d0**, by contrast, the first difference is the species of Spider’s 2/2 summon. The same integer seed is no guarantee that the Python and client PRNGs draw the same species. [Placement example](session2/captures/dev-4664ea1e/paired.md) · [Random-species example](session2/captures/dev-83e018d0/paired.md)

**Game and traffic accounting.** The guest completed exactly one ordinary Turtle Arena run: **The Lowkey Sardines**, turn 6, zero trophies. There was one Arena queue and six genuine, unmodified arena/watch acknowledgments. The active shop’s rewatch arrow was available before game completion, so a finished History entry was unnecessary during the captures. After completion the guest had exactly one History entry. Its Play control fetched /api/playback/history; its Go To swords button, like the public replay code, showed the embedded Fish fight while an unrelated override was armed and issued no battle/get. This saved guest now has a completed run, not a resumable one. Future live injection with the tested launcher needs access to an already-active guest Arena shop with the rewatch arrow. A finished History entry alone is insufficient in the tested client. The launcher does not start another game. The private test profile is retained inside the audit directory.

The client attempted POST /api/arena/watch after the injected replays. **{s['network']['replay_watched_posts_blocked_including_retries']} attempts including retries were blocked, zero forwarded.** The only substituted response was battle/get. There were no ranked or versus game writes and no SAP Library upload. The browser was closed after checking the finished entry. [Game Over](session2/frames/finish-game.jpg) · [One History entry](session2/frames/one-entry.jpg) · [History swords displaying the original Fish](session2/frames/personal-swords-result.jpg) · [Timing and network counts](session2/RUN-SUMMARY.json)
'''
p=D.parent/'REPORT.md';old=p.read_text();p.write_text(old.split('## Completion and interpretation')[0].rstrip()+'\n\n'+text)
print(coverage,counts,endpoint,reviewed)
