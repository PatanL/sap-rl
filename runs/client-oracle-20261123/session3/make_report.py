"""Assemble the session-3 section of REPORT.md from findings.json, part_constructed.md, attribution.json, ACCOUNTING.json."""
import json,re,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parent
subprocess.run([sys.executable,str(H/'findings_build.py')],check=True)
subprocess.run([sys.executable,str(H/'accounting.py')],check=True,stdout=subprocess.DEVNULL)
F=json.loads((H/'findings.json').read_text());A=json.loads((H/'ACCOUNTING.json').read_text())['counts']
head=(H/'part_head.md').read_text();cons=(H/'part_constructed.md').read_text();tail=(H/'part_tail.md').read_text()
def cl(n):return f"[{n}](session2/captures/{n}/timeline.txt)"
fs=F['failing_fights'];ncons=sum('consistent' in f['rule'] or f['rule']=='R5b' for f in fs);nun=sum(f['rule'].startswith(('UNEXPLAINED','NOT')) for f in fs)
import glob as _g
ndiff=sum(r['implied_engine_change'].strip().lower().rstrip('.') not in ('none','none beyond r4') and not r['implied_engine_change'].startswith('None') for r in F['rules'])
head=head.replace('{NRULES}',str(len(F['rules']))).replace('{NDIFF}',str(ndiff)).replace('{NFIRST}',str(len(fs)-ncons-nun)).replace('{NCONS}',str(ncons)).replace('{NUNEXP}',str(nun)).replace('{NLOCAL}',str(A.get('battle_get_fulfilled_locally',0))).replace('{NCONSTR}',str(len([d for d in _g.glob(str(R/'session2/captures/c3-*')) if (Path(d)/'run.json').exists()])))
out=[head,'']
out+=['### Findings','',
 'One row per rule. "Constructed" counts captures of tests built for the rule (controls included); "real" counts currently failing fights whose first real difference is this rule, "+n" those where the rule fits but the difference was not isolated. No capture contradicts a listed rule. Statements, capture labels and notes are in [findings.json](session3/findings.json).','',
 '| Rule | Real game | Engine | Change the rule implies | Evidence | Confidence |','|---|---|---|---|---|---|']
for r in F['rules']:
    e=r['evidence_counts'];ev=f"{e['constructed_captures']} constructed; {e['real_fights']} real"+(f" +{e['real_fights_consistent']}" if e['real_fights_consistent'] else '')
    out.append(f"| {r['id']} ({r['question']}) | {r['statement']} | {r['engine_does']} | {r['implied_engine_change']} | {ev} | {r['confidence']} |")
out+=['',cons,'']
fights=F['failing_fights']
out+=['### The 64 failing fights','',
 f"All {len(fights)} fights that the engine cannot reproduce in 5,000 seeds were captured in the client with their original bytes (55 new, 9 repeated at normal speed). In all of them the client's terminal board agrees with the archived ResultingBoard, up to label lag and one misread digit (41 read as 4). Each capture directory holds `paired.md` (round-by-round table against the engine seed that follows the real fight longest), `timeline.txt` and the movie. D = the engine gives one result on every seed; W = the engine never gives the real winner.",'',
 '| Fight | Set | Flags | First real difference | Rule | What the frames show |','|---|---|---|---|---|---|']
KF=json.loads((H/'attribution_frames.json').read_text()) if (H/'attribution_frames.json').exists() else {}
order=sorted(fights,key=lambda f:(f['rule'].startswith(('UNEXPLAINED','NOT')),f['rule'],f['battle']))
for f in order:
    i=f['battle'][:8];c=f['capture'] or '';flags=('D' if f['deterministic_in_engine'] else '')+('W' if f['engine_wrong_winner'] else '')
    link=f"[{i}](session2/captures/{c}/paired.md)" if c else i
    kf=(KF.get(i) or {}).get('frame');note=(f['note'] or '')+(f" [key frame]({kf})" if kf else '')
    out.append(f"| {link} | {f['set']} | {flags} | {f['first_difference'] or ''} | {f['rule']} | {note} |")
import collections
cnt=collections.Counter()
for f in fights:cnt[f['rule']]+=1
un=[f for f in fights if f['rule'].startswith(('UNEXPLAINED','NOT'))]
out+=['',f"Counts by label: "+', '.join(f"{k} {v}" for k,v in sorted(cnt.items()))+'.','',
 f"**Still unexplained: {len(un)}.** "+'; '.join(f"{f['battle'][:8]}" for f in un)+'. All are fights with random effects; the notes in the table say what was seen.','']
acc=['### Accounting','',
 '| Item | Count |','|---|---:|',
 f"| Ordinary guest Arena games started (arena/queue 200) | {A.get('arena_queue_200 (games started)',0)} |",
 f"| Turns ended in that game (build/end 200) | {A.get('build_end_200 (turns ended)',0)} |",
 f"| Pets bought | {A.get('build_play_minion_200 (pets bought)',0)} |",
 f"| Genuine, unmodified arena/watch acknowledgments (the one ordinary battle) | {A.get('arena_watch_POST_forwarded_mode_arena',0)} |",
 f"| battle/get answered locally (fixtures; nothing sent) | {A.get('battle_get_fulfilled_locally',0)} |",
 f"| battle/get forwarded (the ordinary battle) | {A.get('battle_get_forwarded_mode_arena',0)} |",
 f"| arena/watch POSTs attempted in replay mode: blocked | {A.get('arena_watch_POST_blocked_mode_replay',0)} |",
 f"| arena/watch POSTs attempted in replay mode: forwarded | {A.get('arena_watch_POST_forwarded_mode_replay',0)} |",
 f"| user/login POSTs forwarded (the saved guest session signing in at each page load; unmodified, no typed credentials) | {A.get('POST_forwarded user/login [guest]',0)+A.get('POST_forwarded user/login [replay]',0)} |",
 f"| GET user/participation blocked in replay mode | {A.get('blocked GET user/participation [replay]',0)} |",
 f"| Requests to non-game hosts blocked (Unity analytics, itch.io) | {sum(v for k,v in A.items() if k.startswith('blocked non-game host'))} |",
 '',"Other POSTs forwarded, all in the one ordinary game before the anchor: "+', '.join(f"{k.split(' ')[1]} x{v}" for k,v in sorted(A.items()) if k.startswith('POST_forwarded') and '[arena]' in k)+'. No ranked, versus, registration, purchase or upload request was made. Full counts: [ACCOUNTING.json](session3/ACCOUNTING.json), computed from the lines of `session2/network.jsonl` after line 3677.','']
out+=acc+[tail]
text='\n'.join(out)
rep=(R/'REPORT.md').read_text();mark='\n## Session 3, November 23, 2026 (evening)'
base=rep.split(mark)[0].rstrip('\n')+'\n'
(R/'REPORT.md').write_text(base+'\n'+text.strip('\n')+'\n')
print('report written;',len(text),'chars')
