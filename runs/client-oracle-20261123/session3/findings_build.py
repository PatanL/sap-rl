"""Builds session3/findings.json from findings_rules.py (rule statements, capture globs) and attribution.json (real fights)."""
import json,glob,re
from pathlib import Path
H=Path(__file__).resolve().parent;D=H.parent/'session2'
def caps(*pats):
    out=[]
    for p in pats:out+=sorted(Path(x).name for x in glob.glob(str(D/'captures'/p)) if (Path(x)/'run.json').exists())
    return out
RULES=[]
def rule(id,question,statement,engine,change,supporting,contradicting=(),confidence='medium',real_fights=(),notes=''):
    RULES.append(dict(id=id,question=question,statement=statement,engine_does=engine,implied_engine_change=change,supporting_captures=list(supporting),contradicting_captures=list(contradicting),confidence=confidence,notes=notes))
exec((H/'findings_rules.py').read_text())
att=json.loads((H/'attribution.json').read_text()) if (H/'attribution.json').exists() else {}
man={j['id'][:8]:j for j in json.loads((D/'s3real-manifest.json').read_text()) if j['file'].startswith('s3f')}
def cap_of(i):
    for pre in ('s3f-','dev-'):
        if (D/'captures'/(pre+i)/'run.json').exists():return pre+i
    return None
for r in RULES:
    ex=[];co=[]
    for i,a in att.items():
        toks=re.split(r'[+/]',a['rule'])
        for t in toks:
            base=t.replace('-consistent','')
            if base==r['id'] or (base=='R5b' and r['id']=='R5'):
                (co if ('consistent' in a['rule'] or base=='R5b') else ex).append(i)
    r['real_fights_first_difference_matches_rule']=sorted(ex);r['real_fights_consistent_but_not_isolated']=sorted(co)
    r['supporting_captures']=r['supporting_captures']+[c for c in (cap_of(i) for i in sorted(ex+co)) if c]
    r['evidence_counts']=dict(constructed_captures=len([c for c in r['supporting_captures'] if c.startswith('c3-')]),real_fights=len(ex),real_fights_consistent=len(co),contradicting=len(r['contradicting_captures']))
fights=[]
for i,j in sorted(man.items()):
    a=att.get(i)
    fights.append(dict(battle=j['id'],set=j['set'],deterministic_in_engine=j['deterministic'],engine_wrong_winner=not j['outcomePossible'],capture=cap_of(i),rule=(a or {}).get('rule','NOT ANALYSED'),first_difference=(a or {}).get('first_difference'),note=(a or {}).get('note')))
un=[f for f in fights if f['rule'].startswith(('UNEXPLAINED','NOT'))]
(H/'findings.json').write_text(json.dumps(dict(generated='2026-11-23',client='official web client v0.48, guest session, battle/get fulfilled locally',engine='engine-v31 as pinned in session3/engine-pin (copy of the current sources, not edited)',rules=RULES,failing_fights=fights,unexplained=[f['battle'] for f in un],notes=NOTES),indent=1))
print(len(RULES),'rules;',len(fights),'fights;',len(un),'unexplained or not analysed')
