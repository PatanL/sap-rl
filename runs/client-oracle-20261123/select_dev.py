"""Extract current failure rows after the dev split, prioritizing Wolverine/Tiger and small boards."""
import json
from pathlib import Path
from select_fights import D,R,dev,h
def main():
 bad={v['battleId']:v for v in json.load(open(R/'v31p-result-340.json'))['rows'] if not(v['outcomePossible'] and v['statePossible'])}
 fix={v['battleId']:v for v in json.load(open(R/'fixtures-340.json'))['fixtures'] if v['battleId'] in bad and dev(v['matchId'])}
 src=sorted((R/'raw').glob('*.json'))+sorted((R.parent/'expert-replay-ingest-20260926-v1/corpus/matches').glob('*/BUNDLE.json'))
 jobs=[]
 for path in src:
  for pers in json.load(open(path))['perspectives']['perspectives']:
   raw=pers['raw']
   if not dev(raw['MatchId']):continue
   for a in raw['Actions']:
    if not a.get('Battle'):continue
    b=json.loads(a['Battle']);bid=b['Id'];f=fix.get(bid)
    if not f:continue
    name='dev-'+bid[:8]+'.json';(D/'fixtures'/name).write_bytes(a['Battle'].encode())
    species=[p['name'] for p in f['player']+f['opponent']]
    jobs.append(dict(file=name,id=bid,match=raw['MatchId'],seed=b['Seed'],sha256=h(a['Battle']),priority=bool({'Tiger','Wolverine'}&set(species)),pets=len(species),deterministic=bad[bid]['deterministic']))
 assert len(jobs)==len(fix)==len({j['id'] for j in jobs})
 jobs.sort(key=lambda j:(not j['priority'],j['pets'],not j['deterministic'],j['id']))
 (D/'dev-manifest.json').write_text(json.dumps(jobs,indent=2))
 print(len(jobs),sum(v['priority'] for v in jobs))
if __name__=='__main__':main()
