import json,statistics,collections
from pathlib import Path
D=Path(__file__).resolve().parent/'session2'
def main():
 res=json.load(open(D/'validation-results.json'));man=json.load(open(D/'validation-manifest.json'));net=[json.loads(x) for x in open(D/'network.jsonl')]
 assert len(man)==len(res)==20 and len(set(v['match'] for v in man))==20
 local={v['bodySHA'] for v in net if v['action']=='local'};assert all(v['sha256'] in local for v in man)
 hits=sum(v['outcome_match'] and v['multiset_match'] and v['order_match'] for v in res)
 times=[v['seconds'] for v in res];nav=[];correct=nstat=nspecies=known=0
 out=['| Fight | Winner on screen | Survivors read directly, screen left to right | Seconds |','|---|---|---|---:|']
 for v in res:
  s=v['fight'];r=json.load(open(D/'captures'/s/'run.json'));nav.append(r['seconds_with_navigation'])
  tr=json.load(open(D/'captures'/s/'trace.json'))['readout'][-1]
  p=list(reversed(tr['player']))+tr['opponent']
  if len(p)==len(v['read']):
   for q,(nm,at,hp) in zip(p,v['read']):
    nstat+=1;correct+=((q['attack'],q['health'])==(at,hp));known+=q['pet'] is not None;nspecies+=q['pet']==nm
  st=', '.join(f'{n} {a}/{h}' for n,a,h in v['read']) or 'empty'
  out.append(f'| [{s}](session2/{v["source"]}) | {v["side"]} | {st} | {v["seconds"]:.1f} |')
 doc=dict(tested=len(res),matched=hits,median_seconds=statistics.median(times),min_seconds=min(times),max_seconds=max(times),median_with_navigation=statistics.median(nav),
          outcome_counts=collections.Counter(v['side'] for v in res),visible_survivors=sum(len(v['read']) for v in res),
          diagnostic_only=dict(n=nstat,stat_correct=correct,species_assigned=known,species_correct=nspecies))
 (D/'validation-summary.json').write_text(json.dumps(doc,indent=2))
 paragraph=f'''## Twenty-fight recorded-seed validation

**{hits}/20** reproduced the recorded winner and the entire terminal list of survivors, in order, with exact attack and HP. Each used a distinct dev match and unaltered original Battle bytes. The selection was fixed by SHA-256 before play and excluded known engine misses; this checks the client as a reference, independently of the later failing-fight diagnoses. All 20 file hashes appear in the local-fulfillment receipt. Terminal observations were read visually, written to a separate ledger, and then compared with Outcome and ResultingBoard. [Frame readings](session2/terminal-manual.tsv) · [Per-fight comparison](session2/validation-results.json)

Median time from the verified rewatch click to the first blocked watched acknowledgment was **{doc['median_seconds']:.1f} s** (range {min(times):.1f}–{max(times):.1f}). Including ordinary reload and menu navigation, median wall time was {doc['median_with_navigation']:.1f} s per fixture. This is seed-exact endpoint replay on the tested v0.48 fights. It establishes a real-client reference for intermediate-frame inspection; recognition of each intermediate frame is a separate measurement.

'''+'\n'.join(out)+'\n'
 (D/'validation-section.md').write_text(paragraph)
 report=D.parent/'REPORT.md';s=report.read_text()
 if '## Twenty-fight recorded-seed validation' not in s:
  with report.open('a') as f:f.write('\n'+paragraph)
 print(doc)
if __name__=='__main__':main()
