"""Emit per-fight reviewed comparisons and a linked dev-only index."""
from pathlib import Path
import collections,json,csv,re
D=Path(__file__).resolve().parent/'session2'
def main():
 plan=json.load(open(D/'dev-manifest.json'));manual={}
 for l in (D/'dev-manual.tsv').read_text().splitlines():
  k,r,n=l.split('\t',2);manual[k]=(r,n)
 results=[];rows=[];hist=collections.Counter()
 for j in plan:
  name=Path(j['file']).stem;d=D/'captures'/name;k=j['id'][:8]
  if not (d/'paired.json').exists():continue
  q=json.load(open(d/'paired.json'));stage=q['first_candidate'];manualrow=manual.get(k)
  if manualrow:stage,note=manualrow;quality='frame reviewed'
  else:
   cand=next((x for x in q['rows'] if x['stage']==stage),None)
   note='; '.join(cand['differences'][:4]) if cand else 'No difference identified in readable fields.'
   quality='candidate only'
  hist[stage]+=1
  rr=next((x for x in q['rows'] if x['stage']==stage),q['rows'][-1])
  link=rr['real']['frame']
  if k=='544f869c':link='frame-0015.jpg'
  txt=(d/'paired.md').read_text().split('\n## Frame review')[0]
  if manualrow:
   txt+='\n## Frame review\n\n['+str(stage)+']('+link+'). '+note+'\n'
  (d/'paired.md').write_text(txt)
  rows.append(f'| [{k}](captures/{name}/paired.md) | {j["pets"]} | {"yes" if j["priority"] else ""} | {"yes" if j["deterministic"] else ""} | {stage or "unresolved"} | [{quality}](captures/{name}/{link}) | {note} |')
  results.append(dict(id=j['id'],seed=j['seed'],priority=j['priority'],deterministic=j['deterministic'],stage=stage,qualification=quality,first_difference=note,path='captures/'+name+'/paired.md',evidence='captures/'+name+'/'+link,movie='captures/'+name+'/battle.webm'))
 (D/'DEV-TRACES.md').write_text('# Real-client and engine traces on development misses\n\nThe native side uses the original seed modulo 2³². The viewer uses the original signed seed. These generators and random target choices are not established to be equivalent. A first visible divergence localizes the difference; causation requires more evidence. Start is after start-of-battle effects. Stages marked as candidates depend on damage-marker alignment and OCR; the underlying movie and frames are authoritative. Each fight link includes the side-by-side boards.\n\n| Fight | Pets | Wolverine / Tiger | Engine deterministic | Earliest observed boundary | Evidence | What differed |\n|---|---:|---|---|---|---|---|\n'+'\n'.join(rows)+'\n')
 (D/'dev-results.json').write_text(json.dumps(results,indent=2))
 print(len(results),'of',len(plan),'reviewed',sum(r['qualification']=='frame reviewed' for r in results),dict(hist))
if __name__=='__main__':main()
