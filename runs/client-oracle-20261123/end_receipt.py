from pathlib import Path
import json,statistics,collections,hashlib
D=Path(__file__).resolve().parent/'session2'
def main():
 net=[json.loads(l) for l in open(D/'network.jsonl')];seen=collections.Counter(r['bodySHA'] for r in net if r['action']=='local')
 result={'network':{},'batches':{}}
 for label in ('validation','control','dev','repeat','custom'):
  jobs=json.load(open(D/(label+'-manifest.json')));rs=[]
  for j in jobs:
   d=D/'captures'/Path(j['file']).stem;p=d/'run.json'
   if not p.exists():continue
   r=json.load(open(p));assert seen[r['sha256']] and r['sha256']==hashlib.sha256((D/'fixtures'/j['file']).read_bytes()).hexdigest()
   assert (d/r['terminal_frame']).is_file()
   rs.append(r)
  if rs:result['batches'][label]=dict(expected=len(jobs),captured=len(rs),median_seconds=statistics.median(r['seconds_to_terminal'] for r in rs),
           mean_seconds=statistics.mean(r['seconds_to_terminal'] for r in rs),range_seconds=[min(r['seconds_to_terminal'] for r in rs),max(r['seconds_to_terminal'] for r in rs)],
           median_with_navigation=statistics.median(r['seconds_with_navigation'] for r in rs),frames=sum(len(r['frames']) for r in rs))
 for label,cond in {
   'guest_success':lambda r:r.get('action')=='response' and r.get('op','').endswith('/user/register-guest') and r.get('status')==200,
   'ordinary_arena_queues':lambda r:r.get('action')=='response' and r.get('op','').endswith('/arena/queue') and r.get('status')==200,
   'real_arena_watches_allowed':lambda r:r.get('action')=='allow' and r.get('op','').endswith('/arena/watch') and r.get('mode')=='arena',
   'locally_fulfilled':lambda r:r['action']=='local',
   'replay_watched_posts_blocked_including_retries':lambda r:r['action']=='block' and r.get('method')=='POST' and r.get('op','').endswith('/arena/watch') and r.get('mode')=='replay',
   'replay_watched_posts_forwarded':lambda r:r['action']=='allow' and r.get('method')=='POST' and r.get('op','').endswith('/arena/watch') and r.get('mode')=='replay',
   'versus_or_ranked_game_writes':lambda r:r['action']=='allow' and r.get('method')=='POST' and any(x in r.get('op','') for x in ('/versus/','/ranked/'))
 }.items():result['network'][label]=sum(cond(r) for r in net)
 (D/'RUN-SUMMARY.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result))
if __name__=='__main__':main()
