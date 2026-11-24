"""Replay the UI's identity/position ledger without using simulator mechanics."""
import json,collections as C
from pathlib import Path
P=Path(__file__).resolve().parent
def positions(s):return [v['id'] if v else None for v in s['board']]
def shape_trace(row,n):
 before=row['states'][str(n-1)];after=row['states'][str(n)];acts=row['actions'][str(n)]
 known={int(i):v[1] for i,v in row['known'].items()};board=positions(before)
 steps=[];count=C.Counter();pending_summon=False;issues=[]
 for j,a in enumerate(acts):
  t=a['t']
  if 'orders' in a:
   listed={i for i,pos in a['orders']};have=set(board)-{None}
   if listed!=have:
    count['order membership mismatch after pill' if pending_summon else 'order membership mismatch']+=1
   new=[None]*5
   for i,pos in a['orders']:new[pos]=i
   board=new;pending_summon=False
  op=str(t);i=a.get('id');target=a.get('target');target_before=None
  if t==6:
   x=a['x'];target_before=board[x]
   if i in board:
    op='move';old=board.index(i);board[old],board[x]=board[x],board[old]
   elif target_before is not None:
    op='buy_onto';count['onto known same' if i in known and known.get(target_before)==known[i] else 'onto known different' if i in known and target_before in known else 'onto unresolved']+=1
   else:
    op='buy';board[x]=i
  elif t==7:
   op='board_combine' if i in board else 'shop_combine'
   if i in board:board[board.index(i)]=None
   if target not in board:count['merge target not on board']+=1
  elif t==8:
   op='food'
   if a.get('spell') and a['spell']['name']=='Sleeping Pill':
    pending_summon=True
    if target in board:board[board.index(target)]=None
  elif t==9:
   op='sell'
   if i in board:board[board.index(i)]=None
   else:count['unknown sell target']+=1
  else:op={4:'start',5:'roll',11:'end',12:'name'}.get(t,str(t))
  steps.append([op,target_before])
 if board!=positions(after):count['final shape mismatch after pill' if pending_summon else 'final shape mismatch']+=1
 return steps,count
def go():
 counts=C.Counter();money=C.Counter();ex=[];diff_byop=C.Counter()
 for l in open(P/'seats.jsonl'):
  r=json.loads(l)
  for k,s in r['states'].items():
   n=int(k)
   if n<2 or str(n-1) not in r['states']:continue
   steps,issues=shape_trace(r,n);counts.update(issues);ops=C.Counter(x[0] for x in steps);counts.update(ops)
   cost=ops['roll']+3*sum(ops[i] for i in ('buy','buy_onto','shop_combine'))+sum(a['spell']['cost'] for a in r['actions'][k] if a['t']==8)
   money[cost-s['spent']]+=1
   if cost!=s['spent'] and len(ex)<10:ex.append([r['match'],r['seat'],n,ops,issues,cost,s['spent']])
 print(counts);print(money);print(ex[:3])
if __name__=='__main__':go()
