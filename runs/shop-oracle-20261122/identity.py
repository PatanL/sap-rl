"""Resolve UI buy-to-square versus merges using observed identities, never final stats."""
from collections import Counter
def positions(s):return [v['id'] if v else None for v in s['board']]
def insertion_hole(x,board,rule='near-left'):
 empty=[j for j,v in enumerate(board) if v is None]
 if not empty:return None
 if rule=='left':return max((j for j in empty if j<x),default=min(empty))
 if rule=='right':return min((j for j in empty if j>x),default=max(empty))
 return min(empty,key=lambda j:(abs(x-j),j if rule=='near-left' else -j))
def shift(board,x,hole):
 while hole!=x:
  src=hole+(1 if x>hole else -1);board[hole]=board[src];hole=src
 board[x]=None
def compile_turn(row,n,rule='near-left'):
 before=row['states'][str(n-1)];after=row['states'][str(n)];acts=row['actions'][str(n)]
 names={int(i):v[1] for i,v in row['known'].items()};board=positions(before)
 trace=[];counts=Counter();implied={};problems=[];summoned=False
 later=set(i for i in positions(after) if i is not None)
 future=[]
 for a in reversed(acts):
  future.append(later.copy())
  later.update(i for i,x in a.get('orders',[]))
  if a['t'] in (7,8,9):
   for field in ('target','id') if a['t']!=8 else ('target',):
    if a.get(field) is not None:later.add(a[field])
 future.reverse()
 for j,a in enumerate(acts):
  if 'orders' in a:
   ids={i for i,pos in a['orders']}
   if ids!=set(board)-{None}:
    counts['order membership after pill' if summoned else 'order membership mismatch']+=1
   board=[None]*5
   for i,pos in a['orders']:board[pos]=i
   summoned=False
  t=a['t'];i=a.get('id');target=a.get('target');dest=None;op=''
  if t==6:
   x=a['x'];dest=board[x]
   if i in board:op='move';old=board.index(i);board[old],board[x]=board[x],board[old]
   elif dest is None:op='buy';board[x]=i
   else:
    # Directly witnessed species wins. For an unnamed source, absence from all
    # later board references and the final board proves absorption unless a
    # subsequent pill cascade could independently have destroyed it.
    ns,nd=names.get(i),names.get(dest)
    opaque_cascade=any(v['t']==8 and v.get('spell',{}).get('name')=='Sleeping Pill' for v in acts[j+1:])
    if ns is None and nd and i not in future[j] and not opaque_cascade:
     names[i]=nd;implied[i]=dest;ns=nd
    merge=ns==nd if ns and nd else i not in future[j]
    if merge:
     op='buy merge'
     if i in future[j]:counts['known merge source later on board']+=1
    else:
     op='buy';hole=insertion_hole(x,board,rule)
     if hole is not None:shift(board,x,hole);board[x]=i
     else:problems.append('occupied buy destination on full board')
  elif t==7:
   op='board merge' if i in board else 'shop merge'
   if i in board:board[board.index(i)]=None
   if target not in board:counts['merge target missing']+=1
  elif t==8:
   op='food'
   if a['spell'] and a['spell']['name']=='Sleeping Pill':
    summoned=True
    if target in board:board[board.index(target)]=None
  elif t==9:
   op='sell'
   if i in board:board[board.index(i)]=None
   else:counts['sell id missing']+=1
  else:op={4:'start',5:'roll',11:'end',12:'name'}.get(t,str(t))
  trace.append((op,dest))
 if board!=positions(after):counts['final shape after pill' if summoned else 'final shape mismatch']+=1
 return trace,names,implied,counts,problems
if __name__=='__main__':
 import json
 for rule in ['left','right','near-left','near-right']:
  c=Counter()
  for line in open('seats.jsonl'):
   r=json.loads(line)
   if r['split']!='dev':continue
   for k in r['states']:
    n=int(k)
    if n<2 or str(n-1) not in r['states']:continue
    tr,_,im,counts,problems=compile_turn(r,n,rule);c.update(counts);c.update(problems);c.update(x[0] for x in tr);c['implied']+=len(im)
  print(rule,c)
