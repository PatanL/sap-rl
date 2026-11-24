"""Within-fight appearance matching on pet-body patches, independent of endpoint data."""
from pathlib import Path
import json,numpy as np,cv2
from read_frames import D,badges
from trace_frames import ordered_input
cv2.setNumThreads(1)
def patch(im,z,side):
 x,y=z['x'],z['y']
 if x<69 or x>1850:return None
 crop=im[y-158:y-62,x-68:x+69];return cv2.resize(crop,(22,16),interpolation=cv2.INTER_AREA).astype('float32')
def gather(d):
 receipt=json.load(open(d/'run.json'));b=json.load(open(D/'fixtures'/receipt['fixture']))
 specs=[ordered_input(b,s) for s in (0,1)];samples=[[],[]]
 for fn,t in receipt['frames'][:30]:
  im=cv2.imread(str(d/fn));bd=badges(im)
  for s in (0,1):
   selected=[z for z in bd if int(z['x']>=960)==s];sp=specs[s]
   if len(sp)!=len(selected) or len(samples[s])>=len(sp)*9:continue
   ok=[tuple(q[1:3])==(z['attack'],z['health']) for q,z in zip(sp,selected)]
   if sum(ok)<max(1,len(sp)-1):continue
   for v,z,good in zip(sp,selected,ok):
    feat=patch(im,z,s)
    if good and feat is not None:samples[s].append((v[0],feat))
 return samples
def refine(d):
 j=json.load(open(d/'trace.json'));ss=gather(d)
 for row in j['readout']:
  im=cv2.imread(str(d/row['frame']));bs=badges(im)
  for side,k in enumerate(('player','opponent')):
   for p in row[k]:
    if p['pet'] is not None or not ss[side]:continue
    z=min(bs,key=lambda z:abs(z['x']-p['x'])) if bs else None
    if z is None or abs(z['x']-p['x'])>5:continue
    f=patch(im,z,side)
    if f is None:continue
    ls={}
    for n,r in ss[side]:
     score=np.abs(f-r).mean()/255.
     ls[n]=min(score,ls.get(n,1.))
    vals=sorted(ls.items(),key=lambda v:v[1])
    if vals[0][1]<.069 and (len(vals)==1 or vals[1][1]-vals[0][1]>.035):
     p['pet']=vals[0][0];p['identity_basis']='within-fight observed starting artwork'
 (d/'trace.json').write_text(json.dumps(j,indent=2))
 from paired import pair
 pair(d)
if __name__=='__main__':
 import sys,time
 manifest=sys.argv[1];stop=time.monotonic()+float(sys.argv[2]);jobs=json.load(open(D/(manifest+'-manifest.json')))
 while time.monotonic()<stop:
  todo=0
  for j in jobs:
   d=D/'captures'/Path(j['file']).stem
   if (d/'refined.done').exists():continue
   todo+=1
   if (d/'enhanced-v2.done').exists():refine(d);(d/'refined.done').write_text('ok');print(d.name,flush=True)
  if not todo or (D/'halt').exists():break
  time.sleep(1.5)
