"""Look for visible real-client perk artwork; no assignment solely from the fixture perk."""
from pathlib import Path
import json,cv2,numpy as np
from read_frames import D,badges
cv2.setNumThreads(1)
def templates():
 refs=json.load(open(D/'perk-samples.json'));points={0:(80,123),2:(80,120),4:(79,133),5:(91,141),6:(94,127),8:(125,130),10:(157,132),12:(92,126),13:(32,124),14:(141,128)}
 out=[]
 for i,(u,v) in points.items():
  a=refs[i];im=cv2.imread(str(D/a['path']));x=a['x']-100+u;y=a['y']-270+v
  base=im[y-21:y+21,x-21:x+21]
  for sz in (8,12,16,21,25):
   tile=cv2.resize(base,(sz,sz),interpolation=cv2.INTER_AREA)
   if tile.std()<26:continue
   out.append((a['perk'],tile))
 return out
def identify(im,z,tiles):
 x,y=z['x'],z['y'];patch=im[max(y-239,0):y-45,max(0,x-77):min(1920,x+77)]
 if patch.size==0:return None
 sm=cv2.resize(patch,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
 scored={}
 for name,t in tiles:
  if t.shape[1]>sm.shape[1]:continue
  mat=cv2.matchTemplate(sm,t,cv2.TM_CCOEFF_NORMED);_,peak,_,(xx,yy)=cv2.minMaxLoc(mat)
  if peak<.95:continue
  obs=sm[yy:yy+t.shape[0],xx:xx+t.shape[1]]
  err=np.abs(t.astype('i2')-obs).mean()
  if err<13.0:scored[name]=max(peak,scored.get(name,0))
 return next(iter(scored)) if len(scored)==1 else None
def update(d,t):
 r=json.load(open(d/'trace.json'))
 for st in r['readout']:
  im=cv2.imread(str(d/st['frame']));box=badges(im)
  for pet in st['player']+st['opponent']:
   if pet.get('x') is None:continue
   z=min(box,key=lambda z:abs(z['x']-pet['x'])) if box else None
   if z and abs(z['x']-pet['x'])<18:
    p=identify(im,z,t)
    if p:
     pet['perk']=p;pet['perk_basis']='screen artwork match'
 (d/'trace.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':
 import sys,time
 t=templates();plan=json.load(open(D/'dev-manifest.json'));deadline=time.monotonic()+float(sys.argv[1])
 while time.monotonic()<deadline:
  remain=0
  for j in plan:
   d=D/'captures'/Path(j['file']).stem
   if (d/'perks.done').exists():continue
   remain+=1
   if (d/'refined.done').exists():update(d,t);(d/'perks.done').write_text('ok')
  if not remain or (D/'halt').exists():break
  time.sleep(2)
