"""Label keyframe sprites from observed opening formations only. Never access ResultingBoard."""
import cv2,numpy as np,json,sys,collections
from pathlib import Path
from read_frames import badges,D
from trace_frames import feature,ordered_input
cv2.setNumThreads(1)
pdata={int(p['Id']):p['Name'] for p in json.loads((D.parent.parent.parent/'SAP-Calculator/src/assets/data/perks.json').read_text())};pdata[41]='Bread'
def pkpatch(im,z):
 x,y=z['x'],z['y']
 a=im[y-231:y-186,x-29:x+19]
 if a.shape[:2]!=(45,48):return None
 return cv2.resize(a,(20,20),interpolation=cv2.INTER_AREA)
def compact(f):
 return cv2.resize(f,(32,32),interpolation=cv2.INTER_AREA) if f is not None else None
def anchors(d):
 run=json.load(open(d/'run.json'));b=json.load(open(D/'fixtures'/run['fixture']));inputs=[ordered_input(b,i) for i in (0,1)]
 specimens=[];counts=collections.Counter();hits=[]
 for j,(fn,t) in enumerate(run['frames'][:32]):
  im=cv2.imread(str(d/fn));bb=badges(im)
  for side in (0,1):
   sel=[z for z in bb if int(z['x']>=960)==side];ref=inputs[side]
   if not sel or len(sel)!=len(ref) or counts[side]>=5:continue
   flags=[(z['attack'],z['health'])==tuple(o[1:3]) for z,o in zip(sel,ref)]
   if sum(flags)<max(1,len(flags)-1):continue
   if counts[side] and j-hits[-1][1]<1:continue
   counts[side]+=1;hits.append((side,j))
   for z,o,ok in zip(sel,ref,flags):
    if not ok:continue
    f=compact(feature(im,z));p=pkpatch(im,z)
    if f is not None and p is not None:specimens.append((o[0],pdata.get(o[3],'none' if o[3] is None else '?'),f,p))
  if all(counts[i]>=5 for i in (0,1)):break
 return specimens,hits
def bank_arrays(specs):
 return ([r[0] for r in specs],np.array([r[2].flatten() for r in specs],dtype='float32'),
         [r[1] for r in specs],np.array([r[3].flatten() for r in specs],dtype='float32'))
def predict(a,labels,vec,threshold,margin):
 if a is None or not labels:return None
 v=a.flatten().astype(np.float32)[None,:]
 diff=cv2.batchDistance(v,vec,cv2.CV_32F,normType=cv2.NORM_L1,K=min(len(labels),30))
 ds,ids=diff
 best=labels[ids[0,0]];dist=ds[0,0]/(255*v.size)
 second=next((val/(255*v.size) for val,i in zip(ds[0],ids[0]) if labels[i]!=best),1.)
 if dist>threshold or (dist>threshold/3 and second-dist<margin):return None
 return best
def populate(d,globalbank):
 from video_read import unpack
 from trace_frames import process
 if not (d/'trace.json').exists():process(d)
 base=json.load(open(d/'trace.json'))
 if (d/'battle.webm').exists() and not (d/'video-index.json').exists():unpack(d)
 if (d/'video-index.json').exists():
  v=json.load(open(d/'video-index.json'));stages=v['index']
 else:stages=[dict(stage=r['stage'],seconds=r['time'],frame=r['frame']) for r in base['readout']]
 local,anchors_used=anchors(d)
 additional=[]
 if (D/'token-anchors.json').exists():
  for ent in json.load(open(D/'token-anchors.json')):
   im=cv2.imread(str(D/ent['frame']));items=badges(im)
   cand=min(items,key=lambda z:abs(z['x']-ent['x']))
   if abs(cand['x']-ent['x'])<15:
    f=compact(feature(im,cand))
    if f is not None:additional.append((ent['pet'],ent.get('perk','?'),f,pkpatch(im,cand)))
 names,vec,pk,pkvec=bank_arrays(local+globalbank+additional)
 lnames,lvec,_,_=bank_arrays(local+additional)
 battle=json.load(open(D/'fixtures'/json.load(open(d/'run.json'))['fixture']))
 reachable={o[0] for side in (0,1) for o in ordered_input(battle,side)}
 reachable.update(['Bus','Ram','Chick','Zombie Cricket','Bee','Dirty Rat','Zombie Fly'])
 if 'Spider' in reachable:reachable.update(['Badger','Camel','Dodo','Dog','Dolphin','Elephant','Giraffe','Ox','Rabbit','Sheep'])
 # Only manually witnessed positive perk icons are eligible for automatic labels.
 # Opening-board metadata alone sometimes spans a frame in which the icon was already consumed.
 verified=[];neg=[]
 for v,labels in {
   'validation-01':[(476,None),(655,'Cake')], 'validation-02':[(833,None)],
   'validation-03':[(474,None),(654,'Chili'),(833,None)],'validation-05':[(833,'Bread')],
   'validation-07':[(474,None),(654,'Mushroom'),(833,None)],'validation-08':[(833,None)],
   'validation-12':[(1083,'Bread')],'validation-17':[(654,None)]}.items():
  dd=D/'captures'/v
  if not (dd/'run.json').exists():continue
  im=cv2.imread(str(dd/json.load(open(dd/'run.json'))['terminal_frame']));bb=badges(im)
  for x,k in labels:
   if not bb:continue
   q=min(bb,key=lambda t:abs(x-t['x']))
   if abs(q['x']-x)>15:continue
   pa=pkpatch(im,q)
   if pa is not None:(verified if k else neg).append((k or 'unverified',pa))
 only=verified+neg;pk=[q[0] for q in only];pkvec=np.array([q[1].flatten() for q in only],dtype='float32')
 view=[]
 for s in stages:
  im=cv2.imread(str(d/s['frame']));items=badges(im)
  r=dict(stage=s['stage'],time=s['seconds'],frame=s['frame'],player=[],opponent=[])
  for z in items:
   f=compact(feature(im,z));n=predict(f,lnames,lvec,.095,.020) or predict(f,names,vec,.069,.024)
   if n not in reachable:n=None
   perk=predict(pkpatch(im,z),pk,pkvec,.022,.012)
   if perk=='unverified':perk=None
   r['player' if z['x']<960 else 'opponent'].append(dict(x=z['x'],attack=z['attack'],health=z['health'],pet=n,perk=perk))
  r['player'].reverse();view.append(r)
 base['readout']=view;base['source']='rendered movie' if (d/'video-index.json').exists() else 'rendered JPEG sequence'
 from decorate_trace import decorate
 base=decorate(d,base)
 (d/'trace.json').write_text(json.dumps(base,indent=2))
 from paired import pair
 pair(d)
def globals_from_inputs():
 ss=[]
 for x in sorted((D/'captures').glob('validation-*')):
  if (x/'run.json').exists():ss+=anchors(x)[0]
 return ss
if __name__=='__main__':
 import time
 manifest=sys.argv[1];stop=time.monotonic()+float(sys.argv[2]);jobs=json.load(open(D/(manifest+'-manifest.json')));g=globals_from_inputs()
 while time.monotonic()<stop:
  todo=0
  for j in jobs:
   d=D/'captures'/Path(j['file']).stem
   if (d/'enhanced-v2.done').exists():continue
   todo+=1
   if (d/'run.json').exists():
    populate(d,g);(d/'enhanced-v2.done').write_text('complete');print(d.name,flush=True)
  if not todo or (D/'halt').exists():break
  time.sleep(1.5)
