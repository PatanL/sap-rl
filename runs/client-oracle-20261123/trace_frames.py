"""Pixel-only attack markers and candidate settled boards. Inputs label original sprite templates."""
import json,collections,math,sys,cv2,numpy as np
from pathlib import Path
from read_frames import D,badges
import shutil,os
cv2.setNumThreads(1)
pets={int(v['Id']):v['Name'] for v in json.loads((D.parent.parent.parent/'SAP-Calculator/src/assets/data/pets.json').read_text())}
def ordered_input(b,side):
 board=b['UserBoard' if side==0 else 'OpponentBoard']
 slots=sorted((x for x in board['Mins']['Items'] if x),key=lambda x:x.get('Poi',{}).get('x',0),reverse=bool(side))
 return [(pets.get(z.get('Enu',0),'?'),min(50,sum((z.get('At') or {}).get(k,0) for k in ['Perm','Temp'])),min(50,sum((z.get('Hp') or {}).get(k,0) for k in ['Perm','Temp'])),z.get('Perk')) for z in slots]
def feature(im,b):
 x,y=b['x'],b['y']
 if x<104 or x>1815:return None
 crop=im[y-237:y-25,x-103:x+104].copy();white=(crop.min(2)>211).astype('uint8')
 cs,_=cv2.findContours(white,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
 found=[]
 for c in cs:
  xx,yy,w,h=cv2.boundingRect(c)
  if not(38<w<200 and 40<h<201 and 50<xx+w/2<150 and yy+h>133):continue
  area=cv2.contourArea(c)
  if area>1500:found.append((area,c))
 if not found:return None
 c=max(found,key=lambda z:z[0])[1];xx,yy,w,h=cv2.boundingRect(c)
 mask=np.zeros(crop.shape[:2],np.uint8);cv2.drawContours(mask,[c],0,255,-1)
 crop=cv2.bitwise_and(crop,crop,mask=mask)[yy:yy+h,xx:xx+w]
 scale=60/max(w,h);ww=max(1,round(w*scale));hh=max(1,round(h*scale))
 out=np.zeros((64,64,3),np.uint8);out[(64-hh)//2:(64-hh)//2+hh,(64-ww)//2:(64-ww)//2+ww]=cv2.resize(crop,(ww,hh),interpolation=cv2.INTER_AREA)
 return out if x<960 else cv2.flip(out,1)
def distance(a,b):
 return float(np.abs(a.astype('i2')-b).mean()/255.)
def damage(im):
 bb,gg,rr=cv2.split(im[380:688,730:1205]);mask=((rr>195)&(gg<65)&(bb<60)).astype('uint8')
 _,_,st,_=cv2.connectedComponentsWithStats(mask)
 nums=[(x+w/2+730,y+380) for x,y,w,h,area in st if 14<w<78 and 47<h<100 and area>650 and area/(w*h)>.34]
 pairs=[(y+yy)/2 for x,y in nums for xx,yy in nums if 770<x<906 and 1000<xx<1150 and abs(y-yy)<21]
 return float(np.median(pairs)) if pairs else None
def classify(feat,bank):
 if feat is None or not bank:return None,None
 by={}
 for name,v in bank:
  d=distance(feat,v)
  if name not in by or d<by[name]:by[name]=d
 top=sorted(by.items(),key=lambda v:v[1]);label,d=top[0]
 margin=(top[1][1]-d) if len(top)>1 else 1
 if d>.115 or (d>.041 and margin<.025):return None,round(d,3)
 return label,round(d,3)
def process(dest):
 receipt=json.loads((dest/'run.json').read_text());b=json.loads((D/'fixtures'/receipt['fixture']).read_text())
 ins=[ordered_input(b,s) for s in (0,1)]
 states=[];bank=[];anchored=[False,False];anchors=[];tt=receipt['frames']
 feature_cache=[];impacted=False;anchor_counts=collections.Counter()
 for i,(fn,t) in enumerate(tt):
  im=cv2.imread(str(dest/fn));bd=badges(im);yv=damage(im)
  impacted=impacted or yv is not None
  features=[feature(im,z) for z in bd];feature_cache.append(features)
  for side in (0,1):
   if impacted or anchor_counts[side]>=8:continue
   sels=[(j,z) for j,z in enumerate(bd) if (z['x']>=960)==bool(side)]
   if not ins[side] or len(ins[side])!=len(sels):continue
   if all((z['attack'],z['health'])==tuple(src[1:3]) for (_,z),src in zip(sels,ins[side])) and all(features[j] is not None for j,z in sels):
    bank.extend((src[0],features[j]) for (j,z),src in zip(sels,ins[side]));anchored[side]=True;anchor_counts[side]+=1;anchors.append([side,i])
  states.append(dict(frame=i,file=fn,time=t,badges=bd,impactY=yv))
 for i,st in enumerate(states):
  for v,feat in zip(st['badges'],feature_cache[i]):
   name,score=classify(feat,bank);v['species']=name;v['sprite_distance']=score;v['perk']=None
 # Paired red damage numerals at the two front positions mark candidate impacts.
 attacks=[];last_t=-100;last_y=0
 for i,st in enumerate(states):
  y=st['impactY']
  if y is None:continue
  if st['time']-last_t>1.55 or y-last_y>58:attacks.append(i)
  last_t=st['time'];last_y=y
 def preceding(i,lo):
  candidates=range(max(lo,i-5),i)
  def sc(k):
   a=states[k]['badges'];return sum(q['attack'] is not None and q['health'] is not None for q in a)-.85*max(0,i-k-1)
  return max(candidates,key=sc) if candidates else max(0,i-1)
 keyframes=[]
 for k,at in enumerate(attacks):
  j=preceding(at,attacks[k-1]+1 if k else 0)
  keyframes.append((('start' if k==0 else 'after-attack-%02d'%k),j))
 keyframes.append(('final',len(states)-1))
 trace=[]
 out=dest/'rounds';out.mkdir(exist_ok=True)
 for label,i in keyframes:
  st=states[i];target=out/(label+'.jpg')
  if not target.exists():os.link(dest/st['file'],target)
  row=dict(stage=label,time=st['time'],source=st['file'],frame='rounds/'+target.name,
           status='terminal frame after simulation' if label=='final' else 'estimated boundary; verify adjacent frames',
           player=[],opponent=[])
  for z in st['badges']:
   side='player' if z['x']<960 else 'opponent'
   row[side].append(dict(pet=z['species'],attack=z['attack'],health=z['health'],perk=z['perk'],x=z['x'],distance=z['sprite_distance']))
  row['player'].reverse();trace.append(row)
 emit=dict(id=b['Id'],seed=b['Seed'],start_anchors=anchors,seconds=receipt['seconds_to_terminal'],attack_markers=attacks,
    sides='front to back; null means unreadable or unverified',readout=trace)
 (dest/'trace.json').write_text(json.dumps(emit,indent=2));(dest/'states.json').write_text(json.dumps(states,separators=(',',':')))
 return emit
if __name__=='__main__':
 import os
 for p in sys.argv[1:]:
  v=process(Path(p));print(p,len(v['attack_markers']),v['readout'][-1])
