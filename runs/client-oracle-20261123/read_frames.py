"""Read independent battle stat glyphs. Anchor initial individuals only from submitted board input."""
import cv2,numpy as np,json,collections,sys,os,math
from pathlib import Path
cv2.setNumThreads(1)
D=Path(__file__).resolve().parent/'session2'
sour=D.parent.parent.parent/'sapai-gym/sapai_gym1/deployment/stat_ocr.py'
import importlib.util
spec=importlib.util.spec_from_file_location('board_digit_reader',sour);stats=importlib.util.module_from_spec(spec);sys.modules[spec.name]=stats;spec.loader.exec_module(stats)
archive=np.load(D.parent.parent/'live-vision/stat-digits-v1.npz')
tmpl=archive['templates'].astype(bool);labs=archive['labels']
def digit(mask):
 glyph=stats.normalize_glyph(mask).astype(bool);diff=np.logical_xor(tmpl,glyph).mean(axis=(1,2))
 minima=np.array([diff[labs==i].min() for i in range(10)]);best=minima.argsort()
 d,m=float(minima[best[0]]),float(minima[best[1]]-minima[best[0]])
 return (str(int(best[0])) if d<.14 and m>.010 else '?'),round(d,3),round(m,3)
def badges(im):
 roi=im[672:850];mask=(roi.min(2)>212).astype('uint8')
 _,labels,st,_=cv2.connectedComponentsWithStats(mask)
 large=[(i,*map(int,v[:4])) for i,v in enumerate(st) if 24<v[1]<99 and 135<v[2]<207 and 56<v[3]<90 and 1100<v[4]<6000]
 out=[]
 for i,x,y,w,h in sorted(large,key=lambda v:v[1]):
  # The joined outer white contours surround two black-edged badges.
  vals=[];scores=[]
  for side in (0,1):
   mid=x+w*(.26 if side==0 else .76)
   glyphs=sorted([(int(v[0]),j,v) for j,v in enumerate(st) if mid-37<v[0]+v[2]/2<mid+37 and y+14<v[1]<y+39 and 20<=v[3]<=41 and 5<=v[2]<=36 and 55<=v[4]<=1100],key=lambda z:z[0])
   text='';ds=[]
   for gx,j,(xx,yy,ww,hh,area) in glyphs:
    s,dd,margin=digit(labels[yy:yy+hh,xx:xx+ww]==j);text+=s;ds.append([dd,margin])
   vals.append(int(text) if text.isdigit() and 0<=int(text)<=50 else None);scores.append(ds)
  out.append(dict(x=x+w//2,y=y+672+h//2,attack=vals[0],health=vals[1],quality=scores))
 # A lunging pet can connect the two badges' white outline to its body.
 # Recover those pairs from the HP heart color and the matching gray attack badge.
 b,g,r=cv2.split(roi.astype('i2'))
 red=((r>110)&(g<.43*r)&(b<.43*r)).astype('uint8')
 red=cv2.morphologyEx(red,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
 _,_,reds,_=cv2.connectedComponentsWithStats(red)
 candidates=[(v['x'],v['y']-672) for v in out]
 for x,y,w,h,area in reds[1:]:
  if 48<=w<=74 and 32<=h<=60 and 36<=y<=97 and area>300:
   candidates.append((int(x+w/2-44),int(y+26)))
 for cx,cy in list(candidates):
  if cx<45 or cx>1874:continue
  for off in range(-4,5):
   px=int(round(cx+off*179.7));py=cy
   if not 46<px<1874 or (cx<960)!=(px<960) or any(abs(z['x']-px)<54 for z in out):continue
   hx,ax=px+44,px-41
   redpixels=red[max(0,py-23):py+23,hx-29:hx+29].sum()
   grey=roi[max(0,py-20):py+20,ax-27:ax+27].astype('i2')
   neutral=((grey.max(2)-grey.min(2)<17)&(grey.mean(2)>36)&(grey.mean(2)<142)).sum()
   if redpixels<260 or neutral<240:continue
   vals=[];scores=[]
   for mid in (ax,hx):
    glyphs=sorted([(int(v[0]),j,v) for j,v in enumerate(st) if abs(v[0]+v[2]/2-mid)<37 and abs(v[1]+v[3]/2-py)<8 and 21<=v[3]<=41 and 5<=v[2]<=36 and 55<=v[4]<=1100])
    text='';ds=[]
    for _,j,(xx,yy,ww,hh,area) in glyphs:
     ch,dd,mm=digit(labels[yy:yy+hh,xx:xx+ww]==j);text+=ch;ds.append([dd,mm])
    vals.append(int(text) if text.isdigit() and int(text)<=50 else None);scores.append(ds)
   if any(x is not None for x in vals):out.append(dict(x=px,y=py+672,attack=vals[0],health=vals[1],quality=scores))
 for z in out:
  if z['health'] is None or z['health']==0:continue
  # The minus is a short horizontal bright glyph, immediately left of the HP numerals.
  hx=z['x']+44;cy=z['y']-672
  if any(hx-34<x+w/2<hx-3 and 0<y+h/2-cy<11 and 8<=w<=20 and 3<=h<=9 and 23<=area<=120 for x,y,w,h,area in st):
   z['health']=-z['health']
 return sorted(out,key=lambda z:z['x'])
if __name__=='__main__':
 for f in sys.argv[1:]:
  print(f,[(v['x'],v['attack'],v['health'],v['quality']) for v in badges(cv2.imread(f))])
