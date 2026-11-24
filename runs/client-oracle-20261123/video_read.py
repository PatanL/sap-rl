"""Convert a captured native canvas movie to auditable pre-impact and terminal JPEGs."""
import json,cv2,os
from pathlib import Path
import numpy as np
from trace_frames import damage
from read_frames import badges
cv2.setNumThreads(1)
def unpack(p):
 video=p/'battle.webm'
 if not video.exists():return None
 cap=cv2.VideoCapture(str(video)); marks=[];previous=-1.;last_y=None;stats=[];tick=0;start=0.;last_impact_t=-100.;fps=cap.get(cv2.CAP_PROP_FPS)
 while True:
  ok,im=cap.read()
  if not ok:break
  t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
  if t<=previous:t=previous+1/max(1,min(60,fps or 10))
  previous=t
  if im.shape[:2]!=(1080,1920):im=cv2.resize(im,(1920,1080))
  y=damage(im)
  if y is not None:
   if t-last_impact_t>1.15 or (last_y is not None and y-last_y>51):marks.append(tick)
   last_impact_t=t;last_y=y
  # Badge checks only near possible boundaries during a second pass.
  stats.append((t,y))
  tick+=1
 cap.release()
 if not marks or len(stats)<15:return None
 selected=set(); windows=[]
 for j,k in enumerate(marks):
  a=k-1
  while a>0 and stats[k][0]-stats[a][0]<1.10 and (j==0 or a>marks[j-1]+1):a-=1
  candidates=list(range(max(0,a),k)) or [k]
  windows.append(candidates);selected.update(candidates)
 cap=cv2.VideoCapture(str(video));pot={};final=None
 for i in range(len(stats)):
  ok,im=cap.read()
  if not ok:break
  if i in selected or i==len(stats)-1:
   if im.shape[:2]!=(1080,1920):im=cv2.resize(im,(1920,1080))
   if i==len(stats)-1:final=im
   if i in selected:
    bb=badges(im);positive=[q for q in bb if q['attack'] is not None and q['health'] is not None and q['health']>0]
    # Prefer full readable pairs allocated to stationary pet bodies. Select latest among equal-quality frames.
    score=len(positive)-.5*(len(bb)-len(positive))
    pot[i]=(score,cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,91])[1].tobytes())
 cap.release()
 dest=p/'video-rounds';dest.mkdir(exist_ok=True);index=[]
 for j,w in enumerate(windows):
  if not any(i in pot for i in w):continue
  ix=max((i for i in w if i in pot),key=lambda i:(pot[i][0],i))
  label='start' if not j else f'after-attack-{j:02d}';file=dest/(label+'.jpg');file.write_bytes(pot[ix][1])
  index.append(dict(stage=label,frame=file.relative_to(p).as_posix(),movie_frame=ix,seconds=stats[ix][0],next_damage=stats[marks[j]][0]))
 if final is not None:
  cv2.imwrite(str(dest/'final.jpg'),final);index.append(dict(stage='final',frame='video-rounds/final.jpg',movie_frame=len(stats)-1,seconds=stats[-1][0]))
 info=dict(movie_frames=len(stats),movie_seconds=stats[-1][0],candidate_impacts=len(marks),index=index)
 (p/'video-index.json').write_text(json.dumps(info,indent=2));return info
