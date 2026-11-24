from pathlib import Path
import json
from read_frames import D,badges
from physical import physical
import cv2
cases=[('front-deer',13,'after-attack-01',['Deer','Bus']),('blast-middle',16,'after-attack-01',['Bus','Deer','Rabbit','Duck']),('snipe-gaps',14,'start',['Beaver','Bus','Deer','Pig'])]
out=[]
for name,i,stage,reading in cases:
 path=D/'captures'/('custom-'+name)/f'frame-{i:04d}.jpg';p=sorted((z for z in badges(cv2.imread(str(path))) if z['x']<960),key=lambda z:-z['x'])
 assert len(p)==len(reading)
 observed=[[physical(z['x']),species,z['attack'],z['health']] for z,species in zip(p,reading)]
 native=next(z['player'] for z in json.load(open(D/'engine-traces'/('custom-'+name+'.json'))) if z['stage']==stage)
 src=[(q[0],q[1][4:].title(),q[2],q[3]) for q in native]
 out.append(dict(test=name,stage=stage,real=observed,native=native,match=[tuple(row) for row in observed]==src,frame=path.relative_to(D).as_posix()))
(D/'placement-results.json').write_text(json.dumps(out,indent=2));print([(v['test'],v['match']) for v in out])
