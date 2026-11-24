import json,cv2
from read_frames import D,badges
from physical import physical
out=[]
for j in json.load(open(D/'validation-manifest.json')):
 b=json.load(open(D/'fixtures'/j['file']));d=D/'captures'/j['file'][:-5];r=json.load(open(d/'run.json'));bb=badges(cv2.imread(str(d/r['terminal_frame'])))
 read=[(4-physical(q['x']) if q['x']<960 else 5+physical(q['x'])) if physical(q['x']) is not None else None for q in bb]
 real=[i for i,q in enumerate(b['ResultingBoard']['Mins']['Items']) if q]
 out.append(dict(id=j['id'],rendered_slots=read,recorded_slots=real,match=read==real))
(D/'validation-slots.json').write_text(json.dumps(out,indent=2));print(sum(x['match'] for x in out),len(out))
