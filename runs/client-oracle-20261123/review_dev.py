"""Evidence strips at earliest machine-flagged stages (no reference field is pasted into real images)."""
from pathlib import Path
from PIL import Image,ImageDraw
import json,sys
D=Path(__file__).resolve().parent/'session2'
def board(v):
 return ', '.join(f"{p['pet'] or '?'} {p['attack']}/{p['health']}" for p in v)
def record(j):
 d=D/'captures'/Path(j['file']).stem
 if not (d/'paired.json').exists():return None
 p=json.load(open(d/'paired.json'));rr=p['rows'];idx=next((i for i,v in enumerate(rr) if v['differences']),len(rr)-1)
 return d,rr,idx
def main(start,end):
 jobs=json.load(open(D/'dev-manifest.json'))[start:end];canvas=Image.new('RGB',(1080,190*len(jobs)),'white');txt=ImageDraw.Draw(canvas)
 for ix,j in enumerate(jobs):
  q=record(j)
  if not q:continue
  d,rr,i=q;r=rr[i];e=r['engine'];im=Image.open(d/r['real']['frame']).crop((0,480,1920,822));im.thumbnail((880,155))
  y=ix*190;canvas.paste(im,(0,y+15))
  txt.text((2,y),d.name+' '+r['stage']+'  '+' | '.join(r['differences'][:3]),fill='black')
  if e:txt.text((2,y+170),'Engine F->B: '+board(e['player'])+'   vs   '+board(e['opponent']),fill='black')
 canvas.save(D/'dev-review.jpg')
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))
