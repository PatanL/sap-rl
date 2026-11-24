from rendered_boards import *
from PIL import ImageDraw,Image
def main():
 samples={};ids=json.load(open(D/'validation-manifest.json'))
 for j in ids:
  d=D/'captures'/Path(j['file']).stem;run=json.load(open(d/'run.json'));b=json.load(open(D/'fixtures'/j['file']))
  for fn,t in run['frames'][3:12]:
   im=cv2.imread(str(d/fn));bb=badges(im)
   for s in (0,1):
    items=[q for q in bb if int(q['x']>=960)==s];ai=ordered_input(b,s)
    if len(items)!=len(ai):continue
    for z,(name,a,h,pk) in zip(items,ai):
     if not pk or (a,h)!=(z['attack'],z['health']) or z['x']<105 or z['x']>1815:continue
     k=(pdata.get(pk,str(pk)),s)
     if k not in samples:samples[k]=(d/fn,z)
 ks=sorted(samples);out=Image.new('RGB',(250*4,270*((len(ks)+3)//4)),'white');txt=ImageDraw.Draw(out);refs=[]
 for i,k in enumerate(ks):
  path,z=samples[k];x,y=z['x'],z['y'];crop=Image.open(path).crop((x-100,y-270,x+100,y-20));cx,cy=i%4*250,i//4*270
  out.paste(crop,(cx,cy+20));txt.text((cx,cy),str(i)+' '+str(k),fill='black')
  refs.append(dict(id=i,perk=k[0],side=k[1],path=str(path.relative_to(D)),x=x,y=y))
 out.save(D/'perk-samples.jpg');(D/'perk-samples.json').write_text(json.dumps(refs,indent=2))
if __name__=='__main__':main()
