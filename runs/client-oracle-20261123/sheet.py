from pathlib import Path
from PIL import Image,ImageDraw
import sys,json
p=Path(sys.argv[1]);mode=sys.argv[2] if len(sys.argv)>2 else 'board'
files=[x[0] for x in json.load(open(p/'run.json'))['frames']] if (p/'run.json').exists() else [v.name for v in sorted(p.glob('frame-*.jpg'))]
step=int(sys.argv[3]) if len(sys.argv)>3 else 1
f=files[::step]
w=380;h=132
out=Image.new('RGB',(w*3,(len(f)+2)//3*(h+20)),'white');draw=ImageDraw.Draw(out)
for j,n in enumerate(f):
 im=Image.open(p/n);im=im.crop((0,470,1920,830));im=im.resize((w,h));x,y=j%3*w,j//3*(h+20);out.paste(im,(x,y+20));draw.text((x+3,y+1),n,fill='black')
out.save(p/'sheet.jpg')
