"""Direct thumbnail of each chosen real frame beside native front-to-back data."""
import json,textwrap,sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent/'session2'
def s(side):return ', '.join(p['pet']+' '+str(p['attack'])+'/'+str(p['health']) for p in side) or 'empty'
def sheet(d):
 pairs=json.load(open(d/'paired.json'))['rows']
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
 canvas=Image.new('RGB',(1200,165*len(pairs)),(248,248,246));draw=ImageDraw.Draw(canvas)
 for i,row in enumerate(pairs):
  y=i*165;v=row['real'];e=row['engine'];im=Image.open(d/v['frame']);im=im.crop((0,470,1920,831));im.thumbnail((760,145))
  canvas.paste(im,(0,y+20));draw.text((3,y),f"{d.name[4:]}   {row['stage']}   {v['time']:.2f}s",font=font,fill='black')
  if e:
   txt='Native player\n'+'\n'.join(textwrap.wrap(s(e['player']),50))+'\nNative opponent\n'+'\n'.join(textwrap.wrap(s(e['opponent']),50))
  else:txt='Native simulation already ended.'
  draw.multiline_text((775,y+20),txt,font=font,fill='black',spacing=4)
 canvas.save(d/'rounds-contact.jpg',quality=90)
if __name__=='__main__':
 for p in sys.argv[1:]:sheet(Path(p))
