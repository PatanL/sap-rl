"""Collate actually rendered terminal frames without reading the outcome fields."""
from pathlib import Path
from PIL import Image,ImageDraw
import json,sys
D=Path(__file__).resolve().parent/'session2'
def sheet(names,path):
 w,h=920,196;out=Image.new('RGB',(w,len(names)*(h+25)),'white');dr=ImageDraw.Draw(out)
 for j,name in enumerate(names):
  d=D/'captures'/name;r=json.loads((d/'run.json').read_text());im=Image.open(d/r['terminal_frame'])
  cut=im.crop((0,438,1920,847));cut.thumbnail((w,h))
  dr.text((3,j*(h+25)+3),name+' | '+r['terminal_frame']+' | '+str(r['seconds_to_terminal'])+'s',fill='black')
  out.paste(cut,(0,j*(h+25)+25))
 out.save(path)
if __name__=='__main__':sheet(sys.argv[1:],D/'finals.jpg')
