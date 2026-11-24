import cv2,sys,numpy as np
from trace_frames import damage
cap=cv2.VideoCapture(sys.argv[1])
def outline(im):
 roi=im[500:745,650:1270];b,g,r=cv2.split(roi);m=((r>190)&(g<63)&(b<61)).astype('uint8')
 _,_,s,_=cv2.connectedComponentsWithStats(m)
 xy=[x+w//2+650 for x,y,w,h,area in s if 65<w<226 and 72<h<209 and area>460 and area/(w*h)<.24]
 return any(730<x<920 for x in xy) and any(1002<x<1175 for x in xy)
if __name__=='__main__':
 j=0
 while True:
  ok,im=cap.read()
  if not ok:break
  t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000;y=damage(im);o=outline(im)
  if t>54:print(j,round(t,2),int(y) if y else '-',int(o))
  j+=1
