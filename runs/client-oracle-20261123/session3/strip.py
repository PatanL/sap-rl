"""usage: strip.py <capture> <first> <last> [step] [x0 x1 y0 y1] -> scratchpad v/strip.jpg : movie frames stacked (cropped battle band)"""
import cv2,sys,numpy as np
d='/home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/session2/captures/'+sys.argv[1]
a,b=int(sys.argv[2]),int(sys.argv[3]);step=int(sys.argv[4]) if len(sys.argv)>4 else 1
x0,x1,y0,y1=(int(v) for v in sys.argv[5:9]) if len(sys.argv)>8 else (0,1920,520,860)
cap=cv2.VideoCapture(d+'/battle.webm');i=0;ims=[]
while True:
    ok,im=cap.read()
    if not ok:break
    if a<=i<=b and (i-a)%step==0:
        if im.shape[:2]!=(1080,1920):im=cv2.resize(im,(1920,1080))
        c=im[y0:y1,x0:x1].copy();sc=min(1.0,960/c.shape[1]);c=cv2.resize(c,(int(c.shape[1]*sc),int(c.shape[0]*sc)));cv2.putText(c,str(i),(5,22),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2);ims.append(c)
    i+=1
cv2.imwrite('/tmp/claude-1000/-home-dzliu-pzliu/f7c77898-47a1-451e-a609-998d61e2f9e2/scratchpad/v/strip.jpg',np.vstack(ims));print(len(ims),'frames')
