"""Session 3: read every frame of a capture's canvas movie into a collapsed text timeline.

Each line is one distinct reading of the board (attack/health badges with their physical slot,
front to back, slot 0 = front on both sides).  `n=` is the number of consecutive movie frames
that showed exactly this reading; long runs are settled boards, runs of 1-2 are animation.
`DMG` marks frames with the paired red front-damage numerals (a normal attack exchange).
Off-grid badges (pets moving between slots) are printed with `~x<pixel>`.
Key frames for every reading with n>=MIN_KEEP are saved under <capture>/tl/ for visual checks.
Usage: timeline.py <capture dir> [...]
"""
import json,sys,cv2
from pathlib import Path
from read_frames import badges
from trace_frames import damage
from physical import physical
cv2.setNumThreads(1)
MIN_KEEP=3
def token(z):
    s=physical(z['x']);a='?' if z['attack'] is None else z['attack'];h='?' if z['health'] is None else z['health']
    return (s if s is not None else -1,z['x'],f"{a}/{h}")
def read(im):
    bb=badges(im);out={}
    for side,name in ((True,'P'),(False,'O')):
        sel=[token(z) for z in bb if (z['x']<960)==side]
        sel.sort(key=lambda t:-t[1] if side else t[1])
        out[name]=sel
    return out
def fmt(r):
    def one(t):return (f"s{t[0]}" if t[0]>=0 else f"~x{t[1]}")+f" {t[2]}"
    return 'P: '+' | '.join(one(t) for t in r['P'])+'  ||  O: '+' | '.join(one(t) for t in r['O'])
def key(r):return tuple((t[0],t[2]) for t in r['P']),tuple((t[0],t[2]) for t in r['O'])
def run(d):
    d=Path(d);video=d/'battle.webm';rows=[]
    out=d/'tl';out.mkdir(exist_ok=True)
    if video.exists():
        cap=cv2.VideoCapture(str(video));i=0;prev=-1.
        def frames():
            nonlocal i,prev
            while True:
                ok,im=cap.read()
                if not ok:break
                t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if t<=prev:t=prev+.1
                prev=t
                if im.shape[:2]!=(1080,1920):im=cv2.resize(im,(1920,1080))
                yield i,t,im;i+=1
        src=frames()
    else:
        rr=json.loads((d/'run.json').read_text())
        src=((i,t,cv2.imread(str(d/fn))) for i,(fn,t) in enumerate(rr['frames']))
    cur=None
    for i,t,im in src:
        r=read(im);k=key(r);dmg=damage(im) is not None
        if cur is not None and cur['key']==k:
            cur['n']+=1;cur['dmg']=cur['dmg'] or dmg;cur['last']=im;cur['lastf']=i
        else:
            if cur is not None:rows.append(cur)
            cur=dict(key=k,f=i,t=round(t,2),n=1,dmg=dmg,text=fmt(r),last=im,lastf=i,P=r['P'],O=r['O'])
    if cur is not None:rows.append(cur)
    lines=[]
    for r in rows:
        keep=r['n']>=MIN_KEEP or r is rows[-1]
        if keep:
            fn=f"f{r['lastf']:04d}.jpg";cv2.imwrite(str(out/fn),r['last'][330:900],[cv2.IMWRITE_JPEG_QUALITY,80])
        lines.append(f"f={r['f']:04d} t={r['t']:6.2f} n={r['n']:3d} {'DMG' if r['dmg'] else '   '} {r['text']}"+(f"   [tl/f{r['lastf']:04d}.jpg]" if keep else ''))
    (d/'timeline.txt').write_text('\n'.join(lines)+'\n')
    (d/'timeline.json').write_text(json.dumps([dict(f=r['f'],lastf=r['lastf'],t=r['t'],n=r['n'],dmg=r['dmg'],P=r['P'],O=r['O'],frame=(f"tl/f{r['lastf']:04d}.jpg" if (r['n']>=MIN_KEEP or r is rows[-1]) else None)) for r in rows]))
    return lines
if __name__=='__main__':
    for p in sys.argv[1:]:
        ls=run(p);print(p,len(ls),'readings')
