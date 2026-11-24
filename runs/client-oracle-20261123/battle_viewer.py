"""Run byte-exact local battle fixtures in the real client's rewatch viewer. Extract rendered frames only."""
import base64,collections,hashlib,json,os,re,time
from pathlib import Path
import cv2,numpy as np
cv2.setNumThreads(1)
def emit(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def small(buf):
 return cv2.resize(cv2.imdecode(np.frombuffer(buf,np.uint8),cv2.IMREAD_COLOR),(960,540))
def batch(page,D,manifest):
 plan=json.loads((D/(manifest+'-manifest.json')).read_text());ref=D/'frames'
 def template(f,box):
  im=small((ref/(f+'.jpg')).read_bytes());x,y,z,w=box;return box,im[y:w,x:z].copy()
 tm={k:template(f,box) for k,f,box in [('fast','proof-now',(520,6,555,48)),('main','main3',(362,143,593,209)),('arena','play-modes',(365,143,593,209)),('shop','round3-menu2',(776,488,945,525)),('rewatch','round3-menu2',(870,5,892,28))]}
 def seen(im,k):
  (x,y,z,w),t=tm[k];return np.abs(im[y:w,x:z].astype('i2')-t).mean()<14
 def ready(dest):
  t=time.monotonic();last=0
  while time.monotonic()-t<85:
   buf=page.screenshot(type='jpeg',quality=78,timeout=18000);im=small(buf)
   if seen(im,'shop') and seen(im,'rewatch'):return
   if time.monotonic()-last>3:
    for k in ('main','arena'):
     if seen(im,k):
      page.mouse.click(960,350,delay=130);last=time.monotonic();break
   page.wait_for_timeout(300)
  (dest/'blocked.jpg').write_bytes(buf);raise RuntimeError('The original arena shop with rewatch arrow was not reachable')
 original=page;standby=None
 if os.environ.get('SAP_PRELOAD_MENU')=='1' and manifest!='validation':standby=page.context.new_page()
 for job in plan:
  name=Path(job['file']).stem;dest=D/'captures'/name
  if (dest/'run.json').exists():continue
  dest.mkdir(parents=True,exist_ok=True)
  if (D/'halt').exists():return
  s=(D/'fixtures'/job['file']).read_bytes();sha=hashlib.sha256(s).hexdigest()
  if sha!=job['sha256']:raise ValueError('Fixture changed')
  # The caller armed current.json. Its bytes, including the real ID, are delivered locally.
  temp=D/'fixtures/current.tmp';temp.write_bytes(s);temp.replace(D/'fixtures/current.json')
  first=time.monotonic()
  page.bring_to_front();ready(dest);page.mouse.move(820,100)
  video=False
  if manifest!='validation':
   try:
    video=page.evaluate('''() => {
      const c=document.querySelector('canvas'); const stream=c.captureStream(10);
      const opts={mimeType:'video/webm;codecs=vp8',videoBitsPerSecond:4800000};
      const rec=new MediaRecorder(stream,opts), chunks=[];
      window.__auditVideoDone=new Promise(resolve=>{
        rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
        rec.onstop=()=>{stream.getTracks().forEach(t=>t.stop());let r=new FileReader();
          r.onload=()=>resolve(r.result.split(',')[1]);r.readAsDataURL(new Blob(chunks,{type:'video/webm'}));};
      });window.__auditRecorder=rec;rec.start(1000);return true;
    }''')
   except Exception as e:print('Video unavailable; full-size PNG/JPEG frame collection continues',type(e).__name__,flush=True)
  events=[]
  def request(q):
   from urllib.parse import urlsplit
   path=urlsplit(q.url).path
   if '/api/battle/get/' in path and q.method=='GET':events.append(('get',time.monotonic()))
   if path.endswith('/api/arena/watch') and q.method=='POST':events.append(('watched',time.monotonic()))
  page.on('request',request)
  start=time.monotonic();index=[];end=None;last_click=start;preloaded=False;speed_selected=False
  try:
   page.mouse.click(1764,34,delay=140);page.wait_for_timeout(450);page.mouse.move(1200,150)
   while time.monotonic()-start<170:
    buf=page.screenshot(type='jpeg',quality=88,timeout=20000)
    ix=len(index);namefile=f'frame-{ix:04d}.jpg';(dest/namefile).write_bytes(buf)
    index.append([namefile,round(time.monotonic()-start,3)])
    if (manifest in ('dev','control') or os.environ.get('SAP_FAST')=='1') and not speed_selected and time.monotonic()-start>3 and any(e[0]=='get' for e in events):
     probe=small(buf)[6:48,520:555].min(2)>205;refwhite=tm['fast'][1].min(2)>205
     if (probe&refwhite).sum()/max(1,(probe|refwhite).sum())>.67:
      page.mouse.click(1074,53,delay=120);page.wait_for_timeout(180);page.mouse.move(1200,150);speed_selected=True
    if standby is not None and not preloaded and any(ev[0]=='get' for ev in events):
     # Only preload the normal menu. Exactly one tab invokes rewatch at any time.
     standby.goto('https://html-classic.itch.zone/html/18047452/Production%20WebGL/index.html',wait_until='commit',timeout=60000)
     page.bring_to_front();preloaded=True
    if any(ev[0]=='get' for ev in events) and any(ev[0]=='watched' for ev in events):
     if end is None:end=time.monotonic()
     if time.monotonic()-end>=1.2:break
    elif not any(ev[0]=='get' for ev in events) and time.monotonic()-last_click>2:
     im=small(buf)
     if time.monotonic()-start>18:break
     if seen(im,'shop') and seen(im,'rewatch'):
      page.mouse.click(1764,34,delay=140);page.wait_for_timeout(450);page.mouse.move(1200,150);last_click=time.monotonic()
    page.wait_for_timeout(int(os.environ.get('SAP_SHOT_INTERVAL_MS','120')))
   if end is None:
    emit(dest/'incomplete.json',dict(id=job.get('id'),frames=index,events=[(a,b-start) for a,b in events]))
    raise RuntimeError('Missing real viewer start or terminal event for '+name)
   if video:
    try:
     encoded=page.evaluate('''async () => {window.__auditRecorder.stop();return await window.__auditVideoDone}''')
     (dest/'battle.webm').write_bytes(base64.b64decode(encoded))
    except Exception as e:print('Video export failed',type(e).__name__,flush=True)
   emit(dest/'run.json',dict(fixture=job['file'],sha256=sha,seconds_to_terminal=round(end-start,3),seconds_with_navigation=round(time.monotonic()-first,3),
         fast_button_used=speed_selected,frames=index,events=[(a,round(b-start,3)) for a,b in events],terminal_frame=index[-1][0]))
   print('CAPTURED',name,len(index),round(end-start,1),flush=True)
  finally:page.remove_listener('request',request)
  if (D/'halt').exists():break
  if standby is None:page.reload(wait_until='domcontentloaded',timeout=90000)
  else:page,standby=standby,page
 if standby is not None:
  for q in (page,standby):
   if q!=original:q.close()
  original.reload(wait_until='domcontentloaded',timeout=90000)
