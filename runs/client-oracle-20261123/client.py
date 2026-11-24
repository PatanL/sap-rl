"""Interactive real WebGL client with scoped requests and local-only replay fixtures."""
import json, time, re, sys, shutil, select, os, signal, hashlib
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright
from feasibility import HERE,BUILD,CHROME,safe_path
D=HERE/'session2'
def main(automatic=None):
    d=D/'guest-data'
    if not d.exists():
        d.mkdir(mode=0o700)
        shutil.copytree(HERE.parent/'live-guest-profile/Default',d/'Default',ignore=shutil.ignore_patterns('LOCK','Singleton*'))
    mode='replay' if automatic else 'guest';fixture=D/'fixtures/current.json' if automatic else None;arena_started=False;start_path=None
    start=time.monotonic();log=(D/'network.jsonl').open('a',buffering=1)
    def record(**kw):
        log.write(json.dumps(dict(seconds=round(time.monotonic()-start,3),**kw))+'\n')
    with sync_playwright() as pw:
        ctx=pw.chromium.launch_persistent_context(str(d),executable_path=str(CHROME),headless=True,viewport={'width':1920,'height':1080},device_scale_factor=1,
            service_workers='block',accept_downloads=False,args=['--no-first-run','--disable-extensions','--disable-background-networking','--disable-sync','--disable-component-update','--disable-default-apps','--use-angle=swiftshader','--enable-unsafe-swiftshader','--autoplay-policy=no-user-gesture-required','--mute-audio','--disable-dev-shm-usage'])
        def route(rt):
            nonlocal arena_started,start_path
            q=rt.request;u=urlsplit(q.url);h=u.hostname or '';path=u.path;m=q.method
            static=(h=='html-classic.itch.zone' and path.startswith('/html/18047452/')) or (h=='static.itch.io' and path=='/htmlgame.js')
            api=h=='api.teamwood.games' and re.fullmatch(r'/[\w.\-]+/api/([^?]+)',path)
            op=api.group(1).lower() if api else ''
            if mode=='replay' and fixture and m=='GET' and op.startswith('battle/get/'):
                body=fixture.read_bytes();info=json.loads(body)
                assert all(k in info for k in ['Id','Seed','UserBoard','OpponentBoard','ResultingBoard','Outcome'])
                record(action='local',op=op,source=fixture.name,bodySHA=hashlib.sha256(body).hexdigest())
                rt.fulfill(status=200,headers={'Content-Type':'application/json','Access-Control-Allow-Origin':'https://html-classic.itch.zone','Access-Control-Allow-Credentials':'true'},body=body)
                print('LOCAL',fixture.name,flush=True);return
            allow=m in ('GET','HEAD','OPTIONS') and static
            extra=set((D/'allowed-ops.txt').read_text().splitlines()) if (D/'allowed-ops.txt').exists() else set()
            if api:
                reads=op in ('ping','version/current','history/fetch','playback/participation','news/get','settings/get') or bool(re.search(r'(?:^|/)(?:fetch|list|get|current)(?:/|$)',op)) or op.startswith(('battle/get/','history/get/','user/get','pack/get','pack/fetch'))
                allow=m=='OPTIONS' or (m in ('HEAD','GET') and reads) or (m,op) in {('POST','user/register-guest'),('POST','user/guest'),('POST','user/refresh')} or (m+' '+op) in extra
                if op.startswith(('arena/','build/')) and m=='POST':allow=mode=='arena'
                if (op.startswith(('versus/','ranked/','purchase/','payment/','store/buy')) and not (m=='GET' and op=='versus/datetime')) or op=='user/register':allow=False
                if 'watch' in op and m=='POST' and mode!='arena':allow=False
                if op=='arena/queue':
                    # Counted allowance: one queue originally, plus one per explicit grant line in session3/queue-grants.txt.
                    done=sum('"action": "response"' in v and '/api/arena/queue"' in v and '"status": 200' in v for v in (D/'network.jsonl').read_text().splitlines())
                    gf=HERE/'session3/queue-grants.txt';grants=sum(1 for v in gf.read_text().splitlines() if v.strip() and not v.startswith('#')) if gf.exists() else 0
                    if done>=1+grants:allow=False
                if op=='user/register-guest' and mode!='guest':allow=False
                if allow and m=='POST' and op in ('arena/start','arena/create','arena/join'):
                    if start_path==op and arena_started:allow=False
                    else:start_path=op;arena_started=True
            allow=bool(allow and u.scheme=='https')
            if api or not allow:record(action='allow' if allow else 'block',method=m,host=h,op=safe_path(path),mode=mode)
            if not allow and api:print('BLOCK',m,safe_path(path),flush=True)
            rt.continue_() if allow else rt.abort('blockedbyclient')
        ctx.route('**/*',route);ctx.route_web_socket('**/*',lambda ws:ws.close())
        page=ctx.pages[0] if ctx.pages else ctx.new_page()
        def response(res):
            u=urlsplit(res.url)
            if u.hostname=='api.teamwood.games':
                record(action='response',method=res.request.method,op=safe_path(u.path),status=res.status)
                print(res.status,res.request.method,safe_path(u.path),flush=True)
        page.on('response',response)
        def snap(n):
            assert re.fullmatch(r'[\w-]+',n)
            page.screenshot(path=str(D/'frames'/(n+'.jpg')),type='jpeg',quality=91,timeout=30000)
        def stop(sig,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
        try:
            page.goto(BUILD,wait_until='domcontentloaded',timeout=90000);print('BROWSER READY',flush=True)
            if automatic:
                from battle_viewer import batch
                batch(page,D,automatic)
                return
            while True:
                page.wait_for_timeout(100)
                if not select.select([sys.stdin],[],[],0)[0]:continue
                cmd=sys.stdin.readline()
                if not cmd:break
                a=cmd.strip().split()
                if not a:continue
                if a[0]=='quit':break
                if a[0]=='snap':snap(a[1])
                elif a[0]=='reload':page.reload(wait_until='domcontentloaded',timeout=90000)
                elif a[0]=='mode' and a[1] in ('guest','arena','history','replay'):mode=a[1]
                elif a[0]=='fixture':fixture=D/'fixtures'/a[1]
                elif a[0]=='click':
                    page.mouse.click(int(a[1]),int(a[2]));page.wait_for_timeout(int(a[4]) if len(a)>4 else 1400);snap(a[3])
                elif a[0]=='scroll':
                    page.mouse.move(int(a[1]),int(a[2]));page.mouse.wheel(0,int(a[3]));page.wait_for_timeout(650);snap(a[4])
                elif a[0]=='recordclick':
                    page.mouse.click(int(a[1]),int(a[2]));end=time.monotonic()+float(a[4]);n=0
                    while time.monotonic()<end:
                        snap(a[3]+'-'+str(n).zfill(4));page.wait_for_timeout(120);n+=1
                elif a[0]=='record':
                    end=time.monotonic()+float(a[2]);n=0
                    while time.monotonic()<end:
                        snap(a[1]+'-'+str(n).zfill(4));page.wait_for_timeout(150);n+=1
                if a[0] in ('replaycode','tap','drag','arena-turn','playback-tools'):
                    import importlib,controls
                    try:importlib.reload(controls).action(page,D,a,snap)
                    except Exception as e:print('CONTROL STOPPED',type(e).__name__,str(e)[:180],flush=True)
                print('DONE',cmd.strip(),flush=True)
        finally:ctx.close();log.close()
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--replay-manifest')
    args=ap.parse_args()
    if args.replay_manifest and not re.fullmatch(r'[A-Za-z0-9_-]+',args.replay_manifest):ap.error('Unsafe manifest name')
    main(args.replay_manifest)
