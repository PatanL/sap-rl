"""Local replay-viewer feasibility check. Network default-deny, no browser extensions."""
import argparse,collections,hashlib,json,os,re,shutil,sys,tempfile,time
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent
BUILD='https://html-classic.itch.zone/html/18047452/Production%20WebGL/index.html'
CHROME=Path.home()/'.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome'
def safe_path(path):
    return '/'.join(x if len(x)<25 and re.fullmatch(r'[a-zA-Z0-9._%-]*',x) else '<id>' for x in path.split('/'))
def main(label):
    src=HERE.parent/label
    frames=HERE/label;frames.mkdir(exist_ok=True)
    counts=collections.Counter();beg=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='.session-',dir=HERE) as tmp, sync_playwright() as pw:
        prof=Path(tmp)/'profile';prof.mkdir()
        shutil.copytree(src/'Default',prof/'Default',ignore=shutil.ignore_patterns('LOCK','Singleton*'))
        ctx=pw.chromium.launch_persistent_context(str(prof),executable_path=str(CHROME),headless=True,viewport={'width':1920,'height':1080},device_scale_factor=1,service_workers='block',accept_downloads=False,args=['--no-first-run','--disable-extensions','--disable-background-networking','--disable-sync','--disable-component-update','--disable-default-apps','--use-angle=swiftshader','--enable-unsafe-swiftshader','--autoplay-policy=no-user-gesture-required','--mute-audio','--disable-dev-shm-usage'])
        def record(kind,m,h,path,status=None):
            k=(kind,m,h,safe_path(path),status);counts[k]+=1
            if counts[k]==1:print(json.dumps(k),flush=True)
        def route(rt):
            q=rt.request;u=urlsplit(q.url);h=u.hostname or '';path=u.path;method=q.method
            static=h=='html-classic.itch.zone' and path.startswith('/html/18047452/')
            api=h=='api.teamwood.games' and re.fullmatch(r'/[\w.\-]+/api/(?:ping|version/current|history/fetch|battle/get/[\w-]+|playback/participation|version|settings/get|news/get)',path,re.I)
            if api and '/api/battle/get/' in path and method=='GET' and (HERE/'selected-battle.json').is_file():
                body=(HERE/'selected-battle.json').read_bytes()
                obj=json.loads(body)
                if not {'Seed','UserBoard','OpponentBoard','ResultingBoard','Outcome'}.issubset(obj):
                    rt.abort('blockedbyclient');raise ValueError('Invalid battle fixture')
                record('fulfilled-local',method,h,path)
                rt.fulfill(status=200,headers={'Content-Type':'application/json','Access-Control-Allow-Origin':'https://html-classic.itch.zone','Access-Control-Allow-Credentials':'true'},body=body)
                return
            if u.scheme=='https' and method in ('GET','HEAD','OPTIONS') and (static or api or (h=='static.itch.io' and path=='/htmlgame.js')):
                record('allowed',method,h,path);rt.continue_()
            else:
                record('blocked',method,h,path);rt.abort('blockedbyclient')
        ctx.route('**/*',route)
        ctx.route_web_socket('**/*',lambda ws:ws.close())
        page=ctx.pages[0] if ctx.pages else ctx.new_page()
        page.on('response',lambda res: record('response',res.request.method,urlsplit(res.url).hostname or '',urlsplit(res.url).path,res.status) if res.status>=400 or urlsplit(res.url).hostname=='api.teamwood.games' else None)
        page.on('pageerror',lambda err: print('PAGE ERROR',str(err).split('?')[0][:220],flush=True))
        def snap(name):
            page.screenshot(path=str(frames/(name+'.jpg')),type='jpeg',quality=82,timeout=30000)
            print('frame',name,int(time.monotonic()-beg),'s',flush=True)
        try:
            page.goto(BUILD,wait_until='domcontentloaded',timeout=70000)
            for j,delay in enumerate([2,6,12]):
                page.wait_for_timeout(delay*1000);snap('boot-'+str(j))
            print('READY',flush=True)
            for line in sys.stdin:
                s=line.strip().split()
                if not s:continue
                if s[0]=='quit':break
                if s[0]=='snap':snap(s[1])
                elif s[0]=='click':
                    page.mouse.click(int(s[1]),int(s[2]));page.wait_for_timeout(1300);snap(s[3])
                elif s[0]=='wait':
                    page.wait_for_timeout(int(s[1])*1000);snap(s[2])
        finally:
            ctx.close()
            (frames/'requests.json').write_text(json.dumps([dict(kind=k[0],method=k[1],host=k[2],path=k[3],status=k[4],count=v) for k,v in counts.items()],indent=2))
if __name__=='__main__':main(sys.argv[1])
