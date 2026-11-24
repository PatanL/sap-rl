from pathlib import Path
import json,time,re
def action(page,D,a,snap):
    if a[0]=='replaycode':
        # This field accepts ordinary public playback codes. No account input.
        source=json.loads((D/'host-code.json').read_text())
        assert set(source)=={'Pid','T'} and re.fullmatch(r'[0-9a-f-]{36}',source['Pid'])
        page.mouse.click(int(a[1]),int(a[2]));page.wait_for_timeout(550);page.keyboard.press('Control+A',delay=100);page.wait_for_timeout(120);page.keyboard.type(json.dumps(source,separators=(',',':')),delay=44);page.wait_for_timeout(350);snap(a[3])
    elif a[0]=='tap':
        for x,y,ms in json.loads((D/'tap.json').read_text()):
            page.mouse.click(x,y);page.wait_for_timeout(ms)
        snap(a[1])
    elif a[0]=='drag':
        page.mouse.move(int(a[1]),int(a[2]));page.mouse.down();page.mouse.move(int(a[3]),int(a[4]),steps=12);page.mouse.up();page.wait_for_timeout(900);snap(a[5])
    elif a[0]=='arena-turn':
        import cv2,numpy as np
        def small(buf):return cv2.resize(cv2.imdecode(np.frombuffer(buf,np.uint8),cv2.IMREAD_COLOR),(960,540))
        refs={}
        for name,filename,box in [('shop','one-fish',(773,484,946,527)),('roll','one-fish',(7,484,134,527)),('confirm','end-one',(222,223,738,297)),('defeat','loss-panel',(416,346,547,377)),('tier','arena-end-check2',(351,96,497,122))]:
            x,y,xx,yy=box;refs[name]=(small((D/'frames'/(filename+'.jpg')).read_bytes())[y:yy,x:xx],box)
        previous='';last=0;start=time.monotonic()
        while time.monotonic()-start<140:
            buf=page.screenshot(type='jpeg',quality=83);im=small(buf)
            def seen(key):
                ref,(x,y,xx,yy)=refs[key];diff=np.abs(im[y:yy,x:xx].astype(float)-ref);return diff.mean()<13
            stage='shop' if seen('shop') and seen('roll') else 'confirm' if seen('confirm') else 'defeat' if seen('defeat') else 'tier' if seen('tier') else ''
            if stage and (stage!=previous or time.monotonic()-last>4):
                ix=time.monotonic_ns()
                (D/'frames'/f'ordinary-{ix}-{stage}.jpg').write_bytes(buf)
                xy={'shop':(850,505),'confirm':(625,283),'defeat':(480,105),'tier':(480,102)}[stage]
                page.mouse.click(2*xy[0],2*xy[1]);last=time.monotonic();previous=stage
            if not stage and time.monotonic()-max(last,start)>30:break
            page.wait_for_timeout(450)
        snap(a[1])
    elif a[0]=='playback-tools':
        import importlib,battle_viewer
        importlib.reload(battle_viewer).batch(page,D,a[1])
