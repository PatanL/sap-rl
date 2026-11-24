"""Append the reviewed first difference (attribution.json) to each fight's paired.md and record the key frame.
The automatic lines above it come from stat matching, which a stale label (or its tolerance) can mislead; the reviewed line overrides them."""
import json,re,glob,os
from pathlib import Path
H=Path(__file__).resolve().parent;D=H.parent/'session2'
att=json.loads((H/'attribution.json').read_text());out={}
MARK='\n## Reviewed first difference'
for i,a in att.items():
    note=a['note'];m=re.search(r'((?:s3f|dev)-[0-9a-f]{8})[^.]*?\bf0*(\d+)',note)
    capname=m.group(1) if m else None;fno=int(m.group(2)) if m else None
    frame=None
    if capname:
        fr=sorted(int(Path(p).stem[1:]) for p in glob.glob(str(D/'captures'/capname/'tl'/'f*.jpg')))
        c=[x for x in fr if x>=fno]
        if c:frame=f"session2/captures/{capname}/tl/f{c[0]:04d}.jpg"
    out[i]=dict(frame=frame,cited_capture=capname,cited_movie_frame=fno)
    for pre in ('s3f-','dev-'):
        p=D/'captures'/(pre+i)/'paired.md'
        if not p.exists():continue
        t=p.read_text().split(MARK)[0].rstrip('\n')
        rel=os.path.relpath(H.parent/frame,p.parent) if frame else None
        t+=MARK+'\n\n'+f"This line was written after reading the movie timeline and overrides the automatic lines above (those compare stats only and are misled by stale labels in both directions).\n\n- Rule: **{a['rule']}**\n- Round: {a['first_difference']}\n- {note}\n"+(f"- Key frame: [{Path(frame).name}]({rel})\n" if frame else '')
        p.write_text(t+'\n')
(H/'attribution_frames.json').write_text(json.dumps(out,indent=1));print(len(out),'fights;',sum(1 for v in out.values() if v['frame']),'with a key frame')
