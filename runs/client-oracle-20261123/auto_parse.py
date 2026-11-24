"""Read complete captures on disk as they arrive; exits once manifest complete or deadline."""
from pathlib import Path
import time,json,sys,traceback
from trace_frames import process,D
def main():
 manifest=sys.argv[1];stop=time.monotonic()+int(sys.argv[2]);plan=json.load(open(D/(manifest+'-manifest.json')))
 while time.monotonic()<stop:
  todo=0
  for job in plan:
   dest=D/'captures'/Path(job['file']).stem
   if (dest/'trace.json').exists():continue
   todo+=1
   if (dest/'run.json').exists():
    x=process(dest);print(job['file'],len(x['attack_markers']),flush=True)
  if not todo or (D/'halt').exists():return
  time.sleep(2)
if __name__=='__main__':main()
