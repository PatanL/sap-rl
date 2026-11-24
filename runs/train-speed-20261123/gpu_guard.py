"""Bound a single GPU experiment including imports/startup. Children of spawn keep their usual entry point."""
import os,runpy,signal,sys
def expired(signum,frame):raise TimeoutError('GPU experiment reached its 870-second budget')
if __name__=='__main__':
    if not os.path.exists('/sap/runs/train-live-20261123/STOPPED'):raise SystemExit('Missing operator STOPPED marker')
    signal.signal(signal.SIGALRM,expired);signal.alarm(870)
    sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')
