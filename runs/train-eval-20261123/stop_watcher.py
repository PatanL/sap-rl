"""Gracefully stop only the recorded watcher for this directory."""
import os,signal
from pathlib import Path
root=Path(__file__).resolve().parent;pidfile=root/'watch.pid'
if not pidfile.exists():raise SystemExit('No watcher PID file')
pid=int(pidfile.read_text())
fd=os.pidfd_open(pid)
try:
    proc=Path('/proc')/str(pid)
    args=(proc/'cmdline').read_bytes().split(b'\0')
    if (proc/'cwd').resolve()!=root or b'watch.py' not in args:
        raise SystemExit('PID does not belong to this watcher')
    signal.pidfd_send_signal(fd,signal.SIGTERM)
finally:os.close(fd)
print('Sent TERM; watcher finishes the current scan before exiting.')
