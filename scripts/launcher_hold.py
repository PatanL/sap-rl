"""Hold only a verified launcher PID; its already-running children keep working."""
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import time


def process_identity(pid):
    raw = Path(f'/proc/{pid}/stat').read_text()
    # comm can contain spaces or parentheses; fields after its closing ')'
    # start with state (field 3); starttime is field 22.
    fields = raw[raw.rfind(')') + 2:].split()
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    return fields[19], fields[0], command


@dataclass
class LauncherHold:
    pid: int
    required_command: str
    start_time: str | None = None

    def acquire(self):
        start, state, command = process_identity(self.pid)
        if self.required_command not in command or not command.startswith(('/bin/bash ', 'bash ')):
            raise RuntimeError('Refusing to suspend an unrecognized launcher')
        if state in ('T', 't', 'Z'):
            raise RuntimeError('Launcher is already stopped or exited')
        self.start_time = start
        os.kill(self.pid, signal.SIGSTOP)  # NOT killpg: children must continue.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if self.is_held():
                return
            time.sleep(.01)
        raise RuntimeError('Launcher did not enter stopped state')

    def is_held(self):
        try:
            start, state, _ = process_identity(self.pid)
        except FileNotFoundError:
            return False
        return start == self.start_time and state in ('T', 't')

    def release(self):
        if self.is_held():
            os.kill(self.pid, signal.SIGCONT)
            return True
        return False

    def terminate_finished_launcher(self):
        """Caller must first prove its learner/container finished naturally."""
        if not self.is_held():
            raise RuntimeError('Launcher ownership was lost')
        os.kill(self.pid, signal.SIGTERM)
        os.kill(self.pid, signal.SIGCONT)  # Deliver TERM without starting another chunk.
