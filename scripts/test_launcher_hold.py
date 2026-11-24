"""CPU-only job-control tests; no Docker/GPU/training process is accessed."""
import subprocess
from pathlib import Path
import signal
import tempfile
import time
import unittest

from launcher_hold import LauncherHold


class HoldTest(unittest.TestCase):
    def exercise(self, terminate):
        with tempfile.TemporaryDirectory(prefix='sap-handoff-test-') as directory:
            path = Path(directory)
            child = path / 'child.py'
            child.write_text("import time\nfrom pathlib import Path\np=Path(__file__).parent\n"
                             "(p/'started').touch()\ntime.sleep(.3)\n(p/'finished').touch()\n")
            # The shell command after the child must not execute while held.
            script = path / 'launcher.sh'
            script.write_text(f"#!/bin/bash\npython3 '{child}'\ntouch '{path / 'next_chunk'}'\n")
            process = subprocess.Popen(['/bin/bash', str(script)])
            hold = LauncherHold(process.pid, str(script))
            try:
                deadline = time.monotonic() + 3
                while not (path/'started').exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((path/'started').exists())
                hold.acquire()
                while not (path/'finished').exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((path/'finished').exists(), 'Child stopped with launcher')
                self.assertTrue(hold.is_held())
                self.assertFalse((path/'next_chunk').exists())
                if terminate:
                    hold.terminate_finished_launcher()
                else:
                    self.assertTrue(hold.release())
                process.wait(timeout=3)
                self.assertEqual((path/'next_chunk').exists(), not terminate)
                self.assertFalse(hold.release())  # Never signal a reused/exited PID.
            finally:
                if process.poll() is None:
                    hold.release()
                    process.terminate()
                    process.wait(timeout=3)

    def test_child_continues_and_next_chunk_waits(self):
        self.exercise(False)

    def test_finished_launcher_terminates_without_starting_next_chunk(self):
        self.exercise(True)

    def test_unrecognized_process_is_not_signaled(self):
        process = subprocess.Popen(['sleep', '5'])
        try:
            with self.assertRaises(RuntimeError):
                LauncherHold(process.pid, 'continue_v45_overnight.sh').acquire()
            self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait(timeout=3)


if __name__ == '__main__':
    unittest.main()
