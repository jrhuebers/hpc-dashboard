"""Offline PTY regressions: fake scheduler, no collectors or allocations."""
import fcntl
import os
from pathlib import Path
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
import unittest


CHILD = r'''
import fcntl, getpass, os, sys, termios, threading, time
fcntl.ioctl(0, termios.TIOCSCTTY, 0)
sys.path.insert(0, sys.argv[1])
import slurmjobs as s
status, stall_at = int(sys.argv[2]), int(sys.argv[3])
class FakeQueue:
    def __init__(self, timeout): self.calls = 0
    def fetch(self, user):
        self.calls += 1
        os.write(status, ('FETCH %d\n' % self.calls).encode())
        if stall_at < 0 and self.calls > 1:
            raise RuntimeError('offline scheduler failure\nsecond detail')
        if stall_at > 0 and self.calls >= stall_at:
            threading.Event().wait(60)  # Model a scheduler or filesystem stall.
        return [s.Job('%03d' % i, 'fixture-%02d' % i, getpass.getuser(), 'RUNNING',
                      'fake', 'node', 1, 1024, 'no GPU', 10, 3600)
                for i in range(40)]
    def fetch_steps(self, user): return {}
s.Queue = FakeQueue
original = s.QueueCache
s.QueueCache = lambda queue, user: original(queue, user, interval=0.2)
sys.argv = ['sj', '--no-usage', '--interval', '0.1']
sys.exit(s.main())
'''


class WatchPTYTests(unittest.TestCase):
    def launch(self, stall_at=1):
        self.master, self.slave = pty.openpty()
        attrs = termios.tcgetattr(self.slave)
        attrs[0] |= termios.IXON | termios.IXOFF
        termios.tcsetattr(self.slave, termios.TCSANOW, attrs)
        self.saved = termios.tcgetattr(self.slave)
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack('HHHH', 12, 100, 0, 0))
        self.status, writer = os.pipe()
        self.proc = subprocess.Popen(
            [sys.executable, '-c', CHILD, str(Path(__file__).parent), str(writer), str(stall_at)],
            stdin=self.slave, stdout=self.slave, stderr=self.slave,
            pass_fds=(writer,), start_new_session=True)
        os.close(writer)
        self.output = bytearray()
        self.markers = bytearray()
        self.addCleanup(self.cleanup)
        self.until(lambda: b'FETCH 1\n' in self.markers)
        self.assertFalse(termios.tcgetattr(self.slave)[0] & (termios.IXON | termios.IXOFF))

    def cleanup(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=2)
        for fd in (self.master, self.slave, self.status):
            os.close(fd)

    def pump(self, timeout=0.1):
        ready, _, _ = select.select([self.master, self.status], [], [], timeout)
        for fd in ready:
            try:
                value = os.read(fd, 65536)
            except OSError:
                continue
            (self.output if fd == self.master else self.markers).extend(value)

    def until(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.pump()
        self.assertTrue(predicate(), bytes(self.output[-1000:]))

    def quit(self, key=b'q'):
        start = time.monotonic()
        os.write(self.master, key)
        self.until(lambda: self.proc.poll() is not None, timeout=2)
        self.assertEqual(self.proc.returncode, 0)
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(termios.tcgetattr(self.slave), self.saved)
        self.pump(0)
        self.assertIn(b'\x1b[?1049l', self.output)
        self.assertIn(b'\x1b[?1006l\x1b[?1000l', self.output)

    def test_quit_during_stalled_first_fetch(self):
        self.launch()
        self.quit()
        self.assertEqual(self.markers.count(b'FETCH'), 1)

    def test_escape_during_stalled_first_fetch(self):
        self.launch()
        self.quit(b'\x1b')

    def test_ctrl_c_during_stalled_first_fetch(self):
        self.launch()
        self.quit(b'\x03')

    def test_ctrl_s_does_not_freeze_output_or_quit(self):
        self.launch()
        os.write(self.master, b'\x13')
        self.until(lambda: b'Refreshing 1s' in self.output)
        self.quit()

    def test_failure_is_visible_without_scrolling_to_end(self):
        self.launch(stall_at=-1)
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack('HHHH', 12, 200, 0, 0))
        self.until(lambda: b'Refresh failed' in self.output)
        self.until(lambda: b'offline scheduler failure second detail' in self.output)
        self.assertNotIn(b'fixture-39', self.output)
        self.quit()

    def test_resize_narrow_and_wide_while_refresh_is_stalled(self):
        self.launch(stall_at=2)
        self.until(lambda: b'FETCH 2\n' in self.markers)
        for width in (30, 100):
            self.output.clear()
            fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack('HHHH', 12, width, 0, 0))
            # The normal viewport footer must still occupy a visible physical row.
            self.until(lambda: b'q quit' in self.output)
            os.write(self.master, b'\x1b[F')
            def has_last_job():
                plain = re.sub(rb'\x1b\[[0-?]*[ -/]*[@-~]', b'', self.output)
                # At width 30 the title wraps to many lines; End exposes its
                # unique suffix and details, not necessarily its whole prefix.
                return b'e-39' in re.sub(rb'\s+', b'', plain)
            self.until(has_last_job)
        self.quit()

    def test_scroll_last_good_display_while_refresh_is_stalled(self):
        self.launch(stall_at=2)
        self.until(lambda: b'FETCH 2\n' in self.markers)
        os.write(self.master, b'\x1b[F')
        self.until(lambda: b'fixture-39' in self.output)
        self.until(lambda: b'last display' in self.output)
        self.assertEqual(self.markers.count(b'FETCH'), 2)
        self.quit()


if __name__ == '__main__':
    unittest.main()
