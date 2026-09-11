"""Bounded cross-process key pins; the namespace lock orders acquisition.

One open file description per live restore holds advisory shared byte locks.
The final descriptor close releases its keys, including on process loss.
No per-key files, persisted refcounts or PID liveness guesses are involved.
"""

import ctypes
from dataclasses import dataclass
import fcntl
import os
import stat


class _Flock(ctypes.Structure):
    # Native Linux 64-bit ABI, checked with the kernel on every store open.
    _fields_ = [
        ("type", ctypes.c_short),
        ("whence", ctypes.c_short),
        ("start", ctypes.c_longlong),
        ("length", ctypes.c_longlong),
        ("pid", ctypes.c_int),
    ]


def _range(kind, digest):
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("lease digest is malformed")
    # A collision only conservatively pins an unrelated key. Preserve the
    # even offsets used by previous token-file writers. The file stays empty.
    offset = (int(digest[:16], 16) & ((1 << 62) - 1)) * 2
    return _Flock(kind, os.SEEK_SET, offset, 1, 0)


def open_regular(path):
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        actual, named = os.fstat(fd), path.lstat()
        if not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
            named.st_dev,
            named.st_ino,
        ):
            raise ValueError("lease lock file identity is invalid")
        return fd
    except BaseException:
        os.close(fd)
        raise


class KeyPins:
    def __init__(self, path):
        if ctypes.sizeof(ctypes.c_void_p) != 8 or not hasattr(fcntl, "F_OFD_GETLK"):
            raise OSError("64-bit Linux OFD locks are required for key leases")
        self.path = path
        self.fd = open_regular(path)
        try:
            self.busy("0" * 64)
        except BaseException:
            self.close()
            raise

    def acquire(self, keys):
        # The caller holds the namespace lock while acquiring every key.
        fd = open_regular(self.path)
        try:
            for key in keys:
                fcntl.fcntl(fd, fcntl.F_OFD_SETLK, bytes(_range(fcntl.F_RDLCK, key)))
            return fd
        except BaseException:
            os.close(fd)
            raise

    def busy(self, key):
        return self._query(_range(fcntl.F_WRLCK, key))

    def _query(self, query):
        actual, named = os.fstat(self.fd), self.path.lstat()
        if (actual.st_dev, actual.st_ino) != (
            named.st_dev,
            named.st_ino,
        ) or not stat.S_ISREG(named.st_mode):
            raise ValueError("key lease lock identity changed")
        answer = fcntl.fcntl(self.fd, fcntl.F_OFD_GETLK, bytes(query))
        return _Flock.from_buffer_copy(answer).type != fcntl.F_UNLCK

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


@dataclass
class RestoreView:
    owner: object
    descriptors: tuple
    result: object = None
    active: bool = True
