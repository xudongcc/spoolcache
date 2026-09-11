"""POSIX cache paths without process-lifetime interning of arbitrary digests."""

from pathlib import PosixPath


class CachePath(PosixPath):
    """Keep pathlib behavior while avoiding its high-cardinality string cache.

    The installed Python 3.12 pathlib parser retains two interned strings per
    unique key name in the measured storage workload. A bounded metadata index
    cannot bound that separate retention. Override only POSIX lexical parsing;
    no global pathlib/sys monkeypatch or interned digest catalog is introduced.
    Python 3.12/3.13 use this parser. Earlier pathlib versions do not use this
    hook; RSS qualification applies to the recorded interpreter. Real storage
    remains rank-local Linux buffered I/O.
    """

    @classmethod
    def _parse_path(cls, path):
        # Preserve POSIX's special exactly-two-leading-slashes root, repeated
        # separators, dot elision and literal '..' exactly as pathlib does.
        root = ""
        if path.startswith("/"):
            root = "//" if path.startswith("//") and not path.startswith("///") else "/"
        return "", root, [part for part in path.split("/") if part and part != "."]
