from __future__ import annotations

import mmap
import unittest

from spoolcache.buffers import AlignedBufferPool
from spoolcache.errors import StoreBusyError


class BufferPoolTests(unittest.TestCase):
    def test_pool_never_grows_and_returns_credit(self) -> None:
        with AlignedBufferPool(slot_bytes=mmap.PAGESIZE, slot_count=2) as pool:
            self.assertEqual(pool.budget_bytes, mmap.PAGESIZE * 2)
            with pool.acquire() as first, pool.acquire() as second:
                self.assertNotEqual(first.index, second.index)
                self.assertEqual(pool.borrowed, 2)
                with self.assertRaises(StoreBusyError):
                    with pool.acquire():
                        pass
            self.assertEqual(pool.borrowed, 0)
            with pool.acquire() as returned:
                returned.view[:4] = b"data"
                returned.zero(0, 4)
                self.assertEqual(bytes(returned.view[:4]), b"\0\0\0\0")


if __name__ == "__main__":
    unittest.main()
