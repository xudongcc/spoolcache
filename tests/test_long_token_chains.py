"""Long prefixes have no aggregate header, descriptor-byte or key-count cap."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spoolcache.manifest import TokenFileDescriptor
from tests.token_fixtures import layout_for, open_store


class LongTokenChainTests(unittest.TestCase):
    def test_chain_exceeds_old_key_token_header_and_descriptor_limits(self):
        page_bytes = 512 * 1024
        layout = layout_for(tokens_per_page=256, page_bytes=page_bytes)
        count = 8193
        with tempfile.TemporaryDirectory() as directory, open_store(
            Path(directory) / "rank", layout=layout, slot_bytes=4096,
        ) as store:
            # 128 transfer digests per descriptor exceed the former 64 MiB
            # conservative charge over this chain. Payload I/O is simulated.
            store.validate_prefix_headers(count * 256)

            def open_chunk(key):
                index = int(key, 16)
                return os.open(os.devnull, os.O_RDONLY), TokenFileDescriptor(
                    key, f"{index - 1:064x}" if index > 1 else None,
                    index * 256, store._segments(index * 256), page_bytes,
                    hashlib.sha256(b"x" * 64).hexdigest(), 16384, "b" * 64,
                    block_sha256=(hashlib.sha256(b"x" * 4096).hexdigest(),) * 128,
                )

            with patch.object(store, "_open_chunk", side_effect=open_chunk):
                snapshot = store._chain(f"{count:064x}")
            self.assertEqual(len(snapshot.objects), count)
            self.assertEqual(snapshot.span_tokens, 2097408)
            layout.validate_manifest_coverage(snapshot)
