from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import vllm
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

    from spoolcache.vllm.compat import verify_vllm_runtime
    from spoolcache.vllm.connector import SpoolCacheConnector
except ImportError:
    vllm = None


@unittest.skipIf(vllm is None, "an installed vLLM runtime is required")
class VLLMPipelineRuntimeTests(unittest.TestCase):
    def test_real_connector_contract_exposes_pp_aware_handshake(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runtime-contract.txt").write_text(
                "vllm public PP handshake\n",
                encoding="utf-8",
            )
            receipt = verify_vllm_runtime(
                vllm_module=vllm,
                connector_type=SpoolCacheConnector,
                spec_kind_resolver=get_kv_cache_spec_kind,
                package_roots=(root,),
                require_pp_aware=True,
            )
        self.assertIn(
            "set_xfer_handshake_metadata_pp_aware",
            receipt.capabilities,
        )


if __name__ == "__main__":
    unittest.main()
