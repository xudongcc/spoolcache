from __future__ import annotations

import inspect
import unittest

try:
    from vllm.config import ModelConfig
except ImportError:
    ModelConfig = None


@unittest.skipIf(ModelConfig is None, "an installed vLLM runtime is required")
class VLLMModelNamespaceRuntimeTests(unittest.TestCase):
    def test_model_locator_and_revision_are_public_constructor_fields(self) -> None:
        parameters = inspect.signature(ModelConfig).parameters
        self.assertIn("model", parameters)
        self.assertIn("revision", parameters)


if __name__ == "__main__":
    unittest.main()
