"""Distinguish disk-restore divergence from native prefix-cache divergence."""
import unittest
from benchmarks.probe_gemma_mixed_cache import common_prefix_length, classify


def sample(text, probabilities, cached):
    return {"output_sha256": text, "logprobs": probabilities, "cached_tokens": cached}


class MixedCacheProbeTests(unittest.TestCase):
    def test_common_prefix_uses_tokens_not_rendered_text_length(self):
        self.assertEqual(common_prefix_length([1, 2, 3], [1, 2, 4, 5]), 2)
        self.assertEqual(common_prefix_length([1, 2], [1, 2, 3]), 2)

    def test_native_reproduction_requires_same_span_and_probabilities(self):
        cold = sample("OTHER", [1], 0)
        disk = sample("DOGS", [2], 2560)
        native = sample("DOGS", [2], 2560)
        self.assertEqual(classify(cold, disk, [native]), "divergence-reproduced-by-native-cache")
        with self.assertRaisesRegex(ValueError, "span"):
            classify(cold, disk, [sample("DOGS", [2], 4096)])
        with self.assertRaisesRegex(ValueError, "native"):
            classify(cold, disk, [sample("DOGS", [3], 2560)])

    def test_matching_labels_do_not_prove_probability_equivalence(self):
        cold = sample("OTHER", [1], 0)
        disk = sample("OTHER", [2], 2560)
        native = sample("OTHER", [2], 2560)
        self.assertEqual(classify(cold, disk, [native]), "same-output-with-native-matched-probability-drift")

    def test_stable_native_match_is_distinct_from_cold_equivalence(self):
        cold = sample("OTHER", [1], 0)
        disk = sample("OTHER", [1], 2560)
        self.assertEqual(classify(cold, disk, [disk]), "cold-native-disk-equivalent")
        with self.assertRaises(ValueError):
            classify(cold, disk, [])
