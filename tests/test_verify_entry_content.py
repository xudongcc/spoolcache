from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchmarks.verify_entry_content import (
    VerificationError,
    main as verify_main,
    validate_expected_inputs,
    validate_manifest_identity,
)
from spoolcache.manifest import (
    ObjectDescriptor,
    RankManifest,
    encode_manifest,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def manifest() -> RankManifest:
    object_digest = digest("payload")
    return RankManifest(
        entry_id=digest("entry"),
        deployment_identity_digest=digest("deployment"),
        rank_identity_digest=digest("rank-1"),
        span_tokens=256,
        physical_rank=1,
        topology_digest=digest("topology"),
        profile="vllm-runtime-kv-v1",
        layout_digest=digest("layout"),
        objects=(
            ObjectDescriptor(
                group_index=0,
                layer_name="layer",
                page_start=0,
                page_count=1,
                byte_length=7,
                stored_length=7,
                sha256=object_digest,
                relative_path=(
                    f"objects/{object_digest[:2]}/{object_digest}.spool"
                ),
            ),
        ),
        created_at_unix_ns=1,
    )


class VerifyEntryIdentityTests(unittest.TestCase):
    def _expected(self, value: RankManifest) -> dict[str, object]:
        return {
            "entry_id": value.entry_id,
            "span_tokens": value.span_tokens,
            "deployment_identity_digest": value.deployment_identity_digest,
            "rank_identity_digest": value.rank_identity_digest,
            "physical_rank": value.physical_rank,
            "topology_digest": value.topology_digest,
            "layout_digest": value.layout_digest,
            "tp_degree": 2,
            "pp_degree": 1,
            "dcp_degree": 1,
            "pp_rank": 0,
            "tp_rank": 1,
            "dcp_rank": 0,
        }

    def test_accepts_exact_identity(self) -> None:
        value = manifest()
        validate_manifest_identity(value, **self._expected(value))

    def test_rejects_wrong_deployment_rank_and_layout(self) -> None:
        value = manifest()
        cases = (
            ("deployment_identity_digest", digest("other-deployment"), "deployment"),
            ("rank_identity_digest", digest("other-rank"), "rank identity"),
            ("physical_rank", 0, "physical rank"),
            ("layout_digest", digest("other-layout"), "layout"),
        )
        for field, replacement, message in cases:
            expected = self._expected(value)
            expected[field] = replacement
            with self.subTest(field=field):
                with self.assertRaisesRegex(VerificationError, message):
                    validate_manifest_identity(value, **expected)

    def test_rejects_wrong_topology_and_layout_protocol(self) -> None:
        value = manifest()
        expected = self._expected(value)
        expected["topology_digest"] = digest("other-topology")
        with self.assertRaisesRegex(VerificationError, "topology"):
            validate_manifest_identity(value, **expected)
        with self.assertRaisesRegex(VerificationError, "layout protocol"):
            validate_manifest_identity(
                replace(value, profile="other-profile"),
                **self._expected(value),
            )

    def test_operator_inputs_are_validated_before_path_derivation(self) -> None:
        value = manifest()
        common = {
            "entry_id": value.entry_id,
            "span_tokens": value.span_tokens,
            "deployment_identity_digest": value.deployment_identity_digest,
            "rank_identity_digest": value.rank_identity_digest,
            "physical_rank": value.physical_rank,
            "topology_digest": value.topology_digest,
            "layout_digest": value.layout_digest,
            "groups": 1,
            "layers": 1,
            "group_layers": "1",
            "pages": "1",
            "tp_degree": 2,
            "pp_degree": 1,
            "dcp_degree": 1,
            "pp_rank": 0,
            "tp_rank": 1,
            "dcp_rank": 0,
        }
        self.assertEqual(validate_expected_inputs(**common), ((1,), (1,)))
        stage_local = dict(common)
        stage_local.update(
            groups=2,
            layers=1,
            group_layers="0,1",
            pages="0,1",
        )
        self.assertEqual(
            validate_expected_inputs(**stage_local),
            ((0, 1), (0, 1)),
        )
        cases = (
            ("entry_id", "../" + value.entry_id, "--entry-id"),
            (
                "deployment_identity_digest",
                value.deployment_identity_digest.upper(),
                "deployment-identity",
            ),
            ("rank_identity_digest", "g" * 64, "rank-identity"),
            ("physical_rank", -1, "physical-rank"),
            ("span_tokens", 0, "expected-span"),
            ("groups", 0, "expected-groups"),
            ("group_layers", "2", "group-layers"),
            ("pages", "one", "expected-pages"),
            ("pp_rank", 1, "worker coordinate"),
        )
        for field, replacement, message in cases:
            inputs = dict(common)
            inputs[field] = replacement
            with self.subTest(field=field):
                with self.assertRaisesRegex(VerificationError, message):
                    validate_expected_inputs(**inputs)

    def test_operator_pages_match_stage_local_group_ownership(self) -> None:
        value = manifest()
        common = {
            "entry_id": value.entry_id,
            "span_tokens": value.span_tokens,
            "deployment_identity_digest": value.deployment_identity_digest,
            "rank_identity_digest": value.rank_identity_digest,
            "physical_rank": value.physical_rank,
            "topology_digest": value.topology_digest,
            "layout_digest": value.layout_digest,
            "groups": 2,
            "layers": 1,
            "group_layers": "0,1",
            "tp_degree": 2,
            "pp_degree": 1,
            "dcp_degree": 1,
            "pp_rank": 0,
            "tp_rank": 1,
            "dcp_rank": 0,
        }
        for pages in ("1,1", "0,0"):
            with self.subTest(pages=pages):
                with self.assertRaisesRegex(
                    VerificationError,
                    "zero exactly for stage-local empty groups",
                ):
                    validate_expected_inputs(pages=pages, **common)

    def test_cli_rejects_malformed_entry_before_reading_manifest(self) -> None:
        value = manifest()
        argv = [
            "verify_entry_content.py",
            "--rank-root",
            "/path-that-must-not-be-read",
            "--entry-id",
            "../manifest",
            "--expected-span",
            str(value.span_tokens),
            "--expected-deployment-identity-digest",
            value.deployment_identity_digest,
            "--expected-rank-identity-digest",
            value.rank_identity_digest,
            "--expected-physical-rank",
            str(value.physical_rank),
            "--expected-tp-degree",
            "2",
            "--expected-pp-degree",
            "1",
            "--expected-dcp-degree",
            "1",
            "--expected-pp-rank",
            "0",
            "--expected-tp-rank",
            "1",
            "--expected-dcp-rank",
            "0",
            "--expected-topology-digest",
            value.topology_digest,
            "--expected-layout-digest",
            value.layout_digest,
            "--expected-groups",
            "1",
            "--expected-layers",
            "1",
            "--expected-group-layers",
            "1",
            "--expected-pages",
            "1",
        ]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(SystemExit, "--entry-id"):
                verify_main()

    def test_cli_rejects_identity_before_opening_payloads(self) -> None:
        value = manifest()
        with tempfile.TemporaryDirectory() as directory:
            rank_root = Path(directory)
            manifest_dir = rank_root / "manifests" / value.entry_id[:2]
            manifest_dir.mkdir(parents=True)
            (manifest_dir / f"{value.entry_id}.json").write_bytes(
                encode_manifest(value)
            )
            argv = [
                "verify_entry_content.py",
                "--rank-root",
                str(rank_root),
                "--entry-id",
                value.entry_id,
                "--expected-span",
                str(value.span_tokens),
                "--expected-deployment-identity-digest",
                digest("wrong-deployment"),
                "--expected-rank-identity-digest",
                value.rank_identity_digest,
                "--expected-physical-rank",
                str(value.physical_rank),
                "--expected-tp-degree",
                "2",
                "--expected-pp-degree",
                "1",
                "--expected-dcp-degree",
                "1",
                "--expected-pp-rank",
                "0",
                "--expected-tp-rank",
                "1",
                "--expected-dcp-rank",
                "0",
                "--expected-topology-digest",
                value.topology_digest,
                "--expected-layout-digest",
                value.layout_digest,
                "--expected-groups",
                "1",
                "--expected-layers",
                "1",
                "--expected-group-layers",
                "1",
                "--expected-pages",
                "1",
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(SystemExit, "deployment identity"):
                    verify_main()


if __name__ == "__main__":
    unittest.main()
