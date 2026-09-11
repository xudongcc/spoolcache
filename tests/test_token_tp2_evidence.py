import json
import types
import unittest
from unittest.mock import Mock

from benchmarks.token_tp2_evidence import TokenTP2Evidence, validate_process


class TokenTP2EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.artifact = {"model": "fixture/model", "revision": "a" * 40}
        self.argv = [
            "vllm",
            "serve",
            "fixture/model",
            "--revision",
            "a" * 40,
            "--tensor-parallel-size",
            "2",
        ]
        self.config = {
            "kv_connector": "SpoolCacheConnector",
            "kv_connector_module_path": "spoolcache.vllm.connector",
        }

    def test_pinned_process_and_connector_mode_are_required(self):
        self.assertEqual(validate_process(self.argv, self.artifact, "off"), self.argv)
        on = self.argv + ["--kv-transfer-config", json.dumps(self.config)]
        validate_process(on, self.artifact, "on")
        for command, mode in (
            (self.argv, "on"),
            (on, "off"),
            ([x.replace("a" * 40, "b" * 40) for x in on], "on"),
            (self.argv[:-1] + ["1"], "off"),
        ):
            with (
                self.subTest(command=command, mode=mode),
                self.assertRaises(ValueError),
            ):
                validate_process(command, self.artifact, mode)

    def test_no_reuse_requires_explicitly_disabled_gpu_prefix_cache(self):
        artifact = dict(self.artifact, prefix_caching=False)
        validate_process(self.argv + ["--no-enable-prefix-caching"], artifact, "off")
        for flags in (
            [],
            ["--enable-prefix-caching"],
            ["--enable-prefix-caching", "--no-enable-prefix-caching"],
        ):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate_process(self.argv + flags, artifact, "off")



    def evidence(self):
        args = types.SimpleNamespace(
            head_container="head", worker_container="worker", worker_host="host"
        )
        evidence = TokenTP2Evidence(args, self.artifact)
        evidence.process = Mock(
            side_effect=lambda rank, mode: {"started": "new", "image": "image"}
        )
        evidence.logs = Mock(return_value=[self.log(0), self.log(1)])
        return evidence

    @staticmethod
    def log(rank, tp_rank=None):
        return (
            f"spoolcache: rank identity rank={rank} pp_rank=0 tp_rank={rank if tp_rank is None else tp_rank} dcp_rank=0 "
            f"deployment={'b' * 64} rank_identity={str(rank) * 64} topology={'c' * 64} hma_layout={'d' * 64}\n"
            f"spoolcache: worker ready rank={rank} root=/cache/rank-{rank} transfer_bytes=67108864\n"
            f"spoolcache: HMA runtime layout profile=fixture groups=[{{'index': 0}}] layers=1 alignment=256 logical_digest=eeee physical_digest=dddd\n"
        )

    def test_missing_or_wrong_rank_cannot_qualify(self):
        evidence = self.evidence()
        evidence.logs.return_value[1] = self.log(1).replace(
            "deployment=" + "b" * 64, "deployment=" + "f" * 64
        )
        evidence.refresh()
        self.assertEqual(len(evidence.identities), 2)
        for logs in ([self.log(0)], [self.log(0), self.log(1, tp_rank=0)]):
            evidence.logs.return_value = logs
            with self.subTest(logs=logs), self.assertRaises(ValueError):
                evidence.refresh()

    def test_partial_restart_or_changed_persistent_identity_is_rejected(self):
        evidence = self.evidence()
        evidence.refresh()
        evidence.identities[0]["started"] = "old"
        with self.assertRaisesRegex(ValueError, "both ranks"):
            evidence.refresh(require_restarted=True)
        evidence.identities[1]["started"] = "old"
        evidence.refresh(require_restarted=True)
        for identity in evidence.identities:
            identity["started"] = "older"
        evidence.identities[1]["topology"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "identity changed"):
            evidence.refresh(require_restarted=True)
