"""The fixed live regression must reject misleading cache/output evidence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.run_gemma_e2e import CASES, Regression, check_sample


class Evidence:
    def manifests(self):
        return [[]]

    def verify(self, phase, samples):
        pass


class FakeRegression(Regression):
    def __init__(self, output):
        self.output = Path(output)
        self.nonce = "test"
        self.evidence = Evidence()
        self.events = []
        self.controls = {}
        self.restored = {}

    def reset(self):
        self.events.append("reset")

    def restart(self):
        self.events.append("restart")

    def request(self, case, phase, salt, skip_read=False, skip_write=False):
        self.events.append((case.name, phase, salt, skip_read, skip_write))
        cached = 0
        if phase == "consumer" and not skip_read and "no-write" not in salt:
            cached = case.hit
        return {"cached_tokens": cached, "prompt_tokens": 4128,
                "output_sha256": "a" * 64, "request_id": "test-request"}


class GemmaRegressionTests(unittest.TestCase):
    def test_strict_cached_token_and_output_evidence(self):
        valid = {"cached_tokens": 2048, "prompt_tokens": 4128,
                 "output_sha256": "a" * 64, "request_id": "r"}
        check_sample(valid, 2048, "a" * 64)
        for change in ({"cached_tokens": True}, {"cached_tokens": 0},
                       {"prompt_tokens": 100}, {"output_sha256": "b" * 64},
                       {"request_id": ""}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_sample({**valid, **change}, 2048, "a" * 64)

    def test_full_matrix_restart_and_independent_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRegression(directory)
            runner.execute()
            restart = runner.events.index("restart")
            after = [e for e in runner.events[restart + 1:] if isinstance(e, tuple)]
            self.assertEqual([e[0] for e in after], [c.name for c in CASES])
            self.assertTrue(all(e[1] == "consumer" for e in after))
            requests = [e for e in runner.events if isinstance(e, tuple)]
            self.assertTrue(any(e[3:] == (True, False) for e in requests))
            self.assertTrue(any(e[3:] == (False, True) for e in requests))
            self.assertTrue(any(e[3:] == (True, True) for e in requests))

    def test_unstable_cold_control_stops_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRegression(directory)
            original = runner.request
            def unstable(*args, **kwargs):
                row = original(*args, **kwargs)
                if args[2].endswith("control2"):
                    row["output_sha256"] = "b" * 64
                return row
            with patch.object(runner, "request", side_effect=unstable):
                with self.assertRaisesRegex(ValueError, "output"):
                    runner.execute()
            self.assertNotIn("restart", runner.events)
            self.assertFalse(any(isinstance(e, tuple) and e[1] == "producer"
                                 for e in runner.events))

    def test_changed_post_restart_output_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRegression(directory)
            original = runner.request
            def changed(*args, **kwargs):
                row = original(*args, **kwargs)
                if "restart" in runner.events:
                    row["output_sha256"] = "b" * 64
                return row
            with patch.object(runner, "request", side_effect=changed):
                with self.assertRaisesRegex(ValueError, "output"):
                    runner.execute()
            self.assertFalse((Path(directory) / "summary.json").exists())

    def test_disabled_writes_must_not_create_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRegression(directory)
            with patch.object(runner.evidence, "manifests", side_effect=[[[]], [["new"]]]):
                with self.assertRaisesRegex(ValueError, "manifest"):
                    runner.flags()


class RankEvidenceTests(unittest.TestCase):
    def test_missing_rank_restore_cannot_pass_on_api_hit(self):
        from types import SimpleNamespace
        from benchmarks.gemma_e2e_evidence import Evidence as RankEvidence
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(head_container="head", worker_container="worker",
                                   topology="pp2", output=Path(directory))
            evidence = RankEvidence(args, "model", "revision")
            evidence.identities = [{"root": "/cache", "groups": []}, {}]
            logs = ["spoolcache: hit request=req tokens=2048 entry=abcd\n", ""]
            with patch.object(evidence, "logs", return_value=logs):
                with self.assertRaisesRegex(ValueError, "rank 0 restore"):
                    evidence.verify("restore", {"text": {"request_id": "req", "cached_tokens": 2048}})

    def test_conflicting_scheduler_entries_fail(self):
        from types import SimpleNamespace
        from benchmarks.gemma_e2e_evidence import Evidence as RankEvidence
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(head_container="head", topology="pp1", output=Path(directory))
            evidence = RankEvidence(args, "model", "revision")
            log = ("spoolcache: hit request=req tokens=2048 entry=abcd\n"
                   "spoolcache: hit request=req tokens=2048 entry=efab\n")
            with patch.object(evidence, "logs", return_value=[log]):
                with self.assertRaisesRegex(ValueError, "scheduler"):
                    evidence.verify("restore", {"text": {"request_id": "req", "cached_tokens": 2048}})

    def test_pp_stage_deployments_can_differ_but_restart_must_cover_both(self):
        import json
        from types import SimpleNamespace
        from benchmarks.gemma_e2e_evidence import Evidence as RankEvidence
        args = SimpleNamespace(head_container="head", worker_container="worker", topology="pp2",
                               image_id="image")
        evidence = RankEvidence(args, "model", "revision")
        logs = [f"spoolcache: rank identity rank={rank} pp_rank={rank} deployment=stage{rank} "
                f"rank_identity=rank{rank} topology=shared hma_layout=aa\n"
                f"spoolcache: worker ready rank={rank} root=/cache/{rank}\n"
                "spoolcache: HMA runtime layout profile=fixture groups=[] layers=0 alignment=32 "
                "logical_digest=bb physical_digest=aa\n" for rank in (0, 1)]
        def states(started):
            return [json.dumps({"image": "image", "running": True, "started": start,
                                "cmd": ["model", "--revision", "revision"]}) for start in started]
        with patch.object(evidence, "logs", return_value=logs):
            with patch.object(evidence, "call", side_effect=states(["old", "old"])):
                evidence.refresh()
            self.assertNotEqual(evidence.identities[0]["deployment"], evidence.identities[1]["deployment"])
            with patch.object(evidence, "call", side_effect=states(["new", "old"])):
                with self.assertRaisesRegex(ValueError, "rank 1.*restart"):
                    evidence.refresh(require_restarted=True)
            with patch.object(evidence, "call", side_effect=states(["new", "new"])):
                evidence.refresh(require_restarted=True)

    def test_prompt_is_fixed_and_request_provenance_is_retained(self):
        import json
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), api="http://fixture",
                                   head_container="head", topology="pp1")
            runner = Regression(args)
            self.assertEqual(json.loads((Path(directory) / "run.json").read_text())["nonce"], runner.nonce)
            result = SimpleNamespace(returncode=0, stdout='{"cached_tokens":0}')
            with patch("benchmarks.run_gemma_e2e.subprocess.run", return_value=result):
                first = runner.request(CASES[0], "consumer", "salt-one", True, True)
                runner.nonce = "different-run"
                second = runner.request(CASES[0], "consumer", "salt-two", True, True)
            self.assertEqual(first["nonce"], second["nonce"])
            self.assertNotEqual(first["cache_salt"], second["cache_salt"])
            self.assertIn(first["nonce"], first["command"])
            self.assertIn("--skip-read", first["command"])
            self.assertIn("--skip-write", first["command"])
