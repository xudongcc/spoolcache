from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "gemma-pp2-dev.sh"


class GemmaPp2DevLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_syntax_and_help_do_not_require_lab_access(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        result = subprocess.run(
            [str(LAUNCHER), "--help"],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertNotIn("source-sync", result.stdout)
        self.assertIn("model-sync", result.stdout)
        self.assertIn("image-sync", result.stdout)

    def test_reproducible_g4_development_baseline_is_fixed(self) -> None:
        expected_fragments = (
            'readonly MODEL_ID="google/gemma-4-E2B-it"',
            'readonly MODEL_REVISION="3e22461f65e89153144f8adb70e3b8c2cc9845a7"',
            'readonly IMAGE="spoolcache-dev:vllm-0.28.0"',
            'readonly PP_LAYER_PARTITION="12,23"',
            "--tensor-parallel-size 1",
            "--pipeline-parallel-size 2",
            '\\"kv_connector\\":\\"SpoolCacheConnector\\"',
            '\\"kv_connector_module_path\\":\\"spoolcache.vllm.connector\\"',
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_worker_is_started_before_head_and_stopped_before_head(self) -> None:
        worker_start = self.source.index('remote_exec "${worker_args[@]}"')
        head_start = self.source.index('"${head_args[@]}"')
        self.assertLess(worker_start, head_start)

        stop_function = self.source.split("stop_group() {", 1)[1].split(
            "status_group() {", 1
        )[0]
        self.assertLess(
            stop_function.index('remote_exec docker rm -f "$WORKER_CONTAINER"'),
            stop_function.index('docker rm -f "$HEAD_CONTAINER"'),
        )

    def test_serving_uses_installed_artifact(self) -> None:
        for removed in ("source_sync", "SOURCE_LINK", "PYTHONPATH", "snapshot.XXXXXX"):
            self.assertNotIn(removed, self.source)

    def test_cache_path_resolves_against_each_hosts_home(self) -> None:
        resolver = "spoolcache_host_path() {" + self.source.split(
            "spoolcache_host_path() {", 1
        )[1].split("\n}\n", 1)[0] + "\n}\n"
        default = next(
            line for line in self.source.splitlines()
            if line.startswith("SPOOLCACHE_PATH=")
        )
        for raw, suffix, valid in (
            (None, ".cache/spoolcache", True),
            ("~/custom", "custom", True),
            ("/mnt/nvme/spoolcache", None, True),
            ("", None, False),
            ("relative", None, False),
            ("/cache/$(false)", None, False),
        ):
            for home in ("/home/head", "/home/worker"):
                with self.subTest(raw=raw, home=home):
                    environment = {"PATH": "/usr/bin:/bin"}
                    if raw is not None:
                        environment["SPOOLCACHE_PATH"] = raw
                    result = subprocess.run(
                        ["bash", "-c", "set -eu\n" + resolver + default
                         + '\nspoolcache_host_path "$SPOOLCACHE_PATH" "$1"',
                         "test", home],
                        env=environment, capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0 if valid else 2, result.stderr)
                    if valid:
                        expected = f"{home}/{suffix}" if suffix else raw
                        self.assertEqual(result.stdout.strip(), expected)

    def test_release_gate_freezes_one_image_and_rejects_missing_wheels(self) -> None:
        gate = self.source.split("    local head_image_id worker_image_id", 1)[1].split(
            "    ensure_not_running", 1
        )[0]
        prelude = r'''
set -euo pipefail
IMAGE=mutable-release-tag
fail() { echo "$*" >&2; exit 2; }
docker() {
  if [[ "$*" == *io.spoolcache.wheel.sha256* ]]; then printf '%s\n' "$WHEEL";
  else printf '%s\n' "$LOCAL_IMAGE"; fi
}
remote_exec() { printf '%s\n' "$REMOTE_IMAGE"; }
check_gate() {
'''
        for local, remote, wheel, expected in (
            ("sha256:head", "sha256:head", "a" * 64, 0),
            ("sha256:head", "sha256:other", "a" * 64, 2),
            ("", "sha256:head", "a" * 64, 2),
            ("sha256:head", "sha256:head", "", 2),
            ("sha256:head", "sha256:head", "not-a-sha256", 2),
        ):
            with self.subTest(local=local, remote=remote, wheel=wheel):
                result = subprocess.run(
                    ["bash", "-c", prelude + gate
                     + '\n}\ncheck_gate\n[[ "$RUN_IMAGE_ID" == "sha256:head" ]]'],
                    env={"PATH": "/usr/bin:/bin", "LOCAL_IMAGE": local,
                         "REMOTE_IMAGE": remote, "WHEEL": wheel},
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, expected, result.stderr)

    def test_launcher_uses_cx7_and_has_no_supervision_policy(self) -> None:
        expected_fragments = (
            "--device /dev/infiniband:/dev/infiniband",
            "NCCL_SOCKET_IFNAME",
            "NCCL_IB_HCA",
            "NCCL_CROSS_NIC=1",
            'docker save "$IMAGE" | remote_exec docker load',
            "rsync -a --partial --info=progress2",
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)
        self.assertNotIn("--restart", self.source)
        self.assertNotIn("systemctl", self.source)

    def test_status_fallback_is_valid_json(self) -> None:
        self.assertIn('state="absent"', self.source)
        self.assertIn(
            "{\"head\":\"%s\",\"worker\":\"%s\",\"api_ready\":%s",
            self.source,
        )

    def test_model_policy_does_not_enter_connector_core(self) -> None:
        production_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "src" / "spoolcache").rglob("*.py"))
        ).lower()
        for model_family in ("gemma", "qwen", "deepseek", "glm"):
            with self.subTest(model_family=model_family):
                self.assertNotIn(model_family, production_source)


if __name__ == "__main__":
    unittest.main()
