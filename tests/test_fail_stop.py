from __future__ import annotations

import unittest
from unittest.mock import patch

from spoolcache.fail_stop import (
    FATAL_RESTORE_EXIT_CODE,
    terminate_worker_after_fatal_restore,
)


class FailStopTests(unittest.TestCase):
    def test_fatal_restore_crosses_worker_process_boundary(self) -> None:
        error = RuntimeError("authenticated payload differs")
        with (
            patch("spoolcache.fail_stop.logger.critical") as log_critical,
            patch("spoolcache.fail_stop.os._exit") as process_exit,
            self.assertRaisesRegex(RuntimeError, "os._exit returned unexpectedly"),
        ):
            terminate_worker_after_fatal_restore(
                error,
                rank=1,
                entry_id="a" * 64,
            )

        process_exit.assert_called_once_with(FATAL_RESTORE_EXIT_CODE)
        self.assertEqual(log_critical.call_args.args[-1], "a" * 12)
        self.assertIs(
            log_critical.call_args.kwargs["exc_info"][1],
            error,
        )


if __name__ == "__main__":
    unittest.main()
