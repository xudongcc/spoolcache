"""Process boundary for unrecoverable post-admission restore failures."""

from __future__ import annotations

import logging
import os
from typing import NoReturn


logger = logging.getLogger(__name__)

# EX_SOFTWARE is conventional but is not available on every Python platform.
# Keep the value local and stable for process managers and diagnostics.
FATAL_RESTORE_EXIT_CODE = 70


def terminate_worker_after_fatal_restore(
    error: BaseException,
    *,
    rank: int | None,
    entry_id: str,
) -> NoReturn:
    """Leave vLLM's catch-and-continue worker RPC loop after a fatal restore.

    Raising from a connector hook is insufficient: vLLM's multiprocessing
    worker catches ordinary exceptions, while another TP rank may already be
    blocked in a model collective.  A hard worker exit activates vLLM's worker
    monitor, which tears down the complete executor instead of serving with
    partially restored KV state.
    """

    logger.critical(
        "spoolcache: post-admission restore failed; terminating worker "
        "pid=%d rank=%s entry=%s",
        os.getpid(),
        rank,
        entry_id[:12],
        exc_info=(type(error), error, error.__traceback__),
    )
    os._exit(FATAL_RESTORE_EXIT_CODE)
    raise RuntimeError("os._exit returned unexpectedly")
