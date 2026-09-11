"""Resumable bounded scrub of self-contained token files.

No reference set or object queue is needed: one key has one inode. A durable
lexical key cursor advances only after an entire payload was authenticated.
Files inserted behind the cursor are picked up in the next cycle. The bounded
selection pass trades repeated namespace scans for fixed metadata memory.
"""

import heapq
import os
import time
import uuid
from types import SimpleNamespace

from .errors import ManifestError
from .identity import canonical_json
from .maintenance import ScrubStepReport
from .rank_store import _fsync_directory, _is_digest, _read_small_json


class TokenFileScrubber:
    def __init__(self, store):
        self.store = store
        self.path = store.root / "state" / "token-scrub.json"
        with store._exclusive():
            if not self.path.exists():
                self._save(self._empty())

    def _empty(self):
        return {
            "schema": "spoolcache-token-scrub/v1",
            "binding": self.store._binding,
            "phase": "idle",
            "cycle": 0,
            "cursor": "",
            "target_entry": "",
            "last_completed_unix_ns": 0,
            "last_request_status": "",
        }

    def _load(self):
        state = _read_small_json(self.path, maximum=4096)
        expected = self._empty()
        if (
            not isinstance(state, dict)
            or set(state) != set(expected)
            or state["schema"] != expected["schema"]
            or state["binding"] != expected["binding"]
            or state["phase"] not in ("idle", "files")
            or any(
                type(state[k]) is not int or not 0 <= state[k] < (1 << 63)
                for k in ("cycle", "last_completed_unix_ns")
            )
            or any(
                state[k] != "" and not _is_digest(state[k])
                for k in ("cursor", "target_entry")
            )
            or state["last_request_status"]
            not in ("", "authenticated", "absent", "quarantined")
        ):
            raise ManifestError("invalid token scrub state")
        return state

    def _save(self, state):
        self.store._atomic_small_write(self.path, canonical_json(state))

    def status(self):
        with self.store._exclusive():
            return SimpleNamespace(**self._load())

    def repair_invalid_state(self):
        with self.store._exclusive():
            try:
                self._load()
                return
            except (OSError, ValueError, ManifestError):
                pass
            try:
                os.replace(
                    self.path,
                    self.store.root
                    / "quarantine"
                    / f"token-scrub-{uuid.uuid4().hex}.bad",
                )
                _fsync_directory(self.store.root / "quarantine")
                _fsync_directory(self.path.parent)
            except FileNotFoundError:
                pass
            self._save(self._empty())

    def start_cycle(self):
        with self.store._exclusive():
            state = self._load()
            if state["phase"] == "idle":
                state.update(phase="files", cursor="", cycle=state["cycle"] + 1)
                self._save(state)
            return state["cycle"]

    def request(self, entry_id):
        if not _is_digest(entry_id):
            raise ValueError("malformed token scrub key")
        with self.store._exclusive():
            state = self._load()
            if state["target_entry"]:
                raise ValueError("a token scrub request is already pending")
            state.update(target_entry=entry_id, last_request_status="")
            self._save(state)
        return entry_id

    def has_pending_request(self):
        return bool(self.status().target_entry)

    def close(self):
        pass

    def step(
        self,
        *,
        payload_budget_bytes,
        item_budget,
        on_payload_read=None,
        cancel_requested=None,
    ):
        if type(item_budget) is not int or not 0 < item_budget <= 64:
            raise ValueError("token scrub item bound must be in 1..64")
        if type(payload_budget_bytes) is not int or payload_budget_bytes <= 0:
            raise ValueError("token scrub byte bound must be positive")
        with self.store._exclusive():
            state = self._load()
            target = state["target_entry"]
            keys = (
                [target]
                if target
                else heapq.nsmallest(
                    item_budget,
                    (
                        p.stem
                        for p in self.store.iter_manifest_paths()
                        if p.stem > state["cursor"]
                    ),
                )
            )
        checked = authenticated = quarantined = payload_bytes = released = 0
        for key in keys:
            if cancel_requested is not None and cancel_requested():
                break
            if checked and payload_bytes >= payload_budget_bytes:
                break
            status, length = "absent", 0
            with self.store._exclusive():
                if self._load() != state:
                    # A concurrent target request or another maintenance
                    # process changed durable work. Never overwrite its state.
                    break
                # An active read owns this inode through its final CUDA drain.
                # Defer at this cursor so corruption is not silently skipped.
                if self.store._pins.busy(key):
                    break
                try:
                    fd, desc = self.store._open_chunk(key, allow_withdrawn=True)
                    try:
                        with self.store._pool.acquire(block=True) as slot:
                            self.store._authenticate(fd, desc, slot)
                        length = desc.byte_length
                    finally:
                        os.close(fd)
                    # A complete re-hash of the exact self-contained key is
                    # sufficient to release its provenance-free tombstone.
                    if self.store._inventory_is_withdrawn(key):
                        self.store._clear_inventory_withdrawal(key)
                        released += 1
                    status = "authenticated"
                    authenticated += 1
                except FileNotFoundError:
                    pass
                except (OSError, ManifestError):
                    self.store._quarantine_bad(key)
                    status = "quarantined"
                    quarantined += 1
                if target:
                    state.update(target_entry="", last_request_status=status)
                else:
                    state["cursor"] = key
                self._save(state)
            checked += 1
            payload_bytes += length
            if length and on_payload_read is not None:
                on_payload_read(length)
        completed = not target and not keys and state["phase"] == "files"
        if completed:
            with self.store._exclusive():
                if self._load() == state:
                    state.update(
                        phase="idle", cursor="", last_completed_unix_ns=time.time_ns()
                    )
                    self._save(state)
                else:
                    completed = False
        return ScrubStepReport(
            namespace_items_scanned=checked,
            payload_bytes=payload_bytes,
            manifests_examined=checked,
            manifests_authenticated=authenticated,
            objects_authenticated=authenticated,
            entries_quarantined=quarantined,
            objects_quarantined=quarantined,
            inventory_released=released,
            cycle_completed=completed,
            request_completed=bool(target and checked),
            request_status=state["last_request_status"] if target and checked else "",
        )
