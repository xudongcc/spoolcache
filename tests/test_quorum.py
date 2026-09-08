from __future__ import annotations

import hashlib
import unittest

from spoolcache.quorum import (
    InventoryCheckpoint,
    InventoryDelta,
    InventoryReporter,
    QuorumCatalog,
    WorkerInventoryReport,
)


def key(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def reporter(
    *,
    rank: int = 0,
    generation: str = "boot",
    generation_epoch: int = 1,
    max_entries: int = 10,
    max_report_entries: int = 10,
) -> InventoryReporter:
    return InventoryReporter(
        rank=rank,
        generation=generation,
        generation_epoch=generation_epoch,
        max_entries=max_entries,
        max_report_entries=max_report_entries,
    )


def catalog(
    *, expected_ranks: tuple[int, ...], max_entries: int, max_report_entries: int = 10
) -> QuorumCatalog:
    return QuorumCatalog(
        expected_ranks=expected_ranks,
        max_entries=max_entries,
        max_report_entries=max_report_entries,
    )


class QuorumTests(unittest.TestCase):
    def test_identity_and_batch_bounds_reject_bools_and_long_generations(self) -> None:
        with self.assertRaises(ValueError):
            reporter(
                rank=0,
                generation="g" * 129,
                generation_epoch=1,
            )
        worker = reporter(
            rank=0, generation="boot", generation_epoch=1
        )
        for invalid in (True, 0, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    worker.startup(invalid)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    worker.next_report(invalid)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            catalog(expected_ranks=(0,), max_entries=True)  # type: ignore[arg-type]
        for kwargs in (
            {"max_entries": 0},
            {"max_report_entries": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    reporter(**kwargs)  # type: ignore[arg-type]

        scheduler = catalog(expected_ranks=(0,), max_entries=2)
        with self.assertRaises(ValueError):
            scheduler.apply_startup(
                rank=0,
                generation="g" * 129,
                generation_epoch=1,
                entries=(),
            )

    def test_malformed_generation_identity_withdraws_existing_rank_image(self) -> None:
        entry = key("previously-admitted")
        for transport in ("startup", "report"):
            with self.subTest(transport=transport):
                scheduler = catalog(expected_ranks=(0,), max_entries=2)
                scheduler.apply_startup(
                    rank=0,
                    generation="valid",
                    generation_epoch=1,
                    entries=((entry, 256),),
                )
                self.assertTrue(scheduler.has_quorum(entry))
                with self.assertRaises(ValueError):
                    if transport == "startup":
                        scheduler.apply_startup(
                            rank=0,
                            generation="",
                            generation_epoch=2,
                            entries=(),
                        )
                    else:
                        scheduler.apply_report(
                            WorkerInventoryReport(
                                rank=0,
                                generation="",
                                generation_epoch=2,
                                checkpoint=InventoryCheckpoint(
                                    0, 0, 0, 1, 0, ()
                                ),
                            )
                        )
                self.assertFalse(scheduler.is_ready)
                self.assertFalse(scheduler.has_quorum(entry))

    def test_boolean_or_duplicate_rank_identity_is_rejected(self) -> None:
        for expected in ((0, True), (0, 0), (False,), (1,)):
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError):
                    catalog(expected_ranks=expected, max_entries=10)

        scheduler = catalog(expected_ranks=(0, 1), max_entries=10)
        with self.assertRaises(ValueError):
            scheduler.apply_startup(
                rank=True,
                generation="boot",
                generation_epoch=1,
                entries=(),
            )
        with self.assertRaises(ValueError):
            scheduler.apply_report(
                WorkerInventoryReport(
                    rank=True,
                    generation="boot",
                    generation_epoch=1,
                    checkpoint=InventoryCheckpoint(0, 0, 0, 1, 0, ()),
                )
            )

    def test_all_physical_ranks_and_longest_prefix_are_required(self) -> None:
        short, long = key("short"), key("long")
        scheduler = catalog(expected_ranks=(0, 1), max_entries=10)
        scheduler.apply_startup(
            rank=0,
            generation="boot-a",
            generation_epoch=1,
            entries=((short, 256), (long, 512)),
        )
        self.assertFalse(scheduler.has_quorum(short))
        scheduler.apply_startup(
            rank=1,
            generation="boot-b",
            generation_epoch=1,
            entries=((short, 256), (long, 512)),
        )
        self.assertTrue(scheduler.has_quorum(long, 512))
        self.assertEqual(scheduler.longest(((256, short), (512, long))), (512, long))

    def test_generation_change_withdraws_until_complete_checkpoint(self) -> None:
        first, second = key("first"), key("second")
        scheduler = catalog(expected_ranks=(0, 1), max_entries=10)
        for rank in (0, 1):
            scheduler.apply_startup(
                rank=rank,
                generation="old",
                generation_epoch=1,
                entries=((first, 256),),
            )
        report = WorkerInventoryReport(
            rank=1,
            generation="new",
            generation_epoch=2,
            checkpoint=InventoryCheckpoint(1, 1, 0, 2, 2, ((first, 256),)),
            delta=InventoryDelta(1, 0, ((second, 512),), ()),
        )
        scheduler.apply_report(report)
        self.assertEqual(scheduler.generation_changes_total, 1)
        self.assertEqual(scheduler.ready_rank_count, 1)
        self.assertFalse(scheduler.is_ready)
        self.assertFalse(scheduler.has_quorum(first))
        self.assertIn(1, scheduler.desynchronized_ranks)
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=1,
                generation="new",
                generation_epoch=2,
                checkpoint=InventoryCheckpoint(1, 1, 1, 2, 2, ((second, 512),)),
            )
        )
        self.assertEqual(scheduler.generation_changes_total, 1)
        self.assertEqual(scheduler.ready_rank_count, 2)
        self.assertTrue(scheduler.is_ready)
        self.assertTrue(scheduler.has_quorum(first, 256))
        self.assertNotIn(1, scheduler.desynchronized_ranks)

    def test_delayed_older_generation_cannot_roll_back_current_catalog(self) -> None:
        old, new = key("old-generation"), key("new-generation")
        scheduler = catalog(expected_ranks=(0,), max_entries=10)
        scheduler.apply_startup(
            rank=0,
            generation="old",
            generation_epoch=100,
            entries=((old, 256),),
        )
        scheduler.apply_startup(
            rank=0,
            generation="new",
            generation_epoch=200,
            entries=((new, 512),),
        )
        self.assertTrue(scheduler.has_quorum(new, 512))

        # Both the out-of-band handshake and periodic transport can arrive
        # late. Neither may mutate or transiently withdraw the newer image.
        scheduler.apply_startup(
            rank=0,
            generation="old",
            generation_epoch=100,
            entries=((old, 256),),
        )
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=0,
                generation="old",
                generation_epoch=100,
                checkpoint=InventoryCheckpoint(
                    0, 1, 0, 1, 1, ((old, 256),)
                ),
            )
        )
        self.assertTrue(scheduler.is_ready)
        self.assertTrue(scheduler.has_quorum(new, 512))
        self.assertFalse(scheduler.has_quorum(old))

    def test_unknown_lower_generation_withdraws_instead_of_preserving_ghosts(self) -> None:
        stale = key("state-lost-old")
        scheduler = catalog(expected_ranks=(0,), max_entries=10)
        scheduler.apply_startup(
            rank=0,
            generation="old-process",
            generation_epoch=200,
            entries=((stale, 256),),
        )

        # This UUID was never observed by the scheduler. A lower epoch can be
        # a live replacement after state loss, not a provably delayed packet.
        scheduler.apply_startup(
            rank=0,
            generation="unknown-live-replacement",
            generation_epoch=100,
            entries=(),
        )
        self.assertFalse(scheduler.is_ready)
        self.assertFalse(scheduler.has_quorum(stale))
        self.assertIn(0, scheduler.desynchronized_ranks)

    def test_generation_history_is_bounded_and_only_known_old_identity_is_stale(self) -> None:
        scheduler = catalog(expected_ranks=(0,), max_entries=10)
        for epoch in range(1, 70):
            scheduler.apply_startup(
                rank=0,
                generation=f"generation-{epoch}",
                generation_epoch=epoch,
                entries=((key(str(epoch)), 256),),
            )
        self.assertLessEqual(len(scheduler._generation_history[0]), 64)

        # A recent exact identity is a delayed packet and remains harmless.
        scheduler.apply_startup(
            rank=0,
            generation="generation-68",
            generation_epoch=68,
            entries=((key("unexpected"), 512),),
        )
        self.assertTrue(scheduler.has_quorum(key("69"), 256))

        # An evicted/unknown identity cannot be ordered and must fail closed.
        scheduler.apply_startup(
            rank=0,
            generation="generation-1",
            generation_epoch=1,
            entries=(),
        )
        self.assertFalse(scheduler.is_ready)
        self.assertFalse(scheduler.has_quorum(key("69")))

    def test_equal_epoch_different_generation_stays_withdrawn_until_newer(self) -> None:
        first, conflict, recovered = (
            key("first-generation"),
            key("conflicting-generation"),
            key("recovered-generation"),
        )
        scheduler = catalog(expected_ranks=(0,), max_entries=10)
        scheduler.apply_startup(
            rank=0,
            generation="boot-a",
            generation_epoch=100,
            entries=((first, 256),),
        )
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=0,
                generation="boot-b",
                generation_epoch=100,
                checkpoint=InventoryCheckpoint(
                    0, 1, 0, 1, 1, ((conflict, 512),)
                ),
            )
        )
        self.assertFalse(scheduler.is_ready)
        self.assertEqual(scheduler.quorum_count, 0)

        # Neither contender at the ambiguous epoch can heal the conflict.
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=0,
                generation="boot-a",
                generation_epoch=100,
                checkpoint=InventoryCheckpoint(
                    0, 2, 0, 1, 1, ((first, 256),)
                ),
            )
        )
        self.assertFalse(scheduler.is_ready)

        scheduler.apply_startup(
            rank=0,
            generation="boot-c",
            generation_epoch=101,
            entries=((recovered, 768),),
        )
        self.assertTrue(scheduler.is_ready)
        self.assertTrue(scheduler.has_quorum(recovered, 768))

    def test_delta_gap_cannot_be_healed_by_a_later_delta(self) -> None:
        first, second = key("first"), key("second")
        scheduler = catalog(expected_ranks=(0,), max_entries=10)
        scheduler.apply_startup(
            rank=0,
            generation="boot",
            generation_epoch=1,
            entries=((first, 256),),
        )
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=0,
                generation="boot",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(3, 1, 0, 2, 2, ((first, 256),)),
                delta=InventoryDelta(3, 2, ((second, 512),), ()),
            )
        )
        self.assertEqual(scheduler.generation_changes_total, 0)
        self.assertEqual(scheduler.ready_rank_count, 0)
        self.assertFalse(scheduler.is_ready)
        self.assertIn(0, scheduler.desynchronized_ranks)
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=0,
                generation="boot",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(3, 1, 0, 2, 2, ((first, 256),)),
                delta=InventoryDelta(1, 0, ((second, 512),), ()),
            )
        )
        self.assertFalse(scheduler.has_quorum(second))
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=0,
                generation="boot",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(3, 1, 1, 2, 2, ((second, 512),)),
            )
        )
        self.assertTrue(scheduler.has_quorum(second, 512))

    def test_reporter_bounds_each_checkpoint_page(self) -> None:
        worker = reporter(max_entries=10, max_report_entries=2)
        entries = {key(str(index)): (index + 1) * 256 for index in range(5)}
        worker.replace(entries)
        worker.startup(5)
        pages = [worker.next_report(2).checkpoint for _ in range(3)]
        self.assertTrue(all(len(page.entries) <= 2 for page in pages))
        self.assertEqual({page.index for page in pages}, {0, 1, 2})

    def test_startup_inventory_seeds_the_first_periodic_checkpoint(self) -> None:
        worker = reporter(max_entries=4, max_report_entries=2)
        entries = {key(str(index)): (index + 1) * 256 for index in range(4)}
        worker.replace(entries)
        startup = worker.startup(4)
        first = worker.next_report(2)
        self.assertIsNone(first.delta)
        self.assertEqual(first.checkpoint.sequence, 0)
        self.assertEqual(first.checkpoint.held_count, 4)
        self.assertEqual(first.checkpoint.entries, startup[:2])

    def test_bounded_startup_subset_rolls_forward_without_rank_withdrawal(self) -> None:
        worker = reporter(max_entries=4, max_report_entries=2)
        entries = {key(str(index)): (index + 1) * 256 for index in range(4)}
        worker.replace(entries)
        startup = worker.startup(2)
        scheduler = catalog(
            expected_ranks=(0,), max_entries=4, max_report_entries=2
        )
        scheduler.apply_startup(
            rank=0,
            generation="boot",
            generation_epoch=1,
            entries=startup,
        )

        first = worker.next_report(2)
        self.assertIsNone(first.delta)
        scheduler.apply_report(first)
        self.assertTrue(scheduler.is_ready)
        self.assertEqual(scheduler.quorum_count, 2)

        scheduler.apply_report(worker.next_report(2))
        self.assertTrue(scheduler.is_ready)
        self.assertEqual(scheduler.quorum_count, 4)

    def test_reporter_caps_held_state_and_withdraws_on_oversized_delta(self) -> None:
        initial = {key(label): 256 for label in ("a", "b", "c")}
        replacement = {key(label): 512 for label in ("d", "e", "f", "g")}
        worker = reporter(max_entries=3, max_report_entries=2)
        worker.replace(initial)
        startup = worker.startup(3)
        scheduler = catalog(
            expected_ranks=(0,), max_entries=3, max_report_entries=2
        )
        scheduler.apply_startup(
            rank=0,
            generation="boot",
            generation_epoch=1,
            entries=startup,
        )

        worker.replace(replacement)
        first = worker.next_report(2)
        self.assertIsNotNone(first.delta)
        assert first.delta is not None
        self.assertEqual(first.delta.added, ())
        self.assertEqual(first.delta.removed, ())
        self.assertEqual(first.delta.sequence, first.delta.base_sequence + 1)
        self.assertGreater(first.delta.base_sequence, 0)
        self.assertEqual(first.checkpoint.held_count, 3)
        self.assertLessEqual(len(first.checkpoint.entries), 2)
        scheduler.apply_report(first)
        self.assertFalse(scheduler.is_ready)
        self.assertEqual(scheduler.quorum_count, 0)

        second = worker.next_report(2)
        self.assertLessEqual(len(second.checkpoint.entries), 2)
        scheduler.apply_report(second)
        self.assertTrue(scheduler.is_ready)
        self.assertEqual(scheduler.quorum_count, 3)
        self.assertFalse(any(scheduler.has_quorum(entry) for entry in initial))

    def test_new_withdrawal_delta_is_sent_before_retained_history(self) -> None:
        first, second = key("first-held"), key("second-held")
        worker = reporter(max_entries=4, max_report_entries=4)
        worker.replace({first: 256})
        scheduler = catalog(
            expected_ranks=(0,),
            max_entries=4,
            max_report_entries=4,
        )
        scheduler.apply_startup(
            rank=0,
            generation=worker.generation,
            generation_epoch=worker.generation_epoch,
            entries=worker.startup(4),
        )

        worker.add(second, 512)
        added = worker.next_report(4)
        self.assertIsNotNone(added.delta)
        scheduler.apply_report(added)
        self.assertTrue(scheduler.has_quorum(first))
        self.assertTrue(scheduler.has_quorum(second))

        worker.remove(first)
        withdrawn = worker.next_report(4)
        self.assertIsNotNone(withdrawn.delta)
        assert withdrawn.delta is not None
        self.assertEqual(withdrawn.delta.sequence, 2)
        self.assertEqual(withdrawn.delta.removed, (first,))
        scheduler.apply_report(withdrawn)
        self.assertFalse(scheduler.has_quorum(first))
        self.assertTrue(scheduler.has_quorum(second))

    def test_pending_checkpoint_cannot_accumulate_more_than_held_count(self) -> None:
        old, first, second, overflow = (
            key("old"),
            key("first"),
            key("second"),
            key("overflow"),
        )
        scheduler = catalog(
            expected_ranks=(0,), max_entries=4, max_report_entries=2
        )
        scheduler.apply_startup(
            rank=0,
            generation="boot",
            generation_epoch=1,
            entries=((old, 256),),
        )
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=0,
                generation="boot",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(
                    1, 1, 0, 2, 2, ((first, 256), (second, 512))
                ),
            )
        )
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=0,
                generation="boot",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(
                    1, 1, 1, 2, 2, ((overflow, 768),)
                ),
            )
        )
        self.assertFalse(scheduler.is_ready)
        self.assertEqual(scheduler.quorum_count, 0)
        self.assertNotIn(0, scheduler._pending)

    def test_report_page_and_combined_delta_have_independent_hard_bounds(self) -> None:
        entries = tuple((key(str(index)), 256) for index in range(3))
        malformed = (
            WorkerInventoryReport(
                rank=0,
                generation="boot",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(0, 1, 0, 1, 3, entries),
            ),
            WorkerInventoryReport(
                rank=0,
                generation="boot",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(0, 1, 0, 1, 1, entries[:1]),
                delta=InventoryDelta(1, 0, entries[:2], (entries[2][0],)),
            ),
        )
        for report in malformed:
            with self.subTest(report=report):
                scheduler = catalog(
                    expected_ranks=(0,), max_entries=4, max_report_entries=2
                )
                scheduler.apply_startup(
                    rank=0,
                    generation="boot",
                    generation_epoch=1,
                    entries=((key("old"), 256),),
                )
                scheduler.apply_report(report)
                self.assertFalse(scheduler.is_ready)
                self.assertEqual(scheduler.quorum_count, 0)

    def test_generation_replacement_does_not_leak_recency_state(self) -> None:
        scheduler = catalog(expected_ranks=(0,), max_entries=2)
        for index in range(20):
            scheduler.apply_startup(
                rank=0,
                generation=f"boot-{index}",
                generation_epoch=index,
                entries=((key(str(index)), 256),),
            )
            self.assertLessEqual(len(scheduler._recency), 1)

    def test_startup_materialization_stops_at_one_over_the_catalog_bound(self) -> None:
        scheduler = catalog(expected_ranks=(0,), max_entries=2)
        yielded = 0

        def entries():
            nonlocal yielded
            while True:
                yielded += 1
                yield key(str(yielded)), 256

        with self.assertRaisesRegex(ValueError, "catalog bound"):
            scheduler.apply_startup(
                rank=0,
                generation="boot",
                generation_epoch=1,
                entries=entries(),
            )
        self.assertEqual(yielded, 3)

    def test_stale_rolling_checkpoint_does_not_undo_a_newer_delta(self) -> None:
        first, second, third = key("first"), key("second"), key("third")
        scheduler = catalog(expected_ranks=(0,), max_entries=10)
        scheduler.apply_startup(
            rank=0,
            generation="boot",
            generation_epoch=1,
            entries=((first, 256), (second, 512)),
        )
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=0,
                generation="boot",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(0, 1, 0, 2, 2, ((first, 256),)),
                delta=InventoryDelta(1, 0, ((third, 768),), ()),
            )
        )
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=0,
                generation="boot",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(0, 1, 1, 2, 2, ((second, 512),)),
            )
        )
        self.assertTrue(scheduler.has_quorum(third, 768))

    def test_malformed_checkpoint_withdraws_rank_without_unbounded_state(self) -> None:
        entry = key("entry")
        scheduler = catalog(expected_ranks=(0,), max_entries=2)
        scheduler.apply_startup(
            rank=0,
            generation="boot",
            generation_epoch=1,
            entries=((entry, 256),),
        )
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=0,
                generation="boot",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(0, 1, 0, 1000, 1, ()),
            )
        )
        self.assertFalse(scheduler.has_quorum(entry))
        self.assertIn(0, scheduler.desynchronized_ranks)

    def test_malformed_delta_shape_withdraws_rank_instead_of_escaping(self) -> None:
        entry = key("entry")
        scheduler = catalog(expected_ranks=(0,), max_entries=2)
        scheduler.apply_startup(
            rank=0,
            generation="boot",
            generation_epoch=1,
            entries=((entry, 256),),
        )
        scheduler.apply_report(
            WorkerInventoryReport(
                rank=0,
                generation="boot",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(0, 1, 0, 1, 1, ((entry, 256),)),
                delta=InventoryDelta(
                    1,
                    0,
                    (object(),),  # type: ignore[arg-type]
                    (),
                ),
            )
        )
        self.assertFalse(scheduler.has_quorum(entry))
        self.assertIn(0, scheduler.desynchronized_ranks)


if __name__ == "__main__":
    unittest.main()
