"""Surviving a crash: the checkpoint, and the run loop that writes it."""

from __future__ import annotations

import json

import pytest

from cr_labeler.checkpoint import FORMAT, Checkpoint, CheckpointMismatch
from cr_labeler.geoguessr_io import Location
from cr_labeler.labeler import Labeler, LabelResult

HEADER = {"input": "/maps/in.json", "rows": 3, "zoom": 2}


def test_records_survive_a_reopen(tmp_path):
    path = tmp_path / "run.progress"
    with Checkpoint(path, HEADER) as checkpoint:
        checkpoint.record(0, "2026", "aaa", None)
        checkpoint.record(1, "None", "bbb", None)
        checkpoint.record(2, "unknown", None, "no tiles")

    done = Checkpoint(path, HEADER).load()
    assert set(done) == {0, 1, 2}
    assert done[0].label == "2026"
    assert done[0].pano_id == "aaa"
    assert done[2].error == "no tiles"


def test_reopening_appends_rather_than_truncating(tmp_path):
    path = tmp_path / "run.progress"
    with Checkpoint(path, HEADER) as checkpoint:
        checkpoint.record(0, "2026", "aaa", None)
    with Checkpoint(path, HEADER) as checkpoint:
        checkpoint.record(1, "2025", "bbb", None)

    done = Checkpoint(path, HEADER).load()
    assert set(done) == {0, 1}
    # Exactly one header, however many times the run was restarted.
    header_lines = [
        line for line in path.read_text().splitlines() if '"format"' in line
    ]
    assert len(header_lines) == 1


def test_a_half_written_last_line_costs_one_row(tmp_path):
    """A hard kill can cut a line in half; that must not poison the file."""
    path = tmp_path / "run.progress"
    with Checkpoint(path, HEADER) as checkpoint:
        checkpoint.record(0, "2026", "aaa", None)
        checkpoint.record(1, "2025", "bbb", None)
    with path.open("a") as handle:
        handle.write('{"i": 2, "l": "20')  # power cut, mid-write

    done = Checkpoint(path, HEADER).load()
    assert set(done) == {0, 1}


def test_a_checkpoint_for_different_work_is_refused(tmp_path):
    path = tmp_path / "run.progress"
    with Checkpoint(path, HEADER) as checkpoint:
        checkpoint.record(0, "2026", "aaa", None)

    with pytest.raises(CheckpointMismatch, match="different run"):
        Checkpoint(path, {**HEADER, "zoom": 3}).load()


def test_a_checkpoint_from_another_version_is_refused(tmp_path):
    path = tmp_path / "run.progress"
    path.write_text(json.dumps({"format": FORMAT + 1, **HEADER}) + "\n")

    with pytest.raises(CheckpointMismatch, match="different version"):
        Checkpoint(path, HEADER).load()


def test_missing_checkpoint_is_simply_a_fresh_run(tmp_path):
    assert Checkpoint(tmp_path / "absent", HEADER).load() == {}


# --- the run loop ---------------------------------------------------------


def _locations(n: int) -> list[Location]:
    return [Location(index=i, raw={"panoId": f"pano{i}"}) for i in range(n)]


def _labeler() -> Labeler:
    return Labeler.__new__(Labeler)  # only label_stream is under test


def test_label_stream_returns_every_row(monkeypatch):
    labeler = _labeler()
    monkeypatch.setattr(
        Labeler,
        "label_one",
        lambda self, loc: LabelResult(loc.index, loc.pano_id, "2026", None),
    )
    results = list(labeler.label_stream(_locations(50), workers=4))
    assert sorted(r.index for r in results) == list(range(50))


def test_label_stream_keeps_only_a_window_in_flight(monkeypatch):
    """The point of streaming: 900k rows must not become 900k live futures."""
    live = 0
    high_water = 0

    def one(self, location):
        nonlocal live, high_water
        live += 1
        high_water = max(high_water, live)
        live -= 1
        return LabelResult(location.index, location.pano_id, "2026", None)

    monkeypatch.setattr(Labeler, "label_one", one)
    stream = _labeler().label_stream(_locations(5000), workers=4)
    # Consume lazily; nothing may run far ahead of the consumer.
    for _ in range(10):
        next(stream)
    assert high_water <= 4 * 4 + 4
    stream.close()


def test_stop_drains_what_is_running_and_submits_no_more(monkeypatch):
    monkeypatch.setattr(
        Labeler,
        "label_one",
        lambda self, loc: LabelResult(loc.index, loc.pano_id, "2026", None),
    )
    seen: list[int] = []
    asked = {"stop": False}
    stream = _labeler().label_stream(
        _locations(500), workers=4, stop=lambda: asked["stop"]
    )

    for result in stream:
        seen.append(result.index)
        if len(seen) >= 20:
            asked["stop"] = True

    # It stopped early, and everything it reported is a real completed row.
    assert 20 <= len(seen) < 500
    assert len(set(seen)) == len(seen)


def test_one_exploding_row_does_not_end_the_run(monkeypatch):
    """A 900k run cannot be ended by a single unforeseen failure."""

    def one(self, location):
        if location.index == 3:
            raise RuntimeError("decoder exploded")
        return "ok"

    monkeypatch.setattr(Labeler, "label_pano", lambda self, pid: (_Verdict(), None))
    monkeypatch.setattr(Labeler, "_resolve", one)

    labeler = Labeler.__new__(Labeler)
    results = sorted(
        labeler.label_stream(_locations(6), workers=2), key=lambda r: r.index
    )
    assert len(results) == 6
    assert results[3].error is not None
    assert "decoder exploded" in results[3].error
    assert all(r.error is None for i, r in enumerate(results) if i != 3)


class _Verdict:
    label = "2026"
    quality = 0.9
    instances = 9
    margin = 0.1


# --- the disk guard -------------------------------------------------------


def _guard(tmp_path, floor_gb, free_gb):
    from cr_labeler.cli import _DiskGuard

    guard = _DiskGuard([tmp_path], floor_gb=floor_gb)
    guard.free = lambda: free_gb * 1e9
    return guard


def test_guard_trips_below_the_floor(tmp_path):
    guard = _guard(tmp_path, floor_gb=5.0, free_gb=1.0)
    guard.check(force=True)
    assert guard.exhausted()
    assert "1.0 GB" in guard.reason


def test_guard_stays_quiet_with_room(tmp_path):
    guard = _guard(tmp_path, floor_gb=5.0, free_gb=50.0)
    guard.check(force=True)
    assert not guard.exhausted()
    assert guard.reason is None


def test_guard_rate_limits_but_can_be_forced(tmp_path):
    """Checked on a timer, so a short run still looks and a long one is cheap."""
    guard = _guard(tmp_path, floor_gb=5.0, free_gb=50.0)
    guard.check(force=True)          # arms the timer
    guard.free = lambda: 1.0 * 1e9   # disk fills right after
    guard.check()                    # too soon: skipped
    assert not guard.exhausted()
    guard.check(force=True)
    assert guard.exhausted()


def test_a_tripped_guard_stays_tripped(tmp_path):
    guard = _guard(tmp_path, floor_gb=5.0, free_gb=1.0)
    guard.check(force=True)
    reason = guard.reason
    guard.free = lambda: 500.0 * 1e9
    guard.check(force=True)
    assert guard.reason == reason
