"""Progress reporting of ``run()``, without needing OpenFOAM installed."""

import sys
from pathlib import Path
from typing import Any

import pytest
from rich.progress import Progress, TaskID

from foamlib import AsyncFoamCase, FoamCase
from foamlib._cases._run import _parse_time

Updates = list[tuple[float, float | None]]

CONTROL_DICT = """FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
stopAt endTime;
startTime {start};
endTime {end};
"""


@pytest.fixture
def updates(monkeypatch: pytest.MonkeyPatch) -> Updates:
    """Record the (completed, total) pairs given to the progress bar."""
    seen: Updates = []
    update = Progress.update

    def spy(self: Progress, task: TaskID, **kwargs: Any) -> None:
        if "completed" in kwargs:
            seen.append((kwargs["completed"], kwargs.get("total")))
        update(self, task, **kwargs)

    monkeypatch.setattr(Progress, "update", spy)
    return seen


@pytest.fixture
def case_path(tmp_path: Path) -> Path:
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text(
        CONTROL_DICT.format(start=0, end=10)
    )
    return tmp_path


def echo(line: str) -> list[str]:
    return [sys.executable, "-c", f"print({line!r})"]


def times(updates: Updates) -> list[float]:
    # the last update always marks the run as done
    assert updates[-1] == (1, 1)
    return [completed for completed, _ in updates[:-1]]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", 1.0),
        ("0.001", 0.001),
        ("1e-05", 1e-05),
        ("1s", 1.0),
        ("1e-05s", 1e-05),
        ("-180CAD", -180.0),
    ],
)
def test_parse_time(value: str, expected: float) -> None:
    assert _parse_time(value) == expected


def test_parse_time_rejects_other_text() -> None:
    with pytest.raises(ValueError):
        _parse_time("constant")


@pytest.mark.parametrize("unit", ["", "s"])
def test_progress_with_and_without_a_unit(
    case_path: Path, updates: Updates, unit: str
) -> None:
    FoamCase(case_path).run(echo(f"Time = 5{unit}"), cpus=0, log=False)
    assert (5.0, 10) in updates


def test_progress_ignores_earlier_runs(case_path: Path, updates: Updates) -> None:
    (case_path / "log.old").write_text("Time = 9s\n")
    (case_path / "log.append").write_text("Time = 8s\n")
    FoamCase(case_path).run(echo("Time = 1s"), cpus=0, log="log.append")
    seen = times(updates)
    assert 1.0 in seen
    assert 8.0 not in seen
    assert 9.0 not in seen


def test_progress_reads_a_log_rewritten_by_the_run(
    case_path: Path, updates: Updates
) -> None:
    (case_path / "log.script").write_text("Time = 9s\n" * 100)
    code = "open('log.script', 'w').write('Time = 3s\\n')"
    FoamCase(case_path).run([sys.executable, "-c", code], cpus=0, log=False)
    seen = times(updates)
    assert 3.0 in seen
    assert 9.0 not in seen


def test_progress_from_a_negative_start_time(case_path: Path, updates: Updates) -> None:
    (case_path / "system" / "controlDict").write_text(
        CONTROL_DICT.format(start=-180, end=60)
    )
    FoamCase(case_path).run(echo("Time = -60CAD"), cpus=0, log=False)
    assert (120.0, 240) in updates


def test_progress_keeps_a_positive_start_time(
    case_path: Path, updates: Updates
) -> None:
    (case_path / "system" / "controlDict").write_text(
        CONTROL_DICT.format(start=5, end=10)
    )
    FoamCase(case_path).run(echo("Time = 6s"), cpus=0, log=False)
    assert (6.0, 10) in updates


@pytest.mark.asyncio
async def test_async_progress_ignores_earlier_runs(
    case_path: Path, updates: Updates
) -> None:
    (case_path / "log.old").write_text("Time = 9s\n")
    await AsyncFoamCase(case_path).run(echo("Time = 1s"), cpus=0, log=False)
    seen = times(updates)
    assert 1.0 in seen
    assert 9.0 not in seen
