"""The dependency-source deadline must not depend on machine load (#696).

``dependency_sources`` used to cap its analysis at an unraisable 5 s of wall-clock
time. The same unchanged tree therefore came back complete on an idle machine and
incomplete under load, because the cap was measured in wall-clock seconds. The other
budgets in this codebase are already operator-settable; this one was not.
"""

from __future__ import annotations

import pytest

from skillspector.dependency_sources import (
    DEFAULT_MAX_ANALYSIS_SECONDS,
    MAX_ANALYSIS_SECONDS,
    _max_analysis_seconds_from_environment,
)


def test_default_ceiling_is_unchanged() -> None:
    """The documented 5 s default must not move for operators who set nothing."""
    assert DEFAULT_MAX_ANALYSIS_SECONDS == 5.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, DEFAULT_MAX_ANALYSIS_SECONDS),
        ("120", 120.0),
        ("0.5", 0.5),
        ("0", DEFAULT_MAX_ANALYSIS_SECONDS),
        ("-1", DEFAULT_MAX_ANALYSIS_SECONDS),
        ("nan", DEFAULT_MAX_ANALYSIS_SECONDS),
        ("inf", DEFAULT_MAX_ANALYSIS_SECONDS),
        ("not-a-number", DEFAULT_MAX_ANALYSIS_SECONDS),
    ],
)
def test_dependency_source_seconds_environment_parsing(value: str | None, expected: float) -> None:
    assert _max_analysis_seconds_from_environment(value) == expected


def test_invalid_value_keeps_the_default_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        resolved = _max_analysis_seconds_from_environment("not-a-number")

    assert resolved == DEFAULT_MAX_ANALYSIS_SECONDS
    assert "SKILLSPECTOR_MAX_DEPENDENCY_ANALYSIS_SECONDS" in caplog.text


def test_module_ceiling_matches_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import-time resolution follows the same pattern as the workflow budget."""
    monkeypatch.setenv("SKILLSPECTOR_MAX_DEPENDENCY_ANALYSIS_SECONDS", "42")
    import importlib

    import skillspector.dependency_sources as module

    reloaded = importlib.reload(module)
    try:
        assert reloaded.MAX_ANALYSIS_SECONDS == 42.0
    finally:
        monkeypatch.delenv("SKILLSPECTOR_MAX_DEPENDENCY_ANALYSIS_SECONDS", raising=False)
        importlib.reload(module)

    assert MAX_ANALYSIS_SECONDS == 5.0
