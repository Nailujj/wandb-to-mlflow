"""The adapter must call W&B with signatures W&B actually has.

This is the tier that was missing. Unit tests elsewhere replace the W&B client
with a fake that accepts anything, so they keep passing when the real signature
moves underneath them; the e2e tier can catch that, but only for the code paths
it happens to enable, and it never enabled ``--system-metrics``. That gap let
``scan_history(stream="events")`` survive in the tree long after ``stream`` was
removed from ``scan_history`` -- a call that failed *every run* in the
migration, not merely the system-metric stream.

So these tests bind our real call arguments against the **installed** wandb's
real signatures, using ``inspect.Signature.bind``. No network, no fake that
politely accepts whatever it is handed: if W&B renames or drops a parameter the
adapter passes, this goes red offline and immediately.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from wandb_to_mlflow.source import WandbRun

wandb_public = pytest.importorskip("wandb.apis.public.runs")
WandbApiRun = wandb_public.Run


def signature_of(method_name: str) -> inspect.Signature:
    """The installed wandb's signature for ``Run.<method_name>``, minus ``self``."""
    attr = inspect.getattr_static(WandbApiRun, method_name)
    if isinstance(attr, property):
        pytest.fail(f"Run.{method_name} is a property, not a callable, in this wandb")
    signature = inspect.signature(attr)
    return signature.replace(
        parameters=[p for name, p in signature.parameters.items() if name != "self"]
    )


class SignatureCheckedRun:
    """Stands in for a W&B run, but validates every call against the real thing.

    Each recorded call is bound to the installed wandb's signature for that
    method. A parameter we pass that W&B no longer accepts raises ``TypeError``
    here exactly as it would against the live API.
    """

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        signature_of(name).bind(*args, **kwargs)
        self.calls.append((name, args, kwargs))

    def history(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._record("history", args, kwargs)
        return list(self.rows)

    def scan_history(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._record("scan_history", args, kwargs)
        return list(self.rows)


def adapter(raw: SignatureCheckedRun) -> WandbRun:
    return WandbRun(
        id="abc123",
        name="a-run",
        state="finished",
        created_at="2026-01-01T00:00:00",
        config={},
        summary={},
        tags=[],
        notes=None,
        group=None,
        job_type=None,
        sweep_id=None,
        url="",
        entity="e",
        project="p",
        _raw=raw,
    )


# --------------------------------------------------------------------------- #
# The regression this file exists for
# --------------------------------------------------------------------------- #


def test_system_metrics_call_is_one_wandb_accepts() -> None:
    """The exact failure: ``scan_history(stream=...)`` no longer binds."""
    raw = SignatureCheckedRun([{"_timestamp": 1.0, "system.cpu": 12.5}])
    rows = list(adapter(raw).system_metrics())

    assert rows == [{"_timestamp": 1.0, "system.cpu": 12.5}]
    (name, _, kwargs) = raw.calls[0]
    assert name == "history", "the system stream is reachable via history(), not scan_history()"
    assert kwargs["stream"] == "system"
    assert kwargs["pandas"] is False, "a DataFrame would break the row iteration"


def test_scan_history_is_not_asked_for_a_stream() -> None:
    """Guards the specific dead parameter, in case someone reintroduces it."""
    assert "stream" not in signature_of("scan_history").parameters


def test_history_stream_parameter_still_exists() -> None:
    """If W&B drops this too, system metrics need a new route -- fail loudly."""
    assert "stream" in signature_of("history").parameters


def test_default_history_is_scanned_not_sampled() -> None:
    """``history()`` samples to ~500 points; the data stream must not use it."""
    raw = SignatureCheckedRun([{"_step": 0, "loss": 1.0}])
    list(adapter(raw).history())
    assert [name for name, _, _ in raw.calls] == ["scan_history"]


def test_system_metrics_asks_for_more_than_the_default_sample() -> None:
    from wandb_to_mlflow.source import SYSTEM_METRIC_SAMPLES

    default = signature_of("history").parameters["samples"].default
    assert default < SYSTEM_METRIC_SAMPLES

    raw = SignatureCheckedRun()
    list(adapter(raw).system_metrics())
    assert raw.calls[0][2]["samples"] == SYSTEM_METRIC_SAMPLES


def test_system_metrics_tolerates_none() -> None:
    """``history()`` returns [] or None on an empty stream depending on version."""

    class NoneReturning(SignatureCheckedRun):
        def history(self, *args: Any, **kwargs: Any) -> Any:
            self._record("history", args, kwargs)
            return None

    assert list(adapter(NoneReturning()).system_metrics()) == []
