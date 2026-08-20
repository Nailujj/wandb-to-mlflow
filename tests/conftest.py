"""Shared test configuration.

No test in tiers 1 and 2 may touch the network. The ``no_network`` autouse
fixture enforces that by making socket creation raise, so a regression that
introduces a real W&B or HTTP call fails loudly instead of silently slowing CI.

Tests also assert on CLI output, which Typer renders through rich -- and rich
decides whether to emit colour from the *ambient environment*. That made those
assertions pass locally and fail in CI, which is the worst way for a test to be
wrong. ``no_colour`` pins it.
"""

from __future__ import annotations

import logging
import os
import socket

# MLflow 3.x puts the filesystem tracking backend in maintenance mode and refuses
# to open it unless this is set. Tier 2 uses it deliberately: it is the only
# backend that needs no server and no network. Set before any mlflow import so
# the flag is visible when a store is first constructed.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
from collections.abc import Iterator
from typing import Any

import pytest


class NetworkUseInTestError(RuntimeError):
    """Raised when a non-e2e test tries to open a socket."""


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if request.node.get_closest_marker("e2e"):
        yield
        return

    def guard(*args: Any, **kwargs: Any) -> None:
        raise NetworkUseInTestError(
            "tiers 1 and 2 must not touch the network; mark the test with @pytest.mark.e2e"
        )

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    yield


#: Everything that can talk rich into emitting escape codes. ``FORCE_COLOR`` is
#: the one that bites: GitHub Actions sets it, so CI output is coloured while a
#: developer's piped-to-a-file run is not.
_COLOUR_ENV = ("FORCE_COLOR", "CLICOLOR_FORCE", "CLICOLOR", "TERM", "COLORTERM")


@pytest.fixture(autouse=True)
def no_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make CLI output assertions independent of the terminal they run in.

    With colour on, rich styles option names *inside* the string, so
    ``--manifest`` is emitted as ``ESC[1;36m-ESC[0mESC[1;36m-manifestESC[0m``:
    the two hyphens are separated by an escape sequence and the literal
    substring is simply not there. A plain ``"--manifest" in result.output``
    then fails, on output that is entirely correct.
    """
    for name in _COLOUR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")


def pytest_configure(config: pytest.Config) -> None:
    """Keep MLflow quiet; its startup banners drown the test output."""
    logging.getLogger("mlflow").setLevel(logging.ERROR)
