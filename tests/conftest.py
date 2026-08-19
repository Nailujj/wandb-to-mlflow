"""Shared test configuration.

No test in tiers 1 and 2 may touch the network. The ``no_network`` autouse
fixture enforces that by making socket creation raise, so a regression that
introduces a real W&B or HTTP call fails loudly instead of silently slowing CI.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest


class NetworkUseInTest(RuntimeError):
    """Raised when a non-e2e test tries to open a socket."""


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if request.node.get_closest_marker("e2e"):
        yield
        return

    def guard(*args: Any, **kwargs: Any) -> None:
        raise NetworkUseInTest(
            "tiers 1 and 2 must not touch the network; mark the test with @pytest.mark.e2e"
        )

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    yield
