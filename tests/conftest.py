"""Shared test configuration.

No test in tiers 1 and 2 may touch the network. The ``no_network`` autouse
fixture enforces that by making socket creation raise, so a regression that
introduces a real W&B or HTTP call fails loudly instead of silently slowing CI.
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


def pytest_configure(config: pytest.Config) -> None:
    """Keep MLflow quiet; its startup banners drown the test output."""
    logging.getLogger("mlflow").setLevel(logging.ERROR)
