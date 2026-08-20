"""The source abstraction the migrator depends on.

The migrator is written against :class:`SourceRun`, **never** against
``wandb.apis.public.Run``. That is what makes tiers 1 and 2 of the test suite
possible at all: a hand-built fixture satisfies the protocol, so the entire
mapping can be exercised without a network.

:class:`WandbRun` and friends adapt the real W&B API onto these protocols and
are the only code in the package that imports ``wandb``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wandb.apis.public import Run as WandbApiRun

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: How many pages of history to pull per request. W&B's default samples to ~500
#: points; ``scan_history`` with an explicit page size does not.
SCAN_PAGE_SIZE = 1000

#: How many system-metric samples to ask for. Unlike the default history stream,
#: the system stream has no exhaustive reader: ``scan_history`` takes no
#: ``stream`` argument at all, so ``history(stream="system")`` -- which samples
#: server-side -- is the only way in. Asking for more than exist is harmless and
#: returns what there is, so this is set far above any realistic sample count
#: rather than left at W&B's default of 500.
SYSTEM_METRIC_SAMPLES = 100_000

_RETRY_ATTEMPTS = int(os.environ.get("W2M_RETRY_ATTEMPTS", "5"))
_RETRY_MAX_WAIT = float(os.environ.get("W2M_RETRY_MAX_WAIT", "30"))


class TransientSourceError(RuntimeError):
    """A W&B call failed in a way that is worth retrying."""


# ``OSError`` covers ConnectionError and TimeoutError, but naming them keeps the
# intent readable and survives someone narrowing the tuple later.
_RETRYABLE = (TransientSourceError, ConnectionError, TimeoutError, OSError)


def with_retry(fn: Callable[..., T]) -> Callable[..., T]:
    """Exponential backoff around a W&B network call.

    Large projects hit rate limits; a migration that dies on the 4,000th run
    because of one 429 is not a migration tool.
    """
    wrapped: Callable[..., T] = retry(
        stop=stop_after_attempt(_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, max=_RETRY_MAX_WAIT),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
        before_sleep=lambda state: logger.warning(
            "retrying %s after %s (attempt %d)",
            fn.__name__,
            state.outcome.exception() if state.outcome else "?",
            state.attempt_number,
        ),
    )(fn)
    return wrapped


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #


@runtime_checkable
class SourceFile(Protocol):
    """A file attached to a run (``run.files()``)."""

    name: str
    size: int

    def download(self, dest_dir: Path) -> Path:
        """Fetch the bytes into ``dest_dir`` and return the local path."""
        ...


@runtime_checkable
class SourceArtifact(Protocol):
    """A logged artifact.

    ``is_reference`` and ``size`` are readable **without** downloading, so the
    migrator can honour ``--max-artifact-size`` and skip reference artifacts
    before spending any bytes.
    """

    name: str
    type: str
    version: str
    aliases: list[str]
    digest: str
    size: int
    is_reference: bool
    source_uris: list[str]

    def download(self, dest_dir: Path) -> Path:
        """Fetch the artifact contents into ``dest_dir`` and return the root path."""
        ...


@runtime_checkable
class SourceRun(Protocol):
    """One W&B run, reduced to what the migrator actually needs."""

    id: str
    name: str | None
    state: str
    created_at: str  # ISO-8601
    config: dict[str, Any]
    summary: dict[str, Any]
    tags: list[str]
    notes: str | None
    group: str | None
    job_type: str | None
    sweep_id: str | None
    url: str
    entity: str
    project: str

    def history(self) -> Iterator[dict[str, Any]]: ...
    def files(self) -> Iterator[SourceFile]: ...
    def artifacts(self) -> Iterator[SourceArtifact]: ...
    def system_metrics(self) -> Iterator[dict[str, Any]]: ...


@runtime_checkable
class SourceProject(Protocol):
    """A project's worth of runs."""

    entity: str
    project: str

    def runs(self) -> Iterator[SourceRun]: ...


# --------------------------------------------------------------------------- #
# W&B adapters
# --------------------------------------------------------------------------- #


@dataclass
class WandbFile:
    """Adapts ``wandb.apis.public.File``."""

    name: str
    size: int
    _raw: Any = field(repr=False, default=None)

    def download(self, dest_dir: Path) -> Path:
        @with_retry
        def _download() -> Path:
            self._raw.download(root=str(dest_dir), replace=True, exist_ok=True)
            return dest_dir / self.name

        return _download()


@dataclass
class WandbArtifact:
    """Adapts ``wandb.Artifact`` as returned by ``run.logged_artifacts()``."""

    name: str
    type: str
    version: str
    aliases: list[str]
    digest: str
    size: int
    is_reference: bool
    source_uris: list[str]
    _raw: Any = field(repr=False, default=None)

    @classmethod
    def from_wandb(cls, artifact: Any) -> WandbArtifact:
        no_entries: dict[str, Any] = {}
        no_aliases: list[str] = []
        manifest_entries = _safe(lambda: dict(artifact.manifest.entries), no_entries)
        source_uris = [
            entry.ref
            for entry in manifest_entries.values()
            if getattr(entry, "ref", None) is not None
        ]
        return cls(
            name=str(artifact.name),
            type=str(_safe(lambda: artifact.type, "unknown")),
            version=str(_safe(lambda: artifact.version, "")),
            aliases=_safe(lambda: [str(a) for a in artifact.aliases], no_aliases),
            digest=str(_safe(lambda: artifact.digest, "")),
            size=_size_of(artifact),
            is_reference=bool(source_uris),
            source_uris=source_uris,
            _raw=artifact,
        )

    def download(self, dest_dir: Path) -> Path:
        @with_retry
        def _download() -> Path:
            return Path(self._raw.download(root=str(dest_dir)))

        return _download()


def _size_of(obj: Any) -> int:
    """Byte size of a W&B file or artifact, or 0 when the record does not carry one."""
    try:
        return int(obj.size or 0)
    except Exception:  # deliberately broad: adapter
        return 0


def _safe(fn: Callable[[], T], default: T) -> T:
    """W&B objects raise from property access on partially-populated records."""
    try:
        return fn()
    except Exception as exc:  # deliberately broad: adapter
        logger.debug("attribute access failed, using default: %s", exc)
        return default


@dataclass
class WandbRun:
    """Adapts ``wandb.apis.public.Run`` onto :class:`SourceRun`."""

    id: str
    name: str | None
    state: str
    created_at: str
    config: dict[str, Any]
    summary: dict[str, Any]
    tags: list[str]
    notes: str | None
    group: str | None
    job_type: str | None
    sweep_id: str | None
    url: str
    entity: str
    project: str
    _raw: Any = field(repr=False, default=None)

    @classmethod
    def from_wandb(cls, run: WandbApiRun) -> WandbRun:
        # ``run.summary`` is a lazy SummarySubDict; ``._json_dict`` is the raw mapping.
        empty: dict[str, Any] = {}
        no_tags: list[str] = []
        raw_summary: dict[str, Any] = _safe(lambda: dict(run.summary._json_dict), empty)
        summary = {k: v for k, v in raw_summary.items() if not str(k).startswith("_")}
        # ``_timestamp`` and ``_runtime`` are needed to derive end_time; keep them aside.
        for internal in ("_timestamp", "_runtime", "_step"):
            if internal in raw_summary:
                summary[internal] = raw_summary[internal]
        return cls(
            id=str(run.id),
            name=_safe(lambda: run.name, None),
            state=str(_safe(lambda: run.state, "unknown")),
            created_at=str(_safe(lambda: run.created_at, "")),
            config=_safe(lambda: dict(run.config), empty),
            summary=summary,
            tags=_safe(lambda: [str(t) for t in run.tags], no_tags),
            notes=_safe(lambda: run.notes, None),
            group=_safe(lambda: run.group, None),
            job_type=_safe(lambda: run.job_type, None),
            sweep_id=_safe(lambda: run.sweep.id if run.sweep else None, None),
            url=str(_safe(lambda: run.url, "")),
            entity=str(_safe(lambda: run.entity, "")),
            project=str(_safe(lambda: run.project, "")),
            _raw=run,
        )

    def history(self) -> Iterator[dict[str, Any]]:
        """Full history via ``scan_history``.

        ``run.history()`` samples to roughly 500 points and says nothing about
        having done so. Using it here would silently discard most of a long run.
        """

        @with_retry
        def _scan() -> Iterator[dict[str, Any]]:
            return iter(self._raw.scan_history(page_size=SCAN_PAGE_SIZE))

        yield from _scan()

    def system_metrics(self) -> Iterator[dict[str, Any]]:
        """Server-sampled system metrics. Opt-in; see MAPPING.md.

        Read through ``history(stream="system")``, not ``scan_history``. This
        code once called ``scan_history(stream="events")`` -- a signature that
        has **never existed in any released wandb** (checked against the
        published wheels back to 0.15): ``scan_history`` takes no ``stream``
        argument and raises ``TypeError`` on one, which failed every run in the
        migration rather than just this stream. The flag was broken from the
        day it was written, and no offline test could see it because the fakes
        accepted the phantom argument. There is no exhaustive reader for this
        stream, so these points are server-sampled, as MAPPING.md says.

        Keys arrive already prefixed ``system.`` by W&B. The migrator's own
        prefixing is idempotent, so they are not renamed here.
        """

        @with_retry
        def _scan() -> list[dict[str, Any]]:
            rows = self._raw.history(stream="system", samples=SYSTEM_METRIC_SAMPLES, pandas=False)
            return list(rows or [])

        yield from _scan()

    def files(self) -> Iterator[SourceFile]:
        @with_retry
        def _files() -> list[Any]:
            return list(self._raw.files())

        for raw in _files():
            yield WandbFile(name=str(raw.name), size=_size_of(raw), _raw=raw)

    def artifacts(self) -> Iterator[SourceArtifact]:
        @with_retry
        def _artifacts() -> list[Any]:
            return list(self._raw.logged_artifacts())

        for raw in _artifacts():
            yield WandbArtifact.from_wandb(raw)


@dataclass
class WandbProject:
    """Adapts ``wandb.Api()`` onto :class:`SourceProject`."""

    entity: str
    project: str
    api: Any = field(repr=False, default=None)
    filters: dict[str, Any] | None = None

    @classmethod
    def connect(
        cls, entity: str, project: str, filters: dict[str, Any] | None = None
    ) -> WandbProject:
        import wandb

        return cls(entity=entity, project=project, api=wandb.Api(), filters=filters)

    def runs(self) -> Iterator[SourceRun]:
        @with_retry
        def _runs() -> Any:
            return self.api.runs(f"{self.entity}/{self.project}", filters=self.filters)

        for raw in _runs():
            yield WandbRun.from_wandb(raw)
