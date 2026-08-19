"""The migrator.

Two rules shape this module:

1. **`MlflowClient` only, never the fluent API.** ``mlflow run`` opens its own
   ambient run for the entry point; any use of ``mlflow.start_run`` or
   ``mlflow.log_*`` here would nest every migrated run inside it and pollute the
   target experiment. Every write goes through an explicit client call with an
   explicit experiment id and run id.
2. **A partial failure never kills the migration.** Each run is migrated inside
   its own try/except; failures are collected, reported at the end, and turned
   into a non-zero exit code by the CLI.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

from wandb_to_mlflow.coerce import (
    Drop,
    DropReport,
    as_metric,
    as_param,
    as_tag,
    flatten_config,
    renamed,
    sanitise_key,
    sanitise_keys,
)
from wandb_to_mlflow.limits import Limits, default_limits
from wandb_to_mlflow.source import SourceArtifact, SourceProject, SourceRun
from wandb_to_mlflow.state import RUN_ID_TAG, SWEEP_PARENT_TAG, MigrationState

logger = logging.getLogger(__name__)

#: W&B run state -> MLflow run status. `crashed` and `failed` are distinct in
#: W&B and collapse here; the raw state is preserved in the `wandb.state` tag.
STATE_TO_STATUS: dict[str, str] = {
    "finished": "FINISHED",
    "crashed": "FAILED",
    "failed": "FAILED",
    "killed": "KILLED",
    "running": "RUNNING",
    "preempted": "KILLED",
}
DEFAULT_STATUS = "FINISHED"

SUMMARY_METRIC_PREFIX = "final."
SUMMARY_PARAM_PREFIX = "summary."
TAG_PREFIX = "wandb.tag."
FILES_ARTIFACT_PATH = "wandb_files"
ARTIFACTS_ARTIFACT_PATH = "artifacts"

#: W&B history columns that describe the row rather than being data in it.
INTERNAL_COLUMNS = frozenset({"_step", "_timestamp", "_runtime", "_wandb"})

DEFAULT_MAX_ARTIFACT_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class MigrateOptions:
    """Everything the user can turn on. Bytes-costing features are all opt-in."""

    experiment: str | None = None
    include_files: bool = False
    include_artifacts: bool = False
    include_system_metrics: bool = False
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES
    overwrite: bool = False
    dry_run: bool = False
    workers: int = 1


@dataclass
class RunReport:
    """What happened to one run. This is what `verify` and the CLI report on."""

    wandb_run_id: str
    wandb_name: str | None = None
    wandb_sweep_id: str | None = None
    mlflow_run_id: str | None = None
    status: str = DEFAULT_STATUS
    param_count: int = 0
    metric_keys: list[str] = field(default_factory=list)
    metric_point_counts: dict[str, int] = field(default_factory=dict)
    final_values: dict[str, float] = field(default_factory=dict)
    dropped: DropReport = field(default_factory=DropReport)
    renamed_keys: dict[str, str] = field(default_factory=dict)
    truncated_params: list[str] = field(default_factory=list)
    truncated_tags: list[str] = field(default_factory=list)
    artifacts_migrated: int = 0
    artifacts_skipped: int = 0
    reference_artifacts: list[str] = field(default_factory=list)
    files_migrated: int = 0
    parent_run_id: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "wandb_run_id": self.wandb_run_id,
            "wandb_name": self.wandb_name,
            "wandb_sweep_id": self.wandb_sweep_id,
            "mlflow_run_id": self.mlflow_run_id,
            "status": self.status,
            "param_count": self.param_count,
            "metric_keys": sorted(self.metric_keys),
            "metric_point_counts": dict(sorted(self.metric_point_counts.items())),
            "dropped": self.dropped.as_dict(),
            "renamed_keys": self.renamed_keys,
            "truncated_params": sorted(self.truncated_params),
            "artifacts_migrated": self.artifacts_migrated,
            "artifacts_skipped": self.artifacts_skipped,
            "reference_artifacts": self.reference_artifacts,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "error": self.error,
        }


@dataclass
class MigrationResult:
    experiment_id: str | None = None
    experiment_name: str | None = None
    reports: list[RunReport] = field(default_factory=list)

    @property
    def failures(self) -> list[RunReport]:
        return [r for r in self.reports if r.error is not None]

    @property
    def migrated(self) -> list[RunReport]:
        return [r for r in self.reports if r.error is None and not r.skipped]

    @property
    def skipped(self) -> list[RunReport]:
        return [r for r in self.reports if r.skipped]

    @property
    def ok(self) -> bool:
        return not self.failures


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #


def parse_timestamp(value: str | None) -> int | None:
    """W&B's ISO-8601 ``created_at`` to epoch milliseconds.

    W&B returns these naive but in UTC; treating a naive stamp as local time
    would shift every migrated run by the migrating machine's offset.
    """
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        logger.warning("unparseable timestamp %r; falling back to no start time", value)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _seconds_to_ms(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return int(float(value) * 1000)
    except (OverflowError, ValueError):  # pragma: no cover - defensive
        return None


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def _param(key: str, value: str) -> Param:
    """``Param``/``RunTag`` are unannotated in MLflow; wrap them once rather than
    scattering ignores through the migrator."""
    return Param(key=key, value=value)  # type: ignore[no-untyped-call]


def _chunks(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), max(1, size)):
        yield list(items[start : start + max(1, size)])


# --------------------------------------------------------------------------- #
# Migrator
# --------------------------------------------------------------------------- #


class Migrator:
    """Migrates a :class:`SourceProject` into one MLflow experiment."""

    def __init__(
        self,
        client: MlflowClient,
        options: MigrateOptions | None = None,
        limits: Limits | None = None,
    ) -> None:
        self.client = client
        self.options = options or MigrateOptions()
        self.limits = limits or default_limits()
        self.state: MigrationState | None = None

    # -- experiment ------------------------------------------------------- #

    def ensure_experiment(self, name: str) -> str:
        existing = self.client.get_experiment_by_name(name)
        if existing is not None:
            return str(existing.experiment_id)
        return str(self.client.create_experiment(name))

    # -- entry point ------------------------------------------------------ #

    def migrate_project(self, source: SourceProject) -> MigrationResult:
        name = self.options.experiment or source.project
        result = MigrationResult(experiment_name=name)
        if self.options.dry_run:
            experiment = self.client.get_experiment_by_name(name)
            result.experiment_id = str(experiment.experiment_id) if experiment else None
        else:
            result.experiment_id = self.ensure_experiment(name)

        if result.experiment_id is not None:
            self.state = MigrationState(self.client, result.experiment_id)
            self.state.load()
        else:
            self.state = None

        runs = list(source.runs())
        workers = max(1, self.options.workers)
        if workers == 1:
            result.reports.extend(self._guarded(run, result.experiment_id) for run in runs)
        else:
            # Reports are collected in submission order, so a migration's output
            # does not depend on which worker happened to finish first.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._guarded, run, result.experiment_id) for run in runs]
                result.reports.extend(future.result() for future in futures)
        return result

    def _guarded(self, run: SourceRun, experiment_id: str | None) -> RunReport:
        """One run's migration, with its failure contained to itself."""
        try:
            return self.migrate_run(run, experiment_id)
        except Exception as exc:  # deliberately broad: one bad run must not stop the rest
            logger.exception("run %s failed to migrate", run.id)
            return RunReport(wandb_run_id=run.id, error=repr(exc))

    # -- one run ---------------------------------------------------------- #

    def migrate_run(self, run: SourceRun, experiment_id: str | None) -> RunReport:
        report = RunReport(wandb_run_id=run.id, wandb_name=run.name, wandb_sweep_id=run.sweep_id)
        report.status = STATE_TO_STATUS.get(run.state, DEFAULT_STATUS)

        existing = self.state.lookup(run.id) if self.state else None
        if existing is not None and not self.options.overwrite and existing.reusable:
            # Already migrated in full under the current mapping. Re-reading its
            # history from W&B would cost real time and change nothing.
            report.mlflow_run_id = existing.mlflow_run_id
            report.skipped = True
            report.skip_reason = "already migrated"
            logger.info("skipping W&B run %s: already migrated", run.id)
            return report

        start_time = parse_timestamp(run.created_at)
        points, last_ts, metric_renames = self._collect_metrics(run, report)
        params, param_renames, truncated_params = self._collect_params(run)
        report.renamed_keys = {**metric_renames, **param_renames}
        report.truncated_params = truncated_params
        report.param_count = len(params)
        report.metric_keys = sorted({p.key for p in points})
        for point in points:
            report.metric_point_counts[point.key] = report.metric_point_counts.get(point.key, 0) + 1

        if self.options.dry_run:
            report.skipped = True
            return report

        assert experiment_id is not None  # dry_run is the only path with no experiment
        if existing is not None:
            # Either --overwrite, or a run interrupted before it was marked
            # complete. Both mean the existing run is not to be trusted.
            assert self.state is not None
            self.state.discard(run.id, existing.mlflow_run_id)
            report.skip_reason = "overwritten" if self.options.overwrite else "resumed"

        parent_run_id = self._sweep_parent(run, experiment_id)
        report.parent_run_id = parent_run_id

        tags = self._build_tags(run, report, parent_run_id)
        mlflow_run = self.client.create_run(
            experiment_id=experiment_id,
            start_time=start_time,
            tags=tags,
            run_name=run.name or None,
        )
        run_id = str(mlflow_run.info.run_id)
        report.mlflow_run_id = run_id

        self._log_params(run_id, params)
        self._log_metrics(run_id, points)
        if self.options.include_files:
            report.files_migrated = self._migrate_files(run, run_id, report)
        if self.options.include_artifacts:
            self._migrate_artifacts(run, run_id, report)

        end_time = self._end_time(run, start_time, last_ts, report)
        self.client.set_terminated(run_id, status=report.status, end_time=end_time)
        # Last write, deliberately: a run without this marker is a half-written
        # run, and a resumed migration replaces it rather than trusting it.
        if self.state is not None:
            self.state.mark_complete(run.id, run_id)
        return report

    # -- metrics ---------------------------------------------------------- #

    def _collect_metrics(
        self, run: SourceRun, report: RunReport
    ) -> tuple[list[Metric], int | None, dict[str, str]]:
        """Read all history, then sanitise the key set **once**.

        Keys cannot be sanitised incrementally: a collision is only visible once
        every key is known, and points already written under an un-suffixed name
        could not be retracted. So history is buffered per run -- peak memory is
        O(points in one run), which for the spec's 20,000 x 5 case is tens of MB.
        """
        raw: list[tuple[str, float, int, int]] = []  # key, value, step, timestamp_ms
        last_ts: int | None = None
        start_ms = parse_timestamp(run.created_at) or 0

        streams: list[tuple[Iterable[dict[str, Any]], str]] = [(run.history(), "")]
        if self.options.include_system_metrics:
            streams.append((run.system_metrics(), "system."))

        for stream, prefix in streams:
            for index, row in enumerate(stream):
                step = self._step_of(row, index)
                row_ts = _seconds_to_ms(row.get("_timestamp")) or start_ms
                if prefix == "" and row_ts:
                    last_ts = max(last_ts or 0, row_ts)
                for key, value in row.items():
                    if key in INTERNAL_COLUMNS or str(key).startswith("_"):
                        continue
                    metric, reason, media_type = as_metric(value)
                    if metric is None:
                        assert reason is not None
                        report.dropped.record(reason, media_type)
                        continue
                    raw.append((f"{prefix}{key}", metric, step, row_ts))

        summary_points, summary_renames = self._summary_metrics(run, report, start_ms)
        raw.extend(summary_points)

        key_map = sanitise_keys({key for key, _, _, _ in raw}, self.limits)
        points = [
            Metric(key=key_map[key], value=value, timestamp=ts, step=step)
            for key, value, step, ts in raw
        ]
        for point in points:
            if point.key.startswith(SUMMARY_METRIC_PREFIX):
                report.final_values[point.key] = point.value
        return points, last_ts, {**renamed(key_map), **summary_renames}

    @staticmethod
    def _step_of(row: dict[str, Any], index: int) -> int:
        raw = row.get("_step")
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            return index
        return int(raw)

    def _summary_metrics(
        self, run: SourceRun, report: RunReport, start_ms: int
    ) -> tuple[list[tuple[str, float, int, int]], dict[str, str]]:
        """Numeric summary values become ``final.<k>`` metrics at step 0.

        This is what makes the MLflow runs table sortable by final accuracy,
        which is the first thing anyone looks for after a migration.
        """
        out: list[tuple[str, float, int, int]] = []
        for key, value in run.summary.items():
            if str(key).startswith("_"):
                continue
            metric, reason, media_type = as_metric(value)
            if metric is None:
                if reason in (Drop.NONFINITE, Drop.MEDIA):
                    report.dropped.record(reason, media_type)
                continue
            out.append((f"{SUMMARY_METRIC_PREFIX}{key}", metric, 0, start_ms))
        return out, {}

    def _log_metrics(self, run_id: str, points: Sequence[Metric]) -> None:
        batch_size = min(self.limits.max_metrics_per_batch, self.limits.max_entities_per_batch)
        for chunk in _chunks(points, batch_size):
            self.client.log_batch(run_id, metrics=chunk)

    # -- params ----------------------------------------------------------- #

    def _collect_params(self, run: SourceRun) -> tuple[dict[str, str], dict[str, str], list[str]]:
        raw: dict[str, Any] = dict(flatten_config(run.config))
        for key, value in run.summary.items():
            if str(key).startswith("_"):
                continue
            metric, _, _ = as_metric(value)
            if metric is None:
                raw[f"{SUMMARY_PARAM_PREFIX}{key}"] = value

        key_map = sanitise_keys(raw.keys(), self.limits)
        params: dict[str, str] = {}
        truncated: list[str] = []
        for source_key, value in raw.items():
            target = key_map[source_key]
            rendered, was_truncated = as_param(value, self.limits)
            params[target] = rendered
            if was_truncated:
                truncated.append(target)
        return params, renamed(key_map), truncated

    def _log_params(self, run_id: str, params: dict[str, str]) -> None:
        entries = [_param(k, v) for k, v in params.items()]
        batch_size = min(self.limits.max_params_tags_per_batch, self.limits.max_entities_per_batch)
        for chunk in _chunks(entries, batch_size):
            self.client.log_batch(run_id, params=chunk)

    # -- tags ------------------------------------------------------------- #

    def _build_tags(
        self, run: SourceRun, report: RunReport, parent_run_id: str | None
    ) -> dict[str, str]:
        tags: dict[str, str] = {
            RUN_ID_TAG: run.id,
            "wandb.state": run.state,
            "wandb.entity": run.entity,
            "wandb.project": run.project,
        }
        if run.url:
            tags["wandb.url"] = run.url
        if run.group:
            tags["wandb.group"] = run.group
        if run.job_type:
            tags["wandb.job_type"] = run.job_type
        if run.sweep_id:
            tags["wandb.sweep_id"] = run.sweep_id
        if parent_run_id:
            tags["mlflow.parentRunId"] = parent_run_id
        for tag in run.tags:
            tags[f"{TAG_PREFIX}{sanitise_key(tag, self.limits)}"] = "true"

        truncated_tags: list[str] = []
        if run.notes:
            value, was_truncated = as_tag(run.notes, self.limits)
            tags["mlflow.note.content"] = value
            if was_truncated:
                truncated_tags.append("mlflow.note.content")

        if report.dropped.total:
            tags["wandb.dropped"] = as_tag(report.dropped.as_dict(), self.limits)[0]
        if report.renamed_keys:
            tags["wandb.renamed_keys"] = as_tag(report.renamed_keys, self.limits)[0]
        if report.truncated_params:
            tags["wandb.truncated_params"] = as_tag(sorted(report.truncated_params), self.limits)[0]
        if truncated_tags:
            tags["wandb.truncated_tags"] = as_tag(truncated_tags, self.limits)[0]
        report.truncated_tags = truncated_tags

        # Every tag value is subject to the tag limit, including ones built above.
        return {key: as_tag(value, self.limits)[0] for key, value in tags.items()}

    # -- sweeps ----------------------------------------------------------- #

    def _sweep_parent(self, run: SourceRun, experiment_id: str) -> str | None:
        """Create (once) the synthetic parent run standing in for a W&B sweep.

        The parent has no W&B counterpart beyond the sweep id -- the search
        space and method are not migrated. See MAPPING.md.
        """
        sweep_id = run.sweep_id
        if not sweep_id:
            return None
        if self.state is None:
            return None
        # Held across the create: two children of one sweep migrating in
        # parallel must not each create a parent.
        with self.state.lock:
            cached = self.state.sweep_parent(sweep_id)
            if cached:
                return cached
            return self._create_sweep_parent(run, experiment_id, sweep_id)

    def _create_sweep_parent(self, run: SourceRun, experiment_id: str, sweep_id: str) -> str:
        parent = self.client.create_run(
            experiment_id=experiment_id,
            start_time=parse_timestamp(run.created_at),
            tags={
                "wandb.sweep_id": sweep_id,
                SWEEP_PARENT_TAG: sweep_id,
                "wandb.is_sweep_parent": "true",
                "wandb.entity": run.entity,
                "wandb.project": run.project,
            },
            run_name=f"sweep-{sweep_id}",
        )
        parent_id = str(parent.info.run_id)
        self.client.set_terminated(parent_id, status="FINISHED")
        assert self.state is not None
        self.state.remember_sweep_parent(sweep_id, parent_id)
        return parent_id

    # -- files and artifacts ---------------------------------------------- #

    def _migrate_files(self, run: SourceRun, run_id: str, report: RunReport) -> int:
        count = 0
        with tempfile.TemporaryDirectory(prefix="w2m-files-") as tmp:
            root = Path(tmp)
            for source_file in run.files():
                if source_file.size > self.options.max_artifact_bytes:
                    report.artifacts_skipped += 1
                    logger.warning(
                        "skipping file %s (%d bytes exceeds --max-artifact-size)",
                        source_file.name,
                        source_file.size,
                    )
                    continue
                source_file.download(root)
                count += 1
            if count:
                self.client.log_artifacts(run_id, str(root), artifact_path=FILES_ARTIFACT_PATH)
        return count

    def _migrate_artifacts(self, run: SourceRun, run_id: str, report: RunReport) -> None:
        for artifact in run.artifacts():
            if artifact.is_reference:
                # Bytes live in someone else's bucket. Record the URI; do not reach for it.
                report.reference_artifacts.append(artifact.name)
                logger.info("artifact %s is a reference; recording URI only", artifact.name)
                continue
            if artifact.size > self.options.max_artifact_bytes:
                report.artifacts_skipped += 1
                logger.warning(
                    "skipping artifact %s (%d bytes exceeds --max-artifact-size)",
                    artifact.name,
                    artifact.size,
                )
                continue
            with tempfile.TemporaryDirectory(prefix="w2m-artifact-") as tmp:
                root = Path(tmp)
                artifact.download(root)
                (root / "_wandb_artifact.json").write_text(
                    json.dumps(_artifact_metadata(artifact), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                path = f"{ARTIFACTS_ARTIFACT_PATH}/{sanitise_key(artifact.name, self.limits)}"
                self.client.log_artifacts(run_id, str(root), artifact_path=path)
            report.artifacts_migrated += 1

        if report.reference_artifacts:
            self.client.set_tag(
                run_id,
                "wandb.reference_artifacts",
                as_tag(_reference_summary(run), self.limits)[0],
            )

    # -- end time --------------------------------------------------------- #

    def _end_time(
        self, run: SourceRun, start_time: int | None, last_ts: int | None, report: RunReport
    ) -> int | None:
        """W&B exposes no true end time; this is the documented fallback chain.

        Which source was used is recorded so the approximation stays auditable.
        """
        source = "history"
        end = last_ts
        if end is None:
            end = _seconds_to_ms(run.summary.get("_timestamp"))
            source = "summary._timestamp"
        if end is None and start_time is not None:
            runtime = _seconds_to_ms(run.summary.get("_runtime"))
            if runtime is not None:
                end = start_time + runtime
                source = "start+_runtime"
        if end is None:
            end = start_time
            source = "start_time"
        if end is not None and start_time is not None and end < start_time:
            end = start_time
            source = "start_time"
        if report.mlflow_run_id:
            self.client.set_tag(report.mlflow_run_id, "wandb.end_time_source", source)
        return end


def _artifact_metadata(artifact: SourceArtifact) -> dict[str, Any]:
    """Inert sidecar: MLflow has no home for versions, aliases or lineage."""
    return {
        "name": artifact.name,
        "type": artifact.type,
        "version": artifact.version,
        "aliases": list(artifact.aliases),
        "digest": artifact.digest,
        "size": artifact.size,
        "note": "Versions, aliases, lineage and metadata are not migrated. See MAPPING.md.",
    }


def _reference_summary(run: SourceRun) -> list[dict[str, Any]]:
    return [
        {"name": a.name, "uris": list(a.source_uris)} for a in run.artifacts() if a.is_reference
    ]
