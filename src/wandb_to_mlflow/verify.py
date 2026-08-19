"""The verification oracle.

The distinction this module exists to make: **expected loss** (media, NaN, bools
— things MAPPING.md says do not survive) is not a failure. **Unexpected loss**
(a finite scalar that should have migrated and did not) always is. A verifier
that cannot tell them apart is either permanently red or useless.

Two modes:

- against a manifest — ground truth recorded by the seeder from what it actually
  logged. This is the self-test oracle.
- against a live W&B project — expectations derived by re-planning the migration
  from the source. This only proves the migration matches what the tool would do
  today, which is why the manifest mode exists and is what the self-test uses.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from wandb_to_mlflow.migrate import MigrateOptions, Migrator, RunReport
from wandb_to_mlflow.source import SourceProject
from wandb_to_mlflow.state import RUN_ID_TAG

logger = logging.getLogger(__name__)

#: Final metric values are floats that made a JSON round trip; compare with a
#: tolerance rather than pretending binary equality survives serialisation.
FLOAT_TOLERANCE = 1e-9


class Severity(str, Enum):
    """Whether a difference fails the verification."""

    EXPECTED = "expected"  # documented in MAPPING.md; informational only
    UNEXPECTED = "unexpected"  # real loss or corruption; fails the run


@dataclass(frozen=True)
class Diff:
    wandb_run_id: str
    kind: str
    detail: str
    expected: Any = None
    actual: Any = None
    severity: Severity = Severity.UNEXPECTED

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "severity": self.severity.value}


@dataclass
class VerifyReport:
    experiment: str | None = None
    runs_expected: int = 0
    runs_found: int = 0
    diffs: list[Diff] = field(default_factory=list)

    @property
    def failures(self) -> list[Diff]:
        return [d for d in self.diffs if d.severity is Severity.UNEXPECTED]

    @property
    def expected_losses(self) -> list[Diff]:
        return [d for d in self.diffs if d.severity is Severity.EXPECTED]

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "runs_expected": self.runs_expected,
            "runs_found": self.runs_found,
            "ok": self.ok,
            "failures": [d.as_dict() for d in self.failures],
            "expected_losses": [d.as_dict() for d in self.expected_losses],
        }


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


@dataclass
class ExpectedRun:
    """One run's ground truth, as recorded in ``manifest.json``."""

    wandb_run_id: str
    expected_status: str = "FINISHED"
    expected_param_count: int | None = None
    expected_params: dict[str, str] = field(default_factory=dict)
    expected_metric_keys: list[str] = field(default_factory=list)
    expected_metric_point_counts: dict[str, int] = field(default_factory=dict)
    expected_final_values: dict[str, float] = field(default_factory=dict)
    expected_dropped: dict[str, Any] = field(default_factory=dict)
    expected_parent: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpectedRun:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_report(cls, report: RunReport) -> ExpectedRun:
        """Derive expectations from a planned (dry-run) migration."""
        return cls(
            wandb_run_id=report.wandb_run_id,
            expected_status=report.status,
            expected_param_count=report.param_count,
            expected_metric_keys=sorted(report.metric_keys),
            expected_metric_point_counts=dict(report.metric_point_counts),
            expected_final_values=dict(report.final_values),
            expected_dropped=report.dropped.as_dict(),
            # A sweep child must end up nested. Planning knows the sweep id even
            # though it has not created the synthetic parent run yet.
            expected_parent=report.wandb_sweep_id,
        )


@dataclass
class Manifest:
    wandb: dict[str, Any] = field(default_factory=dict)
    runs: list[ExpectedRun] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            wandb=data.get("wandb", {}),
            runs=[ExpectedRun.from_dict(entry) for entry in data.get("runs", [])],
        )

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def as_dict(self) -> dict[str, Any]:
        return {"wandb": self.wandb, "runs": [asdict(run) for run in self.runs]}


def manifest_from_source(source: SourceProject, options: MigrateOptions) -> Manifest:
    """Expectations derived by planning a migration without writing anything.

    Used by live-mode verification and by ``plan``. Note the limitation called
    out in the module docstring: this compares the migration against the same
    logic that produced it.
    """
    planner = Migrator(MlflowClient(), MigrateOptions(**{**options.__dict__, "dry_run": True}))
    result = planner.migrate_project(source)
    return Manifest(
        wandb={"entity": source.entity, "project": source.project},
        runs=[ExpectedRun.from_report(report) for report in result.reports if not report.error],
    )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


class Verifier:
    def __init__(self, client: MlflowClient) -> None:
        self.client = client

    def verify(self, manifest: Manifest, experiment_name: str) -> VerifyReport:
        report = VerifyReport(experiment=experiment_name, runs_expected=len(manifest.runs))
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            report.diffs.append(
                Diff(
                    wandb_run_id="-",
                    kind="missing_experiment",
                    detail=f"experiment {experiment_name!r} does not exist",
                    expected=experiment_name,
                )
            )
            return report

        actual = self._index(str(experiment.experiment_id))
        report.runs_found = len(actual)
        for expected in manifest.runs:
            found = actual.get(expected.wandb_run_id)
            if found is None:
                report.diffs.append(
                    Diff(
                        wandb_run_id=expected.wandb_run_id,
                        kind="missing_run",
                        detail="no MLflow run carries this wandb.run_id",
                    )
                )
                continue
            report.diffs.extend(self._compare(expected, found))
        return report

    # -- loading the actual state ----------------------------------------- #

    def _index(self, experiment_id: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        token: str | None = None
        while True:
            page = self.client.search_runs(
                experiment_ids=[experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=1000,
                page_token=token,
            )
            for run in page:
                wandb_id = run.data.tags.get(RUN_ID_TAG)
                if wandb_id:
                    out[wandb_id] = run
            token = getattr(page, "token", None)
            if not token:
                return out

    # -- per-run comparison ------------------------------------------------ #

    def _compare(self, expected: ExpectedRun, run: Any) -> list[Diff]:
        diffs: list[Diff] = []
        run_id = expected.wandb_run_id
        tags = dict(run.data.tags)

        if run.info.status != expected.expected_status:
            diffs.append(
                Diff(
                    run_id,
                    "status",
                    "run status differs",
                    expected.expected_status,
                    run.info.status,
                )
            )

        params = dict(run.data.params)
        if (
            expected.expected_param_count is not None
            and len(params) != expected.expected_param_count
        ):
            diffs.append(
                Diff(
                    run_id,
                    "param_count",
                    "number of params differs",
                    expected.expected_param_count,
                    len(params),
                )
            )
        for key, value in expected.expected_params.items():
            if params.get(key) != value:
                diffs.append(
                    Diff(run_id, "param_value", f"param {key!r} differs", value, params.get(key))
                )

        diffs.extend(self._compare_metrics(expected, run))
        diffs.extend(self._compare_dropped(expected, tags))

        parent = tags.get("mlflow.parentRunId")
        if expected.expected_parent is not None:
            expected_parent_tag = expected.expected_parent
            actual_sweep = tags.get("wandb.sweep_id")
            if actual_sweep != expected_parent_tag or parent is None:
                diffs.append(
                    Diff(
                        run_id,
                        "parent",
                        "sweep parentage differs",
                        expected_parent_tag,
                        actual_sweep,
                    )
                )
        elif parent is not None:
            diffs.append(Diff(run_id, "parent", "run has an unexpected parent", None, parent))

        return diffs

    def _compare_metrics(self, expected: ExpectedRun, run: Any) -> list[Diff]:
        diffs: list[Diff] = []
        run_id = expected.wandb_run_id
        actual_keys = set(run.data.metrics)

        missing = sorted(set(expected.expected_metric_keys) - actual_keys)
        if missing:
            diffs.append(
                Diff(
                    run_id, "missing_metrics", "metrics absent from the migrated run", missing, None
                )
            )
        extra = sorted(actual_keys - set(expected.expected_metric_keys))
        if extra:
            # Fabricated series are as bad as missing ones -- this is where a
            # bool logged as 1.0 would show up.
            diffs.append(
                Diff(
                    run_id, "unexpected_metrics", "metrics present that should not be", None, extra
                )
            )

        for key, count in expected.expected_metric_point_counts.items():
            if key not in actual_keys:
                continue  # already reported as missing
            actual_count = len(self.client.get_metric_history(run.info.run_id, key))
            if actual_count != count:
                diffs.append(
                    Diff(
                        run_id,
                        "point_count",
                        f"metric {key!r} has the wrong number of points",
                        count,
                        actual_count,
                    )
                )

        for key, value in expected.expected_final_values.items():
            actual_value = run.data.metrics.get(key)
            if actual_value is None or not _close(actual_value, value):
                diffs.append(
                    Diff(
                        run_id,
                        "final_value",
                        f"metric {key!r} has the wrong value",
                        value,
                        actual_value,
                    )
                )
        return diffs

    def _compare_dropped(self, expected: ExpectedRun, tags: dict[str, str]) -> list[Diff]:
        """Compare what was lost against what MAPPING.md says should be lost.

        Both directions matter. Losing *more* than expected is unexpected loss.
        Losing *less* is worse than it sounds: it means a value the mapping says
        must be rejected (a NaN, a bool) got through and is now fabricated data.
        """
        run_id = expected.wandb_run_id
        if not expected.expected_dropped:
            return []
        raw = tags.get("wandb.dropped")
        actual = json.loads(raw) if raw else {}
        diffs: list[Diff] = []
        for reason, count in expected.expected_dropped.items():
            actual_count = actual.get(reason)
            if actual_count == count:
                diffs.append(
                    Diff(
                        run_id,
                        "expected_loss",
                        f"{reason}: {_describe(count)} (as documented)",
                        count,
                        actual_count,
                        Severity.EXPECTED,
                    )
                )
                continue
            diffs.append(
                Diff(
                    run_id,
                    "dropped_mismatch",
                    f"{reason} drop count differs from the manifest",
                    count,
                    actual_count,
                )
            )
        return diffs


def _close(actual: float, expected: float) -> bool:
    if math.isnan(expected):  # pragma: no cover - manifests never record NaN finals
        return False
    return math.isclose(actual, expected, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)


def _describe(count: Any) -> str:
    if isinstance(count, dict):
        return ", ".join(f"{k}={v}" for k, v in sorted(count.items())) or "none"
    return str(count)


def format_report(report: VerifyReport) -> str:
    """Render a report as a table. Returns the text; the CLI owns the printing."""
    lines: list[str] = []
    header = f"experiment {report.experiment!r}: {report.runs_found} migrated runs found"
    lines.append(header)
    lines.append(f"{report.runs_expected} runs expected by the manifest")
    lines.append("")

    if report.expected_losses:
        lines.append("Expected loss (documented in MAPPING.md, not a failure):")
        for diff in report.expected_losses:
            lines.append(f"  {diff.wandb_run_id:<24} {diff.detail}")
        lines.append("")

    if report.failures:
        lines.append("MISMATCHES:")
        width = max(len(d.kind) for d in report.failures)
        for diff in report.failures:
            lines.append(
                f"  {diff.wandb_run_id:<24} {diff.kind:<{width}}  {diff.detail}"
                f"\n  {'':<24} {'':<{width}}  expected={diff.expected!r} actual={diff.actual!r}"
            )
    else:
        lines.append("No unexpected loss. Every difference is one MAPPING.md documents.")
    return "\n".join(lines)
