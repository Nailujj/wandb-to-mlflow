"""Tier 3: the real loop against real services.

Skipped unless ``W2M_E2E=1`` and ``WANDB_API_KEY`` are both set. This is the
only tier that can catch W&B API drift — a changed field name, a changed
pagination contract, a summary shape that moved — which unit tests structurally
cannot, because they mock the very thing that drifted.

Run it with::

    W2M_E2E=1 W2M_E2E_ENTITY=<your-entity> uv run pytest -m e2e

It creates a timestamped ``w2m-selftest-*`` project, migrates it, verifies the
migration against the seeder's manifest, and deletes the runs afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from wandb_to_mlflow.migrate import MigrateOptions, Migrator
from wandb_to_mlflow.seed import cleanup, default_project_name, seed
from wandb_to_mlflow.source import WandbProject
from wandb_to_mlflow.verify import Verifier, format_report

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("W2M_E2E") != "1" or not os.environ.get("WANDB_API_KEY"),
        reason="set W2M_E2E=1 and WANDB_API_KEY to run the real loop",
    ),
]

ENTITY = os.environ.get("W2M_E2E_ENTITY", "")
#: Full scale is 20,000 steps. Override with W2M_E2E_STEPS to shorten a debug run.
STEPS = int(os.environ.get("W2M_E2E_STEPS", "0")) or None


@pytest.fixture(scope="module")
def seeded() -> Iterator[tuple[str, object]]:
    """Seed a real project, and clean it up even if seeding dies partway.

    Learned the hard way: a seeding failure mid-fixture leaves an orphaned
    project behind, because a plain teardown after `yield` never runs.
    """
    if not ENTITY:
        pytest.skip("set W2M_E2E_ENTITY to the W&B entity to seed into")
    project = default_project_name()
    keep = os.environ.get("W2M_E2E_KEEP") == "1"
    try:
        created, manifest = seed(ENTITY, project, steps=STEPS)
    except BaseException:
        if not keep:
            _cleanup_quietly(project)
        raise
    try:
        yield created, manifest
    finally:
        if not keep:
            _cleanup_quietly(created)


def _cleanup_quietly(project: str) -> None:
    try:
        cleanup(ENTITY, project)
    except Exception as exc:  # cleanup failing must not mask the real failure
        print(f"could not clean up {ENTITY}/{project}: {exc}")


def test_the_whole_loop(seeded: tuple[str, object], tmp_path: Path) -> None:
    """seed -> migrate -> verify against the manifest -> exit clean."""
    project, manifest = seeded
    uri = f"file://{tmp_path / 'mlruns'}"
    client = MlflowClient(tracking_uri=uri)

    options = MigrateOptions(experiment=project, include_artifacts=True, include_files=True)
    result = Migrator(client, options).migrate_project(WandbProject.connect(ENTITY, project))
    assert result.failures == [], [r.error for r in result.failures]

    report = Verifier(client).verify(manifest, project)  # type: ignore[arg-type]
    assert report.failures == [], format_report(report)
    assert report.expected_losses, "the hostile runs must produce documented loss"


def test_scan_history_returns_every_point(seeded: tuple[str, object]) -> None:
    """The regression this exists for: `run.history()` silently samples to ~500.

    Only a real W&B run can catch a change here, which is the entire argument
    for this tier existing.
    """
    project, _ = seeded
    source = WandbProject.connect(ENTITY, project)
    throughput = next(run for run in source.runs() if run.name == "many-steps")
    rows = list(throughput.history())
    assert len(rows) == (STEPS or 20_000)
    assert len({row["_step"] for row in rows}) == len(rows)


def test_re_migration_is_idempotent_against_real_data(
    seeded: tuple[str, object], tmp_path: Path
) -> None:
    project, _ = seeded
    uri = f"file://{tmp_path / 'mlruns'}"
    client = MlflowClient(tracking_uri=uri)
    options = MigrateOptions(experiment=project)
    source = WandbProject.connect(ENTITY, project)

    first = Migrator(client, options).migrate_project(source)
    second = Migrator(client, options).migrate_project(WandbProject.connect(ENTITY, project))
    assert all(report.skipped for report in second.reports)
    assert {r.mlflow_run_id for r in first.reports} == {r.mlflow_run_id for r in second.reports}
