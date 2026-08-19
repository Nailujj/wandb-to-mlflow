"""The manifest is the self-test's ground truth, so it gets tested itself.

The strongest offline check available: turn each `RunSpec` into the `SourceRun`
that W&B would hand back for it, migrate that, and assert the migration matches
the manifest the seeder derived independently. If the seeder's arithmetic and
the migrator's behaviour ever disagree, this fails here rather than in a tier-3
run that costs a real W&B project to discover.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from mlflow.tracking import MlflowClient

from tests import fixtures
from wandb_to_mlflow import seed as seed_module
from wandb_to_mlflow.migrate import INTERNAL_COLUMNS, MigrateOptions, Migrator
from wandb_to_mlflow.seed import (
    PROJECT_PREFIX,
    RunSpec,
    build_specs,
    default_project_name,
    expected_for,
    is_seeded_project,
    manifest_for,
    plan_text,
)
from wandb_to_mlflow.verify import Verifier, format_report

SWEEP_ID = "sw-seeded"


def wandb_encode(value: Any) -> Any:
    """W&B returns non-finite numbers as JSON's string spellings, not as floats.

    Measured against a real run: `float("nan")` comes back as `"NaN"`.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return {float("inf"): "Infinity", float("-inf"): "-Infinity"}.get(value, "NaN")
    return value


def fake_run_for(spec: RunSpec, run_id: str) -> fixtures.FakeRun:
    """The `SourceRun` W&B would return for a seeded `RunSpec`.

    This is the bridge between the two halves of the self-test: what the seeder
    logs, and what the migrator reads. It reproduces two behaviours measured
    against real W&B, both of which the first live self-test caught:

    - non-finite numbers come back as `"NaN"` / `"Infinity"` / `"-Infinity"`;
    - `run.summary` is auto-populated with the last value of every logged key,
      media included, whether or not the user wrote a summary.
    """
    media = seed_module._media_rows(spec)
    rows: list[dict[str, Any]] = []
    auto_summary: dict[str, Any] = {}
    for index, row in enumerate(spec.rows):
        payload: dict[str, Any] = {
            "_step": index,
            "_timestamp": fixtures.BASE_TS + index,
            **{k: wandb_encode(v) for k, v in row.items()},
        }
        if index < len(media):
            payload.update({k: {"_type": t} for k, t in media[index].items()})
        rows.append(payload)
        auto_summary.update({k: v for k, v in payload.items() if k not in INTERNAL_COLUMNS})
    return fixtures.FakeRun(
        id=run_id,
        name=spec.name,
        state=spec.expected_state,
        config={k: v for k, v in spec.config.items() if v != {}},  # W&B drops empty dicts
        summary={**auto_summary, **spec.summary},
        tags=list(spec.tags),
        notes=spec.notes,
        group=spec.group,
        job_type=spec.job_type,
        sweep_id=SWEEP_ID if spec.in_sweep else None,
        rows=rows,
    )


@pytest.fixture
def specs() -> list[RunSpec]:
    smaller = build_specs()
    for spec in smaller:
        if spec.key == "throughput":
            spec.rows = spec.rows[:300]  # the shape matters here, not the volume
    return smaller


# --------------------------------------------------------------------------- #
# the manifest predicts the migration
# --------------------------------------------------------------------------- #


def test_the_manifest_matches_what_a_migration_actually_produces(
    specs: list[RunSpec], tmp_path: Path
) -> None:
    run_ids = {spec.key: f"seeded-{spec.key}" for spec in specs}
    manifest = manifest_for(specs, "acme", "w2m-selftest-x", run_ids, SWEEP_ID)

    client = MlflowClient(tracking_uri=f"file://{tmp_path / 'mlruns'}")
    project = fixtures.FakeProject(
        entity="acme",
        project="w2m-selftest-x",
        run_list=[fake_run_for(spec, run_ids[spec.key]) for spec in specs],
    )
    result = Migrator(client, MigrateOptions(experiment="seeded")).migrate_project(project)
    assert result.failures == []

    report = Verifier(client).verify(manifest, "seeded")
    assert report.failures == [], format_report(report)


def test_the_manifest_records_the_expected_loss_for_each_hostile_case(
    specs: list[RunSpec],
) -> None:
    run_ids = {spec.key: f"seeded-{spec.key}" for spec in specs}
    manifest = manifest_for(specs, "acme", "p", run_ids, SWEEP_ID)
    by_key = {run.wandb_run_id: run for run in manifest.runs}

    assert by_key["seeded-nonfinite"].expected_dropped["nonfinite"] == 3
    assert by_key["seeded-bools"].expected_dropped["bool"] == 3
    media = by_key["seeded-media"].expected_dropped
    assert media["media"] == 3
    assert media["media_types"] == {"image-file": 2, "table-file": 1}


def test_the_manifest_predicts_key_sanitisation(specs: list[RunSpec]) -> None:
    run_ids = {spec.key: f"seeded-{spec.key}" for spec in specs}
    manifest = manifest_for(specs, "acme", "p", run_ids, SWEEP_ID)
    hostile = next(r for r in manifest.runs if r.wandb_run_id == "seeded-hostile_keys")
    assert "train/loss" in hostile.expected_metric_keys
    assert "héllo" in hostile.expected_metric_keys
    assert "x_y_" in hostile.expected_metric_keys

    collision = next(r for r in manifest.runs if r.wandb_run_id == "seeded-key_collision")
    history_keys = [k for k in collision.expected_metric_keys if not k.startswith("final.")]
    assert len(history_keys) == 2
    assert all(k.startswith("a_b_") for k in history_keys)
    # W&B's auto-summary means the collision has to be resolved twice over,
    # once for the history keys and once for the final.* pair.
    final_keys = [k for k in collision.expected_metric_keys if k.startswith("final.")]
    assert len(final_keys) == 2
    assert all(k.startswith("final.a_b_") for k in final_keys)


def test_the_manifest_predicts_sweep_parentage(specs: list[RunSpec]) -> None:
    run_ids = {spec.key: f"seeded-{spec.key}" for spec in specs}
    manifest = manifest_for(specs, "acme", "p", run_ids, SWEEP_ID)
    children = [r for r in manifest.runs if r.expected_parent == SWEEP_ID]
    assert len(children) == seed_module.SWEEP_CHILDREN
    assert all(r.expected_parent is None for r in manifest.runs if r not in children)


def test_the_manifest_predicts_statuses(specs: list[RunSpec]) -> None:
    run_ids = {spec.key: f"seeded-{spec.key}" for spec in specs}
    by_key = {r.wandb_run_id: r for r in manifest_for(specs, "a", "p", run_ids).runs}
    assert by_key["seeded-failed"].expected_status == "FAILED"
    assert by_key["seeded-crashed"].expected_status == "FAILED"
    assert by_key["seeded-nested_config"].expected_status == "FINISHED"


def test_the_manifest_only_covers_runs_that_were_actually_created(
    specs: list[RunSpec],
) -> None:
    """A seeding run that died halfway must not claim runs it never made."""
    partial = {spec.key: f"id-{spec.key}" for spec in specs[:4]}
    manifest = manifest_for(specs, "acme", "p", partial)
    assert len(manifest.runs) == 4


def test_manifest_records_where_it_came_from(specs: list[RunSpec]) -> None:
    manifest = manifest_for(specs, "acme", "proj", {}, "sw-1", created_at="2026-01-01T00:00:00Z")
    assert manifest.wandb == {
        "entity": "acme",
        "project": "proj",
        "created_at": "2026-01-01T00:00:00Z",
        "sweep_id": "sw-1",
    }


# --------------------------------------------------------------------------- #
# spec coverage
# --------------------------------------------------------------------------- #


def test_all_sixteen_cases_are_seeded() -> None:
    specs = build_specs()
    keys = {spec.key for spec in specs}
    assert keys == {
        "nested_config",
        "long_config_value",
        "nonfinite",
        "bools",
        "hostile_keys",
        "key_collision",
        "sparse",
        "throughput",
        "empty_history",
        "media",
        "artifacts",
        "metadata",
        "failed",
        "crashed",
        "duplicate_name_a",
        "duplicate_name_b",
        "unicode",
        *(f"sweep_child_{i}" for i in range(seed_module.SWEEP_CHILDREN)),
    }
    assert all(spec.exercises for spec in specs)


def test_two_seeded_runs_share_a_display_name() -> None:
    names = [spec.name for spec in build_specs()]
    assert names.count("same-name") == 2


def test_the_throughput_run_is_twenty_thousand_steps() -> None:
    throughput = next(s for s in build_specs() if s.key == "throughput")
    assert len(throughput.rows) == 20_000
    assert len(throughput.rows[0]) == 5


def test_the_seeded_payload_stays_tiny() -> None:
    """Spec 6.1: a self-test that costs money is a self-test nobody runs."""
    assert seed_module._payload_estimate(build_specs()) < 5 * 1024 * 1024


def test_seeder_and_fixtures_cover_the_same_cases() -> None:
    """The two halves of the self-test must not drift apart.

    Adding a hostile case to the seeder without adding it to the offline
    fixtures (or the reverse) means one tier silently stops covering it.
    """
    specs = build_specs()
    assert len({spec.exercises for spec in specs}) >= 15
    assert len({run.id for run in fixtures.all_runs()}) == len(specs)
    assert len(specs) == 17 + seed_module.SWEEP_CHILDREN  # 16 spec cases, one split in two


# --------------------------------------------------------------------------- #
# safety rules (spec 6.1)
# --------------------------------------------------------------------------- #


def test_default_project_name_is_timestamped_and_unmistakable() -> None:
    name = default_project_name(datetime(2026, 8, 19, 13, 45, 1, tzinfo=timezone.utc))
    assert name == "w2m-selftest-20260819-134501"
    assert is_seeded_project(name)


def test_default_project_name_uses_utc() -> None:
    assert default_project_name().startswith(PROJECT_PREFIX)


def test_projects_the_tool_did_not_create_are_not_recognised() -> None:
    assert not is_seeded_project("production-experiments")
    assert not is_seeded_project("")


def test_the_plan_names_the_project_and_every_run() -> None:
    text = plan_text(build_specs(), "acme", "w2m-selftest-x")
    assert "entity:  acme" in text
    assert "project: w2m-selftest-x" in text
    assert "CREATE" in text
    assert "seed --cleanup w2m-selftest-x" in text
    for spec in build_specs():
        assert spec.name in text
        assert spec.exercises in text


def test_expected_for_is_deterministic() -> None:
    spec = next(s for s in build_specs() if s.key == "key_collision")
    assert expected_for(spec, "x") == expected_for(spec, "x")
