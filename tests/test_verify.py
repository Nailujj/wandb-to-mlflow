"""Verification must catch injected corruption -- and must not cry wolf about
loss that MAPPING.md documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from mlflow.tracking import MlflowClient

from tests import fixtures
from wandb_to_mlflow.migrate import MigrateOptions, Migrator
from wandb_to_mlflow.verify import (
    ExpectedRun,
    Manifest,
    Severity,
    Verifier,
    format_report,
    manifest_from_source,
)


@pytest.fixture
def client(tmp_path: Path) -> MlflowClient:
    return MlflowClient(tracking_uri=f"file://{tmp_path / 'mlruns'}")


@pytest.fixture
def migrated(client: MlflowClient) -> tuple[MlflowClient, Manifest, Any]:
    """A migrated project plus the manifest describing what should be there."""
    project = fixtures.fake_project()
    manifest = manifest_from_source(project, MigrateOptions(experiment="verified"))
    result = Migrator(client, MigrateOptions(experiment="verified")).migrate_project(project)
    return client, manifest, result


def verify(client: MlflowClient, manifest: Manifest) -> Any:
    return Verifier(client).verify(manifest, "verified")


def expect(manifest: Manifest, wandb_id: str) -> ExpectedRun:
    return next(r for r in manifest.runs if r.wandb_run_id == wandb_id)


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #


def test_a_clean_migration_verifies(migrated: Any) -> None:
    client, manifest, _ = migrated
    report = verify(client, manifest)
    assert report.failures == [], format_report(report)
    assert report.ok
    assert report.runs_expected == len(manifest.runs)


def test_documented_loss_is_reported_but_does_not_fail(migrated: Any) -> None:
    """This distinction is the whole point of verify."""
    client, manifest, _ = migrated
    report = verify(client, manifest)
    assert report.ok
    losses = {d.wandb_run_id for d in report.expected_losses}
    assert "r10-media" in losses  # images and tables
    assert "r03-nonfinite" in losses  # NaN and inf
    assert "r04-bools" in losses  # the bool trap
    assert all(d.severity is Severity.EXPECTED for d in report.expected_losses)


def test_report_text_says_so_when_nothing_is_wrong(migrated: Any) -> None:
    client, manifest, _ = migrated
    text = format_report(verify(client, manifest))
    assert "No unexpected loss" in text
    assert "MISMATCHES" not in text


# --------------------------------------------------------------------------- #
# injected corruption -- each of these must be caught
# --------------------------------------------------------------------------- #


def test_detects_a_missing_run(migrated: Any) -> None:
    client, manifest, result = migrated
    victim = next(r for r in result.reports if r.wandb_run_id == "r12-metadata")
    client.delete_run(victim.mlflow_run_id)
    report = verify(client, manifest)
    assert not report.ok
    assert [d.kind for d in report.failures] == ["missing_run"]
    assert report.failures[0].wandb_run_id == "r12-metadata"


def test_detects_a_missing_metric_series(migrated: Any) -> None:
    client, manifest, _ = migrated
    expect(manifest, "r01-nested").expected_metric_keys.append("ghost")
    expect(manifest, "r01-nested").expected_metric_point_counts["ghost"] = 3
    report = verify(client, manifest)
    assert not report.ok
    assert any(d.kind == "missing_metrics" and "ghost" in d.expected for d in report.failures)


def test_detects_a_fabricated_metric_series(migrated: Any) -> None:
    """Where a bool logged as 1.0 would surface."""
    client, manifest, _ = migrated
    expect(manifest, "r04-bools").expected_metric_keys.remove("loss")
    report = verify(client, manifest)
    assert not report.ok
    assert any(d.kind == "unexpected_metrics" and "loss" in d.actual for d in report.failures)


def test_detects_a_truncated_metric_series(migrated: Any) -> None:
    client, manifest, _ = migrated
    expect(manifest, "r07-sparse").expected_metric_point_counts["dense"] = 99
    report = verify(client, manifest)
    assert not report.ok
    diff = next(d for d in report.failures if d.kind == "point_count")
    assert (diff.expected, diff.actual) == (99, 100)


def test_detects_a_wrong_final_value(migrated: Any) -> None:
    client, manifest, _ = migrated
    expect(manifest, "r01-nested").expected_final_values["final.accuracy"] = 0.5
    report = verify(client, manifest)
    assert not report.ok
    assert any(d.kind == "final_value" for d in report.failures)


def test_detects_a_wrong_status(migrated: Any) -> None:
    client, manifest, _ = migrated
    expect(manifest, "r13-crashed").expected_status = "FINISHED"
    report = verify(client, manifest)
    assert not report.ok
    diff = next(d for d in report.failures if d.kind == "status")
    assert (diff.expected, diff.actual) == ("FINISHED", "FAILED")


def test_detects_a_wrong_param_count(migrated: Any) -> None:
    client, manifest, _ = migrated
    expect(manifest, "r01-nested").expected_param_count = 999
    assert any(d.kind == "param_count" for d in verify(client, manifest).failures)


def test_detects_a_corrupted_param_value(migrated: Any) -> None:
    client, manifest, _ = migrated
    expect(manifest, "r01-nested").expected_params = {"optimizer.kind": "adam"}
    diff = next(d for d in verify(client, manifest).failures if d.kind == "param_value")
    assert (diff.expected, diff.actual) == ("adam", "sgd")


def test_detects_a_value_that_should_have_been_dropped_but_was_not(migrated: Any) -> None:
    """Losing less than expected means a NaN or a bool got through as real data."""
    client, manifest, _ = migrated
    expect(manifest, "r03-nonfinite").expected_dropped["nonfinite"] = 5
    report = verify(client, manifest)
    assert not report.ok
    diff = next(d for d in report.failures if d.kind == "dropped_mismatch")
    assert (diff.expected, diff.actual) == (5, 3)


def test_detects_missing_sweep_parentage(migrated: Any) -> None:
    client, manifest, result = migrated
    child = next(r for r in result.reports if r.wandb_run_id == "r15-sweep-0")
    expect(manifest, "r15-sweep-0").expected_parent = "sw-abc123"
    assert verify(client, manifest).ok  # parentage is correct as migrated

    client.delete_tag(child.mlflow_run_id, "mlflow.parentRunId")
    assert any(d.kind == "parent" for d in verify(client, manifest).failures)


def test_detects_an_unexpected_parent(migrated: Any) -> None:
    client, manifest, result = migrated
    orphan = next(r for r in result.reports if r.wandb_run_id == "r04-bools")
    client.set_tag(orphan.mlflow_run_id, "mlflow.parentRunId", "fabricated")
    diff = next(d for d in verify(client, manifest).failures if d.kind == "parent")
    assert diff.actual == "fabricated"


def test_detects_a_missing_experiment(client: MlflowClient) -> None:
    report = Verifier(client).verify(Manifest(runs=[ExpectedRun("x")]), "never-created")
    assert not report.ok
    assert report.failures[0].kind == "missing_experiment"


def test_failures_are_rendered_with_expected_and_actual(migrated: Any) -> None:
    client, manifest, _ = migrated
    expect(manifest, "r13-crashed").expected_status = "FINISHED"
    text = format_report(verify(client, manifest))
    assert "MISMATCHES" in text
    assert "r13-crashed" in text
    assert "expected='FINISHED'" in text and "actual='FAILED'" in text


# --------------------------------------------------------------------------- #
# manifest round trip
# --------------------------------------------------------------------------- #


def test_manifest_survives_a_json_round_trip(tmp_path: Path, migrated: Any) -> None:
    client, manifest, _ = migrated
    path = tmp_path / "manifest.json"
    manifest.dump(path)
    assert verify(client, Manifest.load(path)).ok


def test_manifest_json_matches_the_documented_shape(tmp_path: Path, migrated: Any) -> None:
    _, manifest, _ = migrated
    path = tmp_path / "manifest.json"
    manifest.dump(path)
    data = json.loads(path.read_text())
    assert set(data) == {"wandb", "runs"}
    entry = next(r for r in data["runs"] if r["wandb_run_id"] == "r01-nested")
    assert set(entry) >= {
        "wandb_run_id",
        "expected_status",
        "expected_param_count",
        "expected_params",
        "expected_metric_keys",
        "expected_metric_point_counts",
        "expected_final_values",
        "expected_dropped",
        "expected_parent",
    }


def test_manifest_load_ignores_unknown_fields(tmp_path: Path) -> None:
    """Forward compatibility: a newer seeder's manifest must not crash an older verifier."""
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"runs": [{"wandb_run_id": "a", "from_the_future": 1}]}))
    assert Manifest.load(path).runs[0].wandb_run_id == "a"


def test_planning_a_manifest_writes_nothing(client: MlflowClient) -> None:
    manifest = manifest_from_source(fixtures.fake_project(), MigrateOptions(experiment="planned"))
    assert client.get_experiment_by_name("planned") is None
    assert len(manifest.runs) == len(fixtures.all_runs())
    assert manifest.wandb == {"entity": "acme", "project": "w2m-selftest"}
