"""Tier 2: fake sources into a real MLflow file store.

No network, but a real MLflow backend -- the assertions read the store back
rather than inspecting the migrator's own bookkeeping, so a bug that reports
success without writing anything cannot pass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlflow
import pytest
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient

from tests import fixtures
from wandb_to_mlflow.coerce import TRUNCATION_MARKER
from wandb_to_mlflow.limits import default_limits
from wandb_to_mlflow.migrate import MigrateOptions, Migrator, parse_timestamp

LIMITS = default_limits()


@pytest.fixture
def client(tmp_path: Path) -> MlflowClient:
    return MlflowClient(tracking_uri=f"file://{tmp_path / 'mlruns'}")


def migrate(
    client: MlflowClient, runs: list[fixtures.FakeRun], **options: Any
) -> tuple[Any, MlflowClient]:
    project = fixtures.FakeProject(run_list=runs)
    migrator = Migrator(client, MigrateOptions(experiment="tests", **options))
    return migrator.migrate_project(project), client


def get_run(client: MlflowClient, result: Any, wandb_id: str) -> Any:
    report = next(r for r in result.reports if r.wandb_run_id == wandb_id)
    assert report.error is None, report.error
    return client.get_run(report.mlflow_run_id)


def metric_history(client: MlflowClient, run_id: str, key: str) -> list[Any]:
    return list(client.get_metric_history(run_id, key))


# --------------------------------------------------------------------------- #
# whole-project smoke
# --------------------------------------------------------------------------- #


def test_every_fixture_migrates_without_error(client: MlflowClient) -> None:
    result, _ = migrate(client, fixtures.all_runs())
    assert result.failures == []
    assert result.ok
    assert len(result.reports) == len(fixtures.all_runs())


def test_experiment_is_created_with_the_requested_name(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_empty_history()])
    experiment = client.get_experiment(result.experiment_id)
    assert experiment.name == "tests"


def test_experiment_defaults_to_the_wandb_project_name(client: MlflowClient) -> None:
    migrator = Migrator(client, MigrateOptions())
    result = migrator.migrate_project(fixtures.FakeProject(run_list=[fixtures.run_bools()]))
    assert client.get_experiment(result.experiment_id).name == "w2m-selftest"


# --------------------------------------------------------------------------- #
# identity and metadata
# --------------------------------------------------------------------------- #


def test_run_id_is_tagged_and_name_is_not_the_key(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_duplicate_name_a(), fixtures.run_duplicate_name_b()])
    a = get_run(client, result, "r14-dup-a")
    b = get_run(client, result, "r14-dup-b")
    assert a.info.run_id != b.info.run_id
    assert a.data.tags["mlflow.runName"] == b.data.tags["mlflow.runName"] == "same-name"
    assert {a.data.tags["wandb.run_id"], b.data.tags["wandb.run_id"]} == {"r14-dup-a", "r14-dup-b"}


def test_metadata_maps_to_tags(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_metadata()])
    tags = get_run(client, result, "r12-metadata").data.tags
    assert tags["wandb.group"] == "ablation-a"
    assert tags["wandb.job_type"] == "train"
    assert tags["wandb.tag.baseline"] == "true"
    assert tags["wandb.tag.v2"] == "true"
    assert tags["wandb.tag.needs review"] == "true"
    assert tags["mlflow.note.content"].startswith("A run with notes.")
    assert tags["wandb.url"].endswith("r12-metadata")
    assert tags["wandb.entity"] == "acme"


def test_unicode_survives_names_notes_and_keys(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_unicode()])
    run = get_run(client, result, "r16-unicode")
    assert run.data.tags["mlflow.runName"] == "ünïcode 🎉 실험"
    assert "🚀" in run.data.tags["mlflow.note.content"]
    assert run.data.params["β"] == "0.9"
    assert "λ/λοιπόν" in run.data.metrics


# --------------------------------------------------------------------------- #
# status mapping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("factory", "wandb_id", "expected"),
    [
        (fixtures.run_crashed, "r13-crashed", "FAILED"),
        (fixtures.run_failed, "r13-failed", "FAILED"),
        (fixtures.run_bools, "r04-bools", "FINISHED"),
    ],
)
def test_state_maps_to_status(
    client: MlflowClient, factory: Any, wandb_id: str, expected: str
) -> None:
    result, _ = migrate(client, [factory()])
    run = get_run(client, result, wandb_id)
    assert run.info.status == expected
    assert run.data.tags["wandb.state"] == factory().state


def test_unknown_state_falls_back_but_is_preserved(client: MlflowClient) -> None:
    run = fixtures.run_bools()
    run.state = "some-new-state"
    result, _ = migrate(client, [run])
    migrated = get_run(client, result, "r04-bools")
    assert migrated.info.status == "FINISHED"
    assert migrated.data.tags["wandb.state"] == "some-new-state"


# --------------------------------------------------------------------------- #
# timestamps
# --------------------------------------------------------------------------- #


def test_backdated_start_time_survives_the_round_trip(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_nested_config()])
    run = get_run(client, result, "r01-nested")
    assert run.info.start_time == parse_timestamp("2023-11-14T22:13:20")


def test_end_time_comes_from_the_last_history_timestamp(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_nested_config()])
    run = get_run(client, result, "r01-nested")
    assert run.info.end_time == int((fixtures.BASE_TS + 2) * 1000)
    assert run.data.tags["wandb.end_time_source"] == "history"


def test_end_time_falls_back_for_a_run_with_no_history(client: MlflowClient) -> None:
    source = fixtures.run_empty_history()
    source.summary = {"_runtime": 120.0}
    result, _ = migrate(client, [source])
    run = get_run(client, result, "r09-empty")
    assert run.data.tags["wandb.end_time_source"] == "start+_runtime"
    assert run.info.end_time == run.info.start_time + 120_000


def test_end_time_never_precedes_start_time(client: MlflowClient) -> None:
    source = fixtures.run_empty_history()
    source.created_at = "2030-01-01T00:00:00"
    source.summary = {"_timestamp": fixtures.BASE_TS}
    result, _ = migrate(client, [source])
    run = get_run(client, result, "r09-empty")
    assert run.info.end_time == run.info.start_time
    assert run.data.tags["wandb.end_time_source"] == "start_time"


def test_naive_timestamps_are_read_as_utc() -> None:
    assert parse_timestamp("2023-11-14T22:13:20") == parse_timestamp("2023-11-14T22:13:20Z")


def test_unparseable_timestamp_degrades_instead_of_crashing() -> None:
    assert parse_timestamp("not a date") is None
    assert parse_timestamp("") is None
    assert parse_timestamp(None) is None


# --------------------------------------------------------------------------- #
# config -> params
# --------------------------------------------------------------------------- #


def test_nested_config_becomes_dotted_params(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_nested_config()])
    params = get_run(client, result, "r01-nested").data.params
    assert params["optimizer.sched.warmup.steps"] == "100"
    assert params["optimizer.kind"] == "sgd"
    assert params["layers"] == "[64, 128, 256]"
    assert params["flags.amp"] == "true"
    assert "_wandb.internal" not in params


def test_long_config_value_is_truncated_and_flagged(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_long_config_value()])
    run = get_run(client, result, "r02-longparam")
    assert len(run.data.params["blob"]) == LIMITS.max_param_val_length
    assert run.data.params["blob"].endswith(TRUNCATION_MARKER)
    assert run.data.params["short"] == "ok"
    assert "blob" in json.loads(run.data.tags["wandb.truncated_params"])


def test_non_numeric_summary_becomes_a_param(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_nested_config()])
    params = get_run(client, result, "r01-nested").data.params
    assert params["summary.notes"] == "done"


# --------------------------------------------------------------------------- #
# summary -> final.* metrics
# --------------------------------------------------------------------------- #


def test_numeric_summary_becomes_a_final_metric_at_step_zero(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_nested_config()])
    run = get_run(client, result, "r01-nested")
    assert run.data.metrics["final.accuracy"] == pytest.approx(0.9137)
    points = metric_history(client, run.info.run_id, "final.accuracy")
    assert [p.step for p in points] == [0]


def test_nonfinite_summary_value_is_dropped_not_logged(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_nonfinite()])
    run = get_run(client, result, "r03-nonfinite")
    assert "final.diverged" not in run.data.metrics
    assert "final.accuracy" in run.data.metrics


# --------------------------------------------------------------------------- #
# history -> metrics
# --------------------------------------------------------------------------- #


def test_history_keeps_original_steps_and_timestamps(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_nested_config()])
    run = get_run(client, result, "r01-nested")
    points = sorted(metric_history(client, run.info.run_id, "loss"), key=lambda p: p.step)
    assert [p.step for p in points] == [0, 1, 2]
    assert [p.value for p in points] == [1.0, 0.5, 0.25]
    assert [p.timestamp for p in points] == [int((fixtures.BASE_TS + i) * 1000) for i in range(3)]


def test_bools_never_become_metrics(client: MlflowClient) -> None:
    """The single most dangerous silent corruption: True logged as 1.0."""
    result, _ = migrate(client, [fixtures.run_bools()])
    run = get_run(client, result, "r04-bools")
    assert "improved" not in run.data.metrics
    assert "final.converged" not in run.data.metrics
    assert "loss" in run.data.metrics
    assert run.data.metrics["final.steps"] == 3.0
    assert run.data.params["use_amp"] == "true"
    assert json.loads(run.data.tags["wandb.dropped"])["bool"] == 3


def test_nonfinite_history_values_are_dropped_and_counted(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_nonfinite()])
    run = get_run(client, result, "r03-nonfinite")
    loss = metric_history(client, run.info.run_id, "loss")
    assert [p.step for p in loss] == [0]
    assert all(math.isfinite(p.value) for p in loss)
    dropped = json.loads(run.data.tags["wandb.dropped"])
    assert dropped["nonfinite"] == 4  # 3 in history, 1 in summary


def test_media_is_dropped_and_reported_by_type(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_media()])
    run = get_run(client, result, "r10-media")
    assert "sample" not in run.data.metrics
    dropped = json.loads(run.data.tags["wandb.dropped"])
    assert dropped["media"] == 3
    assert dropped["media_types"] == {"image-file": 2, "table-file": 1}
    assert len(metric_history(client, run.info.run_id, "loss")) == 2


def test_sparse_keys_keep_their_own_steps(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_sparse()])
    run = get_run(client, result, "r07-sparse")
    dense = metric_history(client, run.info.run_id, "dense")
    sparse = metric_history(client, run.info.run_id, "sparse")
    assert len(dense) == 100
    assert len(sparse) == 10
    assert sorted(p.step for p in sparse) == list(range(0, 100, 10))


def test_empty_history_still_produces_a_run(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_empty_history()])
    run = get_run(client, result, "r09-empty")
    assert run.data.params["lr"] == "0.01"
    assert run.data.metrics == {}
    assert run.info.status == "FINISHED"


def test_many_points_are_batched_and_all_arrive(client: MlflowClient) -> None:
    source = fixtures.run_many_steps(steps=2500)
    result, _ = migrate(client, [source])
    run = get_run(client, result, "r08-manysteps")
    assert len(metric_history(client, run.info.run_id, "m1")) == 2500
    assert run.data.metrics["m5"] == pytest.approx(2499 * 5)


def test_many_params_are_batched(client: MlflowClient) -> None:
    source = fixtures.run_empty_history()
    source.config = {f"p{i}": i for i in range(LIMITS.max_params_tags_per_batch * 3 + 7)}
    result, _ = migrate(client, [source])
    run = get_run(client, result, "r09-empty")
    assert len(run.data.params) == LIMITS.max_params_tags_per_batch * 3 + 7


# --------------------------------------------------------------------------- #
# key sanitisation, end to end
# --------------------------------------------------------------------------- #


def test_hostile_keys_are_sanitised_and_recorded(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_hostile_keys()])
    run = get_run(client, result, "r05-keys")
    assert "train/loss" in run.data.metrics  # legal already: must NOT be renamed
    assert "héllo" in run.data.metrics  # unicode is legal in MLflow
    assert "a b" in run.data.metrics
    assert "x_y_" in run.data.metrics
    renames = json.loads(run.data.tags["wandb.renamed_keys"])
    assert renames["x@y!"] == "x_y_"
    assert all(len(k) <= LIMITS.max_entity_key_length for k in run.data.metrics)


def test_colliding_keys_do_not_merge_into_one_series(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_key_collision()])
    run = get_run(client, result, "r06-collision")
    keys = [k for k in run.data.metrics if k.startswith("a_b")]
    assert len(keys) == 2
    values = {
        tuple(sorted(p.value for p in metric_history(client, run.info.run_id, k))) for k in keys
    }
    assert values == {(1.0, 2.0), (10.0, 20.0)}


# --------------------------------------------------------------------------- #
# sweeps
# --------------------------------------------------------------------------- #


def test_sweep_children_nest_under_one_synthetic_parent(client: MlflowClient) -> None:
    result, _ = migrate(client, fixtures.sweep_children())
    children = [get_run(client, result, f"r15-sweep-{i}") for i in range(3)]
    parents = {c.data.tags["mlflow.parentRunId"] for c in children}
    assert len(parents) == 1
    parent = client.get_run(parents.pop())
    assert parent.data.tags["wandb.is_sweep_parent"] == "true"
    assert parent.data.tags["wandb.sweep_id"] == "sw-abc123"
    for child in children:
        assert child.data.tags["wandb.sweep_id"] == "sw-abc123"


def test_non_sweep_runs_have_no_parent(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_bools()])
    assert "mlflow.parentRunId" not in get_run(client, result, "r04-bools").data.tags


# --------------------------------------------------------------------------- #
# the ambient-run trap (spec 8.2)
# --------------------------------------------------------------------------- #


def test_migration_inside_an_ambient_run_does_not_nest(tmp_path: Path) -> None:
    """`mlflow run` opens its own run. Using the fluent API here would adopt it."""
    uri = f"file://{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    ambient_experiment = client.create_experiment("ambient")
    with mlflow.start_run(experiment_id=ambient_experiment) as ambient:
        result, _ = migrate(client, [fixtures.run_bools(), *fixtures.sweep_children()])
    mlflow.end_run()

    for report in result.reports:
        run = client.get_run(report.mlflow_run_id)
        assert run.info.experiment_id != ambient_experiment
        assert run.data.tags.get("mlflow.parentRunId") != ambient.info.run_id
    non_sweep = get_run(client, result, "r04-bools")
    assert "mlflow.parentRunId" not in non_sweep.data.tags
    ambient_run = client.get_run(ambient.info.run_id)
    assert ambient_run.data.metrics == {}
    assert ambient_run.data.params == {}


def test_no_module_uses_the_mlflow_fluent_api() -> None:
    """Checked against the parsed AST, so a docstring mentioning it cannot pass.

    Spec rule 8.2: the fluent API would adopt `mlflow run`'s ambient run.
    """
    import ast

    import wandb_to_mlflow

    fluent = {
        "start_run",
        "end_run",
        "active_run",
        "log_metric",
        "log_metrics",
        "log_param",
        "log_params",
        "log_artifact",
        "log_artifacts",
        "set_tag",
        "set_tags",
        "set_experiment",
    }
    offenders: list[str] = []
    for path in sorted(Path(wandb_to_mlflow.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "mlflow"
                and node.attr in fluent
            ):
                offenders.append(f"{path.name}:{node.lineno} mlflow.{node.attr}")
            if isinstance(node, ast.ImportFrom) and node.module == "mlflow":
                offenders += [
                    f"{path.name}:{node.lineno} from mlflow import {alias.name}"
                    for alias in node.names
                    if alias.name in fluent
                ]
    assert offenders == []


# --------------------------------------------------------------------------- #
# failure isolation
# --------------------------------------------------------------------------- #


def test_one_broken_run_does_not_stop_the_others(client: MlflowClient) -> None:
    class Exploding(fixtures.FakeRun):
        def history(self) -> Any:
            raise RuntimeError("W&B said no")

    broken = Exploding(id="boom", name="boom")
    result, _ = migrate(client, [fixtures.run_bools(), broken, fixtures.run_metadata()])
    assert len(result.reports) == 3
    assert len(result.failures) == 1
    assert result.failures[0].wandb_run_id == "boom"
    assert "W&B said no" in (result.failures[0].error or "")
    assert not result.ok
    assert get_run(client, result, "r04-bools").info.status == "FINISHED"


# --------------------------------------------------------------------------- #
# dry run
# --------------------------------------------------------------------------- #


def test_dry_run_writes_nothing_but_still_reports(client: MlflowClient) -> None:
    result, _ = migrate(client, fixtures.all_runs(), dry_run=True)
    assert client.get_experiment_by_name("tests") is None
    assert all(report.skipped for report in result.reports)
    nested = next(r for r in result.reports if r.wandb_run_id == "r01-nested")
    assert nested.param_count > 0
    assert "loss" in nested.metric_keys
    assert nested.mlflow_run_id is None
    media = next(r for r in result.reports if r.wandb_run_id == "r10-media")
    assert media.dropped.as_dict()["media"] == 3


# --------------------------------------------------------------------------- #
# files and artifacts
# --------------------------------------------------------------------------- #


def test_files_and_artifacts_are_off_by_default(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_artifacts()])
    run = get_run(client, result, "r11-artifacts")
    assert client.list_artifacts(run.info.run_id) == []


def test_files_migrate_under_wandb_files(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_artifacts()], include_files=True)
    run = get_run(client, result, "r11-artifacts")
    names = [a.path for a in client.list_artifacts(run.info.run_id, "wandb_files")]
    assert "wandb_files/config.yaml" in names


def test_artifacts_migrate_with_a_metadata_sidecar(client: MlflowClient) -> None:
    result, _ = migrate(client, [fixtures.run_artifacts()], include_artifacts=True)
    run = get_run(client, result, "r11-artifacts")
    report = next(r for r in result.reports if r.wandb_run_id == "r11-artifacts")
    assert report.artifacts_migrated == 1
    paths = [a.path for a in client.list_artifacts(run.info.run_id, "artifacts/small-dataset_v0")]
    assert any(p.endswith("data.csv") for p in paths)
    assert any(p.endswith("_wandb_artifact.json") for p in paths)


def test_reference_artifacts_are_recorded_but_never_fetched(client: MlflowClient) -> None:
    """FakeArtifact.download raises for references, so a fetch would fail the test."""
    result, _ = migrate(client, [fixtures.run_artifacts()], include_artifacts=True)
    run = get_run(client, result, "r11-artifacts")
    report = next(r for r in result.reports if r.wandb_run_id == "r11-artifacts")
    assert report.reference_artifacts == ["remote-dataset:v0"]
    recorded = json.loads(run.data.tags["wandb.reference_artifacts"])
    assert recorded[0]["uris"] == ["s3://bucket/huge/"]


def test_max_artifact_size_is_honoured_without_downloading(client: MlflowClient) -> None:
    result, _ = migrate(
        client, [fixtures.run_artifacts()], include_artifacts=True, max_artifact_bytes=4
    )
    report = next(r for r in result.reports if r.wandb_run_id == "r11-artifacts")
    assert report.artifacts_migrated == 0
    assert report.artifacts_skipped == 1


def test_oversized_files_are_skipped(client: MlflowClient) -> None:
    result, _ = migrate(
        client, [fixtures.run_artifacts()], include_files=True, max_artifact_bytes=4
    )
    run = get_run(client, result, "r11-artifacts")
    assert client.list_artifacts(run.info.run_id) == []


# --------------------------------------------------------------------------- #
# system metrics
# --------------------------------------------------------------------------- #


def test_system_metrics_are_opt_in_and_prefixed(client: MlflowClient) -> None:
    source = fixtures.run_bools()
    source.system_rows = [
        {"_timestamp": fixtures.BASE_TS, "gpu.0.memory": 40.0},
        {"_timestamp": fixtures.BASE_TS + 5, "gpu.0.memory": 55.0},
    ]
    off, _ = migrate(client, [source])
    assert "system.gpu.0.memory" not in get_run(client, off, "r04-bools").data.metrics

    on, _ = migrate(client, [source], include_system_metrics=True)
    run = get_run(client, on, "r04-bools")
    assert len(metric_history(client, run.info.run_id, "system.gpu.0.memory")) == 2


def test_system_metrics_do_not_move_the_end_time(client: MlflowClient) -> None:
    """Only the data stream defines when the run stopped; events can outlive it."""
    source = fixtures.run_bools()
    source.system_rows = [{"_timestamp": fixtures.BASE_TS + 10_000, "gpu.0.util": 1.0}]
    result, _ = migrate(client, [source], include_system_metrics=True)
    run = get_run(client, result, "r04-bools")
    assert run.info.end_time == int((fixtures.BASE_TS + 2) * 1000)


def test_status_constants_match_mlflow(client: MlflowClient) -> None:
    from wandb_to_mlflow.migrate import DEFAULT_STATUS, STATE_TO_STATUS

    valid = set(RunStatus.all_status())
    for status in {*STATE_TO_STATUS.values(), DEFAULT_STATUS}:
        assert RunStatus.from_string(status) in valid


# --------------------------------------------------------------------------- #
# backend neutrality
# --------------------------------------------------------------------------- #


def test_the_same_migration_works_against_a_sqlite_backend(tmp_path: Path) -> None:
    """ "Vendor-neutral" has to mean more than "works against the file store".

    The SQL store validates more strictly than the file store -- key charsets and
    value lengths especially -- so this is where a sanitisation bug surfaces.
    """
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = client.create_experiment("sqlite", artifact_location=str(artifacts))
    migrator = Migrator(client, MigrateOptions(experiment="sqlite"))
    result = migrator.migrate_project(fixtures.FakeProject(run_list=fixtures.all_runs()))

    assert result.failures == []
    assert client.get_experiment(result.experiment_id).experiment_id == experiment_id
    keys = get_run(client, result, "r05-keys")
    assert "train/loss" in keys.data.metrics
    assert "héllo" in keys.data.metrics
    nested = get_run(client, result, "r01-nested")
    assert nested.data.params["optimizer.sched.warmup.steps"] == "100"
    assert len(metric_history(client, nested.info.run_id, "loss")) == 3
