"""Re-running a migration must be free and must create nothing.

The dangerous failure here is silent: a second run that quietly doubles every
run, or a resumed migration that keeps a half-written run because it "looked"
present. Both are tested by reading the store back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from tests import fixtures
from wandb_to_mlflow.migrate import MigrateOptions, Migrator
from wandb_to_mlflow.state import COMPLETE_TAG, RUN_ID_TAG, VERSION_TAG, MigrationState


@pytest.fixture
def client(tmp_path: Path) -> MlflowClient:
    return MlflowClient(tracking_uri=f"file://{tmp_path / 'mlruns'}")


def run_migration(client: MlflowClient, runs: list[fixtures.FakeRun], **options: Any) -> Any:
    migrator = Migrator(client, MigrateOptions(experiment="idem", **options))
    return migrator.migrate_project(fixtures.FakeProject(run_list=runs))


def active_runs(client: MlflowClient, experiment_id: str) -> list[Any]:
    return list(
        client.search_runs(
            experiment_ids=[experiment_id], run_view_type=ViewType.ACTIVE_ONLY, max_results=5000
        )
    )


def runs_for(client: MlflowClient, experiment_id: str, wandb_id: str) -> list[Any]:
    return [
        r for r in active_runs(client, experiment_id) if r.data.tags.get(RUN_ID_TAG) == wandb_id
    ]


# --------------------------------------------------------------------------- #
# re-running
# --------------------------------------------------------------------------- #


def test_second_run_creates_zero_duplicates(client: MlflowClient) -> None:
    source = fixtures.all_runs()
    first = run_migration(client, source)
    before = len(active_runs(client, first.experiment_id))

    second = run_migration(client, source)
    assert len(active_runs(client, second.experiment_id)) == before
    assert second.experiment_id == first.experiment_id
    assert all(report.skipped for report in second.reports)
    assert {r.skip_reason for r in second.reports} == {"already migrated"}


def test_second_run_reports_the_same_mlflow_run_ids(client: MlflowClient) -> None:
    source = [fixtures.run_nested_config(), fixtures.run_metadata()]
    first = {r.wandb_run_id: r.mlflow_run_id for r in run_migration(client, source).reports}
    second = {r.wandb_run_id: r.mlflow_run_id for r in run_migration(client, source).reports}
    assert first == second


def test_skipped_runs_are_not_re_read_from_the_source(client: MlflowClient) -> None:
    """Re-reading history from W&B is the expensive part; a skip must avoid it."""
    calls = {"history": 0}

    class Counting(fixtures.FakeRun):
        def history(self) -> Any:
            calls["history"] += 1
            return iter(self.rows)

    source = [Counting(id="counted", name="counted", rows=[{"_step": 0, "loss": 1.0}])]
    run_migration(client, source)
    assert calls["history"] == 1
    run_migration(client, source)
    assert calls["history"] == 1


def test_run_ids_are_the_key_not_run_names(client: MlflowClient) -> None:
    """Two W&B runs sharing a display name must stay two MLflow runs, twice over."""
    source = [fixtures.run_duplicate_name_a(), fixtures.run_duplicate_name_b()]
    first = run_migration(client, source)
    run_migration(client, source)
    assert len(active_runs(client, first.experiment_id)) == 2


def test_sweep_parents_are_not_duplicated_on_re_run(client: MlflowClient) -> None:
    source = fixtures.sweep_children()
    first = run_migration(client, source)
    run_migration(client, source)
    parents = [
        r
        for r in active_runs(client, first.experiment_id)
        if r.data.tags.get("wandb.is_sweep_parent") == "true"
    ]
    assert len(parents) == 1


def test_new_runs_are_added_without_touching_existing_ones(client: MlflowClient) -> None:
    first = run_migration(client, [fixtures.run_bools()])
    original_id = first.reports[0].mlflow_run_id
    second = run_migration(client, [fixtures.run_bools(), fixtures.run_metadata()])
    assert len(active_runs(client, second.experiment_id)) == 2
    assert runs_for(client, second.experiment_id, "r04-bools")[0].info.run_id == original_id
    added = next(r for r in second.reports if r.wandb_run_id == "r12-metadata")
    assert not added.skipped


# --------------------------------------------------------------------------- #
# completion marker
# --------------------------------------------------------------------------- #


def test_completed_runs_carry_the_marker_and_version(client: MlflowClient) -> None:
    result = run_migration(client, [fixtures.run_bools()])
    tags = client.get_run(result.reports[0].mlflow_run_id).data.tags
    assert tags[COMPLETE_TAG] == "true"
    assert tags[VERSION_TAG]


def test_a_failed_run_is_not_marked_complete(client: MlflowClient) -> None:
    class Exploding(fixtures.FakeRun):
        def artifacts(self) -> Any:
            raise RuntimeError("died mid-write")

    result = run_migration(
        client,
        [Exploding(id="half", name="half", rows=[{"_step": 0, "loss": 1.0}])],
        include_artifacts=True,
    )
    assert result.failures
    leftovers = runs_for(client, result.experiment_id, "half")
    assert len(leftovers) == 1
    assert COMPLETE_TAG not in leftovers[0].data.tags


# --------------------------------------------------------------------------- #
# resume after interruption
# --------------------------------------------------------------------------- #


def test_interrupted_run_is_replaced_not_kept(client: MlflowClient) -> None:
    """Simulates a kill between create_run and the completion marker."""

    class KilledError(RuntimeError):
        pass

    class Exploding(fixtures.FakeRun):
        def artifacts(self) -> Any:
            raise KilledError("killed mid-migration")

    partial = Exploding(id="r04-bools", name="bool-trap", rows=[{"_step": 0, "loss": 99.0}])
    interrupted = run_migration(client, [partial], include_artifacts=True)
    assert interrupted.failures
    half_written = runs_for(client, interrupted.experiment_id, "r04-bools")[0]

    resumed = run_migration(client, [fixtures.run_bools()])
    assert resumed.failures == []
    survivors = runs_for(client, resumed.experiment_id, "r04-bools")
    assert len(survivors) == 1
    assert survivors[0].info.run_id != half_written.info.run_id
    assert survivors[0].data.tags[COMPLETE_TAG] == "true"
    loss = client.get_metric_history(survivors[0].info.run_id, "loss")
    assert [p.value for p in loss] == [1.0, 0.5, 0.4]  # the good series, not the 99.0 stub
    assert resumed.reports[0].skip_reason == "resumed"


def test_resume_completes_the_runs_that_never_started(client: MlflowClient) -> None:
    source = fixtures.all_runs()
    run_migration(client, source[:5])
    result = run_migration(client, source)
    assert result.failures == []
    assert len(active_runs(client, result.experiment_id)) >= len(source)
    assert len([r for r in result.reports if r.skipped]) == 5


def test_a_stale_mapping_version_forces_re_migration(client: MlflowClient) -> None:
    result = run_migration(client, [fixtures.run_bools()])
    stale_id = result.reports[0].mlflow_run_id
    client.set_tag(stale_id, VERSION_TAG, "0")

    again = run_migration(client, [fixtures.run_bools()])
    assert again.reports[0].mlflow_run_id != stale_id
    assert len(runs_for(client, again.experiment_id, "r04-bools")) == 1


# --------------------------------------------------------------------------- #
# --overwrite
# --------------------------------------------------------------------------- #


def test_overwrite_replaces_rather_than_duplicating(client: MlflowClient) -> None:
    original = fixtures.run_bools()
    first = run_migration(client, [original])
    original_id = first.reports[0].mlflow_run_id

    changed = fixtures.run_bools()
    changed.rows = [{"_step": 0, "_timestamp": fixtures.BASE_TS, "loss": 0.001}]
    second = run_migration(client, [changed], overwrite=True)

    survivors = runs_for(client, second.experiment_id, "r04-bools")
    assert len(survivors) == 1
    assert survivors[0].info.run_id != original_id
    assert survivors[0].data.metrics["loss"] == 0.001
    assert second.reports[0].skip_reason == "overwritten"


def test_overwrite_soft_deletes_so_nothing_is_actually_lost(client: MlflowClient) -> None:
    first = run_migration(client, [fixtures.run_bools()])
    original_id = first.reports[0].mlflow_run_id
    run_migration(client, [fixtures.run_bools()], overwrite=True)
    assert client.get_run(original_id).info.lifecycle_stage == "deleted"


# --------------------------------------------------------------------------- #
# the state index itself
# --------------------------------------------------------------------------- #


def test_state_index_is_built_with_one_search_not_one_per_run(client: MlflowClient) -> None:
    result = run_migration(client, fixtures.all_runs())
    searches = {"count": 0}
    original = client.search_runs

    def counting(*args: Any, **kwargs: Any) -> Any:
        searches["count"] += 1
        return original(*args, **kwargs)

    client.search_runs = counting  # type: ignore[method-assign]
    state = MigrationState(client, result.experiment_id)
    state.load()
    assert searches["count"] == 1
    assert len(state) == len(fixtures.all_runs())


def test_state_ignores_runs_that_are_not_ours(client: MlflowClient) -> None:
    result = run_migration(client, [fixtures.run_bools()])
    client.create_run(experiment_id=result.experiment_id, run_name="someone-elses-run")
    state = MigrationState(client, result.experiment_id)
    state.load()
    assert len(state) == 1
    assert state.lookup("r04-bools") is not None
