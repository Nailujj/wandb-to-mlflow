"""The fixture set must cover every hostile case in the spec, and must actually
satisfy the protocols the migrator is written against."""

from __future__ import annotations

import math

from tests import fixtures
from wandb_to_mlflow.source import SourceArtifact, SourceFile, SourceProject, SourceRun


def test_every_fixture_satisfies_the_source_run_protocol() -> None:
    for run in fixtures.all_runs():
        assert isinstance(run, SourceRun)


def test_fake_project_satisfies_the_protocol() -> None:
    assert isinstance(fixtures.fake_project(), SourceProject)


def test_files_and_artifacts_satisfy_their_protocols() -> None:
    run = fixtures.run_artifacts()
    assert all(isinstance(f, SourceFile) for f in run.files())
    assert all(isinstance(a, SourceArtifact) for a in run.artifacts())


def test_run_ids_are_unique() -> None:
    ids = [run.id for run in fixtures.all_runs()]
    assert len(ids) == len(set(ids))


def test_all_sixteen_spec_cases_are_present() -> None:
    runs = {run.id: run for run in fixtures.all_runs()}
    # 1 nested config
    assert runs["r01-nested"].config["optimizer"]["sched"]["warmup"]["steps"] == 100
    # 2 long config value
    assert len(runs["r02-longparam"].config["blob"]) == 20_000
    # 3 non-finite metrics
    assert math.isnan(runs["r03-nonfinite"].rows[0]["ratio"])
    assert math.isinf(runs["r03-nonfinite"].rows[1]["loss"])
    # 4 bools in config and history
    assert runs["r04-bools"].config["use_amp"] is True
    assert runs["r04-bools"].rows[0]["improved"] is True
    # 5 hostile keys
    assert {"train/loss", "a b", "héllo", "x@y!"} <= set(runs["r05-keys"].rows[0])
    assert any(len(k) == 300 for k in runs["r05-keys"].rows[0])
    # 6 collision
    assert {"a@b", "a#b"} <= set(runs["r06-collision"].rows[0])
    # 7 sparse
    sparse_rows = [r for r in runs["r07-sparse"].rows if "sparse" in r]
    assert len(sparse_rows) == 10 and len(runs["r07-sparse"].rows) == 100
    # 8 throughput
    assert len(runs["r08-manysteps"].rows) == 200
    # 9 empty history
    assert runs["r09-empty"].rows == [] and runs["r09-empty"].config
    # 10 media
    assert runs["r10-media"].rows[0]["sample"]["_type"] == "image-file"
    assert runs["r10-media"].rows[0]["preds"]["_type"] == "table-file"
    # 11 artifacts
    arts = list(runs["r11-artifacts"].artifacts())
    assert [a.is_reference for a in arts] == [False, True]
    assert list(runs["r11-artifacts"].files())
    # 12 metadata
    meta = runs["r12-metadata"]
    assert meta.tags and meta.notes and meta.group and meta.job_type
    # 13 crashed and failed
    assert runs["r13-crashed"].state == "crashed" and runs["r13-failed"].state == "failed"
    # 14 duplicate display names
    assert runs["r14-dup-a"].name == runs["r14-dup-b"].name
    # 15 sweep children
    sweep = [r for r in runs.values() if r.sweep_id]
    assert len(sweep) == 3 and len({r.sweep_id for r in sweep}) == 1
    # 16 unicode
    assert "🎉" in (runs["r16-unicode"].name or "")


def test_full_scale_throughput_fixture_is_available() -> None:
    """The 20,000-step case exists; only the default set uses a smaller one."""
    assert len(fixtures.run_many_steps().rows) == 20_000


def test_fake_artifact_download_writes_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    artifact = next(iter(fixtures.run_artifacts().artifacts()))
    root = artifact.download(tmp_path / "a")
    assert (root / "data.csv").read_bytes() == b"a,b\n1,2\n"


def test_fake_file_download_writes_bytes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    file = next(iter(fixtures.run_artifacts().files()))
    assert file.download(tmp_path).read_bytes()
