"""The CLI must never write during `plan`, must exit non-zero on failure, and
must own all output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from tests import fixtures
from wandb_to_mlflow import cli
from wandb_to_mlflow.migrate import MigrateOptions, Migrator
from wandb_to_mlflow.verify import manifest_from_source

runner = CliRunner()


@pytest.fixture(autouse=True)
def offline_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """`WandbProject.connect` would open a socket; hand back fixtures instead."""

    def connect(entity: str, project: str, filters: Any = None) -> fixtures.FakeProject:
        return fixtures.FakeProject(entity=entity, project=project, run_list=fixtures.all_runs())

    monkeypatch.setattr(cli.WandbProject, "connect", staticmethod(connect))


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    return f"file://{tmp_path / 'mlruns'}"


def invoke(*args: str) -> Any:
    return runner.invoke(cli.app, list(args))


# --------------------------------------------------------------------------- #
# option parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("word", ["true", "TRUE", "1", "yes", "on", "t"])
def test_truthy_words(word: str) -> None:
    assert cli.parse_bool(word, "--artifacts") is True


@pytest.mark.parametrize("word", ["false", "FALSE", "0", "no", "off", ""])
def test_falsy_words(word: str) -> None:
    assert cli.parse_bool(word, "--artifacts") is False


def test_a_nonsense_boolean_is_rejected_not_guessed() -> None:
    with pytest.raises(Exception, match="expects true or false"):
        cli.parse_bool("maybe", "--artifacts")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("512", 512), ("20MB", 20 * 1024**2), ("1.5GiB", int(1.5 * 1024**3)), ("2kb", 2048)],
)
def test_size_parsing(text: str, expected: int) -> None:
    assert cli.parse_size(text) == expected


def test_a_nonsense_size_is_rejected() -> None:
    with pytest.raises(Exception, match="unparseable size"):
        cli.parse_size("a lot")


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #


def test_plan_writes_nothing(tracking_uri: str) -> None:
    result = invoke("plan", "-e", "acme", "-p", "demo", "--tracking-uri", tracking_uri)
    assert result.exit_code == 0, result.output
    client = MlflowClient(tracking_uri=tracking_uri)
    # MLflow's own store creates "Default" on init; what matters is that the
    # plan created neither the target experiment nor any run.
    assert client.get_experiment_by_name("demo") is None
    assert [e.name for e in client.search_experiments()] == ["Default"]
    assert client.search_runs(["0"], max_results=10) == []


def test_plan_reports_counts_and_warnings(tracking_uri: str) -> None:
    result = invoke("plan", "-e", "acme", "-p", "demo", "--tracking-uri", tracking_uri)
    assert "Runs to migrate:" in result.output
    assert "Metric points:" in result.output
    assert "Will NOT be migrated" in result.output
    assert "media image-file" in result.output
    assert "nonfinite" in result.output
    assert "Keys renamed for MLflow:" in result.output
    assert "Sweeps (become parents):  1" in result.output
    assert "Nothing was written. This was a plan." in result.output


def test_plan_names_the_flags_that_would_include_more(tracking_uri: str) -> None:
    result = invoke("plan", "-e", "acme", "-p", "demo", "--tracking-uri", tracking_uri)
    assert "--artifacts true" in result.output
    assert "--files true" in result.output


def test_plan_json_is_machine_readable(tracking_uri: str) -> None:
    result = invoke("plan", "-e", "acme", "-p", "demo", "--tracking-uri", tracking_uri, "--json")
    payload = json.loads(result.output)
    assert len(payload["plan"]) == len(fixtures.all_runs())
    assert all(entry["skipped"] for entry in payload["plan"])


def test_plan_uses_the_target_experiment_name(tracking_uri: str) -> None:
    result = invoke(
        "plan",
        "-e",
        "acme",
        "-p",
        "demo",
        "--experiment",
        "chosen",
        "--tracking-uri",
        tracking_uri,
    )
    assert "Target experiment: chosen" in result.output


# --------------------------------------------------------------------------- #
# migrate
# --------------------------------------------------------------------------- #


def test_migrate_writes_and_reports(tracking_uri: str) -> None:
    result = invoke(
        "migrate", "-e", "acme", "-p", "demo", "--experiment", "cli", "--tracking-uri", tracking_uri
    )
    assert result.exit_code == 0, result.output
    assert f"Migrated:   {len(fixtures.all_runs())}" in result.output
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("cli")
    assert experiment is not None
    assert len(client.search_runs([experiment.experiment_id], max_results=100)) > len(
        fixtures.all_runs()
    )  # plus the synthetic sweep parent


def test_migrate_surfaces_dropped_values_and_references(tracking_uri: str) -> None:
    result = invoke(
        "migrate",
        "-e",
        "acme",
        "-p",
        "demo",
        "--experiment",
        "cli",
        "--artifacts",
        "true",
        "--tracking-uri",
        tracking_uri,
    )
    assert "Values dropped (documented in MAPPING.md)" in result.output
    assert "Reference artifacts recorded but NOT fetched" in result.output
    assert "remote-dataset:v0" in result.output


def test_migrate_is_idempotent_from_the_command_line(tracking_uri: str) -> None:
    args = (
        "migrate",
        "-e",
        "acme",
        "-p",
        "demo",
        "--experiment",
        "cli",
        "--tracking-uri",
        tracking_uri,
    )
    invoke(*args)
    second = invoke(*args)
    assert f"Skipped:    {len(fixtures.all_runs())}" in second.output
    assert "Migrated:   0" in second.output


def test_migrate_exits_non_zero_when_a_run_fails(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Exploding(fixtures.FakeRun):
        def history(self) -> Any:
            raise RuntimeError("W&B said no")

    monkeypatch.setattr(
        cli.WandbProject,
        "connect",
        staticmethod(
            lambda entity, project, filters=None: fixtures.FakeProject(
                run_list=[fixtures.run_bools(), Exploding(id="boom")]
            )
        ),
    )
    result = invoke(
        "migrate", "-e", "acme", "-p", "demo", "--experiment", "cli", "--tracking-uri", tracking_uri
    )
    assert result.exit_code == 1
    assert "FAILURES:" in result.output
    assert "boom" in result.output
    assert "Migrated:   1" in result.output  # the good run still went through


def test_a_bad_boolean_option_is_rejected(tracking_uri: str) -> None:
    result = invoke(
        "migrate",
        "-e",
        "acme",
        "-p",
        "demo",
        "--artifacts",
        "perhaps",
        "--tracking-uri",
        tracking_uri,
    )
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #


@pytest.fixture
def seeded(tmp_path: Path, tracking_uri: str) -> Path:
    project = fixtures.fake_project()
    manifest = manifest_from_source(project, MigrateOptions(experiment="cli"))
    path = tmp_path / "manifest.json"
    manifest.dump(path)
    Migrator(
        MlflowClient(tracking_uri=tracking_uri), MigrateOptions(experiment="cli")
    ).migrate_project(project)
    return path


def test_verify_against_a_manifest_passes(seeded: Path, tracking_uri: str) -> None:
    result = invoke(
        "verify", "--manifest", str(seeded), "--experiment", "cli", "--tracking-uri", tracking_uri
    )
    assert result.exit_code == 0, result.output
    assert "No unexpected loss" in result.output


def test_verify_exits_non_zero_on_a_real_mismatch(seeded: Path, tracking_uri: str) -> None:
    data = json.loads(seeded.read_text())
    entry = next(r for r in data["runs"] if r["wandb_run_id"] == "r01-nested")
    entry["expected_metric_keys"].append("ghost")
    seeded.write_text(json.dumps(data))
    result = invoke(
        "verify", "--manifest", str(seeded), "--experiment", "cli", "--tracking-uri", tracking_uri
    )
    assert result.exit_code == 1
    assert "MISMATCHES" in result.output
    assert "ghost" in result.output


def test_verify_live_mode_needs_entity_and_project(tracking_uri: str) -> None:
    result = invoke("verify", "--experiment", "cli", "--tracking-uri", tracking_uri)
    assert result.exit_code != 0
    assert "--manifest" in result.output


def test_verify_live_mode_works(tracking_uri: str) -> None:
    invoke(
        "migrate", "-e", "acme", "-p", "demo", "--experiment", "cli", "--tracking-uri", tracking_uri
    )
    result = invoke(
        "verify", "-e", "acme", "-p", "demo", "--experiment", "cli", "--tracking-uri", tracking_uri
    )
    assert result.exit_code == 0, result.output


def test_verify_live_mode_needs_the_same_opt_in_flags_as_the_migration(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A correct `--system-metrics` migration must not verify as broken.

    Live verification re-plans the migration to derive expectations. Planning
    without the flag the migration ran with makes every correctly-migrated
    `system.*` series look like a metric that should not be there.
    """

    def connect(entity: str, project: str, filters: Any = None) -> fixtures.FakeProject:
        run = fixtures.run_bools()
        run.system_rows = [
            {"_timestamp": fixtures.BASE_TS, "system.cpu": 12.5},
            {"_timestamp": fixtures.BASE_TS + 5, "system.cpu": 30.0},
        ]
        return fixtures.FakeProject(entity=entity, project=project, run_list=[run])

    monkeypatch.setattr(cli.WandbProject, "connect", staticmethod(connect))

    migrated = invoke(
        "migrate",
        "-e",
        "acme",
        "-p",
        "demo",
        "--experiment",
        "sysm",
        "--system-metrics",
        "true",
        "--tracking-uri",
        tracking_uri,
    )
    assert migrated.exit_code == 0, migrated.output

    matched = invoke(
        "verify",
        "-e",
        "acme",
        "-p",
        "demo",
        "--experiment",
        "sysm",
        "--system-metrics",
        "true",
        "--tracking-uri",
        tracking_uri,
    )
    assert matched.exit_code == 0, matched.output

    # And the mismatch is still reported when the two genuinely disagree.
    mismatched = invoke(
        "verify",
        "-e",
        "acme",
        "-p",
        "demo",
        "--experiment",
        "sysm",
        "--tracking-uri",
        tracking_uri,
    )
    assert mismatched.exit_code != 0
    assert "system.cpu" in mismatched.output


def test_verify_json_output(seeded: Path, tracking_uri: str) -> None:
    result = invoke(
        "verify",
        "--manifest",
        str(seeded),
        "--experiment",
        "cli",
        "--tracking-uri",
        tracking_uri,
        "--json",
    )
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["failures"] == []
    assert payload["expected_losses"]


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #


def test_version_command() -> None:
    from wandb_to_mlflow import __version__

    assert invoke("version").output.strip() == __version__


def test_bare_invocation_shows_help() -> None:
    result = invoke()
    assert "migrate" in result.output and "verify" in result.output and "plan" in result.output


def test_module_entry_point_exists() -> None:
    """MLproject entry points call `python -m wandb_to_mlflow`."""
    import wandb_to_mlflow.__main__ as module

    assert module.app is cli.app


def test_library_code_never_prints() -> None:
    """Spec rule 10: only the CLI writes to stdout."""
    import ast

    import wandb_to_mlflow

    offenders: list[str] = []
    for path in sorted(Path(wandb_to_mlflow.__file__).parent.glob("*.py")):
        if path.name in {"cli.py", "__main__.py"}:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


# --------------------------------------------------------------------------- #
# seed safety (spec 6.1)
# --------------------------------------------------------------------------- #


def test_seed_prints_the_plan_and_stops_without_confirmation() -> None:
    result = runner.invoke(cli.app, ["seed", "-e", "acme"], input="n\n")
    assert result.exit_code == 1
    assert "This will CREATE" in result.output
    assert "Nothing was created." in result.output


def test_seed_default_project_name_is_timestamped() -> None:
    result = runner.invoke(cli.app, ["seed", "-e", "acme"], input="n\n")
    assert "project: w2m-selftest-" in result.output


def test_seed_refuses_to_clean_up_a_project_it_did_not_create() -> None:
    result = invoke("seed", "-e", "acme", "--cleanup", "production-experiments", "--yes")
    assert result.exit_code != 0
    assert "Refusing to delete" in result.output


def test_seed_cleanup_asks_before_deleting(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}
    monkeypatch.setattr(
        cli.seed_module, "cleanup", lambda entity, project: called.__setitem__("n", 1)
    )
    result = runner.invoke(
        cli.app, ["seed", "-e", "acme", "--cleanup", "w2m-selftest-x"], input="n\n"
    )
    assert result.exit_code == 1
    assert called["n"] == 0
    assert "Nothing was deleted." in result.output


def test_seed_cleanup_says_the_project_shell_remains(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.seed_module, "cleanup", lambda entity, project: 7)
    result = invoke("seed", "-e", "acme", "--cleanup", "w2m-selftest-x", "--yes")
    assert result.exit_code == 0
    assert "Deleted 7 runs." in result.output
    assert "no project delete" in result.output


def test_demo_stops_without_confirmation() -> None:
    result = runner.invoke(cli.app, ["demo", "-e", "acme"], input="n\n")
    assert result.exit_code == 1
    assert "Nothing was created." in result.output


def test_demo_runs_the_whole_loop(
    tracking_uri: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seed is faked; migrate and verify are real."""
    from tests.test_seed import fake_run_for
    from wandb_to_mlflow.seed import build_specs, manifest_for

    specs = [s for s in build_specs() if s.key != "throughput"]
    run_ids = {spec.key: f"demo-{spec.key}" for spec in specs}
    runs = [fake_run_for(spec, run_ids[spec.key]) for spec in specs]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli.seed_module,
        "seed",
        lambda entity, project, manifest_path=None, steps=None: (
            project,
            manifest_for(specs, entity, project, run_ids, "sw-seeded"),
        ),
    )
    monkeypatch.setattr(
        cli.WandbProject,
        "connect",
        staticmethod(lambda entity, project, filters=None: fixtures.FakeProject(run_list=runs)),
    )

    result = invoke(
        "demo",
        "-e",
        "acme",
        "-p",
        "w2m-selftest-demo",
        "--experiment",
        "demo",
        "--tracking-uri",
        tracking_uri,
        "--yes",
    )
    assert result.exit_code == 0, result.output
    assert "[1/3] seeding" in result.output
    assert "[2/3] migrating" in result.output
    assert "[3/3] verifying" in result.output
    assert "No unexpected loss" in result.output
    assert "mlflow ui" in result.output
    assert "seed --cleanup w2m-selftest-demo" in result.output


def test_plan_after_a_migration_still_reports_the_full_picture(tracking_uri: str) -> None:
    """`plan` must not report an already-migrated project as empty work."""
    invoke(
        "migrate",
        "-e",
        "acme",
        "-p",
        "demo",
        "--experiment",
        "demo",
        "--tracking-uri",
        tracking_uri,
    )
    result = invoke("plan", "-e", "acme", "-p", "demo", "--tracking-uri", tracking_uri)
    assert result.exit_code == 0, result.output
    assert "Runs to migrate:   20" in result.output
    assert "Metric points:     0" not in result.output
    assert "Already in MLflow (would be skipped): 20 of 20 runs" in result.output


def test_live_verify_passes_against_a_project_it_just_migrated(tracking_uri: str) -> None:
    args = ("-e", "acme", "-p", "demo", "--experiment", "demo", "--tracking-uri", tracking_uri)
    assert invoke("migrate", *args).exit_code == 0
    result = invoke("verify", *args)
    assert result.exit_code == 0, result.output
    assert "No unexpected loss" in result.output


def test_migrate_points_at_the_flags_that_would_keep_media(tracking_uri: str) -> None:
    """Media cannot become a metric, but its bytes need not be lost too."""
    result = invoke(
        "migrate", "-e", "acme", "-p", "demo", "--experiment", "m1", "--tracking-uri", tracking_uri
    )
    assert "Media was dropped as metrics. To keep more of it:" in result.output
    assert "--files true" in result.output and "wandb_files/media/" in result.output
    assert "per-run history parquet" in result.output


def test_the_hint_disappears_once_both_flags_are_on(tracking_uri: str) -> None:
    result = invoke(
        "migrate",
        "-e",
        "acme",
        "-p",
        "demo",
        "--experiment",
        "m2",
        "--files",
        "true",
        "--artifacts",
        "true",
        "--tracking-uri",
        tracking_uri,
    )
    assert "To keep more of it:" not in result.output


def test_no_media_hint_when_nothing_media_was_dropped(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli.WandbProject,
        "connect",
        staticmethod(
            lambda entity, project, filters=None: fixtures.FakeProject(
                run_list=[fixtures.run_nested_config()]
            )
        ),
    )
    result = invoke(
        "migrate", "-e", "acme", "-p", "demo", "--experiment", "m3", "--tracking-uri", tracking_uri
    )
    assert "To keep more of it:" not in result.output
