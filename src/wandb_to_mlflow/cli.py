"""The command line interface.

The CLI owns all output. Library modules log; only this module prints. That
split is what lets the migrator be embedded in something else without spraying
text at whoever imported it.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from mlflow.tracking import MlflowClient

from wandb_to_mlflow import __version__
from wandb_to_mlflow import seed as seed_module
from wandb_to_mlflow.migrate import MigrateOptions, MigrationResult, Migrator
from wandb_to_mlflow.source import SourceProject, WandbProject
from wandb_to_mlflow.verify import Manifest, Verifier, format_report, manifest_from_source

app = typer.Typer(
    name="wandb-to-mlflow",
    help="Migrate Weights & Biases runs into any MLflow tracking server.",
    no_args_is_help=True,
    add_completion=False,
)

logger = logging.getLogger("wandb_to_mlflow")

TRUE_WORDS = frozenset({"1", "true", "t", "yes", "y", "on"})
FALSE_WORDS = frozenset({"0", "false", "f", "no", "n", "off", ""})


def parse_bool(value: str, name: str) -> bool:
    """Parse a string-valued boolean option.

    These options take a value (``--artifacts true``) rather than being bare
    flags because ``MLproject`` entry points substitute parameters positionally
    into the command line and cannot emit or omit a flag conditionally.
    """
    text = str(value).strip().lower()
    if text in TRUE_WORDS:
        return True
    if text in FALSE_WORDS:
        return False
    raise typer.BadParameter(f"{name} expects true or false, got {value!r}")


def parse_size(value: str) -> int:
    """Parse a byte size, accepting ``512``, ``20MB``, ``1.5GiB``."""
    text = str(value).strip().upper().replace("IB", "B")
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    for suffix in ("TB", "GB", "MB", "KB", "B"):
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            try:
                return int(float(number or "0") * units[suffix])
            except ValueError:
                raise typer.BadParameter(f"unparseable size {value!r}") from None
    try:
        return int(float(text))
    except ValueError:
        raise typer.BadParameter(f"unparseable size {value!r}") from None


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("wandb").setLevel(logging.WARNING)


def echo(text: str) -> None:
    typer.echo(text)


def emit_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


# --------------------------------------------------------------------------- #
# Shared option types
# --------------------------------------------------------------------------- #

EntityOption = Annotated[str, typer.Option("--entity", "-e", help="W&B entity (user or team).")]
ProjectOption = Annotated[str, typer.Option("--project", "-p", help="W&B project name.")]
ExperimentOption = Annotated[
    str,
    typer.Option("--experiment", help="Target MLflow experiment. Defaults to the W&B project."),
]
TrackingUriOption = Annotated[
    str,
    typer.Option("--tracking-uri", help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI."),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")]
VerboseOption = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]


def build_client(tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri or None)


def build_options(
    experiment: str,
    artifacts: str,
    files: str,
    system_metrics: str,
    max_artifact_size: str,
    overwrite: bool,
    dry_run: bool,
    workers: int = 1,
) -> MigrateOptions:
    return MigrateOptions(
        experiment=experiment or None,
        include_artifacts=parse_bool(artifacts, "--artifacts"),
        include_files=parse_bool(files, "--files"),
        include_system_metrics=parse_bool(system_metrics, "--system-metrics"),
        max_artifact_bytes=parse_size(max_artifact_size),
        overwrite=overwrite,
        dry_run=dry_run,
        workers=workers,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def format_plan(result: MigrationResult, options: MigrateOptions) -> str:
    lines: list[str] = []
    reports = [r for r in result.reports if r.error is None]
    lines.append(f"Target experiment: {result.experiment_name}")
    lines.append(f"Runs to migrate:   {len(reports)}")
    lines.append(f"Params:            {sum(r.param_count for r in reports)}")
    lines.append(f"Metric series:     {sum(len(r.metric_keys) for r in reports)}")
    lines.append(f"Metric points:     {sum(sum(r.metric_point_counts.values()) for r in reports)}")

    dropped: dict[str, int] = {}
    media: dict[str, int] = {}
    for report in reports:
        for reason, count in report.dropped.counts.items():
            dropped[reason] = dropped.get(reason, 0) + count
        for media_type, count in report.dropped.media.items():
            media[media_type] = media.get(media_type, 0) + count

    lines.append("")
    if dropped:
        lines.append("Will NOT be migrated (see MAPPING.md):")
        for reason, count in sorted(dropped.items()):
            lines.append(f"  {reason:<12} {count}")
        for media_type, count in sorted(media.items()):
            lines.append(f"    media {media_type:<20} {count}")
    else:
        lines.append("Nothing will be dropped: every logged value is a finite scalar.")

    renamed = sum(len(r.renamed_keys) for r in reports)
    truncated = sum(len(r.truncated_params) for r in reports)
    sweeps = {r.wandb_sweep_id for r in reports if r.wandb_sweep_id}
    lines.append("")
    lines.append(f"Keys renamed for MLflow:  {renamed}")
    lines.append(f"Param values truncated:   {truncated}")
    lines.append(f"Sweeps (become parents):  {len(sweeps)}")
    if not options.include_artifacts:
        lines.append("Artifacts:                skipped (pass --artifacts true)")
    if not options.include_files:
        lines.append("Run files:                skipped (pass --files true)")
    if not options.include_system_metrics:
        lines.append("System metrics:           skipped (pass --system-metrics true)")

    failures = result.failures
    if failures:
        lines.append("")
        lines.append(f"{len(failures)} run(s) could not even be read:")
        lines.extend(f"  {r.wandb_run_id}: {r.error}" for r in failures)
    lines.append("")
    lines.append("Nothing was written. This was a plan.")
    return "\n".join(lines)


def format_migration(result: MigrationResult) -> str:
    lines: list[str] = []
    lines.append(f"Experiment: {result.experiment_name} (id {result.experiment_id})")
    lines.append(f"Migrated:   {len(result.migrated)}")
    lines.append(f"Skipped:    {len(result.skipped)}")
    lines.append(f"Failed:     {len(result.failures)}")

    dropped_runs = [r for r in result.migrated if r.dropped.total]
    if dropped_runs:
        lines.append("")
        lines.append("Values dropped (documented in MAPPING.md):")
        for report in dropped_runs:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(report.dropped.counts.items()))
            lines.append(f"  {report.wandb_run_id:<24} {summary}")

    references = [r for r in result.migrated if r.reference_artifacts]
    if references:
        lines.append("")
        lines.append("Reference artifacts recorded but NOT fetched:")
        for report in references:
            lines.append(f"  {report.wandb_run_id:<24} {', '.join(report.reference_artifacts)}")

    if result.failures:
        lines.append("")
        lines.append("FAILURES:")
        for report in result.failures:
            lines.append(f"  {report.wandb_run_id:<24} {report.error}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


@app.command()
def plan(
    entity: EntityOption,
    project: ProjectOption,
    experiment: ExperimentOption = "",
    artifacts: Annotated[str, typer.Option("--artifacts")] = "false",
    files: Annotated[str, typer.Option("--files")] = "false",
    system_metrics: Annotated[str, typer.Option("--system-metrics")] = "false",
    max_artifact_size: Annotated[str, typer.Option("--max-artifact-size")] = "100MB",
    tracking_uri: TrackingUriOption = "",
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Report what a migration would do. Writes nothing."""
    setup_logging(verbose)
    options = build_options(
        experiment, artifacts, files, system_metrics, max_artifact_size, False, dry_run=True
    )
    source = WandbProject.connect(entity, project)
    result = _run_plan(source, options, tracking_uri)
    if as_json:
        emit_json({"plan": [r.as_dict() for r in result.reports]})
    else:
        echo(format_plan(result, options))
    raise typer.Exit(1 if result.failures else 0)


def _run_plan(source: SourceProject, options: MigrateOptions, tracking_uri: str) -> MigrationResult:
    migrator = Migrator(build_client(tracking_uri), options)
    return migrator.migrate_project(source)


@app.command()
def migrate(
    entity: EntityOption,
    project: ProjectOption,
    experiment: ExperimentOption = "",
    artifacts: Annotated[str, typer.Option("--artifacts")] = "false",
    files: Annotated[str, typer.Option("--files")] = "false",
    system_metrics: Annotated[str, typer.Option("--system-metrics")] = "false",
    max_artifact_size: Annotated[str, typer.Option("--max-artifact-size")] = "100MB",
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace existing runs.")] = False,
    workers: Annotated[int, typer.Option("--workers", help="Runs to migrate in parallel.")] = 1,
    tracking_uri: TrackingUriOption = "",
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Migrate a W&B project into an MLflow experiment."""
    setup_logging(verbose)
    options = build_options(
        experiment,
        artifacts,
        files,
        system_metrics,
        max_artifact_size,
        overwrite,
        dry_run=False,
        workers=workers,
    )
    source = WandbProject.connect(entity, project)
    migrator = Migrator(build_client(tracking_uri), options)
    result = migrator.migrate_project(source)
    if as_json:
        emit_json(
            {"experiment_id": result.experiment_id, "runs": [r.as_dict() for r in result.reports]}
        )
    else:
        echo(format_migration(result))
    raise typer.Exit(1 if result.failures else 0)


@app.command()
def verify(
    experiment: Annotated[str, typer.Option("--experiment", help="MLflow experiment to check.")],
    manifest: Annotated[
        Path | None, typer.Option("--manifest", help="Ground truth from `seed`.")
    ] = None,
    entity: Annotated[str, typer.Option("--entity", "-e")] = "",
    project: Annotated[str, typer.Option("--project", "-p")] = "",
    tracking_uri: TrackingUriOption = "",
    as_json: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Check a migration, either against a seeded manifest or against live W&B."""
    setup_logging(verbose)
    if manifest is None and not (entity and project):
        raise typer.BadParameter("pass either --manifest, or both --entity and --project")
    if manifest is not None:
        loaded = Manifest.load(manifest)
    else:
        loaded = manifest_from_source(
            WandbProject.connect(entity, project), MigrateOptions(experiment=experiment)
        )
    report = Verifier(build_client(tracking_uri)).verify(loaded, experiment)
    if as_json:
        emit_json(report.as_dict())
    else:
        echo(format_report(report))
    raise typer.Exit(0 if report.ok else 1)


@app.command(name="seed")
def seed_command(
    entity: EntityOption,
    project: Annotated[str, typer.Option("--project", "-p", help="Project name to create.")] = "",
    manifest: Annotated[
        Path, typer.Option("--manifest", help="Where to write ground truth.")
    ] = Path("manifest.json"),
    steps: Annotated[int, typer.Option("--steps", help="Rows for the throughput run.")] = 0,
    cleanup: Annotated[str, typer.Option("--cleanup", help="Delete a seeded project's runs.")] = "",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not prompt.")] = False,
    verbose: VerboseOption = False,
) -> None:
    """Create a real W&B project of hostile runs, and the manifest describing them."""
    setup_logging(verbose)
    if cleanup:
        _cleanup(entity, cleanup, yes)
        return

    project_name = project or seed_module.default_project_name()
    specs = seed_module.build_specs()
    echo(seed_module.plan_text(specs, entity, project_name))
    echo("")
    if not yes and not typer.confirm("Create this project?", default=False):
        echo("Nothing was created.")
        raise typer.Exit(1)

    created, built = seed_module.seed(
        entity, project_name, manifest_path=manifest, steps=steps or None
    )
    echo("")
    echo(f"Seeded {len(built.runs)} runs into {entity}/{created}")
    echo(f"Manifest written to {manifest}")


def _cleanup(entity: str, project: str, yes: bool) -> None:
    if not seed_module.is_seeded_project(project):
        raise typer.BadParameter(
            f"{project!r} does not look like a seeded project "
            f"(expected a name starting {seed_module.PROJECT_PREFIX!r}). "
            "Refusing to delete runs from a project this tool did not create."
        )
    echo(f"This will DELETE every run in {entity}/{project}.")
    if not yes and not typer.confirm("Delete them?", default=False):
        echo("Nothing was deleted.")
        raise typer.Exit(1)
    deleted = seed_module.cleanup(entity, project)
    echo(f"Deleted {deleted} runs.")
    echo(
        "The empty project itself remains: W&B's public API has no project delete. "
        "Remove it from the web UI if you want it gone."
    )


@app.command()
def demo(
    entity: EntityOption,
    project: Annotated[str, typer.Option("--project", "-p")] = "",
    experiment: ExperimentOption = "",
    steps: Annotated[int, typer.Option("--steps")] = 2000,
    tracking_uri: TrackingUriOption = "",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not prompt.")] = False,
    keep: Annotated[bool, typer.Option("--keep", help="Do not delete the seeded project.")] = False,
    verbose: VerboseOption = False,
) -> None:
    """Seed, migrate, verify, and print the URL to look at the result."""
    setup_logging(verbose)
    project_name = project or seed_module.default_project_name()
    specs = seed_module.build_specs()
    echo(seed_module.plan_text(specs, entity, project_name))
    echo("")
    if not yes and not typer.confirm("Run the demo?", default=False):
        echo("Nothing was created.")
        raise typer.Exit(1)

    echo("[1/3] seeding W&B ...")
    manifest_path = Path("manifest.json")
    created, built = seed_module.seed(
        entity, project_name, manifest_path=manifest_path, steps=steps or None
    )
    experiment_name = experiment or created

    echo("[2/3] migrating ...")
    client = build_client(tracking_uri)
    options = MigrateOptions(experiment=experiment_name, include_artifacts=True, include_files=True)
    result = Migrator(client, options).migrate_project(WandbProject.connect(entity, created))
    echo(format_migration(result))

    echo("")
    echo("[3/3] verifying against the manifest ...")
    report = Verifier(client).verify(built, experiment_name)
    echo(format_report(report))

    echo("")
    if not keep:
        echo(f"Clean up W&B with: wandb-to-mlflow seed --cleanup {created} -e {entity} --yes")
    echo(f"Now look at it:  mlflow ui --backend-store-uri {tracking_uri or './mlruns'}")
    echo(f"  experiment:    {experiment_name}")
    echo("  checklist:     see the UI acceptance section of README.md")
    raise typer.Exit(0 if (report.ok and result.ok) else 1)


@app.command()
def version() -> None:
    """Print the version."""
    echo(__version__)


def main() -> None:  # pragma: no cover - console-script shim
    app()
