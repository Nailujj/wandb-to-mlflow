"""The seeder: how this tool tests itself.

It creates a **real** W&B project full of deliberately hostile runs and writes
``manifest.json`` describing the exact MLflow state a correct migration must
produce. The manifest is built from what the seeder **actually logged** — the
payloads it handed to ``wandb.log`` — not from a second query to W&B. Comparing
a migration against the same API it was built on would only prove the tool is
self-consistent with itself.

The module is split deliberately:

- :func:`build_specs` and :func:`manifest_for` are pure data. They have no
  ``wandb`` import between them and are fully tested offline.
- :func:`seed` is the only part that touches the network.

Safety (spec 6.1): the default project name carries a UTC timestamp, the plan is
printed before anything is written, non-interactive use requires ``--yes``, and
the total seeded payload is a few kilobytes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wandb_to_mlflow.coerce import (
    Drop,
    DropReport,
    as_metric,
    as_param,
    flatten_config,
    sanitise_keys,
)
from wandb_to_mlflow.limits import Limits, default_limits
from wandb_to_mlflow.migrate import (
    DEFAULT_STATUS,
    INTERNAL_COLUMNS,
    STATE_TO_STATUS,
    SUMMARY_METRIC_PREFIX,
    SUMMARY_PARAM_PREFIX,
)
from wandb_to_mlflow.verify import ExpectedRun, Manifest

logger = logging.getLogger(__name__)

PROJECT_PREFIX = "w2m-selftest-"

#: Kept small on purpose: a self-test that costs money is a self-test nobody runs.
THROUGHPUT_STEPS = 20_000
SWEEP_CHILDREN = 3


def default_project_name(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return f"{PROJECT_PREFIX}{stamp}"


@dataclass
class RunSpec:
    """One run to seed, described as pure data.

    ``exit_code`` drives the W&B run state; ``expected_state`` is what W&B will
    report afterwards, which the manifest turns into an MLflow status.
    """

    key: str
    exercises: str
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    group: str | None = None
    job_type: str | None = None
    exit_code: int = 0
    expected_state: str = "finished"
    in_sweep: bool = False
    artifacts: list[ArtifactSpec] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)


@dataclass
class ArtifactSpec:
    name: str
    type: str = "dataset"
    contents: dict[str, str] = field(default_factory=dict)
    reference: bool = False


def _rows(count: int, **series: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in range(count):
        row: dict[str, Any] = {}
        for key, values in series.items():
            row[key] = values[step] if isinstance(values, list) else values
        out.append(row)
    return out


def build_specs() -> list[RunSpec]:
    """The 16 hostile cases from the build spec, one per failure mode.

    Mirrors ``tests/fixtures.py`` exactly. A case added in one place must be
    added in the other; a test asserts the two stay in step.
    """
    specs: list[RunSpec] = [
        RunSpec(
            key="nested_config",
            exercises="config flattening",
            name="nested-config",
            config={
                "optimizer": {"kind": "sgd", "sched": {"warmup": {"steps": 100, "ratio": 0.1}}},
                "layers": [64, 128, 256],
                "flags": {"amp": True, "compile": False},
                "empty": {},
            },
            rows=_rows(3, loss=[1.0, 0.5, 0.25]),
            summary={"accuracy": 0.9137, "notes": "done"},
        ),
        RunSpec(
            key="long_config_value",
            exercises="param truncation",
            name="long-config-value",
            config={"blob": "x" * 20_000, "short": "ok"},
            rows=_rows(2, loss=[1.0, 0.9]),
        ),
        RunSpec(
            key="nonfinite",
            exercises="NaN and inf rejection",
            name="nonfinite-metrics",
            rows=[
                {"loss": 1.0, "ratio": float("nan")},
                {"loss": float("inf"), "ratio": 0.5},
                {"loss": float("-inf"), "ratio": 0.25},
            ],
            summary={"accuracy": 0.5},
        ),
        RunSpec(
            key="bools",
            exercises="the bool-is-int trap",
            name="bool-trap",
            config={"use_amp": True, "debug": False},
            rows=_rows(3, improved=[True, False, True], loss=[1.0, 0.5, 0.4]),
            summary={"converged": True, "steps": 3},
        ),
        RunSpec(
            key="hostile_keys",
            exercises="key sanitisation",
            name="hostile-keys",
            rows=_rows(
                2,
                **{
                    "train/loss": [1.0, 0.5],
                    "a b": [1, 2],
                    "héllo": [3, 4],
                    "x@y!": [5, 6],
                    "k" * 300: [7, 8],
                },
            ),
        ),
        RunSpec(
            key="key_collision",
            exercises="collision suffixing",
            name="key-collision",
            rows=_rows(2, **{"a@b": [1.0, 2.0], "a#b": [10.0, 20.0]}),
        ),
        RunSpec(
            key="sparse",
            exercises="per-key step alignment",
            name="sparse-logging",
            rows=[
                {"dense": float(step), **({"sparse": step / 10.0} if step % 10 == 0 else {})}
                for step in range(100)
            ],
        ),
        RunSpec(
            key="throughput",
            exercises="batching and throughput",
            name="many-steps",
            rows=[
                {f"m{m}": float(step) * m for m in range(1, 6)} for step in range(THROUGHPUT_STEPS)
            ],
        ),
        RunSpec(
            key="empty_history",
            exercises="the empty-history path",
            name="config-only",
            config={"lr": 0.01},
        ),
        RunSpec(
            key="media",
            exercises="media rejection and reporting",
            name="media-and-tables",
            rows=[{"loss": 1.0}, {"loss": 0.5}],  # media rows are added by the seeder
        ),
        RunSpec(
            key="artifacts",
            exercises="artifact paths and reference handling",
            name="with-artifacts",
            rows=_rows(1, loss=1.0),
            files={"seeded.txt": "a run file\n"},
            artifacts=[
                ArtifactSpec(name="small-dataset", contents={"data.csv": "a,b\n1,2\n"}),
                ArtifactSpec(name="remote-dataset", reference=True),
            ],
        ),
        RunSpec(
            key="metadata",
            exercises="metadata mapping",
            name="rich-metadata",
            rows=_rows(1, loss=1.0),
            tags=["baseline", "v2", "needs review"],
            notes="A run with notes.\nSecond line.",
            group="ablation-a",
            job_type="train",
        ),
        RunSpec(
            key="failed",
            exercises="status mapping (failure)",
            name="failed-run",
            rows=_rows(2, loss=[1.0, 0.9]),
            exit_code=1,
            expected_state="failed",
        ),
        RunSpec(
            key="crashed",
            exercises="status mapping (non-zero exit)",
            name="crashed-run",
            rows=_rows(1, loss=1.0),
            exit_code=2,
            expected_state="failed",
        ),
        RunSpec(
            key="duplicate_name_a",
            exercises="name is not a key",
            name="same-name",
            rows=_rows(1, loss=1.0),
        ),
        RunSpec(
            key="duplicate_name_b",
            exercises="name is not a key",
            name="same-name",
            rows=_rows(1, loss=2.0),
        ),
        RunSpec(
            key="unicode",
            exercises="encoding",
            # No emoji in `name` or `notes`: W&B's own backend stores those two
            # columns as utf8mb3 and rejects 4-byte characters outright with
            # "Error 3988 (HY000): Conversion from collation utf8mb4_unicode_ci
            # into utf8mb3_general_ci impossible". Config values and tags are
            # stored differently and take emoji fine, so that is where the
            # 4-byte case lives. Three-byte scripts (Hangul, Cyrillic, Greek,
            # CJK) are accepted everywhere and stay in name and notes.
            # tests/fixtures.py keeps an emoji-in-name run regardless: the
            # migrator must handle one even though W&B cannot produce one.
            name="ünïcode 실험",
            notes="Notes with кириллица and 中文.",
            config={"β": 0.9, "描述": "中文", "emoji": "🎉🚀"},
            tags=["🎉"],
            rows=_rows(1, **{"λ/λοιπόν": 0.5}),
        ),
    ]
    specs.extend(
        RunSpec(
            key=f"sweep_child_{i}",
            exercises="sweep parent/child nesting",
            name=f"sweep-child-{i}",
            config={"lr": 10**-i},
            rows=_rows(2, loss=[1.0 / (i + 1), 0.5 / (i + 1)]),
            summary={"accuracy": 0.8 + i / 100},
            in_sweep=True,
        )
        for i in range(SWEEP_CHILDREN)
    )
    return specs


# --------------------------------------------------------------------------- #
# Manifest generation -- pure, from what was logged
# --------------------------------------------------------------------------- #


def _media_rows(spec: RunSpec) -> list[dict[str, str]]:
    """The media payloads the seeder adds for the media case, as ``_type`` maps.

    Recorded separately from ``rows`` because the real objects (``wandb.Image``,
    ``wandb.Table``) can only be built with ``wandb`` imported, while the
    manifest must be derivable without it.
    """
    if spec.key != "media":
        return []
    return [{"sample": "image-file", "preds": "table-file"}, {"sample": "image-file"}]


def expected_for(
    spec: RunSpec,
    wandb_run_id: str,
    sweep_id: str | None = None,
    limits: Limits | None = None,
) -> ExpectedRun:
    """Derive one run's ground truth from the payloads the seeder logged.

    Counting is done here with plain arithmetic over the logged rows, not by
    running the migrator: the manifest has to be able to disagree with the
    migrator, or it is not a test.
    """
    lim = limits or default_limits()
    dropped = DropReport()
    media_rows = _media_rows(spec)

    point_counts: dict[str, int] = {}
    last_values: dict[str, Any] = {}
    for index, row in enumerate(spec.rows):
        for key, value in row.items():
            if key in INTERNAL_COLUMNS:
                continue
            last_values[key] = value
            metric, reason, _ = as_metric(value)
            if metric is None:
                assert reason is not None
                dropped.record(reason)
                continue
            point_counts[key] = point_counts.get(key, 0) + 1
        if index < len(media_rows):
            for media_key, media_type in media_rows[index].items():
                dropped.record(Drop.MEDIA, media_type)
                last_values[media_key] = {"_type": media_type}

    # W&B writes the last logged value of every history key into run.summary by
    # itself, media included. Measured against a real run, not assumed -- and
    # missing it is what made the first live self-test report a dozen
    # "unexpected" final.* metrics that were entirely correct.
    summary = {**last_values, **spec.summary}

    final_values: dict[str, float] = {}
    summary_param_keys: set[str] = set()
    for key, value in summary.items():
        metric, _, _ = as_metric(value)
        if metric is None:
            # Not lost: non-numeric summary values become summary.* params.
            summary_param_keys.add(f"{SUMMARY_PARAM_PREFIX}{key}")
            continue
        final_values[f"{SUMMARY_METRIC_PREFIX}{key}"] = metric

    flat_config = {
        key: value
        for key, value in flatten_config(spec.config).items()
        # W&B discards empty-dict config values server-side, so no param for
        # them ever reaches MLflow. Measured, like the rest of this function.
        if value != {}
    }
    param_keys: set[str] = set(flat_config) | summary_param_keys

    metric_map = sanitise_keys([*point_counts, *final_values], lim)
    param_map = sanitise_keys(param_keys, lim)

    counts = {metric_map[k]: v for k, v in point_counts.items()}
    counts.update({metric_map[k]: 1 for k in final_values})

    expected_params: dict[str, str] = {}
    for key, value in flat_config.items():
        rendered, was_truncated = as_param(value, lim)
        if not was_truncated and len(rendered) <= 64:
            expected_params[param_map[key]] = rendered

    return ExpectedRun(
        wandb_run_id=wandb_run_id,
        expected_status=STATE_TO_STATUS.get(spec.expected_state, DEFAULT_STATUS),
        expected_param_count=len(param_map),
        expected_params=expected_params,
        expected_metric_keys=sorted(counts),
        expected_metric_point_counts=counts,
        expected_final_values={metric_map[k]: v for k, v in final_values.items()},
        expected_dropped=dropped.as_dict(),
        expected_parent=sweep_id if spec.in_sweep else None,
    )


def manifest_for(
    specs: list[RunSpec],
    entity: str,
    project: str,
    run_ids: dict[str, str],
    sweep_id: str | None = None,
    created_at: str | None = None,
    limits: Limits | None = None,
) -> Manifest:
    """Assemble the manifest for the runs that were actually created."""
    return Manifest(
        wandb={
            "entity": entity,
            "project": project,
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "sweep_id": sweep_id,
        },
        runs=[
            expected_for(spec, run_ids[spec.key], sweep_id, limits)
            for spec in specs
            if spec.key in run_ids
        ],
    )


# --------------------------------------------------------------------------- #
# The plan the user is shown before anything is written
# --------------------------------------------------------------------------- #


def plan_text(specs: list[RunSpec], entity: str, project: str) -> str:
    lines = [
        "This will CREATE a new W&B project and write real runs into it:",
        "",
        f"  entity:  {entity}",
        f"  project: {project}",
        f"  runs:    {len(specs)} (one sweep with {SWEEP_CHILDREN} children included)",
        f"  points:  ~{sum(len(s.rows) for s in specs):,} logged history rows",
        f"  bytes:   under {_payload_estimate(specs) // 1024 + 1} KiB of artifacts and files",
        "",
        "Runs and what each one exercises:",
    ]
    width = max(len(spec.name) for spec in specs)
    lines.extend(f"  {spec.name:<{width}}  {spec.exercises}" for spec in specs)
    lines.extend(
        [
            "",
            "Nothing outside this project is touched. Remove it afterwards with:",
            f"  wandb-to-mlflow seed --cleanup {project} --entity {entity} --yes",
        ]
    )
    return "\n".join(lines)


def _payload_estimate(specs: list[RunSpec]) -> int:
    total = 0
    for spec in specs:
        total += sum(len(v) for v in spec.files.values())
        for artifact in spec.artifacts:
            total += sum(len(v) for v in artifact.contents.values())
    return total


# --------------------------------------------------------------------------- #
# The part that touches the network
# --------------------------------------------------------------------------- #


def seed(
    entity: str,
    project: str | None = None,
    manifest_path: Path | None = None,
    steps: int | None = None,
) -> tuple[str, Manifest]:
    """Create the project in W&B and write the manifest. Requires network.

    Returns ``(project_name, manifest)``.
    """
    import wandb

    project_name = project or default_project_name()
    specs = build_specs()
    if steps is not None:
        for spec in specs:
            if spec.key == "throughput":
                spec.rows = spec.rows[:steps]

    created_at = datetime.now(timezone.utc).isoformat()
    run_ids: dict[str, str] = {}
    sweep_id: str | None = None

    for spec in specs:
        if spec.in_sweep:
            continue
        run_ids[spec.key] = _seed_one(wandb, entity, project_name, spec)

    sweep_specs = [spec for spec in specs if spec.in_sweep]
    if sweep_specs:
        sweep_id, sweep_ids = _seed_sweep(wandb, entity, project_name, sweep_specs)
        run_ids.update(sweep_ids)

    manifest = manifest_for(specs, entity, project_name, run_ids, sweep_id, created_at)
    if manifest_path is not None:
        manifest.dump(manifest_path)
    return project_name, manifest


def _seed_one(wandb: Any, entity: str, project: str, spec: RunSpec, **init: Any) -> str:
    run = wandb.init(
        entity=entity,
        project=project,
        name=spec.name,
        config=spec.config,
        tags=spec.tags or None,
        notes=spec.notes,
        group=spec.group,
        job_type=spec.job_type,
        reinit=True,
        settings=wandb.Settings(silent=True),
        **init,
    )
    try:
        media = _media_payloads(wandb, spec)
        for index, row in enumerate(spec.rows):
            payload = dict(row)
            if index < len(media):
                payload.update(media[index])
            run.log(payload)
        for key, value in spec.summary.items():
            run.summary[key] = value
        _seed_files(wandb, run, spec)
        _seed_artifacts(wandb, run, spec)
        return str(run.id)
    finally:
        run.finish(exit_code=spec.exit_code)


def _media_payloads(wandb: Any, spec: RunSpec) -> list[dict[str, Any]]:
    """Real ``wandb.Image``/``wandb.Table`` objects for the media case.

    ``wandb.Image`` requires an array with ``.ndim`` -- a nested Python list is
    rejected with ``AttributeError: 'list' object has no attribute 'ndim'``.
    numpy is imported here rather than at module scope because it is a
    transitive dependency (via MLflow), not one this tool declares, and only
    the seeder's network path needs it.
    """
    if spec.key != "media":
        return []
    import numpy as np

    pixels = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)
    table = wandb.Table(columns=["pred", "label"], data=[["a", "b"], ["c", "d"]])
    return [
        {"sample": wandb.Image(pixels), "preds": table},
        {"sample": wandb.Image(pixels)},
    ]


def _seed_files(wandb: Any, run: Any, spec: RunSpec) -> None:
    del wandb
    if not spec.files:
        return
    for name, contents in spec.files.items():
        target = Path(run.dir) / name
        target.write_text(contents, encoding="utf-8")
        run.save(str(target), base_path=run.dir, policy="now")


def _seed_artifacts(wandb: Any, run: Any, spec: RunSpec) -> None:
    import tempfile

    for artifact_spec in spec.artifacts:
        artifact = wandb.Artifact(name=artifact_spec.name, type=artifact_spec.type)
        with tempfile.TemporaryDirectory(prefix="w2m-seed-") as tmp:
            root = Path(tmp)
            if artifact_spec.reference:
                # A local file:// reference: the bytes stay where they are, which
                # is exactly the case the migrator must record and not fetch --
                # and it needs no cloud credentials to set up.
                target = root / "referenced.txt"
                target.write_text("bytes that live elsewhere\n", encoding="utf-8")
                artifact.add_reference(target.resolve().as_uri())
            else:
                for name, contents in artifact_spec.contents.items():
                    target = root / name
                    target.write_text(contents, encoding="utf-8")
                    artifact.add_file(str(target), name=name)
            run.log_artifact(artifact).wait()


def _seed_sweep(
    wandb: Any, entity: str, project: str, specs: list[RunSpec]
) -> tuple[str, dict[str, str]]:
    """A real W&B sweep, so the parent/child mapping is exercised end to end."""
    # The search space *is* the children's configs. A sweep injects its own
    # parameters into each run's config, so sweeping over anything else would
    # leave the seeded runs carrying config the manifest never predicted.
    by_lr = {repr(float(spec.config["lr"])): spec for spec in specs}
    sweep_id = wandb.sweep(
        {
            "method": "grid",
            "metric": {"name": "loss", "goal": "minimize"},
            "parameters": {"lr": {"values": [spec.config["lr"] for spec in specs]}},
        },
        entity=entity,
        project=project,
    )
    ids: dict[str, str] = {}

    def train() -> None:
        run = wandb.init(settings=wandb.Settings(silent=True))
        spec = by_lr[repr(float(run.config["lr"]))]
        run.name = spec.name
        for row in spec.rows:
            run.log(dict(row))
        for key, value in spec.summary.items():
            run.summary[key] = value
        ids[spec.key] = str(run.id)
        run.finish()

    wandb.agent(sweep_id, function=train, count=len(specs), entity=entity, project=project)
    return str(sweep_id), ids


def is_seeded_project(project: str) -> bool:
    """Only projects this tool created carry the seeded prefix."""
    return bool(project) and project.startswith(PROJECT_PREFIX)


class NotASeededProjectError(RuntimeError):
    """Refusing to delete runs from a project this tool did not create."""


def cleanup(entity: str, project: str) -> int:
    """Delete every run in a **seeded** project. Returns how many were deleted.

    The prefix check lives here, in the function that actually deletes, and not
    only in the CLI that calls it. This is the one code path in the package that
    can destroy data, and a guard a library caller can bypass is not a guard.

    W&B's public API has no project delete, so the (now empty) project shell
    stays behind and has to be removed from the web UI. Said plainly rather than
    pretended otherwise.
    """
    if not is_seeded_project(project):
        raise NotASeededProjectError(
            f"{project!r} does not start with {PROJECT_PREFIX!r}, so this tool did not "
            "create it. Refusing to delete its runs. Nothing in wandb-to-mlflow ever "
            "deletes a W&B project it did not seed."
        )
    import wandb

    api = wandb.Api()
    deleted = 0
    for run in api.runs(f"{entity}/{project}"):
        run.delete(delete_artifacts=True)
        deleted += 1
    return deleted
