"""Hand-built :class:`SourceRun` fixtures, one per hostile case in the spec.

These mirror the 16 seeded W&B runs exactly. The point is that tier 2 catches
almost every regression without a network round trip, leaving tier 3 to catch
only what unit tests structurally cannot: W&B API drift.

Anything added to the seeder must be added here too, and vice versa.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wandb_to_mlflow.source import SourceArtifact, SourceFile, SourceRun

BASE_TS = 1_700_000_000.0  # 2023-11-14T22:13:20Z, a fixed point so tests are stable


@dataclass
class FakeFile:
    name: str
    size: int
    content: bytes = b"fake file content\n"

    def download(self, dest_dir: Path) -> Path:
        target = dest_dir / self.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.content)
        return target


@dataclass
class FakeArtifact:
    name: str
    type: str = "dataset"
    version: str = "v0"
    aliases: list[str] = field(default_factory=lambda: ["latest"])
    digest: str = "deadbeef"
    size: int = 42
    is_reference: bool = False
    source_uris: list[str] = field(default_factory=list)
    files: dict[str, bytes] = field(default_factory=lambda: {"data.csv": b"a,b\n1,2\n"})

    def download(self, dest_dir: Path) -> Path:
        if self.is_reference:  # pragma: no cover - the migrator must never call this
            raise AssertionError("reference artifacts must not be downloaded")
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name, content in self.files.items():
            target = dest_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return dest_dir


@dataclass
class FakeRun:
    """A :class:`SourceRun` with no network behind it."""

    id: str
    name: str | None = None
    state: str = "finished"
    created_at: str = "2023-11-14T22:13:20"
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    group: str | None = None
    job_type: str | None = None
    sweep_id: str | None = None
    url: str = ""
    entity: str = "acme"
    project: str = "w2m-selftest"
    rows: list[dict[str, Any]] = field(default_factory=list)
    system_rows: list[dict[str, Any]] = field(default_factory=list)
    file_list: list[FakeFile] = field(default_factory=list)
    artifact_list: list[FakeArtifact] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.url:
            self.url = f"https://wandb.ai/{self.entity}/{self.project}/runs/{self.id}"

    def history(self) -> Iterator[dict[str, Any]]:
        yield from self.rows

    def system_metrics(self) -> Iterator[dict[str, Any]]:
        yield from self.system_rows

    def files(self) -> Iterator[SourceFile]:
        yield from self.file_list

    def artifacts(self) -> Iterator[SourceArtifact]:
        yield from self.artifact_list


@dataclass
class FakeProject:
    entity: str = "acme"
    project: str = "w2m-selftest"
    run_list: list[FakeRun] = field(default_factory=list)

    def runs(self) -> Iterator[SourceRun]:
        yield from self.run_list


def _rows(count: int, **series: Any) -> list[dict[str, Any]]:
    """History rows with the ``_step``/``_timestamp`` columns W&B always emits."""
    out = []
    for step in range(count):
        row: dict[str, Any] = {"_step": step, "_timestamp": BASE_TS + step, "_runtime": float(step)}
        for key, values in series.items():
            row[key] = values[step] if isinstance(values, list) else values
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# 1. nested config, 3 levels deep, with list and dict values
# --------------------------------------------------------------------------- #
def run_nested_config() -> FakeRun:
    return FakeRun(
        id="r01-nested",
        name="nested-config",
        config={
            "optimizer": {"kind": "sgd", "sched": {"warmup": {"steps": 100, "ratio": 0.1}}},
            "layers": [64, 128, 256],
            "flags": {"amp": True, "compile": False},
            "empty": {},
            "_wandb": {"internal": "dropped"},
        },
        summary={"accuracy": 0.9137, "notes": "done"},
        rows=_rows(3, loss=[1.0, 0.5, 0.25]),
    )


# --------------------------------------------------------------------------- #
# 2. a config value of 20,000 characters
# --------------------------------------------------------------------------- #
def run_long_config_value() -> FakeRun:
    return FakeRun(
        id="r02-longparam",
        name="long-config-value",
        config={"blob": "x" * 20_000, "short": "ok"},
        rows=_rows(2, loss=[1.0, 0.9]),
    )


# --------------------------------------------------------------------------- #
# 3. metrics including NaN, inf, -inf
# --------------------------------------------------------------------------- #
def run_nonfinite() -> FakeRun:
    return FakeRun(
        id="r03-nonfinite",
        name="nonfinite-metrics",
        rows=[
            {"_step": 0, "_timestamp": BASE_TS, "loss": 1.0, "ratio": float("nan")},
            {"_step": 1, "_timestamp": BASE_TS + 1, "loss": float("inf"), "ratio": 0.5},
            {"_step": 2, "_timestamp": BASE_TS + 2, "loss": float("-inf"), "ratio": 0.25},
        ],
        summary={"accuracy": 0.5, "diverged": float("nan")},
    )


# --------------------------------------------------------------------------- #
# 4. bools in both config and history -- the bool-is-int trap
# --------------------------------------------------------------------------- #
def run_bools() -> FakeRun:
    return FakeRun(
        id="r04-bools",
        name="bool-trap",
        config={"use_amp": True, "debug": False},
        summary={"converged": True, "steps": 3},
        rows=_rows(3, improved=[True, False, True], loss=[1.0, 0.5, 0.4]),
    )


# --------------------------------------------------------------------------- #
# 5. hostile history keys
# --------------------------------------------------------------------------- #
def run_hostile_keys() -> FakeRun:
    return FakeRun(
        id="r05-keys",
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
    )


# --------------------------------------------------------------------------- #
# 6. two keys that sanitise to the same target
# --------------------------------------------------------------------------- #
def run_key_collision() -> FakeRun:
    return FakeRun(
        id="r06-collision",
        name="key-collision",
        rows=_rows(2, **{"a@b": [1.0, 2.0], "a#b": [10.0, 20.0]}),
    )


# --------------------------------------------------------------------------- #
# 7. sparse logging: key A every step, key B every 10th
# --------------------------------------------------------------------------- #
def run_sparse() -> FakeRun:
    rows = []
    for step in range(100):
        row: dict[str, Any] = {"_step": step, "_timestamp": BASE_TS + step, "dense": float(step)}
        if step % 10 == 0:
            row["sparse"] = float(step) / 10.0
        rows.append(row)
    return FakeRun(id="r07-sparse", name="sparse-logging", rows=rows)


# --------------------------------------------------------------------------- #
# 8. throughput: many steps x several metrics
# --------------------------------------------------------------------------- #
def run_many_steps(steps: int = 20_000) -> FakeRun:
    rows = [
        {
            "_step": step,
            "_timestamp": BASE_TS + step,
            "m1": float(step),
            "m2": float(step) * 2,
            "m3": float(step) * 3,
            "m4": float(step) * 4,
            "m5": float(step) * 5,
        }
        for step in range(steps)
    ]
    return FakeRun(id="r08-manysteps", name="many-steps", rows=rows)


# --------------------------------------------------------------------------- #
# 9. zero metrics, config only
# --------------------------------------------------------------------------- #
def run_empty_history() -> FakeRun:
    return FakeRun(id="r09-empty", name="config-only", config={"lr": 0.01}, rows=[])


# --------------------------------------------------------------------------- #
# 10. media and tables in history
# --------------------------------------------------------------------------- #
def run_media() -> FakeRun:
    return FakeRun(
        id="r10-media",
        name="media-and-tables",
        rows=[
            {
                "_step": 0,
                "_timestamp": BASE_TS,
                "loss": 1.0,
                "sample": {"_type": "image-file", "path": "media/images/a.png"},
                "preds": {"_type": "table-file", "path": "media/table/t.json"},
            },
            {
                "_step": 1,
                "_timestamp": BASE_TS + 1,
                "loss": 0.5,
                "sample": {"_type": "image-file", "path": "media/images/b.png"},
            },
        ],
    )


# --------------------------------------------------------------------------- #
# 11. one small logged artifact, one reference artifact
# --------------------------------------------------------------------------- #
def run_artifacts() -> FakeRun:
    return FakeRun(
        id="r11-artifacts",
        name="with-artifacts",
        rows=_rows(1, loss=1.0),
        file_list=[FakeFile(name="config.yaml", size=18)],
        artifact_list=[
            FakeArtifact(name="small-dataset:v0", size=8),
            FakeArtifact(
                name="remote-dataset:v0",
                size=1_000_000_000,
                is_reference=True,
                source_uris=["s3://bucket/huge/"],
                files={},
            ),
        ],
    )


# --------------------------------------------------------------------------- #
# 12. tags, notes, group, job_type
# --------------------------------------------------------------------------- #
def run_metadata() -> FakeRun:
    return FakeRun(
        id="r12-metadata",
        name="rich-metadata",
        tags=["baseline", "v2", "needs review"],
        notes="A run with notes.\nSecond line.",
        group="ablation-a",
        job_type="train",
        rows=_rows(1, loss=1.0),
    )


# --------------------------------------------------------------------------- #
# 13. a crashed run and a failed run
# --------------------------------------------------------------------------- #
def run_crashed() -> FakeRun:
    return FakeRun(
        id="r13-crashed", name="crashed", state="crashed", rows=_rows(2, loss=[1.0, 0.9])
    )


def run_failed() -> FakeRun:
    return FakeRun(id="r13-failed", name="failed", state="failed", rows=_rows(1, loss=1.0))


# --------------------------------------------------------------------------- #
# 14. two runs sharing a display name -- name is not a key
# --------------------------------------------------------------------------- #
def run_duplicate_name_a() -> FakeRun:
    return FakeRun(id="r14-dup-a", name="same-name", rows=_rows(1, loss=1.0))


def run_duplicate_name_b() -> FakeRun:
    return FakeRun(id="r14-dup-b", name="same-name", rows=_rows(1, loss=2.0))


# --------------------------------------------------------------------------- #
# 15. a sweep with three children
# --------------------------------------------------------------------------- #
def sweep_children(sweep_id: str = "sw-abc123") -> list[FakeRun]:
    return [
        FakeRun(
            id=f"r15-sweep-{i}",
            name=f"sweep-child-{i}",
            sweep_id=sweep_id,
            config={"lr": 10**-i},
            rows=_rows(2, loss=[1.0 / (i + 1), 0.5 / (i + 1)]),
            summary={"accuracy": 0.8 + i / 100},
        )
        for i in range(3)
    ]


# --------------------------------------------------------------------------- #
# 16. unicode and emoji in name and notes
# --------------------------------------------------------------------------- #
def run_unicode() -> FakeRun:
    return FakeRun(
        id="r16-unicode",
        name="ünïcode 🎉 실험",
        notes="Notes with emoji 🚀 and кириллица.",
        config={"β": 0.9, "描述": "中文"},
        rows=_rows(1, **{"λ/λοιπόν": 0.5}),
    )


def all_runs() -> list[FakeRun]:
    """Every fixture case, as one project's worth of runs.

    ``run_many_steps`` is included at a reduced size; the full 20,000-step case
    is exercised by an explicitly-marked slow test rather than by every run of
    the suite.
    """
    return [
        run_nested_config(),
        run_long_config_value(),
        run_nonfinite(),
        run_bools(),
        run_hostile_keys(),
        run_key_collision(),
        run_sparse(),
        run_many_steps(steps=200),
        run_empty_history(),
        run_media(),
        run_artifacts(),
        run_metadata(),
        run_crashed(),
        run_failed(),
        run_duplicate_name_a(),
        run_duplicate_name_b(),
        *sweep_children(),
        run_unicode(),
    ]


def fake_project() -> FakeProject:
    return FakeProject(run_list=all_runs())
