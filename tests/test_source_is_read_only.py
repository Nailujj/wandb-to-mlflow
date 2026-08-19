"""The migration path must never write to W&B.

W&B is the user's insurance policy. They keep it precisely so that if the
migration turns out to be wrong, the originals are still there. A tool that
migrates *and* mutates the source destroys the thing that makes it safe to try.

So this is an enforced invariant, not a convention: the only module allowed to
write to W&B at all is the seeder, and the only deletion in the package is
guarded at the point of deletion.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import wandb_to_mlflow
from wandb_to_mlflow.seed import NotASeededProjectError, cleanup, is_seeded_project

PACKAGE = Path(wandb_to_mlflow.__file__).parent

#: Anything on a W&B object that creates, changes or destroys server-side state.
MUTATING = {
    "delete",
    "init",
    "log",
    "log_artifact",
    "log_code",
    "save",
    "finish",
    "sweep",
    "agent",
    "link_artifact",
    "use_artifact",
    "upsert_run",
    "update",
    "create_run",
    "restore",
    "alert",
}

#: Only the seeder may write to W&B, and only to build its own scratch project.
MAY_WRITE_TO_WANDB = {"seed.py"}


def calls_in(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.func.attr, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]


def test_the_migration_path_never_mutates_wandb() -> None:
    """`source.py` is the only module that talks to W&B during a migration."""
    offenders = [
        f"source.py:{line} .{name}()"
        for name, line in calls_in(PACKAGE / "source.py")
        if name in MUTATING
    ]
    assert offenders == [], f"the source adapter must be read-only: {offenders}"


def test_no_module_outside_the_seeder_writes_to_wandb() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name in MAY_WRITE_TO_WANDB:
            continue
        offenders += [
            f"{path.name}:{line} .{name}()"
            for name, line in calls_in(path)
            # `.update()`/`.save()`/`.log()` are ordinary dict and client method
            # names too, so only the unambiguously W&B-side ones are checked
            # outside the seeder.
            if name in {"delete", "init", "log_artifact", "sweep", "agent", "alert"}
        ]
    assert offenders == []


def test_the_package_deletes_wandb_data_in_exactly_one_place() -> None:
    """If this count ever changes, someone added a way to destroy user data."""
    deletions = [
        f"{path.name}:{line}"
        for path in sorted(PACKAGE.glob("*.py"))
        for name, line in calls_in(path)
        if name == "delete"
    ]
    assert len(deletions) == 1, deletions
    assert deletions[0].startswith("seed.py:")


def test_cleanup_refuses_a_project_it_did_not_seed() -> None:
    """Guarded in the function that deletes, not only in the CLI that calls it."""
    for project in ["production-experiments", "my-research", "", "wandb-selftest"]:
        with pytest.raises(NotASeededProjectError, match="Refusing to delete"):
            cleanup("some-entity", project)


def test_cleanup_guard_cannot_be_reached_by_a_lookalike_name() -> None:
    assert not is_seeded_project("not-w2m-selftest-20260101-000000")
    assert not is_seeded_project("W2M-SELFTEST-20260101-000000")
    assert is_seeded_project("w2m-selftest-20260101-000000")


def test_cleanup_never_gets_as_far_as_the_network_when_refusing() -> None:
    """The no-network fixture would fail this test if the guard ran too late."""
    with pytest.raises(NotASeededProjectError):
        cleanup("entity", "someone-elses-project")
