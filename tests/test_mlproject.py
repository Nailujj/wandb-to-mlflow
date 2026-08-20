"""The MLproject file and the CLI must not drift apart.

An entry point that passes an option the CLI no longer accepts fails only when
someone runs `mlflow run .`, which is exactly the path least likely to be
exercised during development. So it is checked here instead.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.main import get_command

from wandb_to_mlflow import cli

ROOT = Path(__file__).resolve().parents[1]
MLPROJECT = ROOT / "MLproject"
PYTHON_ENV = ROOT / "python_env.yaml"


@pytest.fixture(scope="module")
def project() -> dict[str, Any]:
    return dict(yaml.safe_load(MLPROJECT.read_text(encoding="utf-8")))


def cli_options(command_name: str) -> set[str]:
    group = get_command(cli.app)
    command = group.commands[command_name]  # type: ignore[attr-defined]
    names: set[str] = set()
    for param in command.params:
        names.update(getattr(param, "opts", []))
    return names


def parse_command(command: str) -> tuple[str, list[str]]:
    """The CLI subcommand and the option flags an entry point passes it."""
    # Substitute {placeholders} first; shlex would otherwise choke on quoting.
    tokens = shlex.split(re.sub(r"\{[a-z_]+\}", "PLACEHOLDER", command))
    assert tokens[:3] == ["python", "-m", "wandb_to_mlflow"], tokens
    return tokens[3], [t for t in tokens[4:] if t.startswith("--")]


# --------------------------------------------------------------------------- #


def test_all_five_entry_points_are_declared(project: dict[str, Any]) -> None:
    assert set(project["entry_points"]) == {"migrate", "plan", "seed", "verify", "demo"}


def test_every_entry_point_invokes_the_module_form(project: dict[str, Any]) -> None:
    """`python -m wandb_to_mlflow`, not a console script that may not be on PATH."""
    for entry in project["entry_points"].values():
        assert entry["command"].startswith("python -m wandb_to_mlflow ")


def test_every_entry_point_maps_to_a_real_cli_command(project: dict[str, Any]) -> None:
    group = get_command(cli.app)
    for name, entry in project["entry_points"].items():
        subcommand, _ = parse_command(entry["command"])
        assert subcommand in group.commands, f"{name} calls unknown command {subcommand!r}"  # type: ignore[attr-defined]


def test_every_option_passed_is_one_the_cli_accepts(project: dict[str, Any]) -> None:
    for name, entry in project["entry_points"].items():
        subcommand, options = parse_command(entry["command"])
        accepted = cli_options(subcommand)
        unknown = [opt for opt in options if opt not in accepted]
        assert unknown == [], f"entry point {name!r} passes unknown options {unknown}"


def test_every_declared_parameter_is_actually_substituted(project: dict[str, Any]) -> None:
    for name, entry in project["entry_points"].items():
        declared = set(entry.get("parameters") or {})
        used = set(re.findall(r"\{([a-z_]+)\}", entry["command"]))
        assert declared == used, f"entry point {name!r}: declared {declared}, used {used}"


def test_boolean_parameters_are_string_typed(project: dict[str, Any]) -> None:
    """MLproject cannot emit or omit a bare flag, so these must take a value."""
    artifacts = project["entry_points"]["migrate"]["parameters"]["artifacts"]
    assert artifacts["type"] == "string"
    assert cli.parse_bool(artifacts["default"], "--artifacts") is False
    assert "--artifacts {artifacts}" in project["entry_points"]["migrate"]["command"]


def test_seed_and_demo_pass_yes_so_they_do_not_hang(project: dict[str, Any]) -> None:
    """`mlflow run` has no tty; an unconfirmed prompt would block forever."""
    for name in ("seed", "demo"):
        assert "--yes" in project["entry_points"][name]["command"]


def test_python_env_declares_the_runtime_dependencies() -> None:
    env = yaml.safe_load(PYTHON_ENV.read_text(encoding="utf-8"))
    names = {re.split(r"[><=]", dep)[0].strip() for dep in env["dependencies"]}
    assert {"wandb", "mlflow", "typer", "tenacity"} <= names
    assert "." in names, "the package itself must be installed into the run environment"


def test_python_env_version_satisfies_the_package_floor() -> None:
    env = yaml.safe_load(PYTHON_ENV.read_text(encoding="utf-8"))
    pyproject = ROOT / "pyproject.toml"
    floor = re.search(r'requires-python = ">=([\d.]+)"', pyproject.read_text()).group(1)  # type: ignore[union-attr]
    as_tuple = tuple(int(part) for part in str(env["python"]).split("."))
    assert as_tuple >= tuple(int(part) for part in floor.split("."))


def test_dependencies_match_the_fixed_set() -> None:
    """Spec: the dependency list is fixed. Anything new needs a DECISIONS entry.

    ``tomllib`` is stdlib only from 3.11, while this project supports 3.10 and
    CI runs both. Skipping on 3.10 rather than taking a ``tomli`` dependency:
    what this asserts is the *content of pyproject.toml*, which does not vary by
    interpreter, so the 3.12 leg enforces it for every leg. Adding a dependency
    to test the list of dependencies would also have to be added to the list
    this very test pins.
    """
    tomllib = pytest.importorskip(
        "tomllib", reason="stdlib TOML parser lands in 3.11; the 3.12 leg covers this"
    )

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = {re.split(r"[><=\[]", dep)[0].strip() for dep in pyproject["project"]["dependencies"]}
    assert runtime == {"wandb", "mlflow", "typer", "tenacity"}
    dev = {
        re.split(r"[><=\[]", dep)[0].strip()
        for dep in pyproject["project"]["optional-dependencies"]["dev"]
    }
    assert dev == {"pytest", "pytest-cov", "ruff", "mypy"}
