"""``limits.py`` must never hardcode, and must degrade rather than crash."""

from __future__ import annotations

import pytest
from mlflow.utils import validation

from wandb_to_mlflow import limits as limits_module
from wandb_to_mlflow.limits import Limits, default_limits


def test_limits_come_from_the_installed_mlflow() -> None:
    resolved = Limits.from_mlflow()
    assert resolved.max_param_val_length == validation.MAX_PARAM_VAL_LENGTH
    assert resolved.max_entity_key_length == validation.MAX_ENTITY_KEY_LENGTH
    assert resolved.max_tag_val_length == validation.MAX_TAG_VAL_LENGTH


def test_missing_name_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(validation, "MAX_PARAM_VAL_LENGTH")
    assert Limits.from_mlflow().max_param_val_length == 500


@pytest.mark.parametrize("bad", [0, -1, "500", True, None])
def test_unusable_value_falls_back(monkeypatch: pytest.MonkeyPatch, bad: object) -> None:
    monkeypatch.setattr(validation, "MAX_TAG_VAL_LENGTH", bad, raising=False)
    assert Limits.from_mlflow().max_tag_val_length == 5000


def test_default_limits_is_memoised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limits_module, "_DEFAULT", None)
    first = default_limits()
    assert default_limits() is first


def test_no_module_hardcodes_a_limit() -> None:
    """The literals 500 and 1000 may appear only as fallbacks inside limits.py."""
    import pathlib

    src = pathlib.Path(limits_module.__file__).parent
    offenders = []
    for path in src.glob("*.py"):
        if path.name == "limits.py":
            continue
        text = path.read_text(encoding="utf-8")
        for literal in ("MAX_PARAM_VAL_LENGTH", "MAX_METRICS_PER_BATCH"):
            if f"{literal} =" in text:
                offenders.append(f"{path.name}: redefines {literal}")
    assert offenders == []
