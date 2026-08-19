"""MLflow's own limits, read from the installed package at runtime.

Never hardcode these. MLflow has changed them across versions (``MAX_PARAM_VAL_LENGTH``
was 250, then 500, then 6000) and a tool that guesses will either truncate data it
did not need to or send values the server rejects.

Each limit is read from :mod:`mlflow.utils.validation` with a conservative fallback,
so the tool still works against MLflow versions where a name was moved or removed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

# Conservative fallbacks, used only if the installed MLflow does not expose the name.
_FALLBACKS: Final[dict[str, int]] = {
    "MAX_PARAM_VAL_LENGTH": 500,
    "MAX_ENTITY_KEY_LENGTH": 250,
    "MAX_TAG_VAL_LENGTH": 5000,
    "MAX_METRICS_PER_BATCH": 1000,
    "MAX_PARAMS_TAGS_PER_BATCH": 100,
    "MAX_ENTITIES_PER_BATCH": 1000,
}


def _read(name: str) -> int:
    try:
        from mlflow.utils import validation
    except ImportError:  # pragma: no cover - mlflow is a hard dependency
        logger.warning("mlflow.utils.validation unavailable; using fallback for %s", name)
        return _FALLBACKS[name]
    value = getattr(validation, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        logger.warning(
            "mlflow.utils.validation.%s missing or unusable (%r); using fallback %d",
            name,
            value,
            _FALLBACKS[name],
        )
        return _FALLBACKS[name]
    return value


@dataclass(frozen=True)
class Limits:
    """The subset of MLflow's limits this tool has to respect."""

    max_param_val_length: int
    max_entity_key_length: int
    max_tag_val_length: int
    max_metrics_per_batch: int
    max_params_tags_per_batch: int
    max_entities_per_batch: int

    @classmethod
    def from_mlflow(cls) -> Limits:
        """Read the limits out of the installed MLflow."""
        return cls(
            max_param_val_length=_read("MAX_PARAM_VAL_LENGTH"),
            max_entity_key_length=_read("MAX_ENTITY_KEY_LENGTH"),
            max_tag_val_length=_read("MAX_TAG_VAL_LENGTH"),
            max_metrics_per_batch=_read("MAX_METRICS_PER_BATCH"),
            max_params_tags_per_batch=_read("MAX_PARAMS_TAGS_PER_BATCH"),
            max_entities_per_batch=_read("MAX_ENTITIES_PER_BATCH"),
        )


_DEFAULT: Limits | None = None


def default_limits() -> Limits:
    """The process-wide limits, resolved once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Limits.from_mlflow()
    return _DEFAULT
