"""Value coercion and key sanitisation.

This is where silent data corruption lives, so every rule here is explicit and
every rejection is counted. The guiding principle: **never invent a number**.
A value that is not unambiguously a finite real scalar is dropped and recorded,
never coerced, parsed or approximated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from wandb_to_mlflow.limits import Limits, default_limits

logger = logging.getLogger(__name__)

TRUNCATION_MARKER = "…[truncated]"

# The portable intersection of MLflow's POSIX and Windows key charsets: POSIX also
# allows ":", Windows does not. Sanitising to the intersection keeps a migration
# valid against a tracking server on either platform. See MAPPING.md section 3.
_ALLOWED_KEY_CHARS = re.compile(r"[^/\w.\- ]", re.UNICODE)

# Path segments MLflow's `path_not_unique` check rejects, and their replacements.
_BAD_SEGMENTS = {"": "_", ".": "_", "..": "__"}

_FALLBACK_KEY = "unnamed"
_COLLISION_HASH_LEN = 6


class Drop(str, Enum):
    """Why a value was not migrated as a metric."""

    BOOL = "bool"
    NONE = "none"
    NONFINITE = "nonfinite"
    STRING = "str"
    LIST = "list"
    MEDIA = "media"
    OTHER = "other"


@dataclass
class DropReport:
    """A running tally of everything a run lost, so nothing is dropped silently."""

    counts: Counter[str] = field(default_factory=Counter)
    media: Counter[str] = field(default_factory=Counter)

    def record(self, reason: Drop, media_type: str | None = None) -> None:
        self.counts[reason.value] += 1
        if reason is Drop.MEDIA:
            self.media[media_type or "unknown"] += 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def merge(self, other: DropReport) -> None:
        self.counts.update(other.counts)
        self.media.update(other.media)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: v for k, v in sorted(self.counts.items())}
        if self.media:
            out["media_types"] = dict(sorted(self.media.items()))
        return out


# --------------------------------------------------------------------------- #
# Values -> metrics
# --------------------------------------------------------------------------- #


def media_type_of(value: Any) -> str | None:
    """The W&B ``_type`` of a media/table dict, if this value is one."""
    if isinstance(value, Mapping):
        raw = value.get("_type")
        return raw if isinstance(raw, str) else "unknown"
    return None


def as_metric(value: Any) -> tuple[float | None, Drop | None, str | None]:
    """Coerce a history/summary value to a metric.

    Returns ``(metric_value, drop_reason, media_type)``. Exactly one of
    ``metric_value`` and ``drop_reason`` is not ``None``.

    Order matters: ``bool`` is a subclass of ``int`` in Python, so it must be
    rejected before the numeric check, or every ``True`` silently becomes ``1.0``.
    """
    if isinstance(value, bool):
        return None, Drop.BOOL, None
    if value is None:
        return None, Drop.NONE, None
    if isinstance(value, int | float):
        as_float = float(value)
        if not math.isfinite(as_float):
            return None, Drop.NONFINITE, None
        return as_float, None, None
    if isinstance(value, str):
        # Deliberately NOT parsed. "3" logged as a string is a string.
        return None, Drop.STRING, None
    if isinstance(value, Mapping):
        return None, Drop.MEDIA, media_type_of(value)
    if isinstance(value, list | tuple | set):
        return None, Drop.LIST, None
    return None, Drop.OTHER, None


def is_metric_value(value: Any) -> bool:
    """True if ``value`` would migrate as a metric."""
    metric, _, _ = as_metric(value)
    return metric is not None


# --------------------------------------------------------------------------- #
# Values -> params and tags
# --------------------------------------------------------------------------- #


def truncate(text: str, limit: int) -> tuple[str, bool]:
    """Cut ``text`` to ``limit`` characters, flagging the marker when it was cut."""
    if len(text) <= limit:
        return text, False
    if limit <= len(TRUNCATION_MARKER):
        return text[:limit], True
    return text[: limit - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER, True


def serialise(value: Any) -> str:
    """Render any value as a param/tag string.

    Strings pass through untouched so that a config value of ``"adam"`` becomes
    the param ``adam``, not ``"adam"``.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)


def as_param(value: Any, limits: Limits | None = None) -> tuple[str, bool]:
    """Coerce a value to a param string. Returns ``(value, was_truncated)``."""
    lim = limits or default_limits()
    return truncate(serialise(value), lim.max_param_val_length)


def as_tag(value: Any, limits: Limits | None = None) -> tuple[str, bool]:
    """Coerce a value to a tag string. Returns ``(value, was_truncated)``."""
    lim = limits or default_limits()
    return truncate(serialise(value), lim.max_tag_val_length)


# --------------------------------------------------------------------------- #
# Config flattening
# --------------------------------------------------------------------------- #


def flatten_config(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config dicts into dotted keys.

    Lists are left intact for :func:`serialise` to JSON-encode; indexing them into
    ``k.0``, ``k.1`` would produce unbounded param counts for long lists.
    Keys starting with ``_`` are W&B internals and are dropped at every level.
    """
    out: dict[str, Any] = {}
    for raw_key, value in config.items():
        key = str(raw_key)
        if key.startswith("_"):
            continue
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            out.update(flatten_config(value, prefix=f"{dotted}."))
        elif isinstance(value, Mapping):
            out[dotted] = {}
        else:
            out[dotted] = value
    return out


# --------------------------------------------------------------------------- #
# Key sanitisation
# --------------------------------------------------------------------------- #


def _fix_segments(key: str) -> str:
    """Make ``key`` path-canonical for MLflow's ``path_not_unique`` check.

    Done per segment rather than with ``posixpath.normpath`` because normpath
    would collapse ``a/../b`` to ``b`` and silently discard a path component.
    """
    segments = [_BAD_SEGMENTS.get(seg, seg) for seg in key.split("/")]
    return "/".join(segments)


def _fit(text: str, limit: int) -> str:
    """Make ``text`` path-canonical and no longer than ``limit``.

    Trimming and segment repair interact: cutting a key can leave a trailing
    ``/``, and repairing that would push the key back over the limit. Stripping
    trailing slashes before the second repair pass breaks that cycle, and the
    repair itself never lengthens a non-empty trailing segment (``.`` -> ``_``
    and ``..`` -> ``__`` are both length-preserving), so one pass suffices.
    """
    text = _fix_segments(text).strip()
    if len(text) <= limit:
        return text
    return _fix_segments(text[:limit].rstrip("/ ")).strip()


def sanitise_key(key: str, limits: Limits | None = None) -> str:
    """Deterministically map a source key onto one MLflow will accept.

    Collision handling is *not* done here — see :func:`sanitise_keys`.
    """
    lim = limits or default_limits()
    cleaned = _ALLOWED_KEY_CHARS.sub("_", key).strip()
    if not cleaned:
        return _FALLBACK_KEY
    return _fit(cleaned, lim.max_entity_key_length)


def _suffixed(base: str, original: str, limit: int) -> str:
    """``base``, shortened as needed, plus a short hash of ``original``.

    If the limit leaves no room for a stem the suffix still wins: a key that is
    slightly over length is a server-side error the user can see, whereas two
    distinct series silently merged into one is corruption they cannot.
    """
    digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:_COLLISION_HASH_LEN]
    suffix = f"_{digest}"
    room = max(1, limit - len(suffix))
    return _fit(base[:room], room) + suffix


def sanitise_keys(keys: Iterable[str], limits: Limits | None = None) -> dict[str, str]:
    """Sanitise a whole key set at once, resolving collisions deterministically.

    When several distinct source keys sanitise to the same target, **every** one
    of them — including the first seen — is suffixed with a short hash of its
    original name. Suffixing only the later arrivals would make the output depend
    on iteration order, which is exactly what a migration tool must not do.
    """
    lim = limits or default_limits()
    unique = list(dict.fromkeys(keys))
    naive = {key: sanitise_key(key, lim) for key in unique}

    grouped: dict[str, list[str]] = {}
    for key, target in naive.items():
        grouped.setdefault(target, []).append(key)

    resolved: dict[str, str] = {}
    for target, sources in grouped.items():
        if len(sources) == 1:
            resolved[sources[0]] = target
            continue
        logger.warning(
            "%d source keys sanitise to %r (%s); disambiguating with hash suffixes",
            len(sources),
            target,
            ", ".join(repr(s) for s in sorted(sources)),
        )
        for source in sources:
            resolved[source] = _suffixed(target, source, lim.max_entity_key_length)
    return resolved


def renamed(mapping: Mapping[str, str]) -> dict[str, str]:
    """The subset of a sanitisation map where the key actually changed."""
    return {src: dst for src, dst in mapping.items() if src != dst}
