"""Every rule in MAPPING.md sections 2 and 3, tested in both directions.

Where a rule is about what MLflow accepts, the assertion goes through MLflow's
own validator rather than restating the regex here — otherwise the test would
only prove this module is self-consistent.
"""

from __future__ import annotations

import math

import pytest
from mlflow.utils.validation import _validate_metric_name, _validate_param_name

from wandb_to_mlflow.coerce import (
    NONFINITE_SENTINELS,
    TRUNCATION_MARKER,
    Drop,
    DropReport,
    as_metric,
    as_param,
    as_tag,
    flatten_config,
    is_metric_value,
    media_type_of,
    renamed,
    sanitise_key,
    sanitise_keys,
    serialise,
    truncate,
)
from wandb_to_mlflow.limits import Limits, default_limits

LIMITS = default_limits()


def tiny_limits(**overrides: int) -> Limits:
    base = {
        "max_param_val_length": 20,
        "max_entity_key_length": 12,
        "max_tag_val_length": 20,
        "max_metrics_per_batch": 10,
        "max_params_tags_per_batch": 10,
        "max_entities_per_batch": 10,
    }
    base.update(overrides)
    return Limits(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 5.1 values -> metrics: accepted
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        (0, 0.0),
        (-7, -7.0),
        (1.5, 1.5),
        (-0.0, -0.0),
        (1e308, 1e308),
        (10**20, 1e20),  # int too large for float64 precision, still finite
    ],
)
def test_finite_numbers_become_metrics(value: object, expected: float) -> None:
    metric, reason, media = as_metric(value)
    assert metric == expected
    assert reason is None
    assert media is None
    assert is_metric_value(value)


# --------------------------------------------------------------------------- #
# 5.1 values -> metrics: rejected
# --------------------------------------------------------------------------- #


def test_bool_is_never_a_metric() -> None:
    """bool is a subclass of int; rejecting it late would invent 0/1 data."""
    for value in (True, False):
        metric, reason, _ = as_metric(value)
        assert metric is None
        assert reason is Drop.BOOL
        assert not is_metric_value(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_rejected(value: float) -> None:
    metric, reason, _ = as_metric(value)
    assert metric is None
    assert reason is Drop.NONFINITE


def test_none_rejected() -> None:
    assert as_metric(None) == (None, Drop.NONE, None)


@pytest.mark.parametrize("value", ["3", "3.5", "", "loss", "nan", "inf", "-infinity"])
def test_strings_are_never_parsed(value: str) -> None:
    """A scalar-looking string is still a string. Parsing it would fabricate data.

    The lowercase spellings stay here deliberately: only JSON's exact
    capitalisation is treated as a non-finite sentinel.
    """
    metric, reason, _ = as_metric(value)
    assert metric is None
    assert reason is Drop.STRING


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_wandb_nonfinite_sentinels_are_classified_as_nonfinite(value: str) -> None:
    """W&B returns non-finite numbers as these strings, measured against a real run.

    They are rejected either way -- this only files the rejection under the
    honest reason, so the drop report does not claim the user logged strings.
    """
    metric, reason, _ = as_metric(value)
    assert metric is None
    assert reason is Drop.NONFINITE


def test_the_sentinels_still_never_become_metrics() -> None:
    """Recognising a sentinel must not become a licence to parse it."""
    for value in NONFINITE_SENTINELS:
        assert not is_metric_value(value)


@pytest.mark.parametrize("value", [[1, 2], (1, 2), {1, 2}])
def test_sequences_rejected(value: object) -> None:
    metric, reason, _ = as_metric(value)
    assert metric is None
    assert reason is Drop.LIST


def test_media_rejected_and_type_reported() -> None:
    metric, reason, media = as_metric({"_type": "image-file", "path": "media/x.png"})
    assert metric is None
    assert reason is Drop.MEDIA
    assert media == "image-file"


def test_media_without_type_reported_as_unknown() -> None:
    _, reason, media = as_metric({"a": 1})
    assert reason is Drop.MEDIA
    assert media == "unknown"


def test_media_with_nonstring_type_reported_as_unknown() -> None:
    _, _, media = as_metric({"_type": 7})
    assert media == "unknown"


def test_media_type_of_non_mapping_is_none() -> None:
    assert media_type_of(3) is None


def test_unknown_object_rejected_as_other() -> None:
    metric, reason, _ = as_metric(object())
    assert metric is None
    assert reason is Drop.OTHER


# --------------------------------------------------------------------------- #
# DropReport
# --------------------------------------------------------------------------- #


def test_drop_report_tallies_and_merges() -> None:
    a = DropReport()
    a.record(Drop.NONFINITE)
    a.record(Drop.MEDIA, "image-file")
    a.record(Drop.MEDIA, "image-file")
    b = DropReport()
    b.record(Drop.MEDIA, "table-file")
    b.record(Drop.BOOL)
    a.merge(b)
    assert a.total == 5
    assert a.as_dict() == {
        "bool": 1,
        "media": 3,
        "nonfinite": 1,
        "media_types": {"image-file": 2, "table-file": 1},
    }


def test_drop_report_without_media_omits_media_types() -> None:
    report = DropReport()
    report.record(Drop.STRING)
    assert report.as_dict() == {"str": 1}


def test_sparse_nulls_are_tracked_but_not_reported_as_loss() -> None:
    """W&B pads sparse rows with nulls; a key not logged at a step is not loss.

    Measured on a real run: 25 rows logging an image every 5th epoch come back
    with 20 explicit nulls. Reporting those as dropped values would tell the
    user they lost 20 things they never had.
    """
    report = DropReport()
    for _ in range(20):
        report.record(Drop.NONE)
    assert report.padding == 20
    assert report.total == 0
    assert report.as_dict() == {}

    report.record(Drop.NONFINITE)
    assert report.total == 1
    assert report.as_dict() == {"nonfinite": 1}
    assert report.padding == 20


def test_padding_survives_a_merge() -> None:
    a, b = DropReport(), DropReport()
    a.record(Drop.NONE)
    b.record(Drop.NONE)
    b.record(Drop.BOOL)
    a.merge(b)
    assert (a.padding, a.total, a.as_dict()) == (2, 1, {"bool": 1})


def test_drop_report_media_defaults_to_unknown() -> None:
    report = DropReport()
    report.record(Drop.MEDIA)
    assert report.media["unknown"] == 1


# --------------------------------------------------------------------------- #
# 5.2 values -> params
# --------------------------------------------------------------------------- #


def test_strings_pass_through_unquoted() -> None:
    assert serialise("adam") == "adam"


def test_non_strings_are_json_serialised() -> None:
    assert serialise({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
    assert serialise([1, None, True]) == "[1, null, true]"
    assert serialise(3) == "3"


def test_unserialisable_falls_back_to_str() -> None:
    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    assert serialise(Weird()) == '"<weird>"'


def test_unicode_is_not_escaped() -> None:
    assert serialise(["héllo 🎉"]) == '["héllo 🎉"]'


def test_short_values_are_not_truncated() -> None:
    value, was_truncated = as_param("adam", LIMITS)
    assert value == "adam"
    assert was_truncated is False


def test_long_values_are_truncated_with_marker() -> None:
    value, was_truncated = as_param("x" * (LIMITS.max_param_val_length + 1), LIMITS)
    assert was_truncated is True
    assert len(value) == LIMITS.max_param_val_length
    assert value.endswith(TRUNCATION_MARKER)


def test_truncate_at_exactly_the_limit_is_untouched() -> None:
    text = "y" * LIMITS.max_param_val_length
    assert as_param(text, LIMITS) == (text, False)


def test_truncate_with_limit_below_marker_length_drops_the_marker() -> None:
    value, was_truncated = truncate("abcdef", 3)
    assert value == "abc"
    assert was_truncated is True


def test_tag_truncation_uses_the_tag_limit() -> None:
    value, was_truncated = as_tag("z" * (LIMITS.max_tag_val_length + 5), LIMITS)
    assert was_truncated is True
    assert len(value) == LIMITS.max_tag_val_length


def test_param_and_tag_default_to_process_limits() -> None:
    assert as_param("a") == ("a", False)
    assert as_tag("a") == ("a", False)


# --------------------------------------------------------------------------- #
# config flattening
# --------------------------------------------------------------------------- #


def test_nested_config_flattens_to_dotted_keys() -> None:
    flat = flatten_config({"opt": {"sgd": {"lr": 0.1, "momentum": 0.9}}, "epochs": 3})
    assert flat == {"opt.sgd.lr": 0.1, "opt.sgd.momentum": 0.9, "epochs": 3}


def test_lists_stay_whole() -> None:
    """Indexing lists into k.0, k.1 would make param count unbounded."""
    assert flatten_config({"layers": [1, 2, 3]}) == {"layers": [1, 2, 3]}


def test_empty_nested_dict_is_kept_as_a_leaf() -> None:
    assert flatten_config({"a": {}}) == {"a": {}}


def test_underscore_keys_dropped_at_every_level() -> None:
    flat = flatten_config({"_wandb": {"x": 1}, "a": {"_internal": 1, "b": 2}})
    assert flat == {"a.b": 2}


def test_non_string_config_keys_are_stringified() -> None:
    assert flatten_config({1: "a"}) == {"1": "a"}


# --------------------------------------------------------------------------- #
# 5.4 key sanitisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ["train/loss", "héllo", "a b", "a-b_c", "a.b", "épée/λοιπόν"])
def test_already_legal_keys_survive_unchanged(key: str) -> None:
    """Verified against MLflow's own validator, not against an assumed regex."""
    assert sanitise_key(key, LIMITS) == key
    _validate_metric_name(key)
    _validate_param_name(key)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("x@y!", "x_y_"),
        ("a\tb", "a_b"),
        ("a:b", "a_b"),  # legal on POSIX, illegal on Windows -> sanitised for portability
        ("a//b", "a/_/b"),
        ("../x", "__/x"),
        ("/abs", "_/abs"),
        ("trailing/", "trailing/_"),
        (".", "_"),
        ("a/./b", "a/_/b"),
    ],
)
def test_illegal_keys_are_repaired(key: str, expected: str) -> None:
    result = sanitise_key(key, LIMITS)
    assert result == expected
    _validate_metric_name(result)


def test_empty_key_becomes_a_placeholder() -> None:
    assert sanitise_key("", LIMITS) == "unnamed"
    assert sanitise_key("   ", LIMITS) == "unnamed"


def test_long_keys_are_truncated_to_the_limit() -> None:
    result = sanitise_key("k" * 300, LIMITS)
    assert len(result) == LIMITS.max_entity_key_length
    _validate_metric_name(result)


def test_truncation_that_creates_a_trailing_slash_is_repaired() -> None:
    limits = tiny_limits(max_entity_key_length=4)
    result = sanitise_key("abc/def", limits)
    assert result == "abc"
    _validate_metric_name(result)


def test_truncation_that_creates_a_dot_segment_is_repaired() -> None:
    result = sanitise_key("a/...x", tiny_limits(max_entity_key_length=4))
    assert result == "a/__"
    _validate_metric_name(result)


def test_aggressive_truncation_still_yields_a_valid_key() -> None:
    result = sanitise_key("///", tiny_limits(max_entity_key_length=1))
    assert result == "_"
    _validate_metric_name(result)


def test_sanitisation_output_is_always_accepted_by_mlflow() -> None:
    hostile = ["x@y!", "", "..", "a//b", "k" * 400, "%$#", "🎉", "a\nb", "/", "./."]
    for key in hostile:
        _validate_metric_name(sanitise_key(key, LIMITS))


# --------------------------------------------------------------------------- #
# collisions
# --------------------------------------------------------------------------- #


def test_non_colliding_keys_keep_their_plain_target() -> None:
    assert sanitise_keys(["a@b", "c"], LIMITS) == {"a@b": "a_b", "c": "c"}


def test_colliding_keys_all_get_a_hash_suffix() -> None:
    """Including the first one seen -- otherwise the result is order-dependent."""
    mapping = sanitise_keys(["a@b", "a#b"], LIMITS)
    assert len(set(mapping.values())) == 2
    assert all(v.startswith("a_b_") for v in mapping.values())
    for target in mapping.values():
        _validate_metric_name(target)


def test_collision_resolution_is_order_independent() -> None:
    keys = ["a@b", "a#b", "a b", "z"]
    assert sanitise_keys(keys, LIMITS) == sanitise_keys(list(reversed(keys)), LIMITS)


def test_duplicate_source_keys_are_not_treated_as_a_collision() -> None:
    assert sanitise_keys(["a@b", "a@b"], LIMITS) == {"a@b": "a_b"}


def test_collision_suffix_respects_the_key_length_limit() -> None:
    limits = tiny_limits(max_entity_key_length=12)
    mapping = sanitise_keys(["x" * 40 + "@", "x" * 40 + "#"], limits)
    assert len(set(mapping.values())) == 2
    for target in mapping.values():
        assert len(target) <= limits.max_entity_key_length
        _validate_metric_name(target)


def test_collision_suffix_survives_a_pathological_limit() -> None:
    """The trim loop must terminate even when there is no room for a stem."""
    limits = tiny_limits(max_entity_key_length=3)
    mapping = sanitise_keys(["a/@", "a/#"], limits)
    assert len(set(mapping.values())) == 2
    for target in mapping.values():
        _validate_metric_name(target)


def test_collision_suffix_wins_over_the_length_limit_when_they_conflict() -> None:
    """An over-length key is a visible server error; a silent merge is corruption."""
    limits = tiny_limits(max_entity_key_length=2)
    mapping = sanitise_keys(["a//@", "a//#"], limits)
    assert len(set(mapping.values())) == 2


def test_renamed_reports_only_changed_keys() -> None:
    assert renamed({"a": "a", "b@": "b_"}) == {"b@": "b_"}


# --------------------------------------------------------------------------- #
# limits fallbacks
# --------------------------------------------------------------------------- #


def test_nan_is_not_equal_to_itself_sanity() -> None:
    """Guards the test above: NaN rejection must not rely on equality."""
    assert math.isnan(float("nan"))
