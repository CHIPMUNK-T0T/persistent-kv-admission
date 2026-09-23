"""Reference checks, aggregation, and fixed figures for on-policy runs.

The helpers in this module deliberately do not run an experiment.  With the
exception of :func:`write_csv_rows` and the three ``plot_*`` functions, they
are pure: inputs are not mutated and no files are written.  Reference checks
raise :class:`ReferenceMismatchError` on the first invalid publication gate.

Candidate-log hashes need special care.  Phase 0.97 did not publish hashes for
the individual ``.npz`` candidate logs.  :func:`verify_candidate_log_metadata`
therefore calls a hash computed now ``contemporaneous`` and never promotes it
to historical evidence.  Only a separately supplied, published hash can make
``historical_hash_verified`` true.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Hashable, Iterable, Mapping, Sequence


Row = Mapping[str, Any]


PHASE097_COEFFICIENT_FIELDS = (
    "log_frequency",
    "neg_log_recency",
    "log_prefix_tokens",
    "log_age",
    "log_reuse_rate",
    "single_occurrence",
    "log_mean_interval",
    "log_median_interval",
    "log_std_interval",
    "interval_cv",
    "log_interval_lag1",
    "log_interval_lag2",
    "log_interval_lag3",
    "log_interval_lag4",
    "log_freq_10s",
    "log_freq_60s",
    "log_freq_300s",
    "log_freq_600s",
    "recent_over_lifetime",
    "interval_trend",
    "burstiness",
    "log_fan_out",
    "log_branch_diversity",
)

PI0_REPLAY_KEY_FIELDS = (
    "trace",
    "l1_fraction",
    "l2_multiplier",
    "target",
    "seed",
)

PI0_REPLAY_TEXT_FIELDS = (
    "l1_policy",
    "l2_policy",
    "hit_model",
    "closure",
    "offline_tiebreak",
    "l2_eviction",
)

# Timing, byte-seconds, and derived floating-point summaries are intentionally
# excluded.  These are the capacities, request/L1 counters, and integer replay
# outcomes whose bit-identical reproduction is the Phase 0.97 publication gate.
PI0_REPLAY_EXACT_FIELDS = (
    "l1_capacity_bytes",
    "l2_capacity_bytes",
    "measured_requests",
    "requested_tokens",
    "requested_blocks",
    "avoided_prefill_tokens",
    "l1_avoided_tokens",
    "l2_avoided_tokens",
    "l1_hit_blocks",
    "l2_hit_blocks",
    "l2_present_unusable_blocks",
    "l2_present_unusable_tokens",
    "l1_evictions",
    "l2_admissions",
    "l2_rejections",
    "l2_evictions",
    "l2_ancestor_copy_bytes",
    "l2_already_held",
    "l2_sample_width",
    "l2_seed",
    "l2_decisions",
)

GENERIC_ARMS = frozenset(
    {
        "lru",
        "lfu",
        "lru_2hit",
        "2hit",
        "two_hit",
        "lru_s",
        "lfu_s",
        "lru_2hit_s",
        "2hit_s",
        "two_hit_s",
    }
)

ATTRIBUTION_CATEGORIES = ("rejected", "evicted", "compulsory", "unexplained")
ATTRIBUTION_DECISION_CATEGORIES = ("rejected", "evicted")

_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


class ReferenceMismatchError(AssertionError):
    """A publication reference or completeness barrier did not match."""


@dataclass(frozen=True)
class NumericMismatch:
    """One failed field comparison."""

    field: str
    actual: Any
    expected: Any
    reason: str


@dataclass(frozen=True)
class VerificationSummary:
    """Summary returned after an exact/tolerant row reference check."""

    name: str
    row_count: int
    key_fields: tuple[str, ...]
    exact_fields: tuple[str, ...]
    tolerance_fields: tuple[str, ...]
    text_fields: tuple[str, ...]


@dataclass(frozen=True)
class CandidateLogVerification:
    """Metadata result with historical and contemporaneous hashes separated."""

    metadata_rows: int
    identity_fields: tuple[str, ...]
    count_fields: tuple[str, ...]
    contemporaneous_sha256: str | None
    published_historical_sha256: str | None
    historical_hash_verified: bool
    hash_status: str

    @property
    def publication_hash_gate_passed(self) -> bool:
        """Whether an independently published historical hash was matched."""

        return self.historical_hash_verified


@dataclass(frozen=True)
class MatrixValidation:
    """Completeness result for fixed policy-population cross scoring."""

    groups: int
    rows: int
    iterations: tuple[int, ...]


@dataclass(frozen=True)
class BarrierValidation:
    """Result of the pre-held-out training/model serialization barrier."""

    training_identities: int
    expected_training_identities: int
    model_identities: int
    expected_model_identities: int
    pi0_models: int
    updated_models: int


@dataclass(frozen=True)
class GenericWinner:
    """One generic winner selected from its published per-cell seed mean."""

    group: tuple[Hashable, ...]
    mechanism: str
    arm: str
    mean_avoided_prefill_tokens: float
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class RegisteredOutcome:
    """One pre-registered A/B/C/D/unresolved classification."""

    label: str
    target: str
    utility_pass: bool
    heldout_ranking_pass: bool
    fit_population_ranking_pass: bool
    own_policy_direction_all_positive: bool | None
    strong_support: bool
    reasons: tuple[str, ...]

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestComparison:
    """Path-level difference between start and end manifests."""

    equal: bool
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]


def stable_field_union(
    rows: Iterable[Row], preferred: Sequence[str] = ()
) -> tuple[str, ...]:
    """Return a stable CSV field union without changing any input row.

    ``preferred`` fields come first in the supplied order.  Other fields keep
    first-appearance order across rows and each row's mapping order.
    """

    fields: list[str] = []
    seen: set[str] = set()
    for field in preferred:
        if field not in seen:
            fields.append(str(field))
            seen.add(str(field))
    for row in rows:
        for field in row:
            name = str(field)
            if name not in seen:
                fields.append(name)
                seen.add(name)
    return tuple(fields)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a headered CSV, rejecting missing or duplicate field names."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"CSV has duplicate header fields: {path}")
        return [dict(row) for row in reader]


def write_csv_rows(
    path: str | Path,
    rows: Iterable[Row],
    *,
    fieldnames: Sequence[str] | None = None,
    preferred_fields: Sequence[str] = (),
) -> tuple[str, ...]:
    """Write rows with a stable union header and return that header.

    This is an explicit write helper.  Rows are materialized as shallow copies
    so neither the iterable's mappings nor their key order is modified.
    """

    materialized = [dict(row) for row in rows]
    fields = (
        tuple(str(field) for field in fieldnames)
        if fieldnames is not None
        else stable_field_union(materialized, preferred_fields)
    )
    if not fields:
        raise ValueError("cannot write a CSV without fields")
    unknown = sorted({key for row in materialized for key in row} - set(fields))
    if unknown:
        raise ValueError(f"rows contain fields outside fieldnames: {unknown}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(materialized)
    return fields


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of the current file bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise InvalidOperation
    return Decimal(str(value).strip())


def _canonical(value: Any) -> Hashable:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "true":
            return True
        if stripped.lower() == "false":
            return False
        try:
            number = Decimal(stripped)
        except InvalidOperation:
            return stripped
        if number.is_nan():
            return "NaN"
        return str(number.normalize())
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    if number.is_nan():
        return "NaN"
    return str(number.normalize())


def _numeric_exact_equal(actual: Any, expected: Any) -> bool:
    try:
        left = _decimal(actual)
        right = _decimal(expected)
    except (InvalidOperation, ValueError):
        return False
    if left is None or right is None:
        return left is right
    if left.is_nan() or right.is_nan():
        return left.is_nan() and right.is_nan()
    return left == right


def compare_numeric_fields(
    actual: Row,
    expected: Row,
    *,
    exact_fields: Sequence[str] = (),
    tolerance_fields: Mapping[str, float | tuple[float, float]] | None = None,
    equal_nan: bool = True,
) -> tuple[NumericMismatch, ...]:
    """Compare numeric fields exactly or with ``(abs_tol, rel_tol)``.

    Exact comparison uses :class:`~decimal.Decimal`, so integer counters never
    pass through binary floating point.  A scalar tolerance means absolute
    tolerance with zero relative tolerance.
    """

    mismatches: list[NumericMismatch] = []
    tolerances = tolerance_fields or {}
    for field in exact_fields:
        if field not in actual or field not in expected:
            mismatches.append(
                NumericMismatch(field, actual.get(field), expected.get(field), "missing")
            )
        elif not _numeric_exact_equal(actual[field], expected[field]):
            mismatches.append(
                NumericMismatch(field, actual[field], expected[field], "not numerically exact")
            )
    for field, tolerance in tolerances.items():
        if field not in actual or field not in expected:
            mismatches.append(
                NumericMismatch(field, actual.get(field), expected.get(field), "missing")
            )
            continue
        if isinstance(tolerance, tuple):
            abs_tol, rel_tol = map(float, tolerance)
        else:
            abs_tol, rel_tol = float(tolerance), 0.0
        try:
            left = float(actual[field])
            right = float(expected[field])
        except (TypeError, ValueError):
            mismatches.append(
                NumericMismatch(field, actual[field], expected[field], "not numeric")
            )
            continue
        if math.isnan(left) or math.isnan(right):
            matches = equal_nan and math.isnan(left) and math.isnan(right)
        else:
            matches = math.isclose(left, right, abs_tol=abs_tol, rel_tol=rel_tol)
        if not matches:
            mismatches.append(
                NumericMismatch(
                    field,
                    actual[field],
                    expected[field],
                    f"outside abs_tol={abs_tol:g}, rel_tol={rel_tol:g}",
                )
            )
    return tuple(mismatches)


def assert_numeric_fields(
    actual: Row,
    expected: Row,
    *,
    exact_fields: Sequence[str] = (),
    tolerance_fields: Mapping[str, float | tuple[float, float]] | None = None,
    context: str = "numeric row",
) -> None:
    """Raise :class:`ReferenceMismatchError` for failed numeric fields."""

    mismatches = compare_numeric_fields(
        actual,
        expected,
        exact_fields=exact_fields,
        tolerance_fields=tolerance_fields,
    )
    if mismatches:
        detail = "; ".join(
            f"{item.field}: {item.actual!r} != {item.expected!r} ({item.reason})"
            for item in mismatches[:12]
        )
        raise ReferenceMismatchError(f"{context}: {detail}")


def _row_key(row: Row, fields: Sequence[str]) -> tuple[Hashable, ...]:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ReferenceMismatchError(f"row is missing key fields {missing}: {row!r}")
    return tuple(_canonical(row[field]) for field in fields)


def _index_unique(rows: Iterable[Row], fields: Sequence[str], name: str) -> dict[tuple, Row]:
    indexed: dict[tuple, Row] = {}
    for row in rows:
        key = _row_key(row, fields)
        if key in indexed:
            raise ReferenceMismatchError(f"{name}: duplicate identity {key!r}")
        indexed[key] = row
    return indexed


def verify_reference_rows(
    actual_rows: Iterable[Row],
    expected_rows: Iterable[Row],
    *,
    key_fields: Sequence[str],
    exact_fields: Sequence[str] = (),
    tolerance_fields: Mapping[str, float | tuple[float, float]] | None = None,
    text_fields: Sequence[str] = (),
    expected_count: int | None = None,
    name: str = "reference rows",
) -> VerificationSummary:
    """Verify unique row identities plus exact, tolerant, and text fields."""

    actual = _index_unique(actual_rows, key_fields, f"{name} actual")
    expected = _index_unique(expected_rows, key_fields, f"{name} expected")
    if expected_count is not None and len(expected) != expected_count:
        raise ReferenceMismatchError(
            f"{name}: expected reference count {expected_count}, found {len(expected)}"
        )
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual), key=repr)
        extra = sorted(set(actual) - set(expected), key=repr)
        raise ReferenceMismatchError(
            f"{name}: identity mismatch; missing={missing[:8]!r}, extra={extra[:8]!r}"
        )
    tolerances = tolerance_fields or {}
    for key in sorted(expected, key=repr):
        left, right = actual[key], expected[key]
        assert_numeric_fields(
            left,
            right,
            exact_fields=exact_fields,
            tolerance_fields=tolerances,
            context=f"{name} {key!r}",
        )
        for field in text_fields:
            if field not in left or field not in right:
                raise ReferenceMismatchError(
                    f"{name} {key!r}: missing text field {field!r}"
                )
            if _canonical(left[field]) != _canonical(right[field]):
                raise ReferenceMismatchError(
                    f"{name} {key!r}: {field} {left[field]!r} != {right[field]!r}"
                )
    return VerificationSummary(
        name=name,
        row_count=len(expected),
        key_fields=tuple(key_fields),
        exact_fields=tuple(exact_fields),
        tolerance_fields=tuple(tolerances),
        text_fields=tuple(text_fields),
    )


def phase097_pi0_fit_rows(
    rows: Iterable[Row], *, population_field: str = "population", pi0_name: str = "A_none"
) -> list[dict[str, Any]]:
    """Copy the four published ``A_none`` fit rows from a mixed fit table."""

    return [dict(row) for row in rows if str(row.get(population_field, "")) == pi0_name]


def verify_phase097_pi0_fits(
    reconstructed_rows: Iterable[Row],
    published_rows: Iterable[Row],
    *,
    coefficient_fields: Sequence[str] = PHASE097_COEFFICIENT_FIELDS,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-12,
    traces: Sequence[str] = ("conversation_trace", "toolagent_trace"),
    targets: Sequence[str] = ("binary", "next_use"),
    expected_count: int = 4,
) -> VerificationSummary:
    """Verify reconstructed pi0 rows against the published Phase 0.97 fits.

    Inputs may be full fit tables or already-filtered pi0 rows.  The gate checks
    row/positive counts, fit diagnostics, ranker standardization, horizon and
    split, and all 23 standardized coefficients.  Wall-clock seconds are not a
    reproducibility field.
    """

    reconstructed = [dict(row) for row in reconstructed_rows]
    published = list(published_rows)
    for row in reconstructed:
        nested = row.get("coefficient_row")
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise ReferenceMismatchError("pi0 coefficient_row must be a mapping")
            for field in coefficient_fields:
                if field in row and _canonical(row[field]) != _canonical(nested.get(field)):
                    raise ReferenceMismatchError(
                        f"pi0 row has conflicting flattened and coefficient_row value for {field}"
                    )
                if field in nested:
                    row[field] = nested[field]
    if any("population" in row for row in reconstructed):
        reconstructed = phase097_pi0_fit_rows(reconstructed)
    if any("population" in row for row in published):
        published = phase097_pi0_fit_rows(published)
    allowed_traces = {str(value) for value in traces}
    allowed_targets = {str(value) for value in targets}
    reconstructed = [
        row for row in reconstructed
        if str(row.get("trace", "")) in allowed_traces
        and str(row.get("target", "")) in allowed_targets
    ]
    published = [
        row for row in published
        if str(row.get("trace", "")) in allowed_traces
        and str(row.get("target", "")) in allowed_targets
    ]
    tolerant = {
        "horizon_seconds": (abs_tol, rel_tol),
        "split_ms": (abs_tol, rel_tol),
        "fit_condition_number": (abs_tol, rel_tol),
        **{field: (abs_tol, rel_tol) for field in coefficient_fields},
    }
    return verify_reference_rows(
        reconstructed,
        published,
        key_fields=("trace", "target"),
        exact_fields=("train_rows", "train_positives", "fit_iterations"),
        tolerance_fields=tolerant,
        text_fields=("standardization", "ranker_standardize", "fit_converged"),
        expected_count=expected_count,
        name="Phase 0.97 pi0 fits",
    )


def _select_pi0_replay_rows(rows: Iterable[Row]) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    if any("iteration" in row for row in materialized):
        return [row for row in materialized if _canonical(row.get("iteration")) == "0"]
    if any("arm" in row for row in materialized):
        selected = [
            row for row in materialized if str(row.get("arm", "")) in {"A_none", "pi0"}
        ]
        return selected or materialized
    return materialized


def verify_published_pi0_replay(
    replayed_rows: Iterable[Row],
    published_rows: Iterable[Row],
    *,
    key_fields: Sequence[str] = PI0_REPLAY_KEY_FIELDS,
    exact_fields: Sequence[str] = PI0_REPLAY_EXACT_FIELDS,
    text_fields: Sequence[str] = PI0_REPLAY_TEXT_FIELDS,
    expected_count: int = 120,
) -> VerificationSummary:
    """Verify all pi0 integer/capacity/request/L1 Phase 0.97 outcomes exactly."""

    return verify_reference_rows(
        _select_pi0_replay_rows(replayed_rows),
        _select_pi0_replay_rows(published_rows),
        key_fields=key_fields,
        exact_fields=exact_fields,
        text_fields=text_fields,
        expected_count=expected_count,
        name="published Phase 0.97 pi0 replay",
    )


def verify_candidate_log_metadata(
    actual_rows: Iterable[Row],
    published_metadata_rows: Iterable[Row],
    *,
    identity_fields: Sequence[str] = (
        "trace",
        "l1_fraction",
        "l2_multiplier",
        "behaviour_policy",
    ),
    count_fields: Sequence[str] = (
        "decisions_seen",
        "decisions_kept",
        "rows",
        "l2_decisions",
        "l2_admissions",
        "l2_rejections",
        "l2_evictions",
    ),
    context_fields: Sequence[str] = (),
    artifact_path: str | Path | None = None,
    contemporaneous_sha256: str | None = None,
    published_historical_sha256: str | None = None,
    require_historical_hash: bool = False,
) -> CandidateLogVerification:
    """Verify candidate-log identity/count metadata and report hash provenance.

    ``contemporaneous_sha256`` means a digest observed during the current run.
    It is checked against ``artifact_path`` if both are supplied.  It is never
    used as a substitute for ``published_historical_sha256``.
    """

    actual = [dict(row) for row in actual_rows]
    published = [dict(row) for row in published_metadata_rows]
    verify_reference_rows(
        actual,
        published,
        key_fields=identity_fields,
        exact_fields=count_fields,
        text_fields=context_fields,
        name="Phase 0.97 candidate-log metadata",
    )
    observed = contemporaneous_sha256.lower() if contemporaneous_sha256 else None
    if artifact_path is not None:
        path_hash = sha256_file(artifact_path)
        if observed is not None and observed != path_hash:
            raise ReferenceMismatchError(
                f"candidate log: supplied current hash {observed} != file hash {path_hash}"
            )
        observed = path_hash
    historical = (
        published_historical_sha256.lower() if published_historical_sha256 else None
    )
    if historical is None:
        historical_verified = False
        status = "contemporaneous_only" if observed is not None else "no_hash_evidence"
    elif observed is None:
        historical_verified = False
        status = "historical_hash_not_observed"
    elif observed != historical:
        raise ReferenceMismatchError(
            f"candidate log: current hash {observed} != published historical hash {historical}"
        )
    else:
        historical_verified = True
        status = "published_historical_match"
    if require_historical_hash and not historical_verified:
        raise ReferenceMismatchError(
            "candidate log has no verified published historical hash; a hash computed "
            "during this run is only contemporaneous evidence"
        )
    actual_count = len(_index_unique(actual, identity_fields, "candidate metadata"))
    return CandidateLogVerification(
        metadata_rows=actual_count,
        identity_fields=tuple(identity_fields),
        count_fields=tuple(count_fields),
        contemporaneous_sha256=observed,
        published_historical_sha256=historical,
        historical_hash_verified=historical_verified,
        hash_status=status,
    )


def validate_4x4_matrix(
    rows: Iterable[Row],
    *,
    group_fields: Sequence[str] = ("trace", "cell", "target", "seed", "split"),
    population_iteration_field: str = "population_iteration",
    model_iteration_field: str = "model_iteration",
    iterations: Sequence[int] = (0, 1, 2, 3),
    expected_groups: Iterable[Sequence[Any] | Row] | None = None,
    expected_group_count: int | None = None,
) -> MatrixValidation:
    """Require every group to contain each policy-population/model pair once."""

    iteration_tuple = tuple(int(value) for value in iterations)
    if len(iteration_tuple) != 4 or len(set(iteration_tuple)) != 4:
        raise ValueError("a cross-score matrix needs four distinct iterations")
    required = set(itertools.product(iteration_tuple, repeat=2))
    observed: dict[tuple[Hashable, ...], set[tuple[int, int]]] = defaultdict(set)
    row_count = 0
    for row in rows:
        group = _row_key(row, group_fields)
        try:
            pair = (int(row[population_iteration_field]), int(row[model_iteration_field]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ReferenceMismatchError(f"4x4 matrix {group!r}: invalid iteration fields") from exc
        if pair not in required:
            raise ReferenceMismatchError(
                f"4x4 matrix {group!r}: iteration pair {pair!r} is outside {iteration_tuple!r}"
            )
        if pair in observed[group]:
            raise ReferenceMismatchError(f"4x4 matrix {group!r}: duplicate entry {pair!r}")
        observed[group].add(pair)
        row_count += 1
    for group, pairs in observed.items():
        if pairs != required:
            raise ReferenceMismatchError(
                f"4x4 matrix {group!r}: missing entries {sorted(required - pairs)!r}"
            )
    if expected_groups is not None:
        canonical_expected: set[tuple[Hashable, ...]] = set()
        for value in expected_groups:
            if isinstance(value, Mapping):
                canonical_expected.add(_row_key(value, group_fields))
            else:
                sequence = tuple(value)
                if len(sequence) != len(group_fields):
                    raise ValueError("expected group has the wrong arity")
                canonical_expected.add(tuple(_canonical(item) for item in sequence))
        if set(observed) != canonical_expected:
            missing = sorted(canonical_expected - set(observed), key=repr)
            extra = sorted(set(observed) - canonical_expected, key=repr)
            raise ReferenceMismatchError(
                f"4x4 matrix group mismatch; missing={missing[:8]!r}, extra={extra[:8]!r}"
            )
    if expected_group_count is not None and len(observed) != expected_group_count:
        raise ReferenceMismatchError(
            f"4x4 matrix groups {len(observed)} != expected {expected_group_count}"
        )
    return MatrixValidation(len(observed), row_count, iteration_tuple)


def _cell_key(row: Row, cell_fields: Sequence[str]) -> tuple[Hashable, ...]:
    return _row_key(row, cell_fields)


def _expected_cell_key(cell: Any, cell_fields: Sequence[str]) -> tuple[Hashable, ...]:
    if len(cell_fields) == 1:
        return (_canonical(cell),)
    if isinstance(cell, Mapping):
        return _row_key(cell, cell_fields)
    try:
        values = tuple(cell)
    except TypeError as exc:
        raise ValueError("multi-field cells must be sequences or mappings") from exc
    if len(values) != len(cell_fields):
        raise ValueError(f"cell {cell!r} has the wrong arity for {cell_fields!r}")
    return tuple(_canonical(value) for value in values)


def _require_sha256(row: Row, field: str, context: str) -> None:
    value = str(row.get(field, "")).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ReferenceMismatchError(f"{context}: {field} is not a SHA-256 digest")


def _verify_artifact_path(
    row: Row,
    path_field: str,
    hash_field: str,
    artifact_root: str | Path | None,
    context: str,
) -> None:
    raw_path = row.get(path_field)
    if raw_path is None or str(raw_path) == "":
        raise ReferenceMismatchError(f"{context}: missing artifact path field {path_field!r}")
    path = Path(str(raw_path))
    if not path.is_absolute() and artifact_root is not None:
        path = Path(artifact_root) / path
    if not path.is_file():
        raise ReferenceMismatchError(f"{context}: artifact does not exist: {path}")
    expected = str(row[hash_field]).lower()
    observed = sha256_file(path)
    if observed != expected:
        raise ReferenceMismatchError(
            f"{context}: artifact hash {observed} != manifest hash {expected}"
        )


def validate_training_model_barrier(
    training_population_rows: Iterable[Row],
    model_rows: Iterable[Row],
    *,
    traces: Sequence[Any],
    cells: Sequence[Any],
    targets: Sequence[Any],
    seeds: Sequence[int],
    iterations: Sequence[int] = (0, 1, 2, 3),
    trace_field: str = "trace",
    cell_fields: Sequence[str] = ("cell",),
    target_field: str = "target",
    seed_field: str = "seed",
    iteration_field: str = "iteration",
    population_hash_field: str | None = "artifact_sha256",
    model_hash_field: str = "model_sha256",
    population_path_field: str | None = None,
    model_path_field: str | None = None,
    artifact_root: str | Path | None = None,
) -> BarrierValidation:
    """Validate every training population and frozen serialized model.

    ``pi0`` is unique per ``(trace, target)`` and shared by cells/seeds.  Each
    update model ``pi1..pi3`` is unique per lineage.  The full axes require 480
    training identities and 364 model identities; a one-value smoke requires
    four and four.  Hash fields must contain 64-character SHA-256 strings.
    """

    iteration_values = tuple(int(value) for value in iterations)
    if iteration_values != (0, 1, 2, 3):
        raise ValueError("the registered barrier iterations are exactly (0, 1, 2, 3)")
    trace_values = tuple(_canonical(value) for value in traces)
    cell_values = tuple(_expected_cell_key(value, cell_fields) for value in cells)
    target_values = tuple(_canonical(value) for value in targets)
    seed_values = tuple(int(value) for value in seeds)
    axes = (trace_values, cell_values, target_values, seed_values)
    if any(not axis for axis in axes):
        raise ValueError("barrier axes must be non-empty")
    if any(len(axis) != len(set(axis)) for axis in axes):
        raise ValueError("barrier axes contain duplicates")

    expected_training = {
        (trace, cell, target, seed, iteration)
        for trace, cell, target, seed, iteration in itertools.product(
            trace_values, cell_values, target_values, seed_values, iteration_values
        )
    }
    observed_training: set[tuple[Any, ...]] = set()
    for row in training_population_rows:
        try:
            identity = (
                _canonical(row[trace_field]),
                _cell_key(row, cell_fields),
                _canonical(row[target_field]),
                int(row[seed_field]),
                int(row[iteration_field]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReferenceMismatchError("training population has an invalid identity") from exc
        if identity in observed_training:
            raise ReferenceMismatchError(f"duplicate training population {identity!r}")
        if population_hash_field is not None:
            _require_sha256(row, population_hash_field, f"training population {identity!r}")
            if population_path_field is not None:
                _verify_artifact_path(
                    row,
                    population_path_field,
                    population_hash_field,
                    artifact_root,
                    f"training population {identity!r}",
                )
        observed_training.add(identity)
    if observed_training != expected_training:
        missing = sorted(expected_training - observed_training, key=repr)
        extra = sorted(observed_training - expected_training, key=repr)
        raise ReferenceMismatchError(
            f"training barrier mismatch; missing={missing[:8]!r}, extra={extra[:8]!r}"
        )

    expected_models: set[tuple[Any, ...]] = {
        (trace, None, target, None, 0)
        for trace, target in itertools.product(trace_values, target_values)
    }
    expected_models.update(
        (trace, cell, target, seed, iteration)
        for trace, cell, target, seed, iteration in itertools.product(
            trace_values, cell_values, target_values, seed_values, iteration_values[1:]
        )
    )
    observed_models: set[tuple[Any, ...]] = set()
    for row in model_rows:
        try:
            iteration = int(row[iteration_field])
            if iteration == 0:
                identity = (
                    _canonical(row[trace_field]), None,
                    _canonical(row[target_field]), None, 0,
                )
            else:
                identity = (
                    _canonical(row[trace_field]),
                    _cell_key(row, cell_fields),
                    _canonical(row[target_field]),
                    int(row[seed_field]), iteration,
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReferenceMismatchError("model manifest has an invalid identity") from exc
        if identity in observed_models:
            raise ReferenceMismatchError(f"duplicate model identity {identity!r}")
        _require_sha256(row, model_hash_field, f"model {identity!r}")
        if model_path_field is not None:
            _verify_artifact_path(
                row,
                model_path_field,
                model_hash_field,
                artifact_root,
                f"model {identity!r}",
            )
        observed_models.add(identity)
    if observed_models != expected_models:
        missing = sorted(expected_models - observed_models, key=repr)
        extra = sorted(observed_models - expected_models, key=repr)
        raise ReferenceMismatchError(
            f"model barrier mismatch; missing={missing[:8]!r}, extra={extra[:8]!r}"
        )
    pi0_models = len(trace_values) * len(target_values)
    expected_training_count = math.prod(map(len, axes)) * len(iteration_values)
    expected_model_count = pi0_models + math.prod(map(len, axes)) * 3
    return BarrierValidation(
        training_identities=len(observed_training),
        expected_training_identities=expected_training_count,
        model_identities=len(observed_models),
        expected_model_identities=expected_model_count,
        pi0_models=pi0_models,
        updated_models=expected_model_count - pi0_models,
    )


def select_generic_winners(
    published_rows: Iterable[Row],
    *,
    group_fields: Sequence[str] = ("trace", "l1_fraction", "l2_multiplier"),
    mechanism_field: str = "kind",
    arm_field: str = "arm",
    seed_field: str = "seed",
    value_field: str = "avoided_prefill_tokens",
    generic_arms: Iterable[str] = GENERIC_ARMS,
    mechanisms: Sequence[str] = ("heap", "sampled"),
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> list[GenericWinner]:
    """Choose generic winners by published five-seed cell mean.

    A winner is selected once per cell/mechanism and reused for every seed.
    Ties use lexical arm order, never a per-seed result.
    """

    allowed = {str(arm) for arm in generic_arms}
    mechanism_set = {str(value) for value in mechanisms}
    seed_set = {int(seed) for seed in expected_seeds}
    values: dict[tuple[tuple, str, str], dict[int, float]] = defaultdict(dict)
    groups: set[tuple] = set()
    for row in published_rows:
        mechanism = str(row.get(mechanism_field, ""))
        arm = str(row.get(arm_field, ""))
        if mechanism not in mechanism_set or arm not in allowed:
            continue
        group = _row_key(row, group_fields)
        groups.add(group)
        try:
            seed = int(row[seed_field])
            value = float(row[value_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReferenceMismatchError(
                f"generic reference {group!r}/{mechanism}/{arm}: invalid seed/value"
            ) from exc
        bucket = values[(group, mechanism, arm)]
        if seed in bucket:
            raise ReferenceMismatchError(
                f"generic reference {group!r}/{mechanism}/{arm}: duplicate seed {seed}"
            )
        bucket[seed] = value
    winners: list[GenericWinner] = []
    for group in sorted(groups, key=repr):
        for mechanism in mechanisms:
            candidates: list[tuple[float, str]] = []
            for (candidate_group, candidate_mechanism, arm), by_seed in values.items():
                if candidate_group != group or candidate_mechanism != mechanism:
                    continue
                if set(by_seed) != seed_set:
                    raise ReferenceMismatchError(
                        f"generic reference {group!r}/{mechanism}/{arm}: "
                        f"seeds {sorted(by_seed)} != {sorted(seed_set)}"
                    )
                candidates.append((statistics.fmean(by_seed.values()), arm))
            if not candidates:
                raise ReferenceMismatchError(
                    f"generic reference {group!r}: no eligible {mechanism} arm"
                )
            mean_value, winner = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
            winners.append(
                GenericWinner(group, str(mechanism), winner, float(mean_value), tuple(sorted(seed_set)))
            )
    return winners


def _reference_index(
    rows: Iterable[Row],
    group_fields: Sequence[str],
    arm_field: str,
    seed_field: str,
    allowed_arms: set[str] | None = None,
) -> dict[tuple[tuple, str, int], Row]:
    result: dict[tuple[tuple, str, int], Row] = {}
    for row in rows:
        if allowed_arms is not None and str(row.get(arm_field, "")) not in allowed_arms:
            continue
        try:
            key = (_row_key(row, group_fields), str(row[arm_field]), int(row[seed_field]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ReferenceMismatchError("utility reference has an invalid identity") from exc
        if key in result:
            raise ReferenceMismatchError(f"duplicate utility reference {key!r}")
        result[key] = row
    return result


def _reference_for_seed(
    references: Mapping[tuple[tuple, str, int], Row],
    group: tuple,
    arm: str,
    seed: int,
    *,
    allow_deterministic_fallback: bool = False,
) -> Row:
    exact = references.get((group, arm, seed))
    if exact is not None:
        return exact
    candidates = [
        row
        for (candidate_group, candidate_arm, _), row in references.items()
        if candidate_group == group and candidate_arm == arm
    ]
    if not candidates:
        raise ReferenceMismatchError(f"missing utility reference {group!r}/{arm}/seed={seed}")
    if not allow_deterministic_fallback:
        raise ReferenceMismatchError(
            f"missing same-seed utility reference {group!r}/{arm}/seed={seed}"
        )
    canonical = {
        (_canonical(row.get("avoided_prefill_tokens")), _canonical(row.get("l1_avoided_tokens")))
        for row in candidates
    }
    if len(canonical) != 1:
        raise ReferenceMismatchError(
            f"utility reference {group!r}/{arm} has no seed={seed} and varies by seed"
        )
    return candidates[0]


def derive_seed_utilities(
    replay_rows: Iterable[Row],
    published_reference_rows: Iterable[Row],
    winners: Iterable[GenericWinner],
    *,
    group_fields: Sequence[str] = ("trace", "l1_fraction", "l2_multiplier"),
    target_field: str = "target",
    seed_field: str = "seed",
    iteration_field: str = "iteration",
    arm_field: str = "arm",
    requested_field: str = "requested_tokens",
    avoided_field: str = "avoided_prefill_tokens",
    l1_avoided_field: str = "l1_avoided_tokens",
    l2_avoided_field: str = "l2_avoided_tokens",
    sampled_lru_arm: str = "lru_s",
    offline_arm: str = "offline_next_use",
) -> list[dict[str, Any]]:
    """Add the registered per-seed utility contrasts without mutating rows.

    ``winners`` must come from :func:`select_generic_winners`; their arm is
    fixed by the published cell mean.  Each output seed is then compared with
    that arm's same-seed published row.  The sampled LRU floor is also
    same-seed; a deterministic offline row may be repeated or supplied once.
    """

    rows = [dict(row) for row in replay_rows]
    references = [dict(row) for row in published_reference_rows]
    winner_rows = list(winners)
    needed_arms = {sampled_lru_arm, offline_arm}
    needed_arms.update(winner.arm for winner in winner_rows)
    reference_index = _reference_index(
        references, group_fields, arm_field, seed_field, needed_arms
    )
    winner_index: dict[tuple[tuple, str], GenericWinner] = {}
    for winner in winner_rows:
        key = (tuple(winner.group), winner.mechanism)
        if key in winner_index:
            raise ReferenceMismatchError(f"duplicate generic winner {key!r}")
        winner_index[key] = winner

    identity_fields = (*group_fields, target_field, seed_field, iteration_field)
    _index_unique(rows, identity_fields, "on-policy utility")
    pi0: dict[tuple[tuple, Hashable, int], Row] = {}
    for row in rows:
        try:
            iteration = int(row[iteration_field])
            seed = int(row[seed_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReferenceMismatchError("on-policy utility row has invalid seed/iteration") from exc
        if iteration == 0:
            key = (_row_key(row, group_fields), _canonical(row[target_field]), seed)
            if key in pi0:
                raise ReferenceMismatchError(f"duplicate pi0 utility row {key!r}")
            pi0[key] = row

    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        group = _row_key(row, group_fields)
        target = _canonical(row[target_field])
        seed = int(row[seed_field])
        baseline = pi0.get((group, target, seed))
        if baseline is None:
            raise ReferenceMismatchError(f"missing pi0 utility baseline {group!r}/{target}/seed={seed}")
        floor = _reference_for_seed(reference_index, group, sampled_lru_arm, seed)
        offline = _reference_for_seed(
            reference_index, group, offline_arm, seed, allow_deterministic_fallback=True
        )
        try:
            requested = int(row[requested_field])
            avoided = int(row[avoided_field])
            l1_avoided = int(row[l1_avoided_field])
            l2_avoided = int(row[l2_avoided_field])
            pi0_avoided = int(baseline[avoided_field])
            floor_avoided = int(floor[avoided_field])
            ceiling_avoided = int(offline[avoided_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReferenceMismatchError("utility row contains a non-integer replay outcome") from exc
        if requested <= 0:
            raise ReferenceMismatchError("requested_tokens must be positive")
        if avoided != l1_avoided + l2_avoided:
            raise ReferenceMismatchError(
                f"utility replay identity failed: {avoided} != {l1_avoided} + {l2_avoided}"
            )
        for name, reference in (("pi0", baseline), ("sampled LRU", floor), ("offline", offline)):
            if int(reference[requested_field]) != requested:
                raise ReferenceMismatchError(f"{name} requested-token identity differs for {group!r}")
            if int(reference[l1_avoided_field]) != l1_avoided:
                raise ReferenceMismatchError(f"{name} L1 outcome differs for {group!r}")
        delta = avoided - pi0_avoided
        denominator = ceiling_avoided - floor_avoided
        row["delta_avoided_tokens_vs_pi0"] = delta
        row["input_token_points_vs_pi0"] = 100.0 * delta / requested
        row["extra_l2_avoided_tokens"] = l2_avoided
        row["gain_vs_sampled_lru_tokens"] = avoided - floor_avoided
        row["headroom_closure"] = (
            (avoided - floor_avoided) / denominator if denominator > 0 else math.nan
        )
        for mechanism in ("heap", "sampled"):
            winner = winner_index.get((group, mechanism))
            if winner is None:
                raise ReferenceMismatchError(f"missing {mechanism} generic winner for {group!r}")
            winner_row = _reference_for_seed(
                reference_index,
                group,
                winner.arm,
                seed,
                allow_deterministic_fallback=(mechanism == "heap"),
            )
            if int(winner_row[requested_field]) != requested:
                raise ReferenceMismatchError(
                    f"{mechanism} winner requested-token identity differs for {group!r}"
                )
            row[f"best_generic_{mechanism}_arm"] = winner.arm
            row[f"delta_tokens_vs_best_generic_{mechanism}"] = (
                avoided - int(winner_row[avoided_field])
            )
        output.append(row)
    return output


def _integer_field(row: Row, field: str, context: str) -> int:
    if field not in row:
        raise ReferenceMismatchError(f"{context}: missing integer field {field!r}")
    try:
        value = Decimal(str(row[field]))
    except InvalidOperation as exc:
        raise ReferenceMismatchError(f"{context}: {field} is not numeric") from exc
    if not value.is_finite() or value != value.to_integral_value():
        raise ReferenceMismatchError(f"{context}: {field} is not an exact integer")
    return int(value)


def validate_attribution_partition(
    row: Row,
    *,
    categories: Sequence[str] = ATTRIBUTION_CATEGORIES,
    decision_categories: Sequence[str] = ATTRIBUTION_DECISION_CATEGORIES,
    context: str = "Phase 0.98b attribution",
) -> None:
    """Validate request- and block-level Phase 0.98b partition identities."""

    for category in categories:
        for unit in ("tokens", "blocks"):
            absent = _integer_field(row, f"absent_{category}_{unit}", context)
            root = _integer_field(row, f"root_{category}_{unit}", context)
            downstream = _integer_field(row, f"downstream_{category}_{unit}", context)
            if absent != root + downstream:
                raise ReferenceMismatchError(
                    f"{context}: absent_{category}_{unit} {absent} != "
                    f"root {root} + downstream {downstream}"
                )

    root_loss = sum(_integer_field(row, f"root_{name}_tokens", context) for name in categories)
    unusable = sum(
        _integer_field(row, f"unusable_after_{name}_tokens", context) for name in categories
    )
    absent_loss = sum(
        _integer_field(row, f"absent_{name}_tokens", context) for name in categories
    )
    decision_loss = sum(
        _integer_field(row, f"root_{name}_tokens", context)
        + _integer_field(row, f"unusable_after_{name}_tokens", context)
        for name in decision_categories
    )
    perblock_decision_loss = sum(
        _integer_field(row, f"absent_{name}_tokens", context)
        + _integer_field(row, f"unusable_after_{name}_tokens", context)
        for name in decision_categories
    )
    totals = {
        "root_loss_tokens": root_loss,
        "unusable_tokens": unusable,
        "absent_loss_tokens": absent_loss,
        "decision_loss_tokens": decision_loss,
        "perblock_decision_loss_tokens": perblock_decision_loss,
    }
    for field, expected in totals.items():
        observed = _integer_field(row, field, context)
        if observed != expected:
            raise ReferenceMismatchError(f"{context}: {field} {observed} != {expected}")
    downstream_absent = _integer_field(row, "downstream_absent_tokens", context)
    if absent_loss != root_loss + downstream_absent:
        raise ReferenceMismatchError(
            f"{context}: absent_loss_tokens {absent_loss} != root_loss_tokens "
            f"{root_loss} + downstream_absent_tokens {downstream_absent}"
        )

    optional_identities = (
        ("attribution_measured_requests", "measured_requests"),
        ("l2_hit_tokens", "l2_avoided_tokens"),
        ("l2_hit_blocks_checked", "l2_hit_blocks"),
        ("unusable_tokens", "l2_present_unusable_tokens"),
    )
    for attribution_field, replay_field in optional_identities:
        if attribution_field in row and replay_field in row:
            left = _integer_field(row, attribution_field, context)
            right = _integer_field(row, replay_field, context)
            if left != right:
                raise ReferenceMismatchError(
                    f"{context}: {attribution_field} {left} != {replay_field} {right}"
                )
    if all(field in row for field in ("requested_tokens", "l1_avoided_tokens", "beyond_prefix_tokens")):
        requested = _integer_field(row, "requested_tokens", context)
        l1 = _integer_field(row, "l1_avoided_tokens", context)
        beyond = _integer_field(row, "beyond_prefix_tokens", context)
        if beyond != requested - l1:
            raise ReferenceMismatchError(
                f"{context}: beyond_prefix_tokens {beyond} != requested {requested} - L1 {l1}"
            )
        l2_hit = _integer_field(row, "l2_hit_tokens", context)
        if beyond != l2_hit + absent_loss + unusable:
            raise ReferenceMismatchError(
                f"{context}: request token partition {beyond} != L2 hit {l2_hit} + "
                f"absent {absent_loss} + unusable {unusable}"
            )
    if all(field in row for field in ("beyond_prefix_blocks", "l2_hit_blocks_checked")):
        absent_blocks = sum(
            _integer_field(row, f"absent_{name}_blocks", context) for name in categories
        )
        unusable_blocks = sum(
            _integer_field(row, f"unusable_after_{name}_blocks", context)
            for name in categories
        )
        beyond_blocks = _integer_field(row, "beyond_prefix_blocks", context)
        hit_blocks = _integer_field(row, "l2_hit_blocks_checked", context)
        if beyond_blocks != hit_blocks + absent_blocks + unusable_blocks:
            raise ReferenceMismatchError(
                f"{context}: request block partition {beyond_blocks} != L2 hit "
                f"{hit_blocks} + absent {absent_blocks} + unusable {unusable_blocks}"
            )


def merge_seed_attribution(
    utility_rows: Iterable[Row],
    *attribution_tables: Iterable[Row],
    key_fields: Sequence[str] = ("trace", "cell", "target", "seed", "iteration"),
) -> list[dict[str, Any]]:
    """Exact-join loss/orphaning tables and validate each per-seed partition.

    Each supplementary table must contain exactly the same identities as the
    utility table.  Conflicting duplicate fields are rejected.  This permits
    loss and orphaning rows to be supplied separately without silently dropping
    a lineage.
    """

    base = _index_unique(
        [dict(row) for row in utility_rows], key_fields, "utility attribution base"
    )
    merged = {key: dict(row) for key, row in base.items()}
    for table_number, table in enumerate(attribution_tables, start=1):
        supplement = _index_unique(
            [dict(row) for row in table], key_fields, f"attribution table {table_number}"
        )
        if set(supplement) != set(base):
            missing = sorted(set(base) - set(supplement), key=repr)
            extra = sorted(set(supplement) - set(base), key=repr)
            raise ReferenceMismatchError(
                f"attribution table {table_number} identity mismatch; "
                f"missing={missing[:8]!r}, extra={extra[:8]!r}"
            )
        for key, row in supplement.items():
            target = merged[key]
            for field, value in row.items():
                if field in target and _canonical(target[field]) != _canonical(value):
                    raise ReferenceMismatchError(
                        f"attribution {key!r}: conflicting field {field!r}"
                    )
                target[field] = value
    output = [merged[key] for key in sorted(merged, key=repr)]
    for row in output:
        validate_attribution_partition(row, context=f"Phase 0.98b {_row_key(row, key_fields)!r}")
    return output


def _summary(values: Sequence[float]) -> dict[str, float | int | bool]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("aggregation needs finite, non-empty values")
    count = len(values)
    mean = statistics.fmean(values)
    if count == 1:
        std = 0.0
        ci95 = math.nan
    else:
        std = statistics.stdev(values)
        ci95 = _T_975.get(count - 1, 1.96) * std / math.sqrt(count)
    return {
        "mean": mean,
        "std": std,
        "ci95_half": ci95,
        "min": min(values),
        "max": max(values),
        "all_positive": all(value > 0 for value in values),
        "n": count,
    }


def aggregate_seed_metrics(
    rows: Iterable[Row],
    *,
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
    seed_field: str = "seed",
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    carry_fields: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Aggregate fixed seed sets with mean, sample SD, t-CI, range and signs."""

    expected = {int(seed) for seed in expected_seeds}
    grouped: dict[tuple, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[_row_key(row, group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=repr):
        members = grouped[key]
        by_seed: dict[int, Row] = {}
        for row in members:
            seed = int(row[seed_field])
            if seed in by_seed:
                raise ReferenceMismatchError(f"aggregate {key!r}: duplicate seed {seed}")
            by_seed[seed] = row
        if set(by_seed) != expected:
            raise ReferenceMismatchError(
                f"aggregate {key!r}: seeds {sorted(by_seed)} != {sorted(expected)}"
            )
        result = {field: members[0][field] for field in group_fields}
        result["seeds"] = len(expected)
        for field in carry_fields:
            values = {_canonical(row.get(field)) for row in members}
            if len(values) != 1:
                raise ReferenceMismatchError(f"aggregate {key!r}: {field} varies by seed")
            result[field] = members[0].get(field)
        for field in metric_fields:
            try:
                values = [float(by_seed[seed][field]) for seed in sorted(expected)]
            except (KeyError, TypeError, ValueError) as exc:
                raise ReferenceMismatchError(f"aggregate {key!r}: invalid metric {field}") from exc
            for suffix, value in _summary(values).items():
                result[f"{field}_{suffix}"] = value
        output.append(result)
    return output


def _material_pass(values: Sequence[float], threshold: float) -> bool:
    return bool(values) and statistics.fmean(values) >= threshold and all(value > 0 for value in values)


def _mixed_sign(values: Sequence[float]) -> bool:
    return any(value > 0 for value in values) and any(value < 0 for value in values)


def classify_registered_outcome(
    *,
    target: str,
    utility_point_deltas: Sequence[float],
    heldout_ranking_deltas: Sequence[float],
    fit_population_ranking_deltas: Sequence[float],
    pi3_fit_population_scores: Sequence[float],
    pi3_terminal_train_scores: Sequence[float],
    pi3_test_scores: Sequence[float],
    own_policy_ranking_deltas: Sequence[float] | None = None,
    expected_seed_count: int = 5,
    integrity_ok: bool = True,
    dynamics_resolved: bool = True,
    populations_usable: bool = True,
    numerical_agreement: bool = True,
    fits_valid: bool = True,
) -> RegisteredOutcome:
    """Apply the registered A/B/C/D/unresolved priority for one cell.

    Ranking deltas use the common terminal population.  Fit-population deltas
    use ``D_train(pi2)``, the population that actually fitted pi3.  Scores are
    pi3 absolute metrics on ``D_train(pi2)``, ``D_train(pi3)``, and
    ``D_test(pi3)`` respectively.
    """

    target_name = str(target)
    if target_name not in {"next_use", "binary"}:
        raise ValueError(f"unknown registered target {target_name!r}")
    sequences = {
        "utility": tuple(map(float, utility_point_deltas)),
        "heldout ranking": tuple(map(float, heldout_ranking_deltas)),
        "fit-population ranking": tuple(map(float, fit_population_ranking_deltas)),
        "fit-population score": tuple(map(float, pi3_fit_population_scores)),
        "terminal-train score": tuple(map(float, pi3_terminal_train_scores)),
        "test score": tuple(map(float, pi3_test_scores)),
    }
    if own_policy_ranking_deltas is not None:
        sequences["own-policy ranking"] = tuple(map(float, own_policy_ranking_deltas))
    for name, values in sequences.items():
        if len(values) != expected_seed_count:
            raise ValueError(
                f"{name} needs {expected_seed_count} per-seed values, found {len(values)}"
            )
    nonfinite = [
        name for name, values in sequences.items()
        if any(not math.isfinite(value) for value in values)
    ]
    utility_pass = not nonfinite and _material_pass(sequences["utility"], 0.10)
    heldout_pass = not nonfinite and _material_pass(sequences["heldout ranking"], 0.05)
    fit_pass = not nonfinite and _material_pass(sequences["fit-population ranking"], 0.05)
    own_positive = (
        None
        if own_policy_ranking_deltas is None
        else all(value > 0 for value in sequences["own-policy ranking"])
    )
    blocking_reasons: list[str] = [f"non-finite metric in {name}" for name in nonfinite]
    reasons: list[str] = []
    invalid_flags = {
        "integrity check failed": not integrity_ok,
        "dynamics unresolved": not dynamics_resolved,
        "population cap/labels unusable": not populations_usable,
        "numerical diagonal disagreement": not numerical_agreement,
        "fit pathology": not fits_valid,
    }
    blocking_reasons.extend(name for name, failed in invalid_flags.items() if failed)
    # Mixed signs take priority only for the two registered primary contrasts.
    # Fit-population gain is diagnostic unless it independently satisfies D.
    for name in ("utility", "heldout ranking"):
        if _mixed_sign(sequences[name]):
            blocking_reasons.append(f"mixed seed signs in {name}")
    if _mixed_sign(sequences["fit-population ranking"]):
        reasons.append("mixed seed signs in diagnostic fit-population ranking")
    if utility_pass and not heldout_pass:
        blocking_reasons.append("material utility gain without registered ranking gain")
    if blocking_reasons:
        label = "unresolved"
    elif heldout_pass and utility_pass:
        label = "A"
    elif heldout_pass:
        label = "B"
    elif fit_pass:
        label = "D"
    else:
        high = 0.30 if target_name == "next_use" else 0.70
        low_everywhere = all(
            value < high
            for name in ("fit-population score", "terminal-train score", "test score")
            for value in sequences[name]
        )
        gain_below_threshold = statistics.fmean(sequences["heldout ranking"]) < 0.05
        if low_everywhere and gain_below_threshold and fits_valid and not utility_pass:
            label = "C"
        else:
            label = "unresolved"
            reasons.append("pattern does not satisfy A-D")
    strong_support = label == "A" and own_positive is True
    if label == "A" and own_positive is False:
        reasons.append("own-policy ranking direction is not positive for every seed")
    return RegisteredOutcome(
        label=label,
        target=target_name,
        utility_pass=utility_pass,
        heldout_ranking_pass=heldout_pass,
        fit_population_ranking_pass=fit_pass,
        own_policy_direction_all_positive=own_positive,
        strong_support=strong_support,
        reasons=tuple(blocking_reasons + reasons),
    )


def _manifest_index(manifest: Mapping[str, Any] | Iterable[Row], key_field: str) -> dict[str, Any]:
    if isinstance(manifest, Mapping):
        return {str(key): value for key, value in manifest.items()}
    result: dict[str, Any] = {}
    for row in manifest:
        if key_field not in row:
            raise ValueError(f"manifest row is missing {key_field!r}")
        key = str(row[key_field])
        if key in result:
            raise ValueError(f"manifest has duplicate key {key!r}")
        result[key] = {field: value for field, value in row.items() if field != key_field}
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def compare_manifests(
    start: Mapping[str, Any] | Iterable[Row],
    end: Mapping[str, Any] | Iterable[Row],
    *,
    key_field: str = "path",
) -> ManifestComparison:
    """Compare start/end source manifests without depending on row order."""

    left = _manifest_index(start, key_field)
    right = _manifest_index(end, key_field)
    added = tuple(sorted(set(right) - set(left)))
    removed = tuple(sorted(set(left) - set(right)))
    changed = tuple(
        sorted(
            key
            for key in set(left) & set(right)
            if _canonical_json(left[key]) != _canonical_json(right[key])
        )
    )
    return ManifestComparison(not (added or removed or changed), added, removed, changed)


def assert_manifests_equal(
    start: Mapping[str, Any] | Iterable[Row],
    end: Mapping[str, Any] | Iterable[Row],
    *,
    key_field: str = "path",
) -> ManifestComparison:
    """Return an equal comparison or raise the publication barrier."""

    result = compare_manifests(start, end, key_field=key_field)
    if not result.equal:
        raise ReferenceMismatchError(
            f"start/end manifests differ: added={result.added!r}, "
            f"removed={result.removed!r}, changed={result.changed!r}"
        )
    return result


def _plot_axes(
    rows: Sequence[Row],
    trace_field: str,
    cell_fields: Sequence[str],
    expected_traces: Sequence[Any] | None,
    expected_cells: Sequence[Any] | None,
) -> tuple[tuple[Hashable, ...], tuple[tuple[Hashable, ...], ...]]:
    observed_traces = {_canonical(row[trace_field]) for row in rows}
    observed_cells = {_cell_key(row, cell_fields) for row in rows}
    traces = (
        tuple(_canonical(value) for value in expected_traces)
        if expected_traces is not None
        else tuple(sorted(observed_traces, key=repr))
    )
    cells = (
        tuple(_expected_cell_key(value, cell_fields) for value in expected_cells)
        if expected_cells is not None
        else tuple(sorted(observed_cells, key=repr))
    )
    if observed_traces != set(traces):
        raise ReferenceMismatchError(
            f"figure trace set {sorted(observed_traces, key=repr)!r} != {traces!r}"
        )
    if observed_cells != set(cells):
        raise ReferenceMismatchError(
            f"figure cell set {sorted(observed_cells, key=repr)!r} != {cells!r}"
        )
    return traces, cells


def _cell_label(cell: tuple[Hashable, ...]) -> str:
    return ",".join(str(value) for value in cell)


def _finish_figure(figure: Any, output_path: str | Path, dpi: int) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=dpi)
    return target


def plot_all_iterations_utility(
    rows: Iterable[Row],
    output_path: str | Path,
    *,
    metric_field: str = "input_token_points_vs_pi0",
    trace_field: str = "trace",
    cell_fields: Sequence[str] = ("cell",),
    target_field: str = "target",
    iteration_field: str = "iteration",
    seed_field: str = "seed",
    expected_traces: Sequence[Any] | None = None,
    expected_cells: Sequence[Any] | None = None,
    targets: Sequence[str] = ("next_use", "binary"),
    iterations: Sequence[int] = (0, 1, 2, 3),
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    dpi: int = 180,
) -> Path:
    """Write the fixed all-iteration utility figure for every cell and trace."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("utility figure needs rows")
    traces, cells = _plot_axes(
        materialized, trace_field, cell_fields, expected_traces, expected_cells
    )
    target_values = tuple(str(value) for value in targets)
    iteration_values = tuple(int(value) for value in iterations)
    seed_values = {int(value) for value in expected_seeds}
    grouped: dict[tuple, dict[int, float]] = defaultdict(dict)
    for row in materialized:
        key = (
            _canonical(row[trace_field]),
            str(row[target_field]),
            _cell_key(row, cell_fields),
            int(row[iteration_field]),
        )
        seed = int(row[seed_field])
        if seed in grouped[key]:
            raise ReferenceMismatchError(f"utility figure duplicate {key!r}/seed={seed}")
        grouped[key][seed] = float(row[metric_field])
    expected_keys = set(itertools.product(traces, target_values, cells, iteration_values))
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped), key=repr)
        extra = sorted(set(grouped) - expected_keys, key=repr)
        raise ReferenceMismatchError(
            f"utility figure grid mismatch; missing={missing[:8]!r}, extra={extra[:8]!r}"
        )
    for key, values in grouped.items():
        if set(values) != seed_values:
            raise ReferenceMismatchError(
                f"utility figure {key!r}: seeds {sorted(values)} != {sorted(seed_values)}"
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(target_values), len(traces),
        figsize=(5.2 * len(traces), 3.8 * len(target_values)), squeeze=False,
        sharex=True,
    )
    for target_index, target in enumerate(target_values):
        for trace_index, trace in enumerate(traces):
            axis = axes[target_index][trace_index]
            axis.axhline(0.0, color="black", linewidth=0.8)
            for cell in cells:
                values = [
                    statistics.fmean(grouped[(trace, target, cell, iteration)].values())
                    for iteration in iteration_values
                ]
                axis.plot(iteration_values, values, marker="o", label=_cell_label(cell))
            axis.set_title(f"{trace} — {target}")
            axis.set_xticks(iteration_values)
            axis.grid(True, alpha=0.25)
            if trace_index == 0:
                axis.set_ylabel("pi_i − pi0 input-token points")
            if target_index == len(target_values) - 1:
                axis.set_xlabel("policy iteration")
    axes[0][-1].legend(title="cell", fontsize=7, title_fontsize=8)
    target_path = _finish_figure(figure, output_path, dpi)
    plt.close(figure)
    return target_path


def plot_terminal_rank_utility_thresholds(
    rows: Iterable[Row],
    output_path: str | Path,
    *,
    ranking_delta_field: str = "terminal_common_population_ranking_delta",
    utility_delta_field: str = "input_token_points_vs_pi0",
    trace_field: str = "trace",
    cell_fields: Sequence[str] = ("cell",),
    target_field: str = "target",
    seed_field: str = "seed",
    expected_traces: Sequence[Any] | None = None,
    expected_cells: Sequence[Any] | None = None,
    targets: Sequence[str] = ("next_use", "binary"),
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    ranking_threshold: float = 0.05,
    utility_threshold: float = 0.10,
    dpi: int = 180,
) -> Path:
    """Write the fixed terminal rank-change versus utility-change figure."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("terminal threshold figure needs rows")
    traces, cells = _plot_axes(
        materialized, trace_field, cell_fields, expected_traces, expected_cells
    )
    target_values = tuple(str(value) for value in targets)
    seed_values = {int(value) for value in expected_seeds}
    grouped: dict[tuple, dict[int, tuple[float, float]]] = defaultdict(dict)
    for row in materialized:
        key = (
            _canonical(row[trace_field]),
            str(row[target_field]),
            _cell_key(row, cell_fields),
        )
        seed = int(row[seed_field])
        if seed in grouped[key]:
            raise ReferenceMismatchError(f"threshold figure duplicate {key!r}/seed={seed}")
        grouped[key][seed] = (
            float(row[ranking_delta_field]), float(row[utility_delta_field])
        )
    expected_keys = set(itertools.product(traces, target_values, cells))
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped), key=repr)
        extra = sorted(set(grouped) - expected_keys, key=repr)
        raise ReferenceMismatchError(
            f"threshold figure grid mismatch; missing={missing[:8]!r}, extra={extra[:8]!r}"
        )
    for key, values in grouped.items():
        if set(values) != seed_values:
            raise ReferenceMismatchError(
                f"threshold figure {key!r}: seeds {sorted(values)} != {sorted(seed_values)}"
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(target_values), len(traces),
        figsize=(5.2 * len(traces), 4.0 * len(target_values)), squeeze=False,
        sharex=True, sharey=True,
    )
    for target_index, target in enumerate(target_values):
        for trace_index, trace in enumerate(traces):
            axis = axes[target_index][trace_index]
            axis.axvline(ranking_threshold, color="black", linestyle="--", linewidth=0.9)
            axis.axhline(utility_threshold, color="black", linestyle="--", linewidth=0.9)
            unavailable = []
            for cell in cells:
                values = grouped[(trace, target, cell)]
                ranking = statistics.fmean(value[0] for value in values.values())
                utility = statistics.fmean(value[1] for value in values.values())
                if not (math.isfinite(ranking) and math.isfinite(utility)):
                    unavailable.append(_cell_label(cell))
                    continue
                axis.scatter([ranking], [utility], s=32)
                axis.annotate(_cell_label(cell), (ranking, utility), fontsize=7, xytext=(3, 3),
                              textcoords="offset points")
            if unavailable:
                axis.text(0.02, 0.02, "NA: " + ", ".join(unavailable),
                          transform=axis.transAxes, fontsize=7, va="bottom")
            axis.set_title(f"{trace} — {target}")
            axis.grid(True, alpha=0.25)
            if trace_index == 0:
                axis.set_ylabel("pi3 − pi0 input-token points")
            if target_index == len(target_values) - 1:
                axis.set_xlabel("pi3 − pi0 common-terminal-population ranking")
    target_path = _finish_figure(figure, output_path, dpi)
    plt.close(figure)
    return target_path


def plot_cross_score_heatmaps(
    rows: Iterable[Row],
    output_path: str | Path,
    *,
    target: str = "next_use",
    metric_field: str = "within_decision_metric",
    trace_field: str = "trace",
    cell_fields: Sequence[str] = ("cell",),
    target_field: str = "target",
    split_field: str = "split",
    seed_field: str = "seed",
    population_iteration_field: str = "population_iteration",
    model_iteration_field: str = "model_iteration",
    expected_traces: Sequence[Any] | None = None,
    expected_cells: Sequence[Any] | None = None,
    splits: Sequence[str] = ("train", "test"),
    iterations: Sequence[int] = (0, 1, 2, 3),
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    dpi: int = 180,
) -> Path:
    """Write fixed train/test 4x4 heatmaps for all cells.

    ``target='next_use'`` is the registered paper figure.  Passing
    ``target='binary'`` produces the optional supplementary figure from the
    same complete CSV surface.
    """

    selected = [dict(row) for row in rows if str(row.get(target_field, "")) == str(target)]
    if not selected:
        raise ValueError(f"heatmap has no rows for target {target!r}")
    traces, cells = _plot_axes(
        selected, trace_field, cell_fields, expected_traces, expected_cells
    )
    split_values = tuple(str(value) for value in splits)
    iteration_values = tuple(int(value) for value in iterations)
    seed_values = {int(value) for value in expected_seeds}
    group_fields = (trace_field, *cell_fields, target_field, seed_field, split_field)
    validate_4x4_matrix(
        selected,
        group_fields=group_fields,
        population_iteration_field=population_iteration_field,
        model_iteration_field=model_iteration_field,
        iterations=iteration_values,
        expected_group_count=len(traces) * len(cells) * len(split_values) * len(seed_values),
    )
    grouped: dict[tuple, dict[int, float]] = defaultdict(dict)
    for row in selected:
        key = (
            _canonical(row[trace_field]),
            _cell_key(row, cell_fields),
            str(row[split_field]),
            int(row[population_iteration_field]),
            int(row[model_iteration_field]),
        )
        seed = int(row[seed_field])
        if seed in grouped[key]:
            raise ReferenceMismatchError(f"heatmap duplicate {key!r}/seed={seed}")
        grouped[key][seed] = float(row[metric_field])
    expected_keys = {
        (trace, cell, split, population, model)
        for trace, cell, split, population, model in itertools.product(
            traces, cells, split_values, iteration_values, iteration_values
        )
    }
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped), key=repr)
        extra = sorted(set(grouped) - expected_keys, key=repr)
        raise ReferenceMismatchError(
            f"heatmap grid mismatch; missing={missing[:8]!r}, extra={extra[:8]!r}"
        )
    for key, values in grouped.items():
        if set(values) != seed_values:
            raise ReferenceMismatchError(
                f"heatmap {key!r}: seeds {sorted(values)} != {sorted(seed_values)}"
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    means = {
        key: statistics.fmean(values.values())
        for key, values in grouped.items()
    }
    finite_values = [value for value in means.values() if math.isfinite(value)]
    lower, upper = (min(finite_values), max(finite_values)) if finite_values else (0.0, 1.0)
    if math.isclose(lower, upper):
        lower -= 0.5
        upper += 0.5
    panel_rows = tuple(itertools.product(traces, split_values))
    figure, axes = plt.subplots(
        len(panel_rows), len(cells),
        figsize=(3.0 * len(cells), 2.8 * len(panel_rows)), squeeze=False,
    )
    image = None
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("lightgray")
    for row_index, (trace, split) in enumerate(panel_rows):
        for column_index, cell in enumerate(cells):
            matrix = np.asarray(
                [
                    [means[(trace, cell, split, population, model)] for model in iteration_values]
                    for population in iteration_values
                ],
                dtype=float,
            )
            axis = axes[row_index][column_index]
            image = axis.imshow(np.ma.masked_invalid(matrix), vmin=lower, vmax=upper,
                                cmap=color_map, aspect="auto")
            axis.set_xticks(range(4), [f"pi{value}" for value in iteration_values])
            axis.set_yticks(range(4), [f"D(pi{value})" for value in iteration_values])
            axis.set_title(f"{trace} {split}\n{_cell_label(cell)}", fontsize=9)
            if row_index == len(panel_rows) - 1:
                axis.set_xlabel("scoring model")
            if column_index == 0:
                axis.set_ylabel("logged population")
            for population_index in range(4):
                for model_index in range(4):
                    axis.text(
                        model_index, population_index,
                        (f"{matrix[population_index, model_index]:.2f}"
                         if math.isfinite(matrix[population_index, model_index]) else "NA"),
                        ha="center", va="center", fontsize=6,
                        color="white" if matrix[population_index, model_index] < (lower + upper) / 2 else "black",
                    )
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.65, label=metric_field)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return target_path
