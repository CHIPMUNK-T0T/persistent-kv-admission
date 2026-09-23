#!/usr/bin/env python3
"""Fixed-round on-policy decision-population learning.

The run order is deliberately enforced:

1. reconstruct and verify the four shared pi0 fits;
2. collect every training population and serialize/hash every pi0..pi3 identity;
3. validate the complete training barrier;
4. cold replay every held-out identity;
5. score saved populations and publish only after all integrity checks pass.

Smoke and full runs always write a new directory.  This module never overwrites
Phase 0.97/0.98 artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import numpy as np

from persistent_kv_admission.attribution import AttributionCollector, LOSS_CATEGORIES
from persistent_kv_admission.crossworkload import HYPERPARAMETERS, fit_fixed_model
from persistent_kv_admission.decisionpop import (
    CANDIDATE_CAP,
    DECISION_CAP,
    MAX_TRAIN_ROWS,
    NEGATIVES_PER_POSITIVE,
    SAMPLE_WIDTH,
    SNAPSHOT_COUNT,
    Rows,
    RawFeatureScorer,
    coefficient_row,
    fit_population_ranker,
    horizon_for,
    state_indices,
)
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.onpolicy import (
    DecisionPopulation,
    DecisionPopulationLogger,
    LabelWindowUtilityCollector,
    align_populations,
    coefficient_cosine,
    deserialize_ranker,
    ranker_detail,
    score_population,
    score_recorded_population,
    serialize_ranker,
    sha256_path,
    verify_recorded_argmin,
)
from persistent_kv_admission.onpolicy_reporting import (
    aggregate_seed_metrics,
    assert_manifests_equal,
    classify_registered_outcome,
    derive_seed_utilities,
    merge_seed_attribution,
    plot_all_iterations_utility,
    plot_cross_score_heatmaps,
    plot_terminal_rank_utility_thresholds,
    select_generic_winners,
    validate_4x4_matrix,
    validate_training_model_barrier,
    verify_candidate_log_metadata,
    verify_published_pi0_replay,
    verify_reference_rows,
    write_csv_rows,
)
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

CELLS = tuple(
    (fraction, multiplier)
    for fraction in (0.0025, 0.01, 0.02)
    for multiplier in (1.0, 4.0)
)
TARGETS = ("next_use", "binary")
SEEDS = (0, 1, 2, 3, 4)
ITERATIONS = (0, 1, 2, 3)
SMOKE_CELL = (0.01, 4.0)
SMOKE_TARGETS = ("next_use",)
SMOKE_SEEDS = (0,)
L1_POLICY = "lru"
HIT_MODEL = "tree"
CLOSURE = "union"
FIT_SEED = 0
FIT_TOLERANCE = 1e-12
PREREGISTRATION_COMMIT = "c10256a"
PLAN = REPOSITORY / "docs/onpolicy-learning-plan.md"
PHASE097_CONFIG = REPOSITORY / "results/paper/decision_population_config.json"
PHASE097_FITS = REPOSITORY / "results/paper/decision_population_fits.csv"
PHASE097_REPLAYS = REPOSITORY / "results/paper/decision_population_replay_seeds.csv"
PHASE098_LOSSES = REPOSITORY / "results/paper/decision_attribution_losses_seeds.csv"
PHASE1_CONFIG = REPOSITORY / "results/paper/phase1_intervention_config.json"
CANDIDATE_META = REPOSITORY / "results/decision_population/decision_population_logs.csv"
CANDIDATE_DIR = REPOSITORY / "results/decision_population/logs"
PREEXISTING_DIRTY_DOCS = (
    "README.md",
    "docs/decision-population-findings.md",
    "docs/experiment-plan.md",
)
_SHARED: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/onpolicy_learning")
    )
    parser.add_argument("--paper-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one trace/cell/target/seed through every iteration",
    )
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cell_label(fraction: float, multiplier: float) -> str:
    return f"l1={fraction:g},l2x{multiplier:g}"


def lineage_slug(
    trace: str, fraction: float, multiplier: float, target: str, seed: int
) -> str:
    return (
        f"{trace}__l1_{fraction:g}__l2x_{multiplier:g}"
        f"__{target}__seed_{seed}"
    )


def source_manifest() -> dict[str, object]:
    paths = sorted((REPOSITORY / "src").glob("**/*.py"))
    paths.extend(
        (
            Path(__file__).resolve(),
            PLAN,
            REPOSITORY / "docs/onpolicy-implementation.md",
            REPOSITORY / "tests/test_onpolicy_learning.py",
        )
    )
    files = {
        str(path.relative_to(REPOSITORY)): sha256_path(path)
        for path in sorted(set(paths))
        if path.exists()
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()}


def dirty_doc_hashes() -> dict[str, str]:
    out = {}
    for name in PREEXISTING_DIRTY_DOCS:
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--no-ext-diff", "--", name],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        ).stdout
        out[name] = hashlib.sha256(diff).hexdigest()
    return out


def ensure_execution_tree_committed() -> None:
    paths = (
        "src",
        "scripts/run_onpolicy_learning.py",
        "tests/test_onpolicy_learning.py",
        "docs/onpolicy-learning-plan.md",
    )
    changed = _git("diff", "HEAD", "--name-only", "--", *paths)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *paths)
    if changed or untracked:
        detail = "; ".join(value for value in (changed, untracked) if value)
        raise SystemExit(f"on-policy execution files are not committed: {detail}")
    prereg = _git("rev-parse", f"{PREREGISTRATION_COMMIT}^{{commit}}")
    plan_commit = _git("log", "-1", "--format=%H", "--", str(PLAN.relative_to(REPOSITORY)))
    if plan_commit != prereg:
        raise SystemExit(
            f"pre-registration identity changed: {plan_commit} != {prereg}"
        )


def _same_number(left, right, tolerance: float = 0.0) -> bool:
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance * max(
        1.0, abs(a), abs(b)
    )


def verify_phase097_config(
    config: dict,
    traces: dict,
    splits: dict,
    horizons: dict,
    working_sets: dict,
    cells: tuple[tuple[float, float], ...],
    targets: tuple[str, ...],
) -> None:
    expected = {
        "l1_policy": L1_POLICY,
        "hit_model": HIT_MODEL,
        "closure": CLOSURE,
        "sample_width": SAMPLE_WIDTH,
        "max_train_rows": MAX_TRAIN_ROWS,
        "negatives_per_positive": NEGATIVES_PER_POSITIVE,
        "hyperparameters": HYPERPARAMETERS,
        "snapshot_count": SNAPSHOT_COUNT,
        "candidate_cap": CANDIDATE_CAP,
        "decision_cap": DECISION_CAP,
        "seeds": list(SEEDS),
    }
    errors = [
        f"{field}: {config.get(field)!r} != {value!r}"
        for field, value in expected.items()
        if config.get(field) != value
    ]
    for target in targets:
        if target not in config.get("targets", ()):
            errors.append(f"target {target} absent")
    for fraction, multiplier in cells:
        if fraction not in config.get("l1_budgets", ()):
            errors.append(f"L1 fraction {fraction} absent")
        if multiplier not in config.get("l2_multipliers", ()):
            errors.append(f"L2 multiplier {multiplier} absent")
    for name in traces:
        for field, expected_value in (
            ("measure_from_ms", splits[name]),
            ("horizons_seconds", horizons[name]),
            ("working_set_bytes", working_sets[name]),
        ):
            if not _same_number(config.get(field, {}).get(name), expected_value):
                errors.append(f"{field}/{name} mismatch")
    if errors:
        raise SystemExit("Phase 0.97 config mismatch: " + "; ".join(errors))



def _published_pi0_index() -> dict[tuple[str, str], dict[str, str]]:
    rows = {}
    for row in load_csv(PHASE097_FITS):
        if row["population"] != "A_none":
            continue
        key = (row["trace"], row["target"])
        if key in rows:
            raise AssertionError(f"duplicate published pi0 fit {key}")
        rows[key] = row
    return rows


def rebuild_pi0(
    traces: dict,
    output_dir: Path,
    targets: tuple[str, ...],
) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    """Recreate and verify the four Phase 0.97 A_none fits before any replay."""

    references = _published_pi0_index()
    models: dict[tuple[str, str], dict] = {}
    checks: list[dict] = []
    model_dir = output_dir / "models" / "pi0"
    for name in sorted(traces):
        for target in targets:
            started = time.time()
            ranker, horizon, split_ms, positives, used = fit_fixed_model(
                traces[name],
                600.0,
                snapshot_count=SNAPSHOT_COUNT,
                candidate_cap=CANDIDATE_CAP,
                max_train_rows=MAX_TRAIN_ROWS,
                negatives_per_positive=NEGATIVES_PER_POSITIVE,
                train_fraction=HYPERPARAMETERS["train_fraction"],
                seed=FIT_SEED,
                standardization="ranker",
                target=target,
            )
            reference = references.get((name, target))
            if reference is None:
                raise SystemExit(f"published pi0 fit missing for {(name, target)}")
            observed = {
                "horizon_seconds": float(horizon),
                "split_ms": float(split_ms),
                "train_rows": int(used),
                "train_positives": int(positives),
                "ranker_standardize": bool(ranker.standardize),
                "fit_iterations": int(getattr(ranker, "iterations", -1)),
                "fit_converged": bool(getattr(ranker, "converged", False)),
                "fit_condition_number": float(
                    getattr(ranker, "condition_number", math.nan)
                ),
            }
            errors = [
                f"{field}: {value!r} != {reference.get(field)!r}"
                for field, value in observed.items()
                if not _same_number(value, reference.get(field), FIT_TOLERANCE)
            ]
            coefficients = coefficient_row(ranker)
            deviations = {
                feature: abs(float(value) - float(reference[feature]))
                for feature, value in coefficients.items()
            }
            maximum = max(deviations.values(), default=0.0)
            if maximum > FIT_TOLERANCE:
                errors.append(f"coefficient deviation {maximum:.3e}")
            if errors:
                raise SystemExit(
                    f"Phase 0.97 pi0 fit mismatch {(name, target)}: "
                    + "; ".join(errors)
                )
            path = model_dir / f"{name}__{target}.json"
            digest = serialize_ranker(ranker, path)
            roundtrip = deserialize_ranker(path, digest)
            if ranker_detail(roundtrip) != ranker_detail(ranker):
                raise AssertionError(f"pi0 serialization changed {(name, target)}")
            models[(name, target)] = {
                "path": str(path.resolve()),
                "sha256": digest,
            }
            checks.append(
                {
                    "trace": name,
                    "target": target,
                    **observed,
                    "max_standardized_coefficient_deviation": maximum,
                    "published_fit_matched": True,
                    "model_path": str(path.resolve()),
                    "model_sha256": digest,
                    "seconds": time.time() - started,
                }
            )
    return models, checks


def _capacity(name: str, fraction: float) -> int:
    return max(1, round(_SHARED["working_sets"][name] * fraction))


def _identity(
    name: str,
    fraction: float,
    multiplier: float,
    target: str,
    seed: int,
    iteration: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "trace": name,
        "l1_fraction": fraction,
        "l2_multiplier": multiplier,
        "cell": cell_label(fraction, multiplier),
        "target": target,
        "seed": seed,
    }
    if iteration is not None:
        row["iteration"] = iteration
        row["policy"] = f"pi{iteration}"
    return row


def _peak_rss_mib() -> float:
    # Linux reports ru_maxrss in KiB.  The experiment host is Linux and records
    # this convention explicitly in the run config.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _request_hook(attribution, label_utility):
    if attribution is None:
        return label_utility.on_request if label_utility is not None else None
    if label_utility is None:
        return attribution.on_request

    def observe(*args, **kwargs):
        attribution.on_request(*args, **kwargs)
        label_utility.on_request(*args, **kwargs)

    return observe


def _learned_replay(
    trace,
    ranker,
    l1_bytes: int,
    l2_bytes: int,
    seed: int,
    split_ms: float,
    occurrence_groups,
    logger: DecisionPopulationLogger,
    *,
    stop_before_ms: float | None = None,
    attribution: AttributionCollector | None = None,
    label_utility: LabelWindowUtilityCollector | None = None,
):
    scorer = RawFeatureScorer(trace, ranker)
    return run_two_tier(
        trace,
        L1_POLICY,
        l1_bytes,
        l2_bytes,
        "learned",
        hit_model=HIT_MODEL,
        closure=CLOSURE,
        bytes_per_token=HYPERPARAMETERS["bytes_per_token"],
        size_model=HYPERPARAMETERS["size_model"],
        measure_from_ms=split_ms,
        occurrence_groups=occurrence_groups,
        l2_eviction="sampled",
        l2_sample_width=SAMPLE_WIDTH,
        l2_seed=seed,
        l2_scorer=scorer,
        l2_decision_hook=logger,
        l2_arm="onpolicy",
        observer=logger,
        l2_request_hook=_request_hook(attribution, label_utility),
        l2_removal_hook=attribution,
        l2_arrival_protection="none",
        stop_before_ms=stop_before_ms,
    )


def _training_worker(task) -> dict[str, object]:
    """Collect D_train(pi0..pi3); each update fits only the just-saved D."""

    name, fraction, multiplier, target, seed = task
    trace = _SHARED["traces"][name]
    split_ms = _SHARED["splits"][name]
    horizon = _SHARED["horizons"][name]
    end_ms = trace.end_ms
    l1_bytes = _capacity(name, fraction)
    l2_bytes = _capacity(name, fraction * multiplier)
    slug = lineage_slug(name, fraction, multiplier, target, seed)
    output_dir = Path(_SHARED["output_dir"])
    population_dir = output_dir / "populations" / "train" / slug
    learned_model_dir = output_dir / "models" / "learned" / slug
    current_info = _SHARED["pi0_models"][(name, target)]
    current_ranker = deserialize_ranker(
        current_info["path"], current_info["sha256"]
    )
    model_rows: list[dict] = []
    population_rows: list[dict] = []
    replay_seconds = 0.0
    fit_seconds = 0.0
    serialization_seconds = 0.0

    for iteration in ITERATIONS:
        identity = _identity(
            name, fraction, multiplier, target, seed, iteration
        )
        model_rows.append(
            {
                **identity,
                "model_path": current_info["path"],
                "model_sha256": current_info["sha256"],
                "shared_pi0": iteration == 0,
                "fitted_from_population_iteration": (
                    "" if iteration == 0 else iteration - 1
                ),
                **ranker_detail(current_ranker),
            }
        )
        logger = DecisionPopulationLogger(
            trace,
            horizon,
            split_ms,
            end_ms,
            "train",
            max_decisions=DECISION_CAP,
            seed=seed,
            state_index=_SHARED["state_indices"][name],
        )
        began = time.time()
        result = _learned_replay(
            trace,
            current_ranker,
            l1_bytes,
            l2_bytes,
            seed,
            split_ms,
            _SHARED["groups"][name],
            logger,
            stop_before_ms=split_ms,
        )
        elapsed = time.time() - began
        replay_seconds += elapsed
        population = logger.rows()
        del logger
        if not len(population):
            raise AssertionError(f"empty training population for {identity}")
        argmin = verify_recorded_argmin(population)
        population_path = population_dir / f"pi{iteration}.npz"
        population_hash = population.save(population_path)
        population_rows.append(
            {
                **identity,
                "window": "train",
                "population_path": str(population_path.resolve()),
                "population_sha256": population_hash,
                "feature_dtype": str(population.features.dtype),
                "fit_feature_dtype": str(population.to_fit_rows().features.dtype),
                "rows": len(population),
                "decisions_kept": population.decisions,
                "decisions_offered": population.decisions_offered,
                "decisions_eligible": population.decisions_eligible,
                "decision_cap": population.max_decisions,
                "cap_bound": population.cap_bound,
                "l2_decisions_prefix": result.l2_decisions,
                "replay_seconds": elapsed,
                **argmin,
            }
        )

        del result
        if iteration == ITERATIONS[-1]:
            del population
            continue
        began = time.time()
        next_ranker, positives, used = fit_population_ranker(
            population.to_fit_rows(), target, seed=FIT_SEED
        )
        elapsed = time.time() - began
        fit_seconds += elapsed
        next_path = learned_model_dir / f"pi{iteration + 1}.json"
        began_serialization = time.time()
        next_hash = serialize_ranker(next_ranker, next_path)
        restored = deserialize_ranker(next_path, next_hash)
        serialization_seconds += time.time() - began_serialization
        if ranker_detail(restored) != ranker_detail(next_ranker):
            raise AssertionError(f"model serialization changed {identity}")
        current_ranker = restored
        current_info = {
            "path": str(next_path.resolve()),
            "sha256": next_hash,
        }
        population_rows[-1].update(
            {
                "fit_target": target,
                "fit_seed": FIT_SEED,
                "fit_rows": int(used),
                "fit_positives": int(positives),
                "fit_seconds": elapsed,
                "fit_converged": bool(getattr(restored, "converged", False)),
                "fit_iterations": int(getattr(restored, "iterations", -1)),
                "fit_condition_number": float(
                    getattr(restored, "condition_number", math.nan)
                ),
            }
        )
        del population

    return {
        "task": task,
        "models": model_rows,
        "populations": population_rows,
        "training_replay_seconds": replay_seconds,
        "fit_seconds": fit_seconds,
        "serialization_seconds": serialization_seconds,
        "peak_worker_rss_mib": _peak_rss_mib(),
    }



def _heldout_worker(task) -> dict[str, object]:
    """Cold replay pi0..pi3 after the parent has validated the full barrier."""

    name, fraction, multiplier, target, seed = task
    trace = _SHARED["traces"][name]
    split_ms = _SHARED["splits"][name]
    horizon = _SHARED["horizons"][name]
    l1_bytes = _capacity(name, fraction)
    l2_bytes = _capacity(name, fraction * multiplier)
    slug = lineage_slug(name, fraction, multiplier, target, seed)
    population_dir = Path(_SHARED["output_dir"]) / "populations" / "test" / slug
    populations: list[dict] = []
    utilities: list[dict] = []
    losses: list[dict] = []
    orphaning: list[dict] = []
    replay_seconds = 0.0

    for iteration in ITERATIONS:
        identity = _identity(
            name, fraction, multiplier, target, seed, iteration
        )
        model = _SHARED["model_index"][
            (name, fraction, multiplier, target, seed, iteration)
        ]
        ranker = deserialize_ranker(model["model_path"], model["model_sha256"])
        logger = DecisionPopulationLogger(
            trace,
            horizon,
            split_ms,
            trace.end_ms,
            "test",
            max_decisions=DECISION_CAP,
            seed=seed,
            state_index=_SHARED["state_indices"][name],
        )
        attribution = AttributionCollector(
            trace,
            measure_from_ms=split_ms,
            bytes_per_token=HYPERPARAMETERS["bytes_per_token"],
            size_model=HYPERPARAMETERS["size_model"],
        )
        label_utility = LabelWindowUtilityCollector(
            trace,
            split_ms,
            trace.end_ms - horizon * 1000.0,
        )
        began = time.time()
        result = _learned_replay(
            trace,
            ranker,
            l1_bytes,
            l2_bytes,
            seed,
            split_ms,
            _SHARED["groups"][name],
            logger,
            attribution=attribution,
            label_utility=label_utility,
        )
        elapsed = time.time() - began
        replay_seconds += elapsed
        attribution.check_against(result)
        population = logger.rows()
        del logger
        if not len(population):
            raise AssertionError(f"empty held-out population for {identity}")
        argmin = verify_recorded_argmin(population)
        population_path = population_dir / f"pi{iteration}.npz"
        population_hash = population.save(population_path)
        populations.append(
            {
                **identity,
                "window": "test",
                "population_path": str(population_path.resolve()),
                "population_sha256": population_hash,
                "feature_dtype": str(population.features.dtype),
                "rows": len(population),
                "decisions_kept": population.decisions,
                "decisions_offered": population.decisions_offered,
                "decisions_eligible": population.decisions_eligible,
                "decision_cap": population.max_decisions,
                "cap_bound": population.cap_bound,
                "l2_decisions_full_replay": result.l2_decisions,
                "replay_seconds": elapsed,
                **argmin,
            }
        )
        utility = {
            **identity,
            **result.as_row(),
            **label_utility.row(),
            "replay_seconds": elapsed,
        }
        utilities.append(utility)
        loss = {
            **identity,
            "requested_tokens": result.requested_tokens,
            "requested_blocks": result.requested_blocks,
            "measured_requests": result.measured_requests,
            "l1_avoided_tokens": result.l1_avoided_tokens,
            "l2_avoided_tokens": result.l2_avoided_tokens,
            "avoided_prefill_tokens": result.avoided_prefill_tokens,
            "l2_present_unusable_tokens": result.l2_present_unusable_tokens,
            **attribution.loss_row(),
            "attribution_partition_verified": True,
        }
        requested = max(int(result.requested_tokens), 1)
        for field, value in tuple(loss.items()):
            if field.endswith("_tokens") and isinstance(value, (int, float)):
                loss[f"{field}_share_of_input"] = float(value) / requested
        losses.append(loss)
        orphaning.append(
            {
                **identity,
                "l2_admissions": result.l2_admissions,
                "l2_rejections": result.l2_rejections,
                "l2_evictions": result.l2_evictions,
                "l2_decisions": result.l2_decisions,
                "l1_evictions": result.l1_evictions,
                **attribution.orphaning_row(),
            }
        )
        del population, result, attribution, label_utility, ranker

    return {
        "task": task,
        "populations": populations,
        "utilities": utilities,
        "losses": losses,
        "orphaning": orphaning,
        "heldout_replay_seconds": replay_seconds,
        "peak_worker_rss_mib": _peak_rss_mib(),
    }


def _load_population(metadata: dict) -> DecisionPopulation:
    path = Path(metadata["population_path"])
    actual = sha256_path(path)
    expected = metadata["population_sha256"]
    if actual != expected:
        raise AssertionError(f"population hash mismatch {path}: {actual} != {expected}")
    return DecisionPopulation.load(path)


def _matrix_rows_for_population(
    task,
    window: str,
    population_iteration: int,
    population_metadata: dict,
    models: list,
) -> tuple[list[dict], dict]:
    name, fraction, multiplier, target, seed = task
    population = _load_population(population_metadata)
    logging_identity = _identity(
        name, fraction, multiplier, target, seed, population_iteration
    )
    recorded = {
        **logging_identity,
        "window": window,
        "population_iteration": population_iteration,
        "scoring_iteration": population_iteration,
        "authoritative_recorded_tuple": True,
        **score_recorded_population(population, target),
    }
    rows = []
    for scoring_iteration, ranker in enumerate(models):
        rows.append(
            {
                **logging_identity,
                "window": window,
                "population_iteration": population_iteration,
                "scoring_iteration": scoring_iteration,
                "authoritative_recorded_tuple": False,
                **score_population(
                    ranker,
                    population,
                    target,
                    model_iteration=scoring_iteration,
                ),
            }
        )
    del population
    return rows, recorded


def _crossscore_worker(task) -> dict[str, object]:
    """Load one population at a time and score all four frozen lineage models."""

    name, fraction, multiplier, target, seed = task
    model_rows = [
        _SHARED["model_index"][
            (name, fraction, multiplier, target, seed, iteration)
        ]
        for iteration in ITERATIONS
    ]
    models = [
        deserialize_ranker(row["model_path"], row["model_sha256"])
        for row in model_rows
    ]
    population_index = _SHARED["population_index"]
    matrix: list[dict] = []
    recorded: list[dict] = []
    began = time.time()
    for window in ("train", "test"):
        for population_iteration in ITERATIONS:
            row = population_index[
                (
                    name,
                    fraction,
                    multiplier,
                    target,
                    seed,
                    window,
                    population_iteration,
                )
            ]
            scored, own = _matrix_rows_for_population(
                task,
                window,
                population_iteration,
                row,
                models,
            )
            matrix.extend(scored)
            recorded.append(own)
    similarities = []
    for left in ITERATIONS:
        for right in ITERATIONS:
            similarities.append(
                {
                    **_identity(name, fraction, multiplier, target, seed),
                    "left_iteration": left,
                    "right_iteration": right,
                    "raw_slope_cosine": coefficient_cosine(
                        models[left], models[right]
                    ),
                }
            )
    del models
    return {
        "task": task,
        "matrix": matrix,
        "recorded": recorded,
        "coefficient_similarity": similarities,
        "crossscore_seconds": time.time() - began,
        "peak_worker_rss_mib": _peak_rss_mib(),
    }


def _alignment_worker(task) -> list[dict]:
    name, fraction, multiplier, target, seed = task
    model_rows = [
        _SHARED["model_index"][
            (name, fraction, multiplier, target, seed, iteration)
        ]
        for iteration in ITERATIONS
    ]
    models = [
        deserialize_ranker(row["model_path"], row["model_sha256"])
        for row in model_rows
    ]
    out = []
    for window in ("train", "test"):
        for left in ITERATIONS:
            for right in range(left + 1, len(ITERATIONS)):
                left_meta = _SHARED["population_index"][
                    (name, fraction, multiplier, target, seed, window, left)
                ]
                right_meta = _SHARED["population_index"][
                    (name, fraction, multiplier, target, seed, window, right)
                ]
                left_population = _load_population(left_meta)
                right_population = _load_population(right_meta)
                out.append(
                    {
                        **_identity(name, fraction, multiplier, target, seed),
                        "window": window,
                        "left_iteration": left,
                        "right_iteration": right,
                        **align_populations(
                            left_population,
                            right_population,
                            models[left],
                            models[right],
                        ),
                    }
                )
                del left_population, right_population
    del models
    return out



def _candidate_path(
    name: str, fraction: float, multiplier: float, policy: str
) -> Path:
    return CANDIDATE_DIR / (
        f"candidates_{name}_{fraction:g}_x{multiplier:g}_{policy}.npz"
    )


def _auxiliary_worker(task) -> list[dict]:
    """Score frozen lineage models on the matching historical LRU/LFU logs."""

    name, fraction, multiplier, target, seed = task
    models = [
        deserialize_ranker(
            _SHARED["model_index"][
                (name, fraction, multiplier, target, seed, iteration)
            ]["model_path"],
            _SHARED["model_index"][
                (name, fraction, multiplier, target, seed, iteration)
            ]["model_sha256"],
        )
        for iteration in ITERATIONS
    ]
    split_ms = _SHARED["splits"][name]
    end_ms = _SHARED["traces"][name].end_ms
    output = []
    for policy in ("lru", "lfu"):
        metadata = _SHARED["candidate_index"][
            (name, fraction, multiplier, policy)
        ]
        path = Path(metadata["artifact_path"])
        current_hash = sha256_path(path)
        if current_hash != metadata["current_sha256"]:
            raise AssertionError(f"candidate log changed during run: {path}")
        historical = Rows.load(path)
        if not _same_number(
            historical.horizon_seconds, _SHARED["horizons"][name]
        ):
            raise AssertionError(f"candidate horizon mismatch: {path}")
        for window in ("train", "test"):
            mask = (
                historical.train_mask(split_ms)
                if window == "train"
                else historical.test_mask(split_ms, end_ms)
            )
            selected = historical.select(mask)
            population = DecisionPopulation.from_rows(
                selected,
                window,
                seed=0,
                source_policy=policy,
            )
            if not len(population):
                raise AssertionError(
                    f"empty auxiliary {window} population in {path}"
                )
            for iteration, ranker in enumerate(models):
                output.append(
                    {
                        **_identity(
                            name, fraction, multiplier, target, seed
                        ),
                        "source_policy": policy,
                        "window": window,
                        "scoring_iteration": iteration,
                        "source_artifact": str(path.resolve()),
                        "source_artifact_current_sha256": current_hash,
                        "source_hash_provenance": "contemporaneous_only",
                        "historical_hash_verified": False,
                        **score_population(
                            ranker,
                            population,
                            target,
                            model_iteration=iteration,
                        ),
                    }
                )
            del population, selected
        del historical
    del models
    return output


def _index_unique(rows: list[dict], fields: tuple[str, ...]) -> dict[tuple, dict]:
    out = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in out:
            raise AssertionError(f"duplicate identity {key}")
        out[key] = row
    return out


def model_index(rows: list[dict]) -> dict[tuple, dict]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "l1_fraction": float(row["l1_fraction"]),
                "l2_multiplier": float(row["l2_multiplier"]),
                "seed": int(row["seed"]),
                "iteration": int(row["iteration"]),
            }
        )
    return _index_unique(
        normalized,
        (
            "trace",
            "l1_fraction",
            "l2_multiplier",
            "target",
            "seed",
            "iteration",
        ),
    )


def population_index(rows: list[dict]) -> dict[tuple, dict]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "l1_fraction": float(row["l1_fraction"]),
                "l2_multiplier": float(row["l2_multiplier"]),
                "seed": int(row["seed"]),
                "iteration": int(row["iteration"]),
            }
        )
    return _index_unique(
        normalized,
        (
            "trace",
            "l1_fraction",
            "l2_multiplier",
            "target",
            "seed",
            "window",
            "iteration",
        ),
    )


def validate_crossscore_integrity(
    matrix_rows: list[dict],
    recorded_rows: list[dict],
    expected_tasks: list[tuple],
) -> dict[str, object]:
    """Require the complete task-grid x train/test x 4x4 score table."""

    matrix_keys = set()
    numerical_errors = []
    for row in matrix_rows:
        key = (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            row["target"],
            int(row["seed"]),
            row["window"],
            int(row["population_iteration"]),
            int(row["scoring_iteration"]),
        )
        if key in matrix_keys:
            raise AssertionError(f"duplicate cross-score row {key}")
        matrix_keys.add(key)
        if int(row["population_iteration"]) == int(row["scoring_iteration"]):
            if float(row["recorded_argmin_mismatches"]) != 0.0:
                raise AssertionError(f"recorded tuple is not argmin for {key}")
            if int(row["recomputed_recorded_primary_exact"]) != 1:
                numerical_errors.append(f"{key}: primary score differs")
            if float(row["actual_proposed_victim_agreement_rate"]) != 1.0:
                numerical_errors.append(f"{key}: recomputed victim differs")
    expected_base = {
        (name, float(fraction), float(multiplier), target, int(seed), window)
        for name, fraction, multiplier, target, seed in expected_tasks
        for window in ("train", "test")
    }
    expected = {
        prefix + (population_iteration, scoring_iteration)
        for prefix in expected_base
        for population_iteration in ITERATIONS
        for scoring_iteration in ITERATIONS
    }
    if matrix_keys != expected:
        raise AssertionError(
            f"cross-score matrix incomplete: {len(expected - matrix_keys)} missing, "
            f"{len(matrix_keys - expected)} extra"
        )
    recorded_keys = []
    for row in recorded_rows:
        recorded_keys.append(
            (
                row["trace"],
                float(row["l1_fraction"]),
                float(row["l2_multiplier"]),
                row["target"],
                int(row["seed"]),
                row["window"],
                int(row["population_iteration"]),
            )
        )
    if len(recorded_keys) != len(set(recorded_keys)):
        raise AssertionError("duplicate authoritative recorded-tuple row")
    expected_recorded = {
        prefix + (iteration,)
        for prefix in expected_base
        for iteration in ITERATIONS
    }
    if set(recorded_keys) != expected_recorded:
        raise AssertionError(
            "authoritative recorded-tuple table is incomplete: "
            f"{len(expected_recorded - set(recorded_keys))} missing, "
            f"{len(set(recorded_keys) - expected_recorded)} extra"
        )
    return {
        "matrix_rows": len(matrix_keys),
        "recorded_rows": len(recorded_keys),
        "numerical_agreement": not numerical_errors,
        "numerical_disagreements": numerical_errors,
    }


def _actual_candidate_metadata(
    names: tuple[str, ...],
    cells: tuple[tuple[float, float], ...],
) -> list[dict]:
    rows = []
    for name in names:
        for fraction, multiplier in cells:
            for policy in ("lru", "lfu"):
                path = _candidate_path(name, fraction, multiplier, policy)
                if not path.exists():
                    raise SystemExit(f"missing Phase 0.97 candidate log {path}")
                artifact = Rows.load(path)
                decisions = (
                    int(len(np.unique(artifact.group))) if len(artifact) else 0
                )
                rows.append(
                    {
                        "trace": name,
                        "l1_fraction": fraction,
                        "l2_multiplier": multiplier,
                        "behaviour_policy": policy,
                        "decisions_kept": decisions,
                        "rows": len(artifact),
                        "artifact_path": str(path.resolve()),
                        "current_sha256": sha256_path(path),
                        "hash_provenance": "contemporaneous_only",
                        "historical_hash_verified": False,
                    }
                )
                del artifact
    return rows



def canonical_models_for_barrier(
    rows: list[dict],
    *,
    traces: tuple[str, ...],
    cells: tuple[tuple[float, float], ...],
    targets: tuple[str, ...],
    seeds: tuple[int, ...],
    pi0_models: dict[tuple[str, str], dict],
) -> list[dict]:
    """Verify all lineage links, then collapse shared pi0 rows for the barrier."""

    indexed = model_index(rows)
    expected = {
        (name, fraction, multiplier, target, seed, iteration)
        for name in traces
        for fraction, multiplier in cells
        for target in targets
        for seed in seeds
        for iteration in ITERATIONS
    }
    if set(indexed) != expected:
        raise AssertionError(
            f"lineage model identities incomplete: {len(expected - set(indexed))} "
            f"missing, {len(set(indexed) - expected)} extra"
        )
    update_paths = set()
    pi0_canonical: dict[tuple[str, str], dict] = {}
    for key in sorted(indexed):
        name, fraction, multiplier, target, seed, iteration = key
        row = indexed[key]
        path = Path(row["model_path"])
        if sha256_path(path) != row["model_sha256"]:
            raise AssertionError(f"model hash mismatch for {key}")
        if iteration == 0:
            shared = pi0_models[(name, target)]
            if (
                row["model_path"] != shared["path"]
                or row["model_sha256"] != shared["sha256"]
                or row.get("shared_pi0") is not True
            ):
                raise AssertionError(f"pi0 linkage mismatch for {key}")
            canonical_key = (name, target)
            prior = pi0_canonical.setdefault(canonical_key, row)
            if (
                prior["model_path"] != row["model_path"]
                or prior["model_sha256"] != row["model_sha256"]
            ):
                raise AssertionError(f"pi0 is not shared for {canonical_key}")
        else:
            if row.get("shared_pi0") is not False:
                raise AssertionError(f"updated model marked shared for {key}")
            if int(row["fitted_from_population_iteration"]) != iteration - 1:
                raise AssertionError(f"wrong fit-population link for {key}")
            if row["model_path"] in update_paths:
                raise AssertionError(
                    f"updated model path reused by another lineage: {row['model_path']}"
                )
            update_paths.add(row["model_path"])
    canonical = list(pi0_canonical.values()) + [
        row for key, row in indexed.items() if key[-1] > 0
    ]
    expected_unique = len(traces) * len(targets) + (
        len(traces) * len(cells) * len(targets) * len(seeds) * 3
    )
    if len(canonical) != expected_unique:
        raise AssertionError(
            f"canonical model identities {len(canonical)} != {expected_unique}"
        )
    paths = {row["model_path"] for row in canonical}
    if len(paths) != expected_unique:
        raise AssertionError(
            f"serialized model paths {len(paths)} != {expected_unique}"
        )
    return canonical


def _run_stage(worker, tasks: list[tuple], workers: int):
    """Run one complete phase; callers must finish its checks before the next."""
    if workers == 1:
        for task in tasks:
            yield worker(task)
        return
    context = mp.get_context("fork")
    with context.Pool(min(workers, len(tasks))) as pool:
        yield from pool.imap_unordered(worker, tasks, chunksize=1)


def _alignment_with_rss(task) -> dict:
    rows = _alignment_worker(task)
    return {"rows": rows, "peak_worker_rss_mib": _peak_rss_mib()}


def _auxiliary_with_rss(task) -> dict:
    rows = _auxiliary_worker(task)
    return {"rows": rows, "peak_worker_rss_mib": _peak_rss_mib()}


def _published_sha256(path: Path) -> str:
    """Check a tracked public CSV against its bytes at the frozen code commit."""
    relative = str(path.relative_to(REPOSITORY))
    stored = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=REPOSITORY,
        check=True, capture_output=True,
    ).stdout
    expected = hashlib.sha256(stored).hexdigest()
    if sha256_path(path) != expected:
        raise AssertionError(f"published reference changed from HEAD: {relative}")
    return expected


def _reference_context(names, cells, seeds, trace_files, splits, horizons, working_sets):
    phase1 = json.loads(PHASE1_CONFIG.read_text(encoding="utf-8"))
    reference_paths = (PHASE097_CONFIG, PHASE097_FITS, PHASE097_REPLAYS, PHASE098_LOSSES)
    hashes = {str(path.relative_to(REPOSITORY)): _published_sha256(path)
              for path in reference_paths}
    hashes[str(PHASE1_CONFIG.relative_to(REPOSITORY))] = _published_sha256(PHASE1_CONFIG)
    for name in names:
        recorded = phase1.get("trace_files", {}).get(name, {})
        if recorded.get("sha256") != trace_files[name]["sha256"]:
            raise AssertionError(f"Phase 1 trace hash mismatch: {name}")
        for field, current in (("horizons_seconds", horizons[name]),
                               ("measure_from_ms", splits[name]),
                               ("working_set_bytes", working_sets[name])):
            if not _same_number(phase1.get(field, {}).get(name), current):
                raise AssertionError(f"Phase 1 {field} mismatch: {name}")
    for field, current in (("l1_policy", L1_POLICY), ("hit_model", HIT_MODEL),
                           ("closure", CLOSURE), ("sample_width", SAMPLE_WIDTH),
                           ("size_model", HYPERPARAMETERS["size_model"]),
                           ("bytes_per_token", HYPERPARAMETERS["bytes_per_token"])):
        if phase1.get(field) != current:
            raise AssertionError(f"Phase 1 {field} mismatch")
    published = load_csv(PHASE097_REPLAYS)
    selected = [row for row in published
                if row["trace"] in names
                and (float(row["l1_fraction"]), float(row["l2_multiplier"])) in cells]
    required = {("A_none", "learned"), ("lru_s", "sampled"),
                ("offline_next_use", "heap")}
    for name in names:
        for fraction, multiplier in cells:
            for target in TARGETS:
                for seed in seeds:
                    subset = [row for row in selected if row["trace"] == name
                              and float(row["l1_fraction"]) == fraction
                              and float(row["l2_multiplier"]) == multiplier
                              and (row["target"] == target or row["target"] == "")
                              and int(row["seed"]) == seed]
                    matching = {(row["arm"], row["kind"]) for row in subset}
                    if not required <= matching:
                        raise AssertionError(f"missing published reference for {(name, fraction, multiplier, target, seed)}")
                    for row in subset:
                        if (int(row["l1_capacity_bytes"]) != max(1, round(working_sets[name] * fraction))
                            or int(row["l2_capacity_bytes"]) != max(1, round(working_sets[name] * fraction * multiplier))
                            or row["l1_policy"] != L1_POLICY or row["hit_model"] != HIT_MODEL
                            or row["closure"] != CLOSURE or int(row["l2_sample_width"]) != SAMPLE_WIDTH):
                            raise AssertionError(f"published reference context mismatch: {name}/{fraction}/{multiplier}")
                        if row["kind"] == "sampled" and int(row["l2_seed"]) != seed:
                            raise AssertionError("sampled reference seed mismatch")
    return selected, hashes, phase1


def _verify_candidate_references(names, cells, reference_rows):
    actual = _actual_candidate_metadata(names, cells)
    published = [row for row in load_csv(CANDIDATE_META)
                 if row["trace"] in names
                 and (float(row["l1_fraction"]), float(row["l2_multiplier"])) in cells
                 and row["behaviour_policy"] in ("lru", "lfu")]
    if len(published) != len(actual):
        raise AssertionError("candidate-log metadata identity count mismatch")
    for row in actual:
        match = [published_row for published_row in published
                 if published_row["trace"] == row["trace"]
                 and float(published_row["l1_fraction"]) == row["l1_fraction"]
                 and float(published_row["l2_multiplier"]) == row["l2_multiplier"]
                 and published_row["behaviour_policy"] == row["behaviour_policy"]]
        if len(match) != 1:
            raise AssertionError(f"candidate metadata identity missing or duplicated: {row}")
        source = match[0]
        verification = verify_candidate_log_metadata(
            [row], [source], count_fields=("decisions_kept", "rows"),
            artifact_path=row["artifact_path"],
            contemporaneous_sha256=row["current_sha256"],
        )
        if verification.hash_status != "contemporaneous_only":
            raise AssertionError("unexpected candidate hash provenance")
        if int(source["decisions_seen"]) != int(source["l2_decisions"]):
            raise AssertionError("candidate decision count identity failed")
        arm = source["behaviour_policy"] + "_s"
        matching_replays = [ref for ref in reference_rows
                            if ref["trace"] == row["trace"]
                            and float(ref["l1_fraction"]) == row["l1_fraction"]
                            and float(ref["l2_multiplier"]) == row["l2_multiplier"]
                            and ref["kind"] == "sampled" and ref["arm"] == arm
                            and int(ref["seed"]) == 0]
        count_fields = ("l2_decisions", "l2_admissions", "l2_rejections", "l2_evictions")
        if not matching_replays or any(
            any(int(ref[field]) != int(source[field]) for field in count_fields)
            for ref in matching_replays
        ):
            raise AssertionError("candidate metadata/replay count mismatch")
        row.update({key: source[key] for key in ("decisions_seen", "l2_decisions",
                                                  "l2_admissions", "l2_rejections", "l2_evictions")})
    return actual


def _pi0_loss_reference_check(losses: list[dict], names, cells, targets, seeds):
    published = [row for row in load_csv(PHASE098_LOSSES)
                 if row["arm"] == "A_none" and row["target"] in targets
                 and row["trace"] in names
                 and (float(row["l1_fraction"]), float(row["l2_multiplier"])) in cells
                 and int(row["seed"]) in seeds]
    pi0 = [row for row in losses if row["iteration"] == 0]
    if len(pi0) != len(names) * len(cells) * len(targets) * len(seeds):
        raise AssertionError("pi0 loss identities incomplete")
    published_cells = {(row["trace"], float(row["l1_fraction"]),
                        float(row["l2_multiplier"])) for row in published}
    checked = [row for row in pi0 if (row["trace"], row["l1_fraction"],
                                     row["l2_multiplier"]) in published_cells]
    exact = tuple(field for field in published[0] if field in pi0[0]
                  if (field.endswith(("_tokens", "_blocks"))
                      and not field.startswith(("diff_", "phase097_")))
                  or field in ("measured_requests", "attribution_measured_requests",
                               "fully_served_requests", "unexplained_states",
                               "unexplained_after_promotion")) if published else ()
    verify_reference_rows(
        checked, published,
        key_fields=("trace", "l1_fraction", "l2_multiplier", "target", "seed"),
        exact_fields=exact, expected_count=len(published_cells) * len(targets) * len(seeds),
        name="Phase 0.98b pi0 loss",
    )
    return {"matched_rows": len(checked),
            "reference_unavailable_cells": [cell_label(fraction, multiplier)
                for fraction, multiplier in cells
                if any((name, fraction, multiplier) not in published_cells for name in names)]}


def _rank_metric(row: dict) -> float:
    return float(row["within_decision_macro"])


def _ranking_and_verdicts(matrix: list[dict], recorded: list[dict],
                          utility: list[dict], models: list[dict],
                          populations: list[dict], names, cells, targets, seeds,
                          numerical_check: dict, smoke: bool):
    matrix_index = _index_unique(matrix,
        ("trace", "l1_fraction", "l2_multiplier", "target", "seed", "window",
         "population_iteration", "scoring_iteration"))
    recorded_index = _index_unique(recorded,
        ("trace", "l1_fraction", "l2_multiplier", "target", "seed", "window",
         "population_iteration"))
    utility_index = _index_unique(utility,
        ("trace", "l1_fraction", "l2_multiplier", "target", "seed", "iteration"))
    model_rows = model_index(models)
    population_rows = population_index(populations)
    own = []
    terminal = []
    verdicts = []
    for name in names:
        for fraction, multiplier in cells:
            for target in targets:
                seed_records = []
                for seed in seeds:
                    base = (name, fraction, multiplier, target, seed)
                    for iteration in ITERATIONS:
                        for window in ("train", "test"):
                            row = recorded_index[base + (window, iteration)]
                            own.append({**_identity(name, fraction, multiplier, target, seed, iteration),
                                        "window": window, **{key: value for key, value in row.items()
                                                            if key not in {"iteration", "policy"}}})
                    def m(window, population_iteration, scoring_iteration):
                        return _rank_metric(matrix_index[base +
                            (window, population_iteration, scoring_iteration)])
                    def r(window, population_iteration):
                        return _rank_metric(recorded_index[base +
                            (window, population_iteration)])
                    metrics = {
                        "terminal_common_population_ranking_delta": m("test", 3, 3) - m("test", 3, 0),
                        "fit_population_ranking_delta": m("train", 2, 3) - m("train", 2, 0),
                        "own_policy_test_ranking_delta": r("test", 3) - r("test", 0),
                        "pi3_fit_population_ranking": m("train", 2, 3),
                        "pi3_terminal_train_ranking": r("train", 3),
                        "pi3_test_ranking": r("test", 3),
                        "pi0_test_ranking": r("test", 0),
                    }
                    row = {**_identity(name, fraction, multiplier, target, seed),
                           **metrics,
                           "input_token_points_vs_pi0": utility_index[base + (3,)]["input_token_points_vs_pi0"],
                           "pi3_test_constant_label_share": recorded_index[base + ("test", 3)]["decisions_constant_label_share"],
                           "pi3_test_population_cap_bound": population_rows[base + ("test", 3)]["cap_bound"]}
                    terminal.append(row)
                    seed_records.append(row)
                if smoke:
                    verdicts.append({"trace": name, "l1_fraction": fraction,
                                     "l2_multiplier": multiplier, "cell": cell_label(fraction, multiplier),
                                     "target": target, "label": "smoke_diagnostic_only",
                                     "reason": "one seed cannot support registered five-seed outcome"})
                    continue
                fit_rows = [model_rows[(name, fraction, multiplier, target, seed, iteration)]
                            for seed in seeds for iteration in (1, 2, 3)]
                fits_valid = all(
                    bool(row["converged"]) if target == "binary"
                    else row["condition_number"] is not None
                    and math.isfinite(float(row["condition_number"]))
                    for row in fit_rows
                )
                high = 0.30 if target == "next_use" else 0.70
                shift = any(
                    math.isfinite(row["pi3_fit_population_ranking"])
                    and math.isfinite(row["pi3_terminal_train_ranking"])
                    and row["pi3_fit_population_ranking"] >= high
                    and row["pi3_terminal_train_ranking"] < high
                    for row in seed_records)
                diagonal = [row for row in matrix
                            if row["trace"] == name and row["l1_fraction"] == fraction
                            and row["l2_multiplier"] == multiplier and row["target"] == target
                            and row["seed"] in seeds
                            and row["population_iteration"] == row["scoring_iteration"]]
                numerical_ok = len(diagonal) == len(seeds) * 2 * 4 and all(
                    int(row["recomputed_recorded_primary_exact"]) == 1
                    and float(row["actual_proposed_victim_agreement_rate"]) == 1.0
                    for row in diagonal)
                result = classify_registered_outcome(
                    target=target,
                    utility_point_deltas=[float(row["input_token_points_vs_pi0"]) for row in seed_records],
                    heldout_ranking_deltas=[row["terminal_common_population_ranking_delta"] for row in seed_records],
                    fit_population_ranking_deltas=[row["fit_population_ranking_delta"] for row in seed_records],
                    pi3_fit_population_scores=[row["pi3_fit_population_ranking"] for row in seed_records],
                    pi3_terminal_train_scores=[row["pi3_terminal_train_ranking"] for row in seed_records],
                    pi3_test_scores=[row["pi3_test_ranking"] for row in seed_records],
                    own_policy_ranking_deltas=[row["own_policy_test_ranking_delta"] for row in seed_records],
                    numerical_agreement=numerical_ok,
                    dynamics_resolved=not shift, fits_valid=fits_valid,
                )
                verdicts.append({"trace": name, "l1_fraction": fraction,
                                 "l2_multiplier": multiplier, "cell": cell_label(fraction, multiplier),
                                 "target": target, **result.as_row(),
                                 "fit_to_terminal_train_shift": shift,
                                 "test_population_cap_bound_any": any(
                                     bool(row["pi3_test_population_cap_bound"]) for row in seed_records),
                                 "test_constant_label_share_max": max(
                                     float(row["pi3_test_constant_label_share"])
                                     for row in seed_records),
                                 "population_dominance_manual_review": True})
    # Cross-trace support is descriptive and never changes an individual verdict.
    if not smoke:
        for target in targets:
            for fraction, multiplier in cells:
                rows = [row for row in verdicts if row["target"] == target
                        and row["l1_fraction"] == fraction and row["l2_multiplier"] == multiplier]
                both = len(rows) == len(names) and len(names) == 2 and all(
                    row["label"] == "A" for row in rows)
                for row in rows:
                    row["same_cell_both_traces_A"] = both
        broad = sum(1 for fraction, multiplier in cells
                    if all(row["label"] == "A" for row in verdicts
                           if row["target"] == "next_use"
                           and row["l1_fraction"] == fraction
                           and row["l2_multiplier"] == multiplier))
        for row in verdicts:
            row["primary_next_use_cross_capacity_two_cells"] = broad >= 2
    return own, terminal, verdicts


def _publish_outputs(output_dir: Path, paper_dir: Path, tables: dict[str, list[dict]],
                     config: dict, *, smoke: bool, names, cells, targets, seeds):
    """Write fixed tables and figures only after every publication gate passes."""
    destination = output_dir / "paper" if smoke else paper_dir / "onpolicy_learning"
    if destination.exists():
        raise FileExistsError(f"on-policy publication target already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".onpolicy_staging_{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"on-policy staging target already exists: {staging}")
    staging.mkdir()
    identities = {}
    for name, rows in tables.items():
        path = staging / f"onpolicy_{name}.csv"
        write_csv_rows(path, rows)
        identities[path.name] = {"path": str((destination / path.name).resolve()),
                                 "sha256": sha256_path(path),
                                 "rows": len(rows)}
    figures = (
        ("all_iterations_utility", plot_all_iterations_utility, tables["seed_utility"],
         {"expected_traces": names, "expected_cells": tuple(cell_label(*cell) for cell in cells),
          "targets": targets, "expected_seeds": seeds}),
        ("terminal_rank_utility", plot_terminal_rank_utility_thresholds,
         tables["terminal_ranking"],
         {"expected_traces": names, "expected_cells": tuple(cell_label(*cell) for cell in cells),
          "targets": targets, "expected_seeds": seeds}),
        ("cross_score_next_use", plot_cross_score_heatmaps,
         tables["cross_score_matrix"],
         {"expected_traces": names, "expected_cells": tuple(cell_label(*cell) for cell in cells),
          "expected_seeds": seeds, "split_field": "window",
          "model_iteration_field": "scoring_iteration",
          "metric_field": "within_decision_macro"}),
    )
    for name, function, rows, kwargs in figures:
        path = staging / f"onpolicy_{name}.png"
        function(rows, path, **kwargs)
        identities[path.name] = {"path": str((destination / path.name).resolve()),
                                 "sha256": sha256_path(path)}
    config["output_artifacts"] = identities
    config_path = staging / "onpolicy_learning_config.json"
    config_path.write_text(json.dumps(config, indent=2, allow_nan=True, default=str) + "\n",
                           encoding="utf-8")
    os.replace(staging, destination)
    return destination


def run_experiment(args: argparse.Namespace) -> dict:
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    ensure_execution_tree_committed()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"new run directory required: {output_dir}")
    head_start = _git("rev-parse", "HEAD")
    source_start = source_manifest()
    dirty_start = dirty_doc_hashes()
    began = time.time()

    traces, groups, splits, horizons, working_sets, indices, trace_files = {}, {}, {}, {}, {}, {}, {}
    for path in args.traces:
        absolute = path.resolve()
        trace = load_mooncake_trace(absolute, 512)
        if trace.name in traces:
            raise SystemExit(f"duplicate trace name: {trace.name}")
        horizon, split_ms, _ = horizon_for(trace)
        if not _same_number(horizon, 600.0):
            raise SystemExit(f"fixed H=600 s violated: {trace.name}/{horizon}")
        traces[trace.name] = trace
        groups[trace.name] = _occurrence_groups(trace)
        splits[trace.name] = split_ms
        horizons[trace.name] = horizon
        working_sets[trace.name] = working_set_bytes(trace)
        indices[trace.name] = state_indices(trace)
        trace_files[trace.name] = {"path": str(absolute), "sha256": sha256_path(absolute)}
    expected_names = {"conversation_trace"} if args.smoke else {
        "conversation_trace", "toolagent_trace"
    }
    if set(traces) != expected_names:
        raise SystemExit(f"trace names {sorted(traces)} != fixed grid {sorted(expected_names)}")
    names = tuple(sorted(traces))
    cells = (SMOKE_CELL,) if args.smoke else CELLS
    targets = SMOKE_TARGETS if args.smoke else TARGETS
    seeds = SMOKE_SEEDS if args.smoke else SEEDS
    config097 = json.loads(PHASE097_CONFIG.read_text(encoding="utf-8"))
    verify_phase097_config(config097, traces, splits, horizons, working_sets, cells, targets)
    references, reference_hashes, phase1 = _reference_context(
        names, cells, seeds, trace_files, splits, horizons, working_sets)
    candidate_meta_hash = sha256_path(CANDIDATE_META)
    candidate_rows = _verify_candidate_references(names, cells, references)
    candidate_index = _index_unique(candidate_rows,
        ("trace", "l1_fraction", "l2_multiplier", "behaviour_policy"))

    output_dir.mkdir(parents=True)
    _SHARED.clear()
    _SHARED.update(traces=traces, groups=groups, splits=splits, horizons=horizons,
                   working_sets=working_sets, state_indices=indices,
                   output_dir=str(output_dir), candidate_index=candidate_index)
    pi0_models, fit_checks = rebuild_pi0(traces, output_dir, targets)
    _SHARED["pi0_models"] = pi0_models
    tasks = [(name, fraction, multiplier, target, seed)
             for name in names for fraction, multiplier in cells
             for target in targets for seed in seeds]
    if len(tasks) != len(names) * len(cells) * len(targets) * len(seeds):
        raise AssertionError("lineage task grid incomplete")

    # Training phase: every lineage finishes its four cold prefix replays.
    training_models, training_populations, stage_stats = [], [], {}
    started = time.time()
    training_workers = []
    for done, result in enumerate(_run_stage(_training_worker, tasks, args.workers), 1):
        training_models.extend(result["models"])
        training_populations.extend(result["populations"])
        training_workers.append(result)
        if done % 10 == 0 or done == len(tasks):
            print(f"training {done}/{len(tasks)}: {time.time() - started:.1f}s", flush=True)
    stage_stats["training_wall_seconds"] = time.time() - started
    stage_stats["training_replay_seconds"] = sum(
        float(result["training_replay_seconds"]) for result in training_workers)
    stage_stats["fit_seconds"] = sum(float(result["fit_seconds"])
                                     for result in training_workers)
    stage_stats["model_serialization_seconds"] = sum(
        float(result["serialization_seconds"]) for result in training_workers)
    canonical_models = canonical_models_for_barrier(
        training_models, traces=names, cells=cells, targets=targets,
        seeds=seeds, pi0_models=pi0_models)
    barrier = validate_training_model_barrier(
        training_populations, canonical_models,
        traces=names, cells=[cell_label(*cell) for cell in cells],
        targets=targets, seeds=seeds,
        population_hash_field="population_sha256", population_path_field="population_path",
        model_path_field="model_path", artifact_root=output_dir)
    if (barrier.training_identities != len(tasks) * 4 or
        barrier.model_identities != len(names) * len(targets) + len(tasks) * 3):
        raise AssertionError("pre-held-out barrier counts mismatch")
    _SHARED["model_index"] = model_index(training_models)
    write_csv_rows(output_dir / "training_model_links.csv", training_models)
    write_csv_rows(output_dir / "training_population_manifest.csv", training_populations)
    write_csv_rows(output_dir / "canonical_model_manifest.csv", canonical_models)
    (output_dir / "training_barrier.json").write_text(
        json.dumps(asdict(barrier), indent=2) + "\n", encoding="utf-8")
    training_peak_rss = max(float(result["peak_worker_rss_mib"])
                            for result in training_workers)
    del training_workers

    # Held-out starts only after the complete serialized training barrier above.
    heldout_populations, utility_raw, losses, orphaning = [], [], [], []
    heldout_workers = []
    started = time.time()
    for done, result in enumerate(_run_stage(_heldout_worker, tasks, args.workers), 1):
        heldout_populations.extend(result["populations"])
        utility_raw.extend(result["utilities"])
        losses.extend(result["losses"])
        orphaning.extend(result["orphaning"])
        heldout_workers.append(result)
        if done % 10 == 0 or done == len(tasks):
            print(f"held-out {done}/{len(tasks)}: {time.time() - started:.1f}s", flush=True)
    stage_stats["heldout_wall_seconds"] = time.time() - started
    stage_stats["heldout_replay_seconds"] = sum(
        float(result["heldout_replay_seconds"]) for result in heldout_workers)
    pi0_reference = [row for row in references
                     if row["arm"] == "A_none" and row["target"] in targets
                     and int(row["seed"]) in seeds]
    pi0_check = verify_published_pi0_replay(
        utility_raw, pi0_reference, expected_count=len(tasks))
    loss_pi0_count = _pi0_loss_reference_check(losses, names, cells, targets, seeds)
    unavailable_loss_cells = set(loss_pi0_count["reference_unavailable_cells"])
    for row in losses:
        row["published_pi0_loss_reference_status"] = (
            "matched" if row["iteration"] == 0 and row["cell"] not in unavailable_loss_cells
            else "unavailable_for_cell" if row["iteration"] == 0
            else "not_applicable_to_updated_policy"
        )
    _SHARED["population_index"] = population_index(training_populations + heldout_populations)

    # Cross-score and alignment load one or two spooled populations per worker.
    matrix, recorded, cosine, crossscore_workers = [], [], [], []
    started = time.time()
    for result in _run_stage(_crossscore_worker, tasks, args.workers):
        matrix.extend(result["matrix"])
        recorded.extend(result["recorded"])
        cosine.extend(result["coefficient_similarity"])
        crossscore_workers.append(result)
    stage_stats["crossscore_wall_seconds"] = time.time() - started
    stage_stats["crossscore_worker_seconds"] = sum(
        float(result["crossscore_seconds"]) for result in crossscore_workers)
    numerical = validate_crossscore_integrity(matrix, recorded, tasks)
    validate_4x4_matrix(
        matrix, group_fields=("trace", "cell", "target", "seed", "window"),
        model_iteration_field="scoring_iteration",
        expected_group_count=len(tasks) * 2)
    started = time.time()
    alignment, alignment_peak_rss = [], 0.0
    for result in _run_stage(_alignment_with_rss, tasks, args.workers):
        alignment.extend(result["rows"])
        alignment_peak_rss = max(alignment_peak_rss, result["peak_worker_rss_mib"])
    stage_stats["alignment_wall_seconds"] = time.time() - started
    if len(alignment) != len(tasks) * 2 * 6:
        raise AssertionError("decision-alignment grid incomplete")
    started = time.time()
    auxiliary, auxiliary_peak_rss = [], 0.0
    for result in _run_stage(_auxiliary_with_rss, tasks, args.workers):
        auxiliary.extend(result["rows"])
        auxiliary_peak_rss = max(auxiliary_peak_rss, result["peak_worker_rss_mib"])
    stage_stats["auxiliary_wall_seconds"] = time.time() - started
    if len(auxiliary) != len(tasks) * 2 * 2 * 4:
        raise AssertionError("auxiliary LRU/LFU score grid incomplete")

    winners = select_generic_winners(references, expected_seeds=SEEDS)
    seed_utility = derive_seed_utilities(utility_raw, references, winners)
    attribution = merge_seed_attribution(utility_raw, losses, orphaning)
    aggregate_utility = aggregate_seed_metrics(
        seed_utility,
        group_fields=("trace", "l1_fraction", "l2_multiplier", "cell", "target", "iteration"),
        metric_fields=("avoided_prefill_tokens", "l2_avoided_tokens",
                       "input_token_points_vs_pi0", "label_window_avoided_prefill_tokens"),
        expected_seeds=seeds)
    own, terminal, verdicts = _ranking_and_verdicts(
        matrix, recorded, seed_utility, training_models,
        training_populations + heldout_populations, names, cells, targets, seeds,
        numerical, args.smoke)
    if (len(own) != len(tasks) * 2 * 4 or len(terminal) != len(tasks)
        or len(verdicts) != len(names) * len(cells) * len(targets)
        or len(cosine) != len(tasks) * 16):
        raise AssertionError("ranking or stability grid incomplete")
    # A non-finite ranking is an unresolved scientific result, never a missing row.
    aggregate_ranking = []
    for name in names:
        for fraction, multiplier in cells:
            for target in targets:
                selected = [row for row in terminal if row["trace"] == name
                            and row["l1_fraction"] == fraction
                            and row["l2_multiplier"] == multiplier
                            and row["target"] == target]
                aggregate_ranking.append({
                    "trace": name, "l1_fraction": fraction, "l2_multiplier": multiplier,
                    "cell": cell_label(fraction, multiplier), "target": target,
                    "seeds": len(selected),
                    **{field + "_mean": (statistics.fmean(float(row[field]) for row in selected)
                                           if all(math.isfinite(float(row[field])) for row in selected)
                                           else math.nan)
                       for field in ("terminal_common_population_ranking_delta",
                                     "fit_population_ranking_delta", "own_policy_test_ranking_delta")},
                    "ranking_nonfinite": any(
                        not math.isfinite(float(row["terminal_common_population_ranking_delta"]))
                        for row in selected),
                })

    # Publication is an all-or-nothing gate for references, source, docs and
    # complete row identities.  Intermediate populations remain available for audit.
    if len(training_populations) != len(tasks) * 4 or len(heldout_populations) != len(tasks) * 4:
        raise AssertionError("population identities incomplete")
    if len(utility_raw) != len(tasks) * 4 or len(losses) != len(utility_raw):
        raise AssertionError("replay/attribution identities incomplete")
    head_end = _git("rev-parse", "HEAD")
    source_end = source_manifest()
    dirty_end = dirty_doc_hashes()
    if head_start != head_end or dirty_start != dirty_end:
        raise AssertionError("code commit or pre-existing dirty documentation changed during run")
    assert_manifests_equal(source_start["files"], source_end["files"])
    for info in trace_files.values():
        if sha256_path(info["path"]) != info["sha256"]:
            raise AssertionError(f"trace changed during run: {info['path']}")
    for relative, digest in reference_hashes.items():
        if sha256_path(REPOSITORY / relative) != digest:
            raise AssertionError(f"reference changed during run: {relative}")
    if sha256_path(CANDIDATE_META) != candidate_meta_hash:
        raise AssertionError("candidate metadata changed during run")
    for row in candidate_rows:
        if sha256_path(row["artifact_path"]) != row["current_sha256"]:
            raise AssertionError(f"candidate artifact changed during run: {row['artifact_path']}")

    config = {
        "phase": "onpolicy_learning", "smoke": bool(args.smoke),
        "preregistration_commit": _git("rev-parse", f"{PREREGISTRATION_COMMIT}^{{commit}}"),
        "code_commit": head_start, "git_head_start": head_start, "git_head_end": head_end,
        "source_manifest_start": source_start, "source_manifest_end": source_end,
        "preexisting_document_diff_sha256_start": dirty_start,
        "preexisting_document_diff_sha256_end": dirty_end,
        "trace_files": trace_files, "reference_sha256": reference_hashes,
        "phase1_reference_code_commit": phase1.get("code_commit"),
        "trace_end_ms": {name: traces[name].end_ms for name in names},
        "candidate_metadata_sha256": candidate_meta_hash,
        "candidate_hash_provenance": "contemporaneous_only; Phase 0.97 did not publish per-log hashes",
        "cells": cells, "targets": targets, "seeds": seeds,
        "iterations": ITERATIONS, "horizons_seconds": horizons,
        "measure_from_ms": splits, "working_set_bytes": working_sets,
        "l1_policy": L1_POLICY, "hit_model": HIT_MODEL, "closure": CLOSURE,
        "l2_arrival_protection": "none", "size_model": HYPERPARAMETERS["size_model"],
        "bytes_per_token": HYPERPARAMETERS["bytes_per_token"],
        "sample_width": SAMPLE_WIDTH, "decision_cap": DECISION_CAP,
        "max_train_rows": MAX_TRAIN_ROWS,
        "negatives_per_positive": NEGATIVES_PER_POSITIVE,
        "fit_seed": FIT_SEED, "collection_seed_role": "lineage seed",
        "heldout_seed_role": "same lineage seed; cold replay",
        "training_barrier": asdict(barrier),
        "serialized_model_bytes": sum(Path(row["model_path"]).stat().st_size
                                      for row in canonical_models),
        "spooled_population_bytes": sum(Path(row["population_path"]).stat().st_size
                                        for row in training_populations + heldout_populations),
        "run_directory_bytes_before_publication": sum(
            path.stat().st_size for path in output_dir.rglob("*") if path.is_file()),
        "training_prefix_replays": len(tasks) * 4,
        "heldout_replays": len(tasks) * 4,
        "published_pi0_rows_matched": pi0_check.row_count,
        "phase098b_pi0_reference": loss_pi0_count,
        "crossscore_validation": numerical,
        "workers": args.workers,
        "worker_rss_unit": "MiB; Linux ru_maxrss KiB / 1024",
        "stage_peak_worker_rss_mib": {
            "training": training_peak_rss,
            "heldout": max(float(result["peak_worker_rss_mib"])
                           for result in heldout_workers),
            "crossscore": max(float(result["peak_worker_rss_mib"])
                              for result in crossscore_workers),
            "alignment": alignment_peak_rss,
            "auxiliary": auxiliary_peak_rss,
        },
        "peak_worker_rss_mib": max(
            training_peak_rss, alignment_peak_rss, auxiliary_peak_rss,
            *(float(result["peak_worker_rss_mib"])
              for result in heldout_workers + crossscore_workers)),
        "stage_seconds": stage_stats,
        "wall_seconds": time.time() - began,
    }
    tables = {
        "pi0_fit_checks": fit_checks,
        "model_links": training_models,
        "canonical_models": canonical_models,
        "training_populations": training_populations,
        "test_populations": heldout_populations,
        "cross_score_matrix": matrix,
        "own_policy_ranking": own,
        "terminal_ranking": terminal,
        "coefficient_similarity": cosine,
        "decision_alignment": alignment,
        "auxiliary_lru_lfu_ranking": auxiliary,
        "seed_utility": seed_utility,
        "aggregate_utility": aggregate_utility,
        "aggregate_ranking": aggregate_ranking,
        "attribution_loss": losses,
        "attribution_orphaning": orphaning,
        "attribution_complete": attribution,
        "candidate_reference_checks": candidate_rows,
        "registered_verdicts": verdicts,
    }
    destination = _publish_outputs(output_dir, args.paper_dir.resolve(), tables, config,
                                   smoke=args.smoke, names=names, cells=cells,
                                   targets=targets, seeds=seeds)
    print(f"on-policy artifacts: {destination}", flush=True)
    return config


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
