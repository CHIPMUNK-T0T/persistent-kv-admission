#!/usr/bin/env python3
"""Run the pre-registered Phase 1 arrival-protection intervention."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from persistent_kv_admission.crossworkload import HYPERPARAMETERS, TARGETS
from persistent_kv_admission.decisionpop import (
    MAX_TRAIN_ROWS,
    NEGATIVES_PER_POSITIVE,
    RawFeatureScorer,
    Rows,
    coefficient_row,
    fit_population_ranker,
    horizon_for,
    state_indices,
)
from persistent_kv_admission.gap import summarize, working_set_bytes
from persistent_kv_admission.intervention import (
    Phase1Collector,
    PROTECTION_EVENTS,
    paired_request_summary,
)
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.temporal import FEATURE_NAMES
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

PHASE097_SCRIPT = REPOSITORY / "scripts/run_decision_population.py"
PHASE097_CONFIG = REPOSITORY / "results/paper/decision_population_config.json"
PHASE097_FITS = REPOSITORY / "results/paper/decision_population_fits.csv"
PHASE097_REPLAYS = REPOSITORY / "results/paper/decision_population_replay_seeds.csv"
PHASE098_REPLAYS = REPOSITORY / "results/paper/decision_attribution_losses_seeds.csv"
VICTIM_LOG_DIR = REPOSITORY / "results/decision_population/logs"
PLAN_PATH = REPOSITORY / "docs/phase1-intervention-plan.md"
PREEXISTING_DIRTY_DOCS = (
    "README.md",
    "docs/decision-population-findings.md",
    "docs/experiment-plan.md",
)

CELLS = ((0.0025, 1.0), (0.01, 4.0), (0.02, 4.0))
INTERVENTION_TARGETS = ("next_use", "binary")
SEEDS = (0, 1, 2, 3, 4)
VARIANTS = ("none", "direct_child", "all")
PAIR_DIRECTIONS = (
    ("direct_child", "none"),
    ("all", "none"),
    ("direct_child", "all"),
)
SAMPLE_WIDTH = 16
REFERENCE_ARMS = (
    ("sampled", "lru_s"),
    ("sampled", "lfu_s"),
    ("sampled", "lru_2hit_s"),
    ("heap", "lru"),
    ("heap", "lfu"),
    ("heap", "lru_2hit"),
    ("heap", "offline_next_use"),
)
FIT_TOLERANCE = 1e-12
REFERENCE_TOLERANCE = 1e-9
SMOKE_CELL = (0.01, 4.0)
SMOKE_TARGETS = ("next_use",)
SMOKE_SEEDS = (0,)
SHARED: dict[str, object] = {}


def _load_phase097():
    spec = importlib.util.spec_from_file_location("run_decision_population", PHASE097_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rdp = _load_phase097()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/phase1_intervention")
    )
    parser.add_argument("--paper-dir", type=Path, default=Path("results/paper"))
    parser.add_argument(
        "--workers", type=int, default=max(1, min(20, os.cpu_count() or 1))
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one trace, one cell, one target and one seed; never copied to paper-dir",
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


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> dict[str, object]:
    paths = sorted((REPOSITORY / "src").glob("**/*.py"))
    paths.extend((Path(__file__).resolve(), PHASE097_SCRIPT))
    files = {
        str(path.relative_to(REPOSITORY)): sha256_path(path)
        for path in sorted(set(paths))
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()}


def dirty_doc_hashes() -> dict[str, str]:
    out = {}
    for name in PREEXISTING_DIRTY_DOCS:
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--", name],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        ).stdout
        out[name] = hashlib.sha256(diff).hexdigest()
    return out


def ensure_execution_tree_committed() -> None:
    paths = (
        "src",
        "scripts/run_phase1_intervention.py",
        "scripts/run_decision_population.py",
        "tests/test_phase1_intervention.py",
        "docs/phase1-intervention-plan.md",
    )
    changed = _git("diff", "HEAD", "--name-only", "--", *paths)
    if changed:
        raise SystemExit(f"Phase 1 execution files have uncommitted changes: {changed}")
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *paths)
    if untracked:
        raise SystemExit(f"Phase 1 execution files are untracked: {untracked}")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def same_number(left, right, tolerance: float) -> bool:
    try:
        a = float(left)
        b = float(right)
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
    working_set: dict,
) -> list[str]:
    errors: list[str] = []
    expected_scalars = {
        "l1_policy": "lru",
        "hit_model": "tree",
        "closure": "union",
        "sample_width": SAMPLE_WIDTH,
        "max_train_rows": MAX_TRAIN_ROWS,
        "negatives_per_positive": NEGATIVES_PER_POSITIVE,
    }
    for key, expected in expected_scalars.items():
        if config.get(key) != expected:
            errors.append(f"config {key}: {config.get(key)!r} != {expected!r}")
    if config.get("hyperparameters") != HYPERPARAMETERS:
        errors.append("config hyperparameters differ from current fixed values")
    for target in INTERVENTION_TARGETS:
        if target not in config.get("targets", []):
            errors.append(f"config lacks target {target}")
    for fraction, multiplier in CELLS:
        if fraction not in config.get("l1_budgets", []):
            errors.append(f"config lacks L1 fraction {fraction}")
        if multiplier not in config.get("l2_multipliers", []):
            errors.append(f"config lacks L2 multiplier {multiplier}")
    for name in traces:
        checks = (
            ("measure_from_ms", splits[name]),
            ("horizons_seconds", horizons[name]),
            ("working_set_bytes", working_set[name]),
        )
        for field, expected in checks:
            actual = config.get(field, {}).get(name)
            if not same_number(actual, expected, 0.0):
                errors.append(f"config {field}/{name}: {actual!r} != {expected!r}")
    return errors


def published_fit_index() -> dict[tuple[str, str, float], dict[str, str]]:
    out = {}
    for row in load_csv(PHASE097_FITS):
        if row["population"] != "B":
            continue
        key = (row["trace"], row["target"], float(row["l1_fraction"]))
        if key in out:
            raise AssertionError(f"duplicate published B fit {key}")
        out[key] = row
    return out


def fit_detail(ranker) -> dict[str, object]:
    return {
        "ranker_class": type(ranker).__name__,
        "indices": list(ranker.indices),
        "l2": float(ranker.l2),
        "standardize": bool(ranker.standardize),
        "mean": np.asarray(ranker.mean, dtype=float).tolist(),
        "scale": np.asarray(ranker.scale, dtype=float).tolist(),
        "coefficients": np.asarray(ranker.coefficients, dtype=float).tolist(),
        "intercept": float(ranker.intercept),
        "iterations": int(getattr(ranker, "iterations", -1)),
        "converged": bool(getattr(ranker, "converged", False)),
        "condition_number": float(getattr(ranker, "condition_number", math.nan)),
    }


def rebuild_b_fits(
    traces: dict,
    splits: dict,
    horizons: dict,
    fractions: tuple[float, ...],
    targets: tuple[str, ...],
) -> tuple[dict, list[dict], list[dict]]:
    published = published_fit_index()
    rankers: dict[tuple, object] = {}
    checks: list[dict] = []
    details: list[dict] = []
    for name in sorted(traces):
        for fraction in fractions:
            path = VICTIM_LOG_DIR / f"victims_{name}_{fraction:g}.npz"
            if not path.exists():
                raise SystemExit(f"missing fixed victim artifact {path}")
            rows = Rows.load(path)
            if not same_number(rows.horizon_seconds, horizons[name], 0.0):
                raise SystemExit(f"victim horizon mismatch in {path}")
            train = rows.select(rows.train_mask(splits[name]))
            for target in targets:
                started = time.time()
                ranker, positives, used = fit_population_ranker(train, target, seed=0)
                key = (name, target, fraction)
                reference = published.get(key)
                if reference is None:
                    raise SystemExit(f"published Phase 0.97 B fit missing: {key}")
                metadata = {
                    "horizon_seconds": horizons[name],
                    "split_ms": splits[name],
                    "train_rows": int(used),
                    "train_positives": int(positives),
                    "l1_fraction": fraction,
                    "population_rows": int(len(rows)),
                    "train_pool_rows": int(len(train)),
                }
                mismatches = []
                for field, value in metadata.items():
                    if not same_number(value, reference.get(field), FIT_TOLERANCE):
                        mismatches.append(
                            f"{field}: {value} != {reference.get(field)}"
                        )
                coefficients = coefficient_row(ranker)
                deviations = {
                    feature: abs(float(value) - float(reference[feature]))
                    for feature, value in coefficients.items()
                }
                maximum = max(deviations.values(), default=0.0)
                if maximum > FIT_TOLERANCE:
                    mismatches.append(f"coefficient deviation {maximum:.3e}")
                if mismatches:
                    raise SystemExit(
                        f"Phase 0.97 B fit mismatch {key}: " + "; ".join(mismatches)
                    )
                ranker_key = rdp.ranker_key(name, target, "B", fraction, 1.0)
                rankers[ranker_key] = ranker
                identity = {
                    "trace": name,
                    "target": target,
                    "population": "B",
                    "cell": rdp.cell_label(fraction),
                    "l1_fraction": fraction,
                }
                checks.append(
                    {
                        **identity,
                        **metadata,
                        "victim_artifact": str(path.relative_to(REPOSITORY)),
                        "victim_artifact_sha256": sha256_path(path),
                        "max_coefficient_deviation": maximum,
                        "reference_matched": True,
                        "seconds": time.time() - started,
                    }
                )
                details.append(
                    {
                        **identity,
                        "victim_artifact_sha256": sha256_path(path),
                        **fit_detail(ranker),
                        "standardized_coefficients": coefficients,
                    }
                )
    return rankers, checks, details


def _worker(task):
    name, fraction, multiplier, target, seed = task
    trace = SHARED["traces"][name]
    split_ms = SHARED["splits"][name]
    ranker = SHARED["rankers"][
        rdp.ranker_key(name, target, "B", fraction, multiplier)
    ]
    l1_bytes = rdp._capacity(name, fraction)
    l2_bytes = rdp._capacity(name, fraction * multiplier)
    replay_rows = []
    attribution_rows = []
    outcomes = {}
    for variant in VARIANTS:
        collector = Phase1Collector(trace, measure_from_ms=split_ms)
        scorer = RawFeatureScorer(trace, ranker)
        started = time.time()
        result = run_two_tier(
            trace,
            rdp.L1_POLICY,
            l1_bytes,
            l2_bytes,
            "learned",
            hit_model=rdp.HIT_MODEL,
            closure=rdp.CLOSURE,
            occurrence_groups=SHARED["groups"][name],
            measure_from_ms=split_ms,
            l2_eviction="sampled",
            l2_sample_width=SAMPLE_WIDTH,
            l2_seed=seed,
            l2_scorer=scorer,
            l2_arm=f"B_{variant}",
            l2_request_hook=collector.on_request,
            l2_removal_hook=collector,
            l2_arrival_protection=variant,
            l2_protection_hook=collector.on_protection,
        )
        collector.finish(result)
        seconds = time.time() - started
        identity = {
            "trace": name,
            "l1_fraction": fraction,
            "l2_multiplier": multiplier,
            "cell": rdp.cell_label(fraction, multiplier),
            "target": target,
            "seed": seed,
            "arm": "B",
            "variant": variant,
        }
        replay_rows.append(
            {
                **identity,
                **result.as_row(),
                **collector.protection_row(),
                "seconds": seconds,
            }
        )
        attribution_rows.append(
            {
                **identity,
                "requested_tokens": result.requested_tokens,
                "measured_requests": result.measured_requests,
                "l1_avoided_tokens": result.l1_avoided_tokens,
                "l2_avoided_tokens": result.l2_avoided_tokens,
                "avoided_prefill_tokens": result.avoided_prefill_tokens,
                "l2_present_unusable_tokens": result.l2_present_unusable_tokens,
                **collector.attribution.loss_row(),
                **collector.attribution.orphaning_row(),
            }
        )
        outcomes[variant] = collector.outcomes
    pair_rows = []
    for left, right in PAIR_DIRECTIONS:
        pair_rows.append(
            {
                "trace": name,
                "l1_fraction": fraction,
                "l2_multiplier": multiplier,
                "cell": rdp.cell_label(fraction, multiplier),
                "target": target,
                "seed": seed,
                "left_variant": left,
                "right_variant": right,
                "comparison": f"{left}-{right}",
                **paired_request_summary(outcomes[left], outcomes[right]),
            }
        )
    return {
        "task": task,
        "replay": replay_rows,
        "attribution": attribution_rows,
        "pairs": pair_rows,
    }


def build_tasks(names, cells, targets, seeds):
    return [
        (name, fraction, multiplier, target, seed)
        for name in names
        for fraction, multiplier in cells
        for target in targets
        for seed in seeds
    ]


def original_reference_checks(
    replay_rows: list[dict],
    attribution_rows: list[dict],
    smoke: bool,
) -> tuple[int, int]:
    phase097 = {}
    for row in load_csv(PHASE097_REPLAYS):
        if row["arm"] == "B" and row["target"] in INTERVENTION_TARGETS:
            key = (
                row["trace"],
                float(row["l1_fraction"]),
                float(row["l2_multiplier"]),
                row["target"],
                int(row["seed"]),
            )
            phase097[key] = row
    phase098 = {}
    for row in load_csv(PHASE098_REPLAYS):
        if row["arm"] == "B":
            key = (
                row["trace"],
                float(row["l1_fraction"]),
                float(row["l2_multiplier"]),
                row["target"],
                int(row["seed"]),
            )
            phase098[key] = row
    originals = [row for row in replay_rows if row["variant"] == "none"]
    attrs = {
        (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            row["target"],
            int(row["seed"]),
        ): row
        for row in attribution_rows
        if row["variant"] == "none"
    }
    matched097 = matched098 = 0
    for row in originals:
        key = (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            row["target"],
            int(row["seed"]),
        )
        ref097 = phase097.get(key)
        if ref097 is None:
            raise SystemExit(f"Phase 0.97 B reference missing: {key}")
        if int(row["avoided_prefill_tokens"]) != int(ref097["avoided_prefill_tokens"]):
            raise SystemExit(
                f"Phase 0.97 B reproduction failed {key}: "
                f"{row['avoided_prefill_tokens']} != {ref097['avoided_prefill_tokens']}"
            )
        matched097 += 1
        ref098 = phase098.get(key)
        if ref098 is None:
            raise SystemExit(f"Phase 0.98b B reference missing: {key}")
        mine = attrs[key]
        checked = (
            "requested_tokens",
            "measured_requests",
            "l1_avoided_tokens",
            "l2_avoided_tokens",
            "avoided_prefill_tokens",
            "l2_present_unusable_tokens",
        )
        checked += tuple(
            column
            for column in mine
            if column.startswith(
                (
                    "attribution_",
                    "fully_",
                    "beyond_",
                    "l2_hit_",
                    "downstream_",
                    "root_",
                    "unusable_",
                    "absent_",
                    "perblock_",
                    "unexplained_",
                )
            )
        )
        for column in checked:
            if column not in ref098 or not same_number(
                mine[column], ref098[column], REFERENCE_TOLERANCE
            ):
                raise SystemExit(
                    f"Phase 0.98b B attribution failed {key}/{column}: "
                    f"{mine.get(column)!r} != {ref098.get(column)!r}"
                )
        matched098 += 1
    expected = 1 if smoke else 60
    if matched097 != expected or matched098 != expected:
        raise SystemExit(
            f"original B reference count {matched097}/{matched098}, expected {expected}"
        )
    return matched097, matched098


def select_and_verify_references(
    replay_rows: list[dict],
    names: tuple[str, ...],
    cells: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
) -> list[dict]:
    wanted = set(REFERENCE_ARMS)
    originals = {
        (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            int(row["seed"]),
        ): row
        for row in replay_rows
        if row["variant"] == "none"
    }
    selected = []
    seen = set()
    for row in load_csv(PHASE097_REPLAYS):
        identity = (row["kind"], row["arm"])
        key = (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            int(row["seed"]),
        )
        if (
            identity not in wanted
            or row["trace"] not in names
            or (key[1], key[2]) not in cells
            or key[3] not in seeds
        ):
            continue
        original = originals.get(key)
        if original is None:
            raise SystemExit(f"reference has no matching original B row: {key}")
        exact_fields = (
            "l1_capacity_bytes",
            "l2_capacity_bytes",
            "requested_tokens",
            "l1_avoided_tokens",
        )
        for field in exact_fields:
            if int(row[field]) != int(original[field]):
                raise SystemExit(
                    f"Phase 0.97 reference mismatch {key}/{identity}/{field}"
                )
        for field, expected in (
            ("l1_policy", "lru"),
            ("hit_model", "tree"),
            ("closure", "union"),
        ):
            if row[field] != expected:
                raise SystemExit(
                    f"Phase 0.97 reference mismatch {key}/{identity}/{field}"
                )
        full_key = key + identity
        if full_key in seen:
            raise SystemExit(f"duplicate Phase 0.97 reference {full_key}")
        seen.add(full_key)
        selected.append(
            {
                "trace": row["trace"],
                "l1_fraction": float(row["l1_fraction"]),
                "l2_multiplier": float(row["l2_multiplier"]),
                "cell": rdp.cell_label(key[1], key[2]),
                "seed": int(row["seed"]),
                "kind": row["kind"],
                "arm": row["arm"],
                "l1_capacity_bytes": int(row["l1_capacity_bytes"]),
                "l2_capacity_bytes": int(row["l2_capacity_bytes"]),
                "requested_tokens": int(row["requested_tokens"]),
                "l1_avoided_tokens": int(row["l1_avoided_tokens"]),
                "l2_avoided_tokens": int(row["l2_avoided_tokens"]),
                "avoided_prefill_tokens": int(row["avoided_prefill_tokens"]),
                "l2_present_unusable_tokens": int(
                    row["l2_present_unusable_tokens"]
                ),
            }
        )
    expected = len(names) * len(cells) * len(seeds) * len(REFERENCE_ARMS)
    if len(selected) != expected:
        raise SystemExit(
            f"Phase 0.97 generic/offline references {len(selected)}, expected {expected}"
        )
    return selected


def attach_reference_context(
    replay_rows: list[dict],
    pair_rows: list[dict],
    references: list[dict],
) -> None:
    by_cell: dict[tuple, dict[str, dict]] = defaultdict(dict)
    sampled_values: dict[tuple, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in references:
        key = (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            int(row["seed"]),
        )
        by_cell[key][row["arm"]] = row
        if row["arm"] in ("lru_s", "lfu_s", "lru_2hit_s"):
            sampled_values[key[:3]][row["arm"]].append(
                int(row["avoided_prefill_tokens"])
            )
    best_sampled_by_cell = {
        key: max(
            ("lru_s", "lfu_s", "lru_2hit_s"),
            key=lambda arm: (
                sum(arms[arm]) / len(arms[arm]),
                -REFERENCE_ARMS.index(("sampled", arm)),
            ),
        )
        for key, arms in sampled_values.items()
    }
    for row in replay_rows:
        key = (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            int(row["seed"]),
        )
        refs = by_cell[key]
        best_heap = max(
            (refs[arm] for arm in ("lru", "lfu", "lru_2hit")),
            key=lambda item: int(item["avoided_prefill_tokens"]),
        )
        best_sampled = refs[best_sampled_by_cell[key[:3]]]
        offline = refs["offline_next_use"]
        row.update(
            best_heap_arm=best_heap["arm"],
            best_heap_avoided_tokens=best_heap["avoided_prefill_tokens"],
            delta_vs_best_heap_tokens=int(row["avoided_prefill_tokens"])
            - int(best_heap["avoided_prefill_tokens"]),
            best_sampled_arm=best_sampled["arm"],
            best_sampled_avoided_tokens=best_sampled["avoided_prefill_tokens"],
            delta_vs_best_sampled_tokens=int(row["avoided_prefill_tokens"])
            - int(best_sampled["avoided_prefill_tokens"]),
            offline_avoided_tokens=offline["avoided_prefill_tokens"],
        )
    replay_index = {
        (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            row["target"],
            int(row["seed"]),
            row["variant"],
        ): row
        for row in replay_rows
    }
    for row in pair_rows:
        key = (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            row["target"],
            int(row["seed"]),
            row["left_variant"],
        )
        left = replay_index[key]
        row.update(
            best_heap_arm=left["best_heap_arm"],
            best_heap_avoided_tokens=left["best_heap_avoided_tokens"],
            left_delta_vs_best_heap_tokens=left["delta_vs_best_heap_tokens"],
            best_sampled_arm=left["best_sampled_arm"],
            best_sampled_avoided_tokens=left["best_sampled_avoided_tokens"],
            left_delta_vs_best_sampled_tokens=left["delta_vs_best_sampled_tokens"],
            offline_avoided_tokens=left["offline_avoided_tokens"],
        )

def aggregate(
    rows: list[dict],
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
    carry: tuple[str, ...] = (),
    directional_metric: str | None = None,
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key, members in sorted(groups.items()):
        entry = dict(zip(keys, key))
        entry["seeds"] = len(members)
        for name in carry:
            values = {str(member[name]) for member in members}
            if len(values) != 1:
                raise AssertionError(f"carry field {name} varies in group {key}")
            entry[name] = members[0][name]
        for metric in metrics:
            values = [float(member[metric]) for member in members]
            stats = summarize(values)
            entry[f"{metric}_mean"] = stats["mean"]
            entry[f"{metric}_std"] = stats["std"]
            entry[f"{metric}_ci95_half"] = stats["ci95_half"]
            entry[f"{metric}_seed_min"] = min(values)
            entry[f"{metric}_seed_max"] = max(values)
        if directional_metric is not None:
            values = [float(member[directional_metric]) for member in members]
            entry["all_seeds_positive"] = all(value > 0 for value in values)
            entry["all_seeds_negative"] = all(value < 0 for value in values)
            entry["mixed_or_zero_seed_sign"] = not (
                entry["all_seeds_positive"] or entry["all_seeds_negative"]
            )
        out.append(entry)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def output_files(
    directory: Path,
    replay_rows: list[dict],
    replay_summary: list[dict],
    attribution_rows: list[dict],
    attribution_summary: list[dict],
    pair_rows: list[dict],
    pair_summary: list[dict],
    references: list[dict],
    fit_checks: list[dict],
    fit_details: list[dict],
    config: dict,
) -> None:
    write_csv(directory / "phase1_intervention_replay_seeds.csv", replay_rows)
    write_csv(directory / "phase1_intervention_replay.csv", replay_summary)
    write_csv(
        directory / "phase1_intervention_attribution_seeds.csv", attribution_rows
    )
    write_csv(
        directory / "phase1_intervention_attribution.csv", attribution_summary
    )
    write_csv(directory / "phase1_intervention_pairs_seeds.csv", pair_rows)
    write_csv(directory / "phase1_intervention_pairs.csv", pair_summary)
    write_csv(directory / "phase1_intervention_references.csv", references)
    write_csv(directory / "phase1_intervention_fit_checks.csv", fit_checks)
    (directory / "phase1_intervention_fits.json").write_text(
        json.dumps(fit_details, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (directory / "phase1_intervention_config.json").write_text(
        json.dumps(config, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    ensure_execution_tree_committed()
    head_start = _git("rev-parse", "HEAD")
    source_start = source_manifest()
    dirty_start = dirty_doc_hashes()
    clock = time.time()

    traces = {}
    groups = {}
    splits = {}
    horizons = {}
    working_set = {}
    indices = {}
    trace_hashes = {}
    for path in args.traces:
        trace = load_mooncake_trace(path, 512)
        if trace.name in traces:
            raise SystemExit(f"duplicate trace name {trace.name}")
        horizon, split_ms, _ = horizon_for(trace)
        traces[trace.name] = trace
        groups[trace.name] = _occurrence_groups(trace)
        splits[trace.name] = split_ms
        horizons[trace.name] = horizon
        working_set[trace.name] = working_set_bytes(trace)
        indices[trace.name] = state_indices(trace)
        trace_hashes[trace.name] = {
            "path": str(path.resolve()),
            "sha256": sha256_path(path),
        }
        print(
            f"loaded {trace.name}: requests={len(trace.requests)} "
            f"H={horizon:g}s split={split_ms:.1f}ms",
            flush=True,
        )

    expected_names = {"conversation_trace"} if args.smoke else {
        "conversation_trace",
        "toolagent_trace",
    }
    if set(traces) != expected_names:
        raise SystemExit(
            f"trace names {sorted(traces)} do not match fixed grid {sorted(expected_names)}"
        )
    cells = (SMOKE_CELL,) if args.smoke else CELLS
    targets = SMOKE_TARGETS if args.smoke else INTERVENTION_TARGETS
    seeds = SMOKE_SEEDS if args.smoke else SEEDS
    names = tuple(sorted(traces))
    fractions = tuple(sorted({fraction for fraction, _ in cells}))

    config097 = json.loads(PHASE097_CONFIG.read_text(encoding="utf-8"))
    config_errors = verify_phase097_config(
        config097, traces, splits, horizons, working_set
    )
    if config_errors:
        raise SystemExit("Phase 0.97 config mismatch: " + "; ".join(config_errors))

    rdp._SHARED.update(
        traces=traces,
        groups=groups,
        splits=splits,
        horizons=horizons,
        working_set=working_set,
        state_index=indices,
        targets=targets,
    )
    rankers, fit_checks, fit_details = rebuild_b_fits(
        traces, splits, horizons, fractions, targets
    )
    rdp._SHARED["rankers"] = rankers
    SHARED.update(
        traces=traces,
        groups=groups,
        splits=splits,
        rankers=rankers,
    )

    tasks = build_tasks(names, cells, targets, seeds)
    print(
        f"{len(tasks)} paired tasks = {len(tasks) * len(VARIANTS)} replays "
        f"on {args.workers} workers",
        flush=True,
    )
    replay_rows: list[dict] = []
    attribution_rows: list[dict] = []
    pair_rows: list[dict] = []
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(min(args.workers, len(tasks))) as pool:
        for done, result in enumerate(
            pool.imap_unordered(_worker, tasks, chunksize=1), 1
        ):
            replay_rows.extend(result["replay"])
            attribution_rows.extend(result["attribution"])
            pair_rows.extend(result["pairs"])
            if done % 5 == 0 or done == len(tasks):
                print(
                    f"[{done}/{len(tasks)}] elapsed={time.time() - started:.0f}s "
                    f"last={result['task']}",
                    flush=True,
                )
    replay_seconds = time.time() - started

    matched097, matched098 = original_reference_checks(
        replay_rows, attribution_rows, args.smoke
    )
    references = select_and_verify_references(
        replay_rows, names, cells, seeds
    )
    attach_reference_context(replay_rows, pair_rows, references)

    replay_rows.sort(
        key=lambda row: (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            row["target"],
            row["variant"],
            int(row["seed"]),
        )
    )
    attribution_rows.sort(
        key=lambda row: (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            row["target"],
            row["variant"],
            int(row["seed"]),
        )
    )
    pair_rows.sort(
        key=lambda row: (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            row["target"],
            tuple(PAIR_DIRECTIONS).index(
                (row["left_variant"], row["right_variant"])
            ),
            int(row["seed"]),
        )
    )
    references.sort(
        key=lambda row: (
            row["trace"],
            float(row["l1_fraction"]),
            float(row["l2_multiplier"]),
            int(row["seed"]),
            row["kind"],
            row["arm"],
        )
    )

    replay_metrics = (
        "avoided_prefill_tokens",
        "l2_avoided_tokens",
        "l2_present_unusable_tokens",
        "l2_admissions",
        "l2_rejections",
        "l2_evictions",
        "l2_decisions",
        "delta_vs_best_heap_tokens",
        "delta_vs_best_sampled_tokens",
        "best_sampled_avoided_tokens",
        "seconds",
    ) + tuple(
        f"{event}{suffix}"
        for event in PROTECTION_EVENTS
        for suffix in ("_count", "_window_count")
    )
    replay_summary = aggregate(
        replay_rows,
        ("trace", "l1_fraction", "l2_multiplier", "cell", "target", "variant"),
        replay_metrics,
        carry=(
            "requested_tokens",
            "l1_avoided_tokens",
            "best_heap_arm",
            "best_heap_avoided_tokens",
            "best_sampled_arm",
            "offline_avoided_tokens",
        ),
    )
    attribution_metrics = tuple(
        field
        for field, value in attribution_rows[0].items()
        if field
        not in {
            "trace",
            "l1_fraction",
            "l2_multiplier",
            "cell",
            "target",
            "seed",
            "arm",
            "variant",
            "requested_tokens",
            "measured_requests",
            "l1_avoided_tokens",
        }
        and isinstance(value, (int, float))
    )
    attribution_summary = aggregate(
        attribution_rows,
        ("trace", "l1_fraction", "l2_multiplier", "cell", "target", "variant"),
        attribution_metrics,
        carry=("requested_tokens", "measured_requests", "l1_avoided_tokens"),
    )
    pair_metrics = (
        "left_avoided_tokens",
        "right_avoided_tokens",
        "saved_requests",
        "lost_requests",
        "unchanged_requests",
        "differing_requests",
        "saved_tokens",
        "lost_tokens",
        "net_avoided_tokens",
        "net_fraction_of_input",
        "net_input_percentage_points",
        "left_delta_vs_best_heap_tokens",
        "left_delta_vs_best_sampled_tokens",
        "best_sampled_avoided_tokens",
    )
    pair_summary = aggregate(
        pair_rows,
        (
            "trace",
            "l1_fraction",
            "l2_multiplier",
            "cell",
            "target",
            "left_variant",
            "right_variant",
            "comparison",
        ),
        pair_metrics,
        carry=(
            "requested_tokens",
            "measured_requests",
            "best_heap_arm",
            "best_heap_avoided_tokens",
            "best_sampled_arm",
            "offline_avoided_tokens",
        ),
        directional_metric="net_avoided_tokens",
    )

    source_end = source_manifest()
    dirty_end = dirty_doc_hashes()
    head_end = _git("rev-parse", "HEAD")
    if source_end != source_start:
        raise SystemExit("execution source changed during the run; no output written")
    if dirty_end != dirty_start:
        raise SystemExit(
            "pre-existing documentation diffs changed during the run; no output written"
        )
    if head_end != head_start:
        raise SystemExit("HEAD changed during the run; no output written")

    prereg_commit = _git(
        "log",
        "-1",
        "--format=%H",
        "--",
        str(PLAN_PATH.relative_to(REPOSITORY)),
    )
    config = {
        "phase": "1",
        "smoke": bool(args.smoke),
        "preregistration_commit": prereg_commit,
        "code_commit": head_start,
        "git_head_start": head_start,
        "git_head_end": head_end,
        "git_dirty": bool(_git("status", "--porcelain")),
        "source_manifest_start": source_start,
        "source_manifest_end": source_end,
        "preexisting_document_diff_sha256_start": dirty_start,
        "preexisting_document_diff_sha256_end": dirty_end,
        "trace_files": trace_hashes,
        "cells": [list(cell) for cell in cells],
        "targets": list(targets),
        "seeds": list(seeds),
        "variants": list(VARIANTS),
        "pair_directions": [list(pair) for pair in PAIR_DIRECTIONS],
        "replays": len(replay_rows),
        "paired_tasks": len(tasks),
        "workers": args.workers,
        "replay_seconds": replay_seconds,
        "wall_seconds": time.time() - clock,
        "l1_policy": rdp.L1_POLICY,
        "hit_model": rdp.HIT_MODEL,
        "closure": rdp.CLOSURE,
        "size_model": HYPERPARAMETERS["size_model"],
        "bytes_per_token": HYPERPARAMETERS["bytes_per_token"],
        "sample_width": SAMPLE_WIDTH,
        "hyperparameters": HYPERPARAMETERS,
        "horizons_seconds": horizons,
        "measure_from_ms": splits,
        "working_set_bytes": working_set,
        "phase097_config": str(PHASE097_CONFIG),
        "phase097_config_verified": True,
        "phase097_fit_reference": str(PHASE097_FITS),
        "fit_checks": len(fit_checks),
        "fit_max_coefficient_deviation": max(
            check["max_coefficient_deviation"] for check in fit_checks
        ),
        "phase097_replay_reference": str(PHASE097_REPLAYS),
        "phase097_original_b_matched": matched097,
        "phase098b_replay_reference": str(PHASE098_REPLAYS),
        "phase098b_original_b_matched": matched098,
        "generic_offline_reference_rows": len(references),
        "reference_tolerance": REFERENCE_TOLERANCE,
        "note": (
            "The three variants use the same reconstructed and coefficient-verified "
            "Phase 0.97 B ranker. Request differences are paired trajectory outcomes, "
            "not one-decision causal attributions. Seed intervals describe sampling-seed "
            "variability only."
        ),
    }

    output_files(
        args.output_dir,
        replay_rows,
        replay_summary,
        attribution_rows,
        attribution_summary,
        pair_rows,
        pair_summary,
        references,
        fit_checks,
        fit_details,
        config,
    )
    if not args.smoke:
        output_files(
            args.paper_dir,
            replay_rows,
            replay_summary,
            attribution_rows,
            attribution_summary,
            pair_rows,
            pair_summary,
            references,
            fit_checks,
            fit_details,
            config,
        )
    print(
        f"verified original B: Phase0.97={matched097}, Phase0.98b={matched098}; "
        f"references={len(references)}; done in {time.time() - clock:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
