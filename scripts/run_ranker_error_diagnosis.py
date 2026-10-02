#!/usr/bin/env python3
"""Ranker-error diagnosis on saved decision logs: selection or population, and which resident errors.

The pre-registration is `docs/ranker-error-diagnosis-plan.md` with its
pre-computation addendum; nothing here may change it, and the run config
records the last commit touching it. The run reads the held-out decision
populations `pi0` and `pi3` of the published on-policy run for every trace x
cell x target x seed, the published `pi0` and `pi3` models of the same lineage
and the published per-seed utilities. It replays nothing and fits nothing; the
arithmetic is `persistent_kv_admission.errordiag`.

Before anything is derived the run checks that every published table read
carries the SHA-256 the on-policy run recorded for it; that the canonical
models, the per-lineage model links and the run's own model manifest agree;
that every model and every population has the published SHA-256 (a population
is hashed on the exact bytes it is loaded from); that every population has the
published row and decision counts, the test window, H = 600 s and its lineage
seed; and that the published utilities are matched to the populations on
trace, cell, target, seed and iteration with none missing (with the identities
the published table carries: avoided = L1 + L2 on both windows, and requested
and L1-avoided tokens shared by `pi0` and `pi3`). Any failure exits without
writing anything. Then a scorer applied to its own population must reproduce
the logged victim: if the mismatches exceed 0.1% of the decisions of any
population, the diagnosis stops as unresolved; the check table, the README and
the run config are written and no reading is derived.

The run refuses an existing output directory, a tree that is not clean and
more than 12 workers. Each worker holds one population at a time.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing as mp
import resource
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import numpy as np

from persistent_kv_admission import errordiag
from persistent_kv_admission.errordiag import (
    CLASSES,
    RATES,
    SUBSETS,
    baseline_reading,
    has_sign_of,
    seed_signs,
    selection_reading,
    window_contradicts,
)
from persistent_kv_admission.mechanism import sign_reading
from persistent_kv_admission.onpolicy import (
    DecisionPopulation,
    deserialize_ranker,
    sha256_path,
    verify_recorded_argmin,
)

PLAN_PATH = REPOSITORY / "docs/ranker-error-diagnosis-plan.md"
# Published inputs: the on-policy run's tables, the config that records their
# SHA-256, and the run's own model manifest (an ignored file beside the models).
PUBLISHED = REPOSITORY / "results/paper/onpolicy_learning"
PUBLISHED_CONFIG = PUBLISHED / "onpolicy_learning_config.json"
TEST_POPULATIONS = PUBLISHED / "onpolicy_test_populations.csv"
CANONICAL_MODELS = PUBLISHED / "onpolicy_canonical_models.csv"
MODEL_LINKS = PUBLISHED / "onpolicy_model_links.csv"
SEED_UTILITY = PUBLISHED / "onpolicy_seed_utility.csv"
RUN_MODEL_MANIFEST = REPOSITORY / "results/onpolicy_full_feb30eb_001/canonical_model_manifest.csv"
DEFAULT_OUTPUT = REPOSITORY / "results/paper/ranker_error_diagnosis_001"

# --- the fixed grid (docs/ranker-error-diagnosis-plan.md, "Fixed surface") ----------------------

TRACES = ("conversation_trace", "toolagent_trace")
FRACTIONS = (0.0025, 0.01, 0.02)
MULTIPLIERS = (1.0, 4.0)
CELLS = tuple((fraction, multiplier) for fraction in FRACTIONS for multiplier in MULTIPLIERS)
TARGETS = ("next_use", "binary")
SEEDS = (0, 1, 2, 3, 4)
FIRST, LAST = 0, 3
ITERATIONS = (FIRST, LAST)
HORIZON_SECONDS = 600.0
STATISTICS = ("m3", "m4")
VICTIM_SOURCES = ("logged", f"scorer_pi{FIRST}", f"scorer_pi{LAST}")
DEFAULT_WORKERS = 10
# The machine overheats above this.
MAX_WORKERS = 12
UTILITY_FIELDS = (
    "requested_tokens", "avoided_prefill_tokens", "l1_avoided_tokens", "l2_avoided_tokens",
    "label_window_requested_tokens", "label_window_avoided_prefill_tokens",
    "label_window_l1_avoided_tokens", "label_window_l2_avoided_tokens",
)
# Shared by pi0 and pi3 of one lineage: the request stream and the L1 tier,
# which never consults L2.
SHARED_UTILITY_FIELDS = ("requested_tokens", "l1_avoided_tokens",
                         "label_window_requested_tokens", "label_window_l1_avoided_tokens")
DELTA_U_TOLERANCE = 1e-9
SHARED: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="new run directory; refused if it exists")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"worker processes, at most {MAX_WORKERS}")
    return parser.parse_args()


# --- provenance ---------------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=REPOSITORY, check=True,
                          capture_output=True, text=True).stdout.strip()


def published_tables() -> tuple[Path, ...]:
    return (TEST_POPULATIONS, CANONICAL_MODELS, MODEL_LINKS, SEED_UTILITY)


def execution_sources() -> list[Path]:
    """Every file the run executes from this repository."""
    paths = set((REPOSITORY / "src").glob("**/*.py"))
    paths.add(Path(__file__).resolve())
    return sorted(paths)


def source_manifest() -> dict[str, object]:
    files = {str(path.relative_to(REPOSITORY)): sha256_path(path) for path in execution_sources()}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()}


def sources_differing_from_head() -> list[str]:
    """Execution files, the plan and the tracked published inputs whose working
    copy is not byte-identical to HEAD's (an untracked file differs by definition)."""
    differing = []
    for path in execution_sources() + [PLAN_PATH, PUBLISHED_CONFIG, *published_tables()]:
        relative = str(path.relative_to(REPOSITORY))
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPOSITORY,
                                   capture_output=True)
        if committed.returncode != 0 or committed.stdout != path.read_bytes():
            differing.append(relative)
    return differing


def _peak_rss_mib(who: int) -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(who).ru_maxrss) / 1024.0


def _proc_mib(path: str, field: str) -> float:
    """One kB field of a /proc file in MiB, nan where it is not available."""
    try:
        with open(path, encoding="ascii") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return math.nan


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _display(path) -> str:
    """A path relative to the repository when it lies inside it."""
    try:
        return str(Path(path).resolve().relative_to(REPOSITORY))
    except ValueError:
        return str(Path(path).resolve())


# --- identities ---------------------------------------------------------------------------------


def cell_label(fraction: float, multiplier: float) -> str:
    return f"l1={fraction:g},l2x{multiplier:g}"


def lineages() -> list[tuple]:
    """(trace, l1 fraction, l2 multiplier, target, seed) in table order."""
    return [(trace, fraction, multiplier, target, seed) for trace in TRACES
            for fraction, multiplier in CELLS for target in TARGETS for seed in SEEDS]


def grid_cells() -> list[tuple]:
    """(trace, l1 fraction, l2 multiplier, target): the units of the readings."""
    return [(trace, fraction, multiplier, target) for trace in TRACES
            for fraction, multiplier in CELLS for target in TARGETS]


def population_keys() -> list[tuple]:
    return [lineage + (iteration,) for lineage in lineages() for iteration in ITERATIONS]


def _key(row) -> tuple:
    return (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]), row["target"],
            int(row["seed"]), int(row["iteration"]))


def _name(key: tuple) -> str:
    trace, fraction, multiplier, target, *rest = key
    text = f"{trace}/{cell_label(fraction, multiplier)}/{target}"
    if rest:
        text += f"/s{rest[0]}"
    if len(rest) > 1:
        text += f"/pi{rest[1]}"
    return text


def _cell_identity(cell: tuple) -> dict[str, object]:
    trace, fraction, multiplier, target = cell
    return {"trace": trace, "l1_fraction": fraction, "l2_multiplier": multiplier,
            "cell": cell_label(fraction, multiplier), "target": target}


def _identity(lineage: tuple) -> dict[str, object]:
    return {**_cell_identity(lineage[:4]), "seed": lineage[4]}


# --- the published inputs and their checks ------------------------------------------------------


def index_rows(rows: list[dict], label: str) -> tuple[dict[tuple, dict], list[str]]:
    """The rows of a published per-lineage table for every population of the
    grid, exactly one each; rows outside the grid (pi1, pi2) are ignored."""
    wanted = set(population_keys())
    index: dict[tuple, dict] = {}
    problems = []
    for row in rows:
        try:
            key = _key(row)
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"{label}: unreadable identity ({error!r})")
            continue
        if key not in wanted:
            continue
        if key in index:
            problems.append(f"{label}: duplicate {_name(key)}")
            continue
        index[key] = row
        if row.get("cell") != cell_label(key[1], key[2]):
            problems.append(f"{label}: {_name(key)} has cell {row.get('cell')!r}")
        if row.get("policy") != f"pi{key[5]}":
            problems.append(f"{label}: {_name(key)} has policy {row.get('policy')!r}")
    for key in sorted(wanted - set(index)):
        problems.append(f"{label}: missing {_name(key)}")
    return index, problems


def load_published() -> tuple[dict[str, object], list[str]]:
    """Every published table, with its SHA-256 checked against the one the
    on-policy run recorded in its config at publication."""
    problems = []
    config = json.loads(PUBLISHED_CONFIG.read_text(encoding="utf-8"))
    artifacts = config.get("output_artifacts", {})
    tables, hashes = {}, {}
    for path in published_tables():
        actual = sha256_path(path)
        recorded = artifacts.get(path.name, {}).get("sha256")
        if actual != recorded:
            problems.append(f"{path.name}: SHA-256 {actual} != {recorded} recorded at publication")
        tables[path] = _read(path)
        hashes[_display(path)] = {"sha256": actual, "recorded_sha256": recorded}
    hashes[_display(PUBLISHED_CONFIG)] = {"sha256": sha256_path(PUBLISHED_CONFIG)}
    if RUN_MODEL_MANIFEST.is_file():
        run_manifest = _read(RUN_MODEL_MANIFEST)
        hashes[_display(RUN_MODEL_MANIFEST)] = {"sha256": sha256_path(RUN_MODEL_MANIFEST)}
    else:
        run_manifest = None
        problems.append(f"the run's model manifest {RUN_MODEL_MANIFEST} is missing")
    return {"tables": tables, "run_manifest": run_manifest, "hashes": hashes}, problems


def population_index(rows: list[dict]) -> tuple[dict[tuple, dict], list[str]]:
    index, problems = index_rows(rows, "test populations")
    for key, row in index.items():
        if row.get("window") != "test":
            problems.append(f"test populations: {_name(key)} has window {row.get('window')!r}")
        if not Path(row["population_path"]).is_file():
            problems.append(f"test populations: {_name(key)} file {row['population_path']} missing")
    return index, problems


_MANIFEST_FIELDS = ("trace", "l1_fraction", "l2_multiplier", "target", "seed", "iteration",
                    "model_path", "model_sha256")


def model_index(canonical: list[dict], links: list[dict],
                run_manifest: list[dict] | None) -> tuple[dict[tuple, dict], list[str]]:
    """The published model of every lineage at `pi0` and `pi3`.

    `pi0` is shared per trace x target and published once in the canonical
    table; every lineage's own link must name the same file and SHA-256, and
    the run's own manifest must hold exactly the canonical rows.
    """
    problems = []
    shared, updated = {}, {}
    for row in canonical:
        try:
            key = _key(row)
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"canonical models: unreadable identity ({error!r})")
            continue
        model = {"model_path": row["model_path"], "model_sha256": row["model_sha256"]}
        if key[5] == FIRST:
            if row.get("shared_pi0") != "True":
                problems.append(f"canonical models: {_name(key)} is pi0 but not shared")
            if (key[0], key[3]) in shared:
                problems.append(f"canonical models: two pi0 rows for {key[0]}/{key[3]}")
            shared[(key[0], key[3])] = model
        elif key[5] == LAST:
            if key in updated:
                problems.append(f"canonical models: duplicate {_name(key)}")
            updated[key] = model
    index = {}
    for key in population_keys():
        model = shared.get((key[0], key[3])) if key[5] == FIRST else updated.get(key)
        if model is None:
            problems.append(f"canonical models: no model for {_name(key)}")
        else:
            index[key] = model
    linked, found = index_rows(links, "model links")
    problems.extend(found)
    for key, row in linked.items():
        model = index.get(key)
        if model is not None and (row["model_path"], row["model_sha256"]) != (
                model["model_path"], model["model_sha256"]):
            problems.append(f"model links: {_name(key)} names {row['model_path']} "
                            f"{row['model_sha256']}, the canonical table another")
    if run_manifest is not None:
        published = {tuple(row.get(field) for field in _MANIFEST_FIELDS) for row in canonical}
        recorded = {tuple(row.get(field) for field in _MANIFEST_FIELDS) for row in run_manifest}
        if published != recorded or len(run_manifest) != len(canonical):
            problems.append("the run's model manifest differs from the published canonical table")
    return index, problems


def _points(tokens: int, requested: int) -> float:
    return 100.0 * tokens / requested


def utility_values(rows: list[dict]) -> tuple[dict[tuple, dict], list[str]]:
    """The published utilities of every lineage, matched to its populations on
    trace, cell, target, seed and iteration.

    `ΔU` is the published `input_token_points_vs_pi0` of the `pi3` row, checked
    against the token columns; `ΔU_label` is the change of label-window extra
    avoided tokens (avoided minus L1-avoided) in points of label-window input
    tokens. Extra tokens are avoided minus L1-avoided, i.e. L2-avoided.
    """
    index, problems = index_rows(rows, "seed utility")
    values = {}
    for lineage in lineages():
        pair = [index.get(lineage + (iteration,)) for iteration in ITERATIONS]
        if None in pair:
            continue
        name = _name(lineage)
        found = []
        try:
            first, last = ({field: int(row[field]) for field in UTILITY_FIELDS} for row in pair)
            delta_u = float(pair[1]["input_token_points_vs_pi0"])
            baseline = float(pair[0]["input_token_points_vs_pi0"])
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"seed utility: {name} unreadable ({error!r})")
            continue
        for policy, row in zip(("pi0", "pi3"), (first, last)):
            if row["avoided_prefill_tokens"] != row["l1_avoided_tokens"] + row["l2_avoided_tokens"]:
                found.append(f"{policy} avoided != L1 + L2")
            if row["label_window_avoided_prefill_tokens"] != (
                    row["label_window_l1_avoided_tokens"] + row["label_window_l2_avoided_tokens"]):
                found.append(f"{policy} label-window avoided != L1 + L2")
            if row["requested_tokens"] <= 0 or row["label_window_requested_tokens"] <= 0:
                found.append(f"{policy} requested tokens not positive")
        for field in SHARED_UTILITY_FIELDS:
            if first[field] != last[field]:
                found.append(f"{field} differs between pi0 and pi3")
        if baseline != 0.0:
            found.append(f"pi0 input_token_points_vs_pi0 is {baseline}")
        if found:
            problems.extend(f"seed utility: {name}: {text}" for text in found)
            continue
        requested = first["requested_tokens"]
        window = first["label_window_requested_tokens"]
        recomputed = _points(last["avoided_prefill_tokens"] - first["avoided_prefill_tokens"],
                             requested)
        if not abs(recomputed - delta_u) <= DELTA_U_TOLERANCE:
            problems.append(f"seed utility: {name}: published delta {delta_u} != {recomputed}")
            continue
        extra = [row["avoided_prefill_tokens"] - row["l1_avoided_tokens"] for row in (first, last)]
        window_extra = [row["label_window_avoided_prefill_tokens"]
                        - row["label_window_l1_avoided_tokens"] for row in (first, last)]
        values[lineage] = {
            "requested_tokens": requested,
            "extra_tokens_pi0": extra[0], "extra_tokens_pi3": extra[1],
            "u_pi0_points": _points(extra[0], requested),
            "u_pi3_points": _points(extra[1], requested),
            "delta_u_points": delta_u,
            "label_window_requested_tokens": window,
            "label_window_extra_tokens_pi0": window_extra[0],
            "label_window_extra_tokens_pi3": window_extra[1],
            "u_label_pi0_points": _points(window_extra[0], window),
            "u_label_pi3_points": _points(window_extra[1], window),
            "delta_u_label_points": _points(window_extra[1] - window_extra[0], window),
        }
    return values, problems


def load_rankers(models: dict[tuple, dict]) -> tuple[dict[tuple, object], dict, list[str]]:
    """Every model, read once per (file, SHA-256) and checked by
    `deserialize_ranker` against the published SHA-256."""
    loaded, rankers, hashes, problems = {}, {}, {}, []
    for key, model in models.items():
        identity = (model["model_path"], model["model_sha256"])
        if identity not in loaded:
            try:
                loaded[identity] = deserialize_ranker(*identity)
            except (OSError, ValueError, KeyError) as error:
                problems.append(f"model {model['model_path']}: {error}")
                loaded[identity] = None
            hashes[_display(model["model_path"])] = model["model_sha256"]
        if loaded[identity] is not None:
            rankers[key] = loaded[identity]
    return rankers, hashes, problems


# --- one population -----------------------------------------------------------------------------


def population_statistics(population: DecisionPopulation, rankers: dict[int, object],
                          own: int, readings: bool) -> dict[str, object]:
    """Everything the readings need from one population, as small numbers.

    `rankers` maps a policy iteration to its published model; `own` is the
    policy that logged the population. `readings` adds readings 3-5 (on the
    `pi0` populations), all by the logged victim.
    """
    decisions = errordiag.decisions_of(population)
    logged = errordiag.logged_victims(population.victim, decisions)
    out: dict[str, object] = {
        "rows": len(population), "decisions": len(decisions),
        "decisions_eligible": population.decisions_eligible, "cap_bound": population.cap_bound,
        "window": population.window, "horizon_seconds": population.horizon_seconds,
        "population_seed": population.seed,
        "recorded_argmin_mismatches": int(verify_recorded_argmin(
            population, raise_on_mismatch=False)["recorded_argmin_mismatches"]),
    }
    matrix = {"logged": errordiag.statistics(decisions, logged)}
    for iteration in ITERATIONS:
        scores = errordiag.scorer_scores(rankers[iteration], population)
        victims = errordiag.scorer_victims(scores, decisions)
        matrix[f"scorer_pi{iteration}"] = errordiag.statistics(decisions, victims)
        if iteration == own:
            out["own_victim_mismatches"] = int((victims != logged).sum())
            out["own_score_exact"] = bool(np.array_equal(scores, population.arm_score))
            out["own_score_max_abs_diff"] = float(np.abs(scores - population.arm_score).max())
        del scores, victims
    out["matrix"] = matrix
    if readings:
        out["classes"] = errordiag.class_rows(decisions, logged)
        out["rates"] = {
            "ranker": errordiag.conditional_rates(decisions, logged),
            "recency": errordiag.conditional_rates(decisions,
                                                   errordiag.recency_victims(decisions)),
            "uniform": errordiag.uniform_rates(decisions),
        }
        out["concentration"] = errordiag.concentration_rows(decisions, logged)
    return out


def lineage_worker(task) -> dict[str, object]:
    """Both populations of one lineage, one at a time, each hashed on the bytes
    it is loaded from; nothing is computed on a population whose hash differs."""
    started = time.time()
    task = tuple(task)
    rankers = {iteration: SHARED["rankers"][task + (iteration,)] for iteration in ITERATIONS}
    populations = {}
    for iteration in ITERATIONS:
        metadata = SHARED["populations"][task + (iteration,)]
        payload = Path(metadata["population_path"]).read_bytes()
        entry: dict[str, object] = {"sha256": hashlib.sha256(payload).hexdigest(),
                                    "bytes": len(payload)}
        if entry["sha256"] == metadata["population_sha256"]:
            population = DecisionPopulation.load(io.BytesIO(payload))
            del payload
            entry.update(population_statistics(population, rankers, iteration,
                                               readings=iteration == FIRST))
            del population
        populations[iteration] = entry
    return {"task": task, "populations": populations, "seconds": time.time() - started,
            "peak_rss_mib": _peak_rss_mib(resource.RUSAGE_SELF)}


def run_lineages(tasks: list[tuple], workers: int):
    """Every lineage, in a fork pool (or in-process with one worker)."""
    if workers <= 1:
        for task in tasks:
            yield lineage_worker(task)
        return
    pool = mp.get_context("fork").Pool(min(workers, len(tasks)))
    try:
        yield from pool.imap_unordered(lineage_worker, tasks, chunksize=1)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()


def check_populations(results: dict[tuple, dict], populations: dict[tuple, dict]) -> list[str]:
    """The loaded populations against the published manifest."""
    problems = []
    for key in population_keys():
        lineage, iteration = key[:5], key[5]
        entry = results[lineage]["populations"][iteration]
        metadata = populations[key]
        if entry["sha256"] != metadata["population_sha256"]:
            problems.append(f"{_name(key)}: population SHA-256 {entry['sha256']} != "
                            f"published {metadata['population_sha256']}")
            continue
        for field, published in (("rows", "rows"), ("decisions", "decisions_kept"),
                                 ("decisions_eligible", "decisions_eligible")):
            if int(entry[field]) != int(metadata[published]):
                problems.append(f"{_name(key)}: {field} {entry[field]} != published "
                                f"{metadata[published]}")
        if entry["window"] != "test":
            problems.append(f"{_name(key)}: window {entry['window']!r}")
        if float(entry["horizon_seconds"]) != HORIZON_SECONDS:
            problems.append(f"{_name(key)}: horizon {entry['horizon_seconds']} s")
        if int(entry["population_seed"]) != key[4]:
            problems.append(f"{_name(key)}: reservoir seed {entry['population_seed']}")
    return problems


# --- derived tables -----------------------------------------------------------------------------


def _mean(values) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def _summary(values) -> dict[str, float]:
    """Mean, min and max over seeds; all nan when any seed is nan."""
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        return {"mean": math.nan, "min": math.nan, "max": math.nan}
    return {"mean": _mean(values), "min": min(values), "max": max(values)}


def check_rows(results, populations, models) -> list[dict]:
    out = []
    for key in population_keys():
        lineage, iteration = key[:5], key[5]
        entry = results[lineage]["populations"][iteration]
        metadata = populations[key]
        decisions = int(entry["decisions"])
        mismatches = int(entry["own_victim_mismatches"])
        out.append({
            **_identity(lineage), "iteration": iteration, "policy": f"pi{iteration}",
            "population_path": metadata["population_path"],
            "population_sha256": entry["sha256"],
            "published_sha256": metadata["population_sha256"],
            "sha256_matches": entry["sha256"] == metadata["population_sha256"],
            "rows": entry["rows"], "decisions": decisions,
            "decisions_eligible": entry["decisions_eligible"], "cap_bound": entry["cap_bound"],
            "recorded_argmin_mismatches": entry["recorded_argmin_mismatches"],
            "own_model_path": models[key]["model_path"],
            "own_model_sha256": models[key]["model_sha256"],
            "own_victim_mismatches": mismatches,
            "own_victim_mismatch_share": mismatches / decisions,
            "exceeds_stop_rule": errordiag.exceeds_stop_rule(mismatches, decisions),
            "own_score_exact": entry["own_score_exact"],
            "own_score_max_abs_diff": entry["own_score_max_abs_diff"],
        })
    return out


def matrix_rows(results) -> list[dict]:
    """The four-way matrix and the logged victims, per seed."""
    out = []
    for lineage in lineages():
        for iteration in ITERATIONS:
            matrix = results[lineage]["populations"][iteration]["matrix"]
            for source in VICTIM_SOURCES:
                values = matrix[source]
                out.append({
                    **_identity(lineage), "population": f"pi{iteration}",
                    "victim_source": source, "own_scorer": source == f"scorer_pi{iteration}",
                    **{field: values[field] for field in
                       ("decisions", "excess_sum", "m3", "avoidable_count", "m4")},
                })
    return out


def selection_seed_rows(results, utilities) -> list[dict]:
    """Per seed: the utilities and the seed-paired changes of reading 1 and 2."""
    out = []
    for lineage in lineages():
        matrix = {iteration: results[lineage]["populations"][iteration]["matrix"]
                  for iteration in ITERATIONS}
        row = {**_identity(lineage), **utilities[lineage]}
        for statistic in STATISTICS:
            for iteration in ITERATIONS:
                for source in VICTIM_SOURCES:
                    row[f"{statistic}_pi{iteration}pop_{source}"] = matrix[iteration][source][statistic]
            old, new = f"scorer_pi{FIRST}", f"scorer_pi{LAST}"
            row[f"delta_sel_{statistic}_pi{FIRST}pop"] = (
                matrix[FIRST][new][statistic] - matrix[FIRST][old][statistic])
            row[f"delta_sel_{statistic}_pi{LAST}pop"] = (
                matrix[LAST][new][statistic] - matrix[LAST][old][statistic])
            row[f"delta_own_{statistic}"] = matrix[LAST][new][statistic] - matrix[FIRST][old][statistic]
            row[f"delta_own_{statistic}_logged"] = (
                matrix[LAST]["logged"][statistic] - matrix[FIRST]["logged"][statistic])
        out.append(row)
    return out


def _by_lineage(rows: list[dict]) -> dict[tuple, dict]:
    return {(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["target"], row["seed"]): row
            for row in rows}


def _seeds(index: dict[tuple, dict], cell: tuple) -> list[dict]:
    return [index[cell + (seed,)] for seed in SEEDS]


def selection_rows(seed_rows: list[dict]) -> list[dict]:
    """Reading 1 per trace x cell x target x statistic."""
    index = _by_lineage(seed_rows)
    out = []
    for cell in grid_cells():
        seeds = _seeds(index, cell)
        delta_u = [row["delta_u_points"] for row in seeds]
        mean_u = _mean(delta_u)
        for statistic in STATISTICS:
            row = {**_cell_identity(cell), "statistic": statistic, "seeds": len(seeds),
                   "delta_u_mean": mean_u, "delta_u_seed_signs": seed_signs(delta_u),
                   "delta_u_reading": sign_reading(delta_u)}
            agree = []
            for iteration in ITERATIONS:
                prefix = f"delta_sel_pi{iteration}pop"
                values = [seed[f"delta_sel_{statistic}_pi{iteration}pop"] for seed in seeds]
                mean = _mean(values)
                agree.append(has_sign_of(-mean, mean_u))
                row.update({f"{prefix}_mean": mean, f"{prefix}_seed_signs": seed_signs(values),
                            f"{prefix}_reading": sign_reading(values),
                            f"{prefix}_agrees": agree[-1]})
            own = [seed[f"delta_own_{statistic}"] for seed in seeds]
            row.update({
                "delta_own_mean": _mean(own), "delta_own_seed_signs": seed_signs(own),
                "delta_own_reading": sign_reading(own),
                "delta_own_agrees": has_sign_of(-_mean(own), mean_u),
                "delta_own_logged_mean": _mean([seed[f"delta_own_{statistic}_logged"]
                                                for seed in seeds]),
                "selection_reading": selection_reading(*agree),
            })
            out.append(row)
    return out


def selection_count_rows(rows: list[dict]) -> list[dict]:
    out = []
    for target in TARGETS:
        for statistic in STATISTICS:
            members = [row for row in rows
                       if row["target"] == target and row["statistic"] == statistic]
            readings = [row["selection_reading"] for row in members]
            out.append({
                "target": target, "statistic": statistic, "cells": len(members),
                "carried_by_selection": readings.count("carried_by_selection"),
                "population_dependent": readings.count("population_dependent"),
                "not_carried_by_selection": readings.count("not_carried_by_selection"),
                "agrees_pi0pop": sum(row[f"delta_sel_pi{FIRST}pop_agrees"] for row in members),
                "agrees_pi3pop": sum(row[f"delta_sel_pi{LAST}pop_agrees"] for row in members),
                "delta_own_agrees": sum(row["delta_own_agrees"] for row in members),
            })
    return out


def window_rows(seed_rows: list[dict]) -> list[dict]:
    """Reading 2 per trace x cell x target, on m3."""
    index = _by_lineage(seed_rows)
    out = []
    for cell in grid_cells():
        seeds = _seeds(index, cell)
        own = [row["delta_own_m3"] for row in seeds]
        label = [row["delta_u_label_points"] for row in seeds]
        full = [row["delta_u_points"] for row in seeds]
        oriented = -_mean(own)
        label_reading = sign_reading(label)
        out.append({
            **_cell_identity(cell), "seeds": len(seeds),
            "delta_own_m3_mean": _mean(own), "delta_own_m3_seed_signs": seed_signs(own),
            "delta_u_label_mean": _mean(label), "delta_u_label_seed_signs": seed_signs(label),
            "delta_u_label_reading": label_reading,
            "delta_u_mean": _mean(full), "delta_u_seed_signs": seed_signs(full),
            "delta_u_reading": sign_reading(full),
            "agrees_label": has_sign_of(oriented, _mean(label)),
            "agrees_full": has_sign_of(oriented, _mean(full)),
            "contradicts_label": window_contradicts(label_reading, oriented),
        })
    return out


def window_count_rows(rows: list[dict]) -> list[dict]:
    out = []
    for target in TARGETS:
        members = [row for row in rows if row["target"] == target]
        contradicting = [row for row in members if row["contradicts_label"]]
        out.append({
            "target": target, "cells": len(members),
            "agrees_label": sum(row["agrees_label"] for row in members),
            "agrees_full": sum(row["agrees_full"] for row in members),
            "contradictions": len(contradicting),
            "contradiction_list": "; ".join(f"{row['trace']}/{row['cell']}"
                                            for row in contradicting),
        })
    return out


def class_seed_rows(results) -> list[dict]:
    return [{**_identity(lineage), **row} for lineage in lineages()
            for row in results[lineage]["populations"][FIRST]["classes"]]


def class_summary_rows(seed_rows: list[dict]) -> list[dict]:
    """Reading 3 per trace x cell x target x subset x class, over seeds."""
    index = {}
    for row in seed_rows:
        index[(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["target"],
               row["seed"], row["subset"], row["class"])] = row
    out = []
    for cell in grid_cells():
        for subset in SUBSETS:
            for name in CLASSES:
                seeds = [index[cell + (seed, subset, name)] for seed in SEEDS]
                row = {**_cell_identity(cell), "subset": subset, "class": name,
                       "seeds": len(seeds),
                       "decisions_mean": _mean([seed["decisions"] for seed in seeds])}
                for field in ("decision_share", "excess_share"):
                    for statistic, value in _summary([seed[field] for seed in seeds]).items():
                        row[f"{field}_{statistic}"] = value
                for field in ("decision_share_of_population", "excess_share_of_population"):
                    row[f"{field}_mean"] = _summary([seed[field] for seed in seeds])["mean"]
                row["overlap_decisions_total"] = sum(seed["overlap_decisions"] for seed in seeds)
                out.append(row)
    return out


def rate_seed_rows(results) -> list[dict]:
    out = []
    for lineage in lineages():
        rates = results[lineage]["populations"][FIRST]["rates"]
        for rate in RATES:
            eligible = {rates[chooser][f"{rate}_decisions"] for chooser in rates}
            if len(eligible) != 1:
                raise AssertionError(f"{_name(lineage)}: choosers see different {rate} decisions")
            out.append({**_identity(lineage), "rate": rate, "eligible_decisions": eligible.pop(),
                        **{chooser: rates[chooser][f"{rate}_rate"]
                           for chooser in errordiag.CHOOSERS}})
    return out


def rate_rows(seed_rows: list[dict]) -> list[dict]:
    """Reading 4 per trace x cell x target x rate."""
    index = {(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["target"],
              row["seed"], row["rate"]): row for row in seed_rows}
    out = []
    for cell in grid_cells():
        for rate in RATES:
            seeds = [index[cell + (seed, rate)] for seed in SEEDS]
            ranker = [seed["ranker"] for seed in seeds]
            row = {**_cell_identity(cell), "rate": rate, "seeds": len(seeds),
                   "eligible_decisions_mean": _mean([seed["eligible_decisions"] for seed in seeds])}
            for chooser in errordiag.CHOOSERS:
                row[f"{chooser}_mean"] = _mean([seed[chooser] for seed in seeds])
            for baseline in ("recency", "uniform"):
                other = [seed[baseline] for seed in seeds]
                differences = [r - b for r, b in zip(ranker, other)]
                row.update({f"ranker_minus_{baseline}_mean": _mean(differences),
                            f"ranker_minus_{baseline}_seed_signs": seed_signs(differences),
                            f"vs_{baseline}": baseline_reading(ranker, other)})
            out.append(row)
    return out


def rate_count_rows(rows: list[dict]) -> list[dict]:
    out = []
    for target in TARGETS:
        for rate in RATES:
            members = [row for row in rows if row["target"] == target and row["rate"] == rate]
            row = {"target": target, "rate": rate, "cells": len(members)}
            for baseline in ("recency", "uniform"):
                readings = [member[f"vs_{baseline}"] for member in members]
                row.update({f"better_than_{baseline}": readings.count("better"),
                            f"worse_than_{baseline}": readings.count("worse"),
                            f"mixed_vs_{baseline}": readings.count("mixed")})
            out.append(row)
    return out


def concentration_seed_rows(results) -> list[dict]:
    return [{**_identity(lineage), **row} for lineage in lineages()
            for row in results[lineage]["populations"][FIRST]["concentration"]]


def concentration_rows(seed_rows: list[dict]) -> list[dict]:
    """Reading 5 per trace x cell x target x subset, over seeds."""
    index = {(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["target"],
              row["seed"], row["subset"]): row for row in seed_rows}
    out = []
    for cell in grid_cells():
        for subset in SUBSETS:
            seeds = [index[cell + (seed, subset)] for seed in SEEDS]
            row = {**_cell_identity(cell), "subset": subset, "seeds": len(seeds),
                   "evictions_mean": _mean([seed["evictions"] for seed in seeds]),
                   "distinct_states_mean": _mean([seed["distinct_states"] for seed in seeds])}
            for field in ("repeat_share", "top_share"):
                for statistic, value in _summary([seed[field] for seed in seeds]).items():
                    row[f"{field}_{statistic}"] = value
            out.append(row)
    return out


def derive_tables(results, utilities) -> dict[str, list[dict]]:
    selection_seeds = selection_seed_rows(results, utilities)
    selection = selection_rows(selection_seeds)
    window = window_rows(selection_seeds)
    classes_seeds = class_seed_rows(results)
    rates_seeds = rate_seed_rows(results)
    rates = rate_rows(rates_seeds)
    concentration_seeds = concentration_seed_rows(results)
    return {
        "matrix_seeds": matrix_rows(results),
        "selection_seeds": selection_seeds,
        "selection": selection,
        "selection_counts": selection_count_rows(selection),
        "window": window,
        "window_counts": window_count_rows(window),
        "classes_seeds": classes_seeds,
        "classes": class_summary_rows(classes_seeds),
        "rates_seeds": rates_seeds,
        "rates": rates,
        "rates_counts": rate_count_rows(rates),
        "concentration_seeds": concentration_seeds,
        "concentration": concentration_rows(concentration_seeds),
    }


# --- the tables and their columns ---------------------------------------------------------------

_CELL = [
    ("trace", "trace name"),
    ("l1_fraction", "L1 capacity as a fraction of the trace's packed working set"),
    ("l2_multiplier", "L2 capacity as a multiple of the L1 capacity"),
    ("cell", "cell label `l1=<fraction>,l2x<multiplier>`"),
    ("target", "training target of the published rankers (`next_use` or `binary`)"),
]
_SEED = [("seed", "lineage seed: the replays' sampling seed and the reservoir's seed")]


def _summary_columns(field: str, text: str) -> list[tuple[str, str]]:
    return [(f"{field}_{statistic}", f"{label} over the seeds of {text} (nan when any seed is nan)")
            for statistic, label in (("mean", "mean"), ("min", "minimum"), ("max", "maximum"))]


def _matrix_columns() -> list[tuple[str, str]]:
    out = []
    for statistic in STATISTICS:
        for iteration in ITERATIONS:
            for source in VICTIM_SOURCES:
                out.append((f"{statistic}_pi{iteration}pop_{source}",
                            f"{statistic} of the pi{iteration} population with the victim of "
                            f"`{source}` (as in `matrix_seeds.csv`)"))
        out += [
            (f"delta_sel_{statistic}_pi{FIRST}pop",
             f"Δsel(pi0) = {statistic}(pi0 population, pi3 scorer) - {statistic}(pi0 population, "
             "pi0 scorer)"),
            (f"delta_sel_{statistic}_pi{LAST}pop",
             f"Δsel(pi3) = {statistic}(pi3 population, pi3 scorer) - {statistic}(pi3 population, "
             "pi0 scorer)"),
            (f"delta_own_{statistic}",
             f"Δown = {statistic}(pi3 population, pi3 scorer) - {statistic}(pi0 population, pi0 "
             "scorer), recomputed victims"),
            (f"delta_own_{statistic}_logged",
             "the same change with the logged victims of both populations"),
        ]
    return out


def _selection_population_columns() -> list[tuple[str, str]]:
    out = []
    for iteration in ITERATIONS:
        prefix = f"delta_sel_pi{iteration}pop"
        population = f"pi{iteration} population"
        out += [
            (f"{prefix}_mean", f"five-seed mean of Δsel(pi{iteration}) = statistic({population}, "
                               f"pi3 scorer) - statistic({population}, pi0 scorer)"),
            (f"{prefix}_seed_signs", f"signs of Δsel(pi{iteration}), seeds 0-4"),
            (f"{prefix}_reading", f"sign reading of Δsel(pi{iteration}) as published "
                                  "(consistent_gain: positive in every seed, i.e. the pi3 scorer's "
                                  "victims have the larger statistic; consistent_loss; mixed)"),
            (f"{prefix}_agrees", f"the mean of -Δsel(pi{iteration}) has the sign of the mean ΔU "
                                 "(zero matching zero only)"),
        ]
    return out


TABLES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "checks": (
        "Required checks, one row per population (trace x cell x target x seed x pi0/pi3).",
        _CELL + _SEED + [
            ("iteration", "policy iteration that logged the population (0 or 3)"),
            ("policy", "pi0 or pi3"),
            ("population_path", "the population file named by `onpolicy_test_populations.csv`"),
            ("population_sha256", "SHA-256 of the bytes the population was loaded from"),
            ("published_sha256", "SHA-256 recorded in `onpolicy_test_populations.csv`"),
            ("sha256_matches", "whether the two are equal (every row must be True)"),
            ("rows", "candidate rows of the population"),
            ("decisions", "logged decisions in the reservoir"),
            ("decisions_eligible", "eligible held-out decisions of the replay the reservoir drew from"),
            ("cap_bound", "whether more decisions were eligible than kept (the reservoir is a sample)"),
            ("recorded_argmin_mismatches",
             "decisions whose logged victim is not the first minimum of the recorded "
             "(arm_score, arm_tiebreak); not a stop criterion"),
            ("own_model_path", "the published model of the population's own policy"),
            ("own_model_sha256", "its published SHA-256"),
            ("own_victim_mismatches",
             "decisions in which that model's victim (first minimum of score and stored "
             "tie-break) differs from the logged victim"),
            ("own_victim_mismatch_share", "own_victim_mismatches / decisions"),
            ("exceeds_stop_rule",
             "own_victim_mismatches > 0.1% of decisions; any True stops the diagnosis as unresolved"),
            ("own_score_exact", "the recomputed score equals the stored arm_score in every row"),
            ("own_score_max_abs_diff", "largest |recomputed score - stored arm_score|"),
        ],
    ),
    "matrix_seeds": (
        "The four-way matrix: m3 and m4 of every population under each scorer's victims, and "
        "under the logged victims, per seed.",
        _CELL + _SEED + [
            ("population", "the policy that logged the decisions (pi0 or pi3)"),
            ("victim_source",
             "`logged` (the logged victim), `scorer_pi0` or `scorer_pi3` (the first minimum of "
             "that model's score and the stored tie-break, in stored candidate order)"),
            ("own_scorer", "the scorer is the population's own policy"),
            ("decisions", "decisions of the population"),
            ("excess_sum", "sum over decisions of label(victim) - min label"),
            ("m3", "excess_sum / decisions"),
            ("avoidable_count", "decisions whose victim is reusable while some candidate is not"),
            ("m4", "avoidable_count / decisions"),
        ],
    ),
    "selection_seeds": (
        "Readings 1 and 2 per seed: published utilities and the seed-paired changes of m3 and m4.",
        _CELL + _SEED + [
            ("requested_tokens", "input tokens of the full evaluation window (shared by pi0 and pi3)"),
            ("extra_tokens_pi0", "pi0 avoided_prefill_tokens - l1_avoided_tokens, full window"),
            ("extra_tokens_pi3", "the same for pi3"),
            ("u_pi0_points", "U(pi0) = 100 x extra_tokens_pi0 / requested_tokens"),
            ("u_pi3_points", "U(pi3), the same"),
            ("delta_u_points",
             "ΔU = U(pi3) - U(pi0): the published `input_token_points_vs_pi0` of the pi3 row"),
            ("label_window_requested_tokens", "input tokens of the label window (shared by pi0 and pi3)"),
            ("label_window_extra_tokens_pi0",
             "pi0 label_window_avoided_prefill_tokens - label_window_l1_avoided_tokens"),
            ("label_window_extra_tokens_pi3", "the same for pi3"),
            ("u_label_pi0_points", "100 x label_window_extra_tokens_pi0 / label_window_requested_tokens"),
            ("u_label_pi3_points", "the same for pi3"),
            ("delta_u_label_points",
             "ΔU_label = 100 x (label_window_extra_tokens_pi3 - label_window_extra_tokens_pi0) / "
             "label_window_requested_tokens"),
        ] + _matrix_columns(),
    ),
    "selection": (
        "Reading 1, per trace x cell x target x statistic (m3 and m4).",
        _CELL + [
            ("statistic", "m3 or m4"),
            ("seeds", "seed-paired values"),
            ("delta_u_mean", "five-seed mean of ΔU (points of full-window input tokens)"),
            ("delta_u_seed_signs", "signs of ΔU, seeds 0-4"),
            ("delta_u_reading", "consistent_gain / consistent_loss / mixed of ΔU"),
        ] + _selection_population_columns() + [
            ("delta_own_mean", "five-seed mean of Δown (recomputed victims)"),
            ("delta_own_seed_signs", "signs of Δown, seeds 0-4"),
            ("delta_own_reading", "sign reading of Δown as published (as for Δsel)"),
            ("delta_own_agrees", "the mean of -Δown has the sign of the mean ΔU"),
            ("delta_own_logged_mean", "five-seed mean of Δown with the logged victims"),
            ("selection_reading",
             "carried_by_selection (both delta_sel_*_agrees), population_dependent (exactly one) "
             "or not_carried_by_selection (neither)"),
        ],
    ),
    "selection_counts": (
        "Reading 1 counts, per target x statistic, out of the trace x cells.",
        [("target", "training target"), ("statistic", "m3 or m4"),
         ("cells", "trace x cells counted (12)"),
         ("carried_by_selection", "cells reading carried_by_selection"),
         ("population_dependent", "cells reading population_dependent"),
         ("not_carried_by_selection", "cells reading not_carried_by_selection"),
         ("agrees_pi0pop", "cells where the mean of -Δsel(pi0) has the sign of the mean ΔU"),
         ("agrees_pi3pop", "the same on the pi3 population"),
         ("delta_own_agrees", "cells where the mean of -Δown has the sign of the mean ΔU")],
    ),
    "window": (
        "Reading 2, per trace x cell x target, on m3 (Δown from recomputed victims).",
        _CELL + [
            ("seeds", "seed-paired values"),
            ("delta_own_m3_mean", "five-seed mean of Δown on m3; the rule uses its negative"),
            ("delta_own_m3_seed_signs", "signs of Δown, seeds 0-4"),
            ("delta_u_label_mean", "five-seed mean of ΔU_label"),
            ("delta_u_label_seed_signs", "signs of ΔU_label, seeds 0-4"),
            ("delta_u_label_reading", "consistent_gain / consistent_loss / mixed of ΔU_label"),
            ("delta_u_mean", "five-seed mean of ΔU"),
            ("delta_u_seed_signs", "signs of ΔU, seeds 0-4"),
            ("delta_u_reading", "consistent_gain / consistent_loss / mixed of ΔU"),
            ("agrees_label", "the mean of -Δown has the sign of the mean ΔU_label"),
            ("agrees_full", "the mean of -Δown has the sign of the mean ΔU"),
            ("contradicts_label",
             "ΔU_label is consistent while the mean of -Δown has the other, non-zero sign"),
        ],
    ),
    "window_counts": (
        "Reading 2 counts, per target, out of the trace x cells.",
        [("target", "training target"), ("cells", "trace x cells counted (12)"),
         ("agrees_label", "cells where the mean of -Δown has the sign of the mean ΔU_label"),
         ("agrees_full", "cells where it has the sign of the mean ΔU"),
         ("contradictions", "cells with contradicts_label"),
         ("contradiction_list", "those cells, trace/cell")],
    ),
    "classes_seeds": (
        "Reading 3 per seed: the four classes of the pi0 population's decisions by their logged "
        "victim, for every decision and separately for decisions whose victim is the arrival or "
        "a resident.",
        _CELL + _SEED + [
            ("subset", "`all`, `arrival` (the victim is the arrival) or `resident`"),
            ("class",
             "`reference_victim` (the victim is the first minimum of (label, tie-break)), "
             "`other_tiebreak` (minimum label, not that first minimum), `avoidable_reusable` "
             "(the victim is reusable and some candidate is not), `order_error` (every candidate "
             "is reusable and the victim's label is above the minimum)"),
            ("decisions", "decisions of the subset in the class"),
            ("excess_sum", "their summed label excess"),
            ("overlap_decisions",
             "decisions of this class that also meet a later class's definition: a victim whose "
             "next use is exactly at the horizon has no excess and is also an avoidable reusable "
             "eviction; the plan's order assigns it to the no-excess class"),
            ("subset_decisions", "decisions in the subset"),
            ("subset_excess_sum", "summed label excess of the subset"),
            ("decision_share", "decisions / subset_decisions"),
            ("excess_share", "excess_sum / subset_excess_sum"),
            ("decision_share_of_population", "decisions / all decisions of the population"),
            ("excess_share_of_population", "excess_sum / the population's total label excess"),
        ],
    ),
    "classes": (
        "Reading 3 over seeds, per trace x cell x target x subset x class.",
        _CELL + [
            ("subset", "as in classes_seeds.csv"), ("class", "as in classes_seeds.csv"),
            ("seeds", "seeds"),
            ("decisions_mean", "mean decisions in the class"),
        ] + _summary_columns("decision_share", "the class's share of the subset's decisions")
          + _summary_columns("excess_share", "the class's share of the subset's label excess") + [
            ("decision_share_of_population_mean",
             "mean over seeds of decisions / all decisions of the population"),
            ("excess_share_of_population_mean",
             "mean over seeds of excess_sum / the population's total label excess"),
            ("overlap_decisions_total", "overlap_decisions summed over seeds"),
        ],
    ),
    "rates_seeds": (
        "Reading 4 per seed, on the pi0 population.",
        _CELL + _SEED + [
            ("rate",
             "`avoidable`: share of the decisions with both a reusable and a non-reusable "
             "candidate whose victim is reusable; `order`: share of the decisions whose "
             "candidates are all reusable and not all of one label whose victim's label is above "
             "the minimum"),
            ("eligible_decisions", "decisions the rate is taken over"),
            ("ranker", "the rate of the frozen ranker (pi0), by its logged victim"),
            ("recency", "the rate of recency: the first minimum of the stored tie-break"),
            ("uniform", "the expected rate of a victim drawn uniformly from the candidates"),
        ],
    ),
    "rates": (
        "Reading 4 over seeds, per trace x cell x target x rate.",
        _CELL + [
            ("rate", "as in rates_seeds.csv"), ("seeds", "seeds"),
            ("eligible_decisions_mean", "mean eligible decisions"),
            ("ranker_mean", "five-seed mean of the ranker's rate"),
            ("recency_mean", "five-seed mean of recency's rate"),
            ("uniform_mean", "five-seed mean of the uniform expectation"),
            ("ranker_minus_recency_mean", "mean of the seed-paired ranker - recency"),
            ("ranker_minus_recency_seed_signs", "signs of ranker - recency, seeds 0-4"),
            ("vs_recency",
             "better (ranker lower in all five seeds), worse (higher in all five) or mixed"),
            ("ranker_minus_uniform_mean", "mean of the seed-paired ranker - uniform"),
            ("ranker_minus_uniform_seed_signs", "signs of ranker - uniform, seeds 0-4"),
            ("vs_uniform", "the same rule against the uniform expectation; descriptive, not "
                           "registered (the plan fixes the rule against recency)"),
        ],
    ),
    "rates_counts": (
        "Reading 4 counts, per target x rate, out of the trace x cells.",
        [("target", "training target"), ("rate", "avoidable or order"),
         ("cells", "trace x cells counted (12)"),
         ("better_than_recency", "cells reading better against recency"),
         ("worse_than_recency", "cells reading worse against recency"),
         ("mixed_vs_recency", "cells reading mixed against recency"),
         ("better_than_uniform", "cells reading better against the uniform expectation "
                                 "(descriptive, not registered)"),
         ("worse_than_uniform", "cells reading worse against it (descriptive, not registered)"),
         ("mixed_vs_uniform", "cells reading mixed against it (descriptive, not registered)")],
    ),
    "concentration_seeds": (
        "Reading 5 per seed: the avoidable reusable evictions of the pi0 population (the "
        "`avoidable_reusable` class of classes_seeds.csv) by victim state, for all of them and "
        "separately for those whose victim is the arrival or a resident; each subset counts its "
        "own distinct states, repeats and top states.",
        _CELL + _SEED + [
            ("subset", "`all` (every avoidable reusable eviction), `arrival` (the victim is the "
                       "arrival) or `resident` (the victim is a resident), as in classes_seeds.csv"),
            ("evictions", "avoidable reusable evictions of the subset in the sample"),
            ("distinct_states", "distinct victim states among them"),
            ("repeated_states", "states that are such a victim more than once in the subset"),
            ("repeat_share", "share of the subset's evictions on those states"),
            ("top_states", "10% of distinct_states, rounded up"),
            ("top_share", "share of the subset's evictions on the top_states states with the most"),
        ],
    ),
    "concentration": (
        "Reading 5 over seeds, per trace x cell x target x subset.",
        _CELL + [
            ("subset", "as in concentration_seeds.csv"),
            ("seeds", "seeds"),
            ("evictions_mean", "mean avoidable reusable evictions in the sample"),
            ("distinct_states_mean", "mean distinct victim states"),
        ] + _summary_columns("repeat_share", "repeat_share")
          + _summary_columns("top_share", "top_share"),
    ),
}

README_HEAD = """# Ranker-error diagnosis on saved decision logs

Pre-registration: `docs/ranker-error-diagnosis-plan.md`. Logs: the held-out decision populations of `pi0` and `pi3` of the published on-policy run, 2 traces x 6 cells x 2 targets x 5 seeds, each a reservoir of at most 40,000 whole decisions drawn uniformly from the decisions at or after the split and at least 600 s before the trace end, loaded by the SHA-256 the published table records. Scorers: the published `pi0` and `pi3` models of the same lineage. A scorer's victim on a logged decision is the first minimum, in stored candidate order, of (score on the stored features, stored tie-break); no hybrid, protection or override. Labels: `label = -log1p(min(next_use_delta_s, 600))`; a candidate is reusable when its next use is at most 600 s away. `m3` is the mean of `label(victim) - min label` over a set of decisions, `m4` the share whose victim is reusable while some candidate is not. `U` is extra avoided prefill tokens (avoided minus L1-avoided) in points (100 x share) of input tokens: `ΔU` on the full evaluation window, `ΔU_label` on the label window. "Consistent" means the same sign in all five seed-paired values; seed signs list seeds 0-4 as +, -, 0, or n for nan; "has the sign of" matches zero with zero only.

Nothing was replayed or fitted. A scorer applied to another policy's population is evaluated on states that policy kept, not on a trajectory of its own. A class share is a share of a statistic, not of lost utility. Repeat counts in a reservoir are lower bounds, and runs of consecutive decisions are not estimated.
"""

UNRESOLVED_HEAD = """# Ranker-error diagnosis on saved decision logs: UNRESOLVED

Pre-registration: `docs/ranker-error-diagnosis-plan.md`. A scorer applied to its own population did not reproduce the logged victim in more than 0.1% of the decisions of at least one population (`checks.csv`, `exceeds_stop_rule`), so the diagnosis stops as unresolved by the plan's rule and no reading is derived.
"""


def readme(names: tuple[str, ...], head: str) -> str:
    sections = [head]
    for name in names:
        description, columns = TABLES[name]
        lines = [f"## `{name}.csv`", "", description, ""]
        lines += [f"- `{column}`: {text}" for column, text in columns]
        sections.append("\n".join(lines) + "\n")
    sections.append("## `run_config.json`\n\nPlan and code commits, source manifest, the "
                    "SHA-256 of every population, model and published table read, check "
                    "counts, timing and memory.\n")
    return "\n".join(sections)


def _write(path: Path, rows: list[dict], name: str) -> None:
    """A table with exactly its documented columns, in their documented order."""
    fields = [column for column, _ in TABLES[name][1]]
    for row in rows:
        if set(row) != set(fields):
            raise AssertionError(f"{name}: columns {sorted(set(row) ^ set(fields))} are not "
                                 "documented or not written")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def publish(directory: Path, tables: dict[str, list[dict]], config: dict,
            names: tuple[str, ...], head: str) -> None:
    """Write into a staging directory beside the target and move it into place."""
    if directory.exists():
        raise FileExistsError(f"output directory {directory} exists")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.staging-{os.getpid()}"
    staging.mkdir()
    for name in names:
        _write(staging / f"{name}.csv", tables[name], name)
    (staging / "README.md").write_text(readme(names, head), encoding="utf-8")
    (staging / "run_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n",
                                             encoding="utf-8")
    os.replace(staging, directory)


# --- main ---------------------------------------------------------------------------------------


def _fail(problems: list[str]) -> None:
    for line in problems[:40]:
        print(f"    CHECK {line}", flush=True)
    raise SystemExit(f"{len(problems)} checks failed; nothing derived or published")


def main() -> None:
    args = parse_args()
    clock = time.time()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.workers > MAX_WORKERS:
        raise SystemExit(f"--workers {args.workers} is above the cap of {MAX_WORKERS}")
    if args.output_dir.exists():
        raise SystemExit(f"output directory {args.output_dir} exists; choose a new one")
    status = _git("status", "--porcelain")
    if status:
        raise SystemExit(f"the working tree is not clean:\n{status}")
    differing = sources_differing_from_head()
    if differing:
        raise SystemExit(f"execution source or published input differs from HEAD: {differing}")
    head_start = _git("rev-parse", "HEAD")
    plan_commit = _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(REPOSITORY)))
    manifest_start = source_manifest()
    available_start = _proc_mib("/proc/meminfo", "MemAvailable")

    # Checks on the published tables and the models, before any population is read.
    published, problems = load_published()
    tables = published["tables"]
    populations, found = population_index(tables[TEST_POPULATIONS])
    problems += found
    models, found = model_index(tables[CANONICAL_MODELS], tables[MODEL_LINKS],
                                published["run_manifest"])
    problems += found
    utilities, found = utility_values(tables[SEED_UTILITY])
    problems += found
    rankers, model_hashes, found = load_rankers(models)
    problems += found
    if problems:
        _fail(problems)
    print(f"published inputs checked: {len(populations)} populations, {len(model_hashes)} "
          f"models, {len(utilities)} utility pairs", flush=True)

    SHARED.clear()
    SHARED.update(populations=populations, rankers=rankers)
    tasks = lineages()
    workers = min(args.workers, len(tasks))
    results, started = {}, time.time()
    for done, result in enumerate(run_lineages(tasks, workers), 1):
        results[tuple(result["task"])] = result
        if done % 10 == 0 or done == len(tasks):
            print(f"  {done}/{len(tasks)} lineages: {time.time() - started:.1f}s", flush=True)
    lineage_seconds = time.time() - started

    problems = check_populations(results, populations)
    if problems:
        _fail(problems)
    checks = check_rows(results, populations, models)
    exceeded = [row for row in checks if row["exceeds_stop_rule"]]
    if source_manifest() != manifest_start or _git("rev-parse", "HEAD") != head_start:
        raise SystemExit("execution source or HEAD changed during the run; nothing published")
    for relative, entry in published["hashes"].items():
        if sha256_path(REPOSITORY / relative) != entry["sha256"]:
            raise SystemExit(f"published input {relative} changed during the run")

    config = {
        "phase": "ranker_error_diagnosis",
        "status": "unresolved" if exceeded else "complete",
        "plan": str(PLAN_PATH.relative_to(REPOSITORY)),
        "plan_sha256": sha256_path(PLAN_PATH),
        "plan_commit": plan_commit,
        "code_commit": head_start,
        "source_manifest": manifest_start,
        "published_inputs": published["hashes"],
        "models": model_hashes,
        "populations": {_name(key): {"path": _display(populations[key]["population_path"]),
                                     "sha256": populations[key]["population_sha256"]}
                        for key in population_keys()},
        "traces": list(TRACES), "cells": [list(cell) for cell in CELLS],
        "targets": list(TARGETS), "seeds": list(SEEDS), "iterations": list(ITERATIONS),
        "horizon_seconds": HORIZON_SECONDS,
        "checks": {
            "published_tables_matching_recorded_sha256": len(published_tables()),
            "populations_hashed": len(checks),
            "populations_matching_published_sha256": sum(row["sha256_matches"] for row in checks),
            "models_matching_published_sha256": len(model_hashes),
            "utility_pairs_matched": len(utilities),
            "own_victim_mismatches_total": sum(row["own_victim_mismatches"] for row in checks),
            "own_victim_mismatch_share_max": max(row["own_victim_mismatch_share"] for row in checks),
            "populations_exceeding_stop_rule": len(exceeded),
            "stop_rule": "own_victim_mismatches > 0.1% of a population's decisions",
            "recorded_argmin_mismatches_total": sum(row["recorded_argmin_mismatches"]
                                                    for row in checks),
            "own_score_exact_populations": sum(row["own_score_exact"] for row in checks),
            "decisions_total": sum(row["decisions"] for row in checks),
        },
        "workers": workers,
        "timing": {"lineage_wall_seconds": lineage_seconds,
                   "lineage_worker_seconds": sum(r["seconds"] for r in results.values()),
                   "lineage_worker_seconds_max": max(r["seconds"] for r in results.values()),
                   "wall_seconds": time.time() - clock},
        "memory": {"rss_unit": "MiB (Linux ru_maxrss KiB / 1024)",
                   "peak_worker_rss_mib": max(r["peak_rss_mib"] for r in results.values()),
                   "children_peak_rss_mib": _peak_rss_mib(resource.RUSAGE_CHILDREN),
                   "parent_peak_rss_mib": _peak_rss_mib(resource.RUSAGE_SELF),
                   "mem_available_mib_at_start": available_start},
        "note": "Arithmetic on saved held-out decision logs and published tables. Nothing is "
                "replayed, fitted or proposed.",
    }
    if exceeded:
        publish(args.output_dir, {"checks": checks}, config, ("checks",), UNRESOLVED_HEAD)
        raise SystemExit(f"unresolved: {len(exceeded)} populations exceed the 0.1% stop rule; "
                         f"checks written to {args.output_dir}")
    derived = derive_tables(results, utilities)
    for row in derived["selection_counts"]:
        print(f"  reading 1 {row['target']}/{row['statistic']}: carried "
              f"{row['carried_by_selection']}, population-dependent {row['population_dependent']}, "
              f"not carried {row['not_carried_by_selection']} of {row['cells']}", flush=True)
    publish(args.output_dir, {"checks": checks, **derived}, config, tuple(TABLES), README_HEAD)
    print(f"written to {args.output_dir}; done in {time.time() - clock:.0f}s", flush=True)


if __name__ == "__main__":
    main()
