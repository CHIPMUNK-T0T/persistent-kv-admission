#!/usr/bin/env python3
"""Matched-horizon diagnosis on saved decision logs: the frozen ranker's errors at the reuse boundary that works for the cell.

The pre-registration is `docs/matched-horizon-diagnosis-plan.md`; nothing here
may change it, and the run config records the last commit touching it. The
run reads the held-out `pi0` decision populations of the published on-policy
run for every trace x cell x target x seed (120; no `pi3` population is read),
the published `pi0` model of each lineage, the published ranker-error
diagnosis tables it must reproduce at 600 s, and the published `all16` replay
rows of `label` and of each ranker's own replay (`learned` for `next_use`,
`pi0_binary` for `binary`; error-location control) and `label_binary_h`
(horizon control, and its fill-in for 150 s) for the bridge. It replays
nothing and fits nothing; the arithmetic is
`persistent_kv_admission.matcheddiag` on `persistent_kv_admission.errordiag`.

The ranker-error diagnosis runner (`scripts/run_ranker_error_diagnosis.py`) is
imported by path, unchanged: its published-input loading and checks (every
on-policy table against the SHA-256 the on-policy config records, the
canonical models, the model links and the run's own model manifest, the
population table), its model loading by SHA-256 and its derivation of the
class and rate tables, so that at h = 600 the arithmetic is the published one.

Before anything is derived the run checks those published inputs; that every
`pi0` population has the published SHA-256 (hashed on the exact bytes it is
loaded from), row, decision and eligible counts, the test window, H = 600 s
and its lineage seed; and that the published replay rows the bridge reads are
there exactly once, under `all16` with eligibility `all` and width 16, with
extra = avoided - L1-avoided and the same arm-independent identifiers for
every arm of a trace x cell x seed. Any failure exits without writing
anything. The `pi0` model applied to its own population must reproduce the
logged victim: more than 0.1% mismatches in any population stops the
diagnosis as unresolved (the check table, README and run config are written;
no reading is derived). Then the h = 600 rows of the class, rate and per-seed
rate tables must equal the published diagnosis tables on their common
columns, value for value as written (exact float equality, no tolerance);
any difference publishes nothing.

The run refuses an existing output directory, a tree that is not clean (or
whose execution sources, plan or published inputs differ from HEAD) and more
than 12 workers; it re-checks HEAD, the execution sources and the published
inputs at the end. Each worker holds one population at a time.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import importlib.util
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

from persistent_kv_admission import errordiag, matcheddiag
from persistent_kv_admission.gap import summarize
from persistent_kv_admission.matcheddiag import (
    COMPOSITION_SHARE,
    DIAGNOSIS_HORIZON_SECONDS,
    HORIZONS_SECONDS,
    MATCHED_HORIZON_SECONDS,
    UNIFORM_CONCORDANCE,
    VICTIM_SOURCES,
    matched_horizon,
)
from persistent_kv_admission.mechanism import sign_reading
from persistent_kv_admission.onpolicy import DecisionPopulation, sha256_path, verify_recorded_argmin

PARENT_SCRIPT = REPOSITORY / "scripts/run_ranker_error_diagnosis.py"


def _load_parent():
    """Import the ranker-error diagnosis runner by path, unchanged, for its
    published-input checks, model loading and table derivations. `main` is
    behind the usual guard, so nothing runs and nothing is read at import."""
    spec = importlib.util.spec_from_file_location("run_ranker_error_diagnosis", PARENT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The grid (traces, cells, targets, seeds) is always read from `red` at call
# time: it is the diagnosis's grid, and one place to restrict it in tests.
red = _load_parent()

PLAN_PATH = REPOSITORY / "docs/matched-horizon-diagnosis-plan.md"
DEFAULT_PAPER_DIR = REPOSITORY / "results/paper"
DEFAULT_OUTPUT = DEFAULT_PAPER_DIR / "matched_horizon_diagnosis_001"
FIRST = red.FIRST
DEFAULT_WORKERS = 10
# The machine overheats above this.
MAX_WORKERS = 12
assert (DEFAULT_WORKERS, MAX_WORKERS) == (red.DEFAULT_WORKERS, red.MAX_WORKERS)
assert red.HORIZON_SECONDS == DIAGNOSIS_HORIZON_SECONDS

# The published diagnosis tables reproduced at h = 600, with the columns that
# identify a row.
DIAGNOSIS_SOURCE = "ranker_error_diagnosis_001"
REPRODUCED = {
    "classes": ("trace", "l1_fraction", "l2_multiplier", "target", "subset", "class"),
    "rates": ("trace", "l1_fraction", "l2_multiplier", "target", "rate"),
    "rates_seeds": ("trace", "l1_fraction", "l2_multiplier", "target", "seed", "rate"),
}
# The published replay rows of the bridge, by source directory under
# --paper-dir: the arms read from each (mechanism all16, variant main).
REPLAY_TABLE = "replay_seeds.csv"
REPLAY_MECHANISM = ("all16", "all", "16")
REPLAY_ARMS = {
    "error_location_001": ("label", "learned", "pi0_binary"),
    "horizon_control_001": ("label_binary_60", "label_binary_300", "label_binary_600"),
    "horizon_fill_001": ("label_binary_150",),
}
# U(label_binary_h) of each horizon: (source, arm). 150 s was replayed only on
# the cells whose h* is 150 s (the fill-in's cells).
LABEL_BINARY_ARMS = {
    60.0: ("horizon_control_001", "label_binary_60"),
    150.0: ("horizon_fill_001", "label_binary_150"),
    300.0: ("horizon_control_001", "label_binary_300"),
    600.0: ("horizon_control_001", "label_binary_600"),
}
assert tuple(LABEL_BINARY_ARMS) == HORIZONS_SECONDS
# Reading 4: each target's ranker's own all16 replay in the error-location
# control, the arm its U gaps are taken against.
RANKER_ARMS = {"next_use": "learned", "binary": "pi0_binary"}
assert set(RANKER_ARMS.values()) <= set(REPLAY_ARMS["error_location_001"])
# Arm-independent identifiers every replay row of a trace x cell x seed shares.
REPLAY_IDENTIFIERS = ("l1_capacity_bytes", "l2_capacity_bytes", "requested_tokens",
                      "l1_avoided_tokens")
REPLAY_FIELDS = ("avoided_prefill_tokens", "extra_avoided_tokens") + REPLAY_IDENTIFIERS
U_GAPS = ("label", "label_binary")
M4_SUBSETS = ("all", "resident")
SHARED: dict[str, object] = {}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="new run directory; refused if it exists")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"worker processes (default {DEFAULT_WORKERS}, at most {MAX_WORKERS})")
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR,
                        help=f"directory holding the published {DIAGNOSIS_SOURCE}/ tables and the "
                             f"{', '.join(REPLAY_ARMS)} replay tables that are read")
    return parser.parse_args(argv)


# --- provenance ---------------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=REPOSITORY, check=True,
                          capture_output=True, text=True).stdout.strip()


def execution_sources() -> list[Path]:
    """Every file the run executes from this repository: every `src` module,
    this script and the diagnosis runner it imports."""
    paths = set((REPOSITORY / "src").glob("**/*.py"))
    paths.update((Path(__file__).resolve(), PARENT_SCRIPT.resolve()))
    return sorted(paths)


def source_manifest() -> dict[str, object]:
    files = {str(path.relative_to(REPOSITORY)): sha256_path(path) for path in execution_sources()}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()}


def reference_paths(paper_dir) -> dict[str, Path]:
    """The published tables read from --paper-dir, by `<source>/<file>`."""
    paper_dir = Path(paper_dir)
    out = {f"{DIAGNOSIS_SOURCE}/{name}.csv": paper_dir / DIAGNOSIS_SOURCE / f"{name}.csv"
           for name in REPRODUCED}
    out.update({f"{source}/{REPLAY_TABLE}": paper_dir / source / REPLAY_TABLE
                for source in REPLAY_ARMS})
    return out


def tracked_inputs(paper_dir) -> list[Path]:
    """The plan and every tracked published input: the on-policy tables and
    config the diagnosis runner reads, and the tables read from --paper-dir."""
    return [PLAN_PATH, red.PUBLISHED_CONFIG, *red.published_tables(),
            *reference_paths(paper_dir).values()]


def sources_differing_from_head(paper_dir) -> list[str]:
    """Execution files, the plan and the tracked published inputs whose working
    copy is not byte-identical to HEAD's (an untracked file, or one outside the
    repository, differs by definition)."""
    differing = []
    for path in execution_sources() + tracked_inputs(paper_dir):
        path = Path(path).resolve()
        try:
            relative = str(path.relative_to(REPOSITORY))
        except ValueError:
            differing.append(str(path))
            continue
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPOSITORY,
                                   capture_output=True)
        if committed.returncode != 0 or not path.is_file() or committed.stdout != path.read_bytes():
            differing.append(relative)
    return differing


# --- identities ---------------------------------------------------------------------------------


def first_keys() -> list[tuple]:
    """The population keys of the run: every lineage at pi0."""
    return [lineage + (FIRST,) for lineage in red.lineages()]


def _horizon_fields(fraction: float, multiplier: float, h: float) -> dict[str, object]:
    star = matched_horizon(fraction, multiplier)
    return {"h": h, "h_star": star, "matched": h == star}


def _with_horizon(row: dict, h: float) -> dict:
    return {**row, **_horizon_fields(row["l1_fraction"], row["l2_multiplier"], h)}


def _cell_text(cell: tuple) -> str:
    trace, fraction, multiplier = cell
    return f"{trace}/{red.cell_label(fraction, multiplier)}"


# --- the published inputs -----------------------------------------------------------------------


def population_index(rows: list[dict]) -> tuple[dict[tuple, dict], list[str]]:
    """The published row of every pi0 population. The table's rows for the
    whole diagnosis grid (pi0 and pi3) are checked by `red.index_rows`, as the
    diagnosis checks them; only pi0 rows are kept, and only their files must
    exist (no pi3 population is read)."""
    index, problems = red.index_rows(rows, "test populations")
    index = {key: row for key, row in index.items() if key[5] == FIRST}
    for key, row in index.items():
        if row.get("window") != "test":
            problems.append(f"test populations: {red._name(key)} has window {row.get('window')!r}")
        if not Path(row["population_path"]).is_file():
            problems.append(f"test populations: {red._name(key)} file "
                            f"{row['population_path']} missing")
    return index, problems


def load_reproduced(paper_dir) -> tuple[dict[str, dict], list[str]]:
    """The published diagnosis tables reproduced at 600 s, as written: header,
    rows of strings, path and SHA-256."""
    tables, problems = {}, []
    for name in REPRODUCED:
        path = Path(paper_dir) / DIAGNOSIS_SOURCE / f"{name}.csv"
        if not path.is_file():
            problems.append(f"published {DIAGNOSIS_SOURCE}/{name}.csv is missing ({path})")
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = list(reader.fieldnames or ())
        missing = [column for column in REPRODUCED[name] if column not in columns]
        if missing:
            problems.append(f"published {DIAGNOSIS_SOURCE}/{name}.csv lacks {missing}")
        tables[name] = {"path": path, "sha256": sha256_path(path), "columns": columns, "rows": rows}
    return tables, problems


def load_replays(paper_dir) -> tuple[dict[tuple, dict], list[str]]:
    """`(source, arm, trace, fraction, multiplier, seed) -> integer columns`
    for every published replay row the bridge reads, each exactly once.

    `label`, `learned`, `pi0_binary` and `label_binary_{60,300,600}` must cover
    every trace x cell x seed of the grid; `label_binary_150` every cell whose h* is 150 s,
    and any other cell either fully or not at all. Rows outside the grid are
    skipped. Every row must be `all16` (eligibility `all`, width 16) with extra
    = avoided - L1-avoided, and every row of a trace x cell x seed must carry
    the same arm-independent identifiers.
    """
    grid = [(trace, fraction, multiplier, seed) for trace in red.TRACES
            for fraction, multiplier in red.CELLS for seed in red.SEEDS]
    wanted = set(grid)
    rows: dict[tuple, dict] = {}
    problems: list[str] = []
    for source, arms in REPLAY_ARMS.items():
        path = Path(paper_dir) / source / REPLAY_TABLE
        if not path.is_file():
            problems.append(f"published {source}/{REPLAY_TABLE} is missing ({path})")
            continue
        for row in red._read(path):
            if (row.get("mechanism") != REPLAY_MECHANISM[0] or row.get("arm") not in arms
                    or row.get("variant", "main") != "main"):
                continue
            try:
                key = (row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
                       int(row["seed"]))
            except (KeyError, TypeError, ValueError) as error:
                problems.append(f"{source}: unreadable identity ({error!r})")
                continue
            if key not in wanted:
                continue
            name = f"{source}/{row['arm']}/{_cell_text(key[:3])}/s{key[3]}"
            if (row.get("eligibility"), row.get("width")) != REPLAY_MECHANISM[1:]:
                problems.append(f"{name}: eligibility/width {row.get('eligibility')}/"
                                f"{row.get('width')}")
            if row.get("cell") != red.cell_label(key[1], key[2]):
                problems.append(f"{name}: cell {row.get('cell')!r}")
            full = (source, row["arm"]) + key
            if full in rows:
                problems.append(f"{name}: duplicate row")
                continue
            try:
                entry = {field: int(row[field]) for field in REPLAY_FIELDS}
            except (KeyError, TypeError, ValueError) as error:
                problems.append(f"{name}: unreadable ({error!r})")
                continue
            if entry["extra_avoided_tokens"] != (entry["avoided_prefill_tokens"]
                                                 - entry["l1_avoided_tokens"]):
                problems.append(f"{name}: extra != avoided - L1-avoided")
            if entry["requested_tokens"] <= 0:
                problems.append(f"{name}: requested tokens not positive")
            rows[full] = entry
    for source, arms in REPLAY_ARMS.items():
        for arm in arms:
            for cell in sorted({key[:3] for key in grid}):
                present = [(source, arm) + cell + (seed,) in rows for seed in red.SEEDS]
                required = arm != LABEL_BINARY_ARMS[150.0][1] or matched_horizon(*cell[1:]) == 150.0
                absent = [seed for seed, there in zip(red.SEEDS, present) if not there]
                if absent and (required or len(absent) < len(present)):
                    problems.append(f"published {source}/{arm}/{_cell_text(cell)}: seeds "
                                    f"{absent} missing")
    for key in grid:
        entries = {full[:2]: entry for full, entry in rows.items() if full[2:] == key}
        identifiers = {tuple(entry[field] for field in REPLAY_IDENTIFIERS)
                       for entry in entries.values()}
        if len(identifiers) > 1:
            problems.append(f"{_cell_text(key[:3])}/s{key[3]}: the replay rows differ in "
                            f"{REPLAY_IDENTIFIERS}")
    return rows, problems


def utility_gaps(replays: dict[tuple, dict]) -> dict[tuple, dict]:
    """Per trace x cell and ranker arm (`RANKER_ARMS`), seed-ordered:
    `U(label) - U(ranker arm)` and, per horizon, `U(label_binary_h) - U(ranker
    arm)` (None where not published), each seed's integer token difference in
    points of its input tokens, as the published location and horizon tables
    compute it."""
    out = {}
    for trace in red.TRACES:
        for fraction, multiplier in red.CELLS:
            cell = (trace, fraction, multiplier)
            for ranker_arm in sorted(set(RANKER_ARMS.values())):
                own = [replays[("error_location_001", ranker_arm) + cell + (seed,)]
                       for seed in red.SEEDS]

                def gap(source: str, arm: str):
                    entries = [replays.get((source, arm) + cell + (seed,)) for seed in red.SEEDS]
                    if None in entries:
                        return None
                    return [red._points(entry["extra_avoided_tokens"]
                                        - base["extra_avoided_tokens"], entry["requested_tokens"])
                            for entry, base in zip(entries, own)]

                out[cell + (ranker_arm,)] = {
                    "label": gap("error_location_001", "label"),
                    "label_binary": {h: gap(*source) for h, source in LABEL_BINARY_ARMS.items()}}
    return out


# --- one population -----------------------------------------------------------------------------


def population_statistics(population: DecisionPopulation, ranker) -> dict[str, object]:
    """Everything the readings need from one pi0 population, as small numbers.

    Victims and key ranks do not depend on the horizon and are computed once:
    the logged victim, the published model's recomputed victim (first minimum
    of score and stored tie-break), recency's. At each horizon the decisions
    are rebuilt from the stored arrays (`matcheddiag.decisions_at`); at h = 600
    they are the diagnosis's.
    """
    base = matcheddiag.decisions_at(population, DIAGNOSIS_HORIZON_SECONDS)
    logged = errordiag.logged_victims(population.victim, base)
    scores = errordiag.scorer_scores(ranker, population)
    recomputed = errordiag.scorer_victims(scores, base)
    recency = errordiag.recency_victims(base)
    ranker_ranks = matcheddiag.ranker_key_ranks(scores, base)
    recency_ranks = matcheddiag.recency_key_ranks(base)
    out: dict[str, object] = {
        "rows": len(population), "decisions": len(base),
        "decisions_eligible": population.decisions_eligible, "cap_bound": population.cap_bound,
        "window": population.window, "horizon_seconds": population.horizon_seconds,
        "population_seed": population.seed,
        "recorded_argmin_mismatches": int(verify_recorded_argmin(
            population, raise_on_mismatch=False)["recorded_argmin_mismatches"]),
        "own_victim_mismatches": int((recomputed != logged).sum()),
        "own_score_exact": bool(np.array_equal(scores, population.arm_score)),
        "own_score_max_abs_diff": float(np.abs(scores - population.arm_score).max()),
    }
    del scores
    horizons = {}
    for h in HORIZONS_SECONDS:
        decisions = (base if h == DIAGNOSIS_HORIZON_SECONDS
                     else matcheddiag.decisions_at(population, h))
        horizons[h] = matcheddiag.horizon_statistics(decisions, logged, recomputed, recency,
                                                     ranker_ranks, recency_ranks)
        del decisions
    out["horizons"] = horizons
    return out


def lineage_worker(task) -> dict[str, object]:
    """The pi0 population of one lineage, hashed on the bytes it is loaded
    from; nothing is computed on a population whose hash differs."""
    started = time.time()
    task = tuple(task)
    key = task + (FIRST,)
    metadata = SHARED["populations"][key]
    payload = Path(metadata["population_path"]).read_bytes()
    entry: dict[str, object] = {"sha256": hashlib.sha256(payload).hexdigest(),
                                "bytes": len(payload)}
    if entry["sha256"] == metadata["population_sha256"]:
        population = DecisionPopulation.load(io.BytesIO(payload))
        del payload
        entry.update(population_statistics(population, SHARED["rankers"][key]))
        del population
    return {"task": task, "population": entry, "seconds": time.time() - started,
            "peak_rss_mib": red._peak_rss_mib(resource.RUSAGE_SELF)}


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
    """The loaded pi0 populations against the published manifest (the
    diagnosis's checks, for pi0)."""
    problems = []
    for key in first_keys():
        entry = results[key[:5]]["population"]
        metadata = populations[key]
        if entry["sha256"] != metadata["population_sha256"]:
            problems.append(f"{red._name(key)}: population SHA-256 {entry['sha256']} != "
                            f"published {metadata['population_sha256']}")
            continue
        for field, published in (("rows", "rows"), ("decisions", "decisions_kept"),
                                 ("decisions_eligible", "decisions_eligible")):
            if int(entry[field]) != int(metadata[published]):
                problems.append(f"{red._name(key)}: {field} {entry[field]} != published "
                                f"{metadata[published]}")
        if entry["window"] != "test":
            problems.append(f"{red._name(key)}: window {entry['window']!r}")
        if float(entry["horizon_seconds"]) != DIAGNOSIS_HORIZON_SECONDS:
            problems.append(f"{red._name(key)}: horizon {entry['horizon_seconds']} s")
        if int(entry["population_seed"]) != key[4]:
            problems.append(f"{red._name(key)}: reservoir seed {entry['population_seed']}")
    return problems


def check_rows(results, populations, models) -> list[dict]:
    """The diagnosis's check table, for the pi0 populations."""
    out = []
    for key in first_keys():
        lineage = key[:5]
        entry = results[lineage]["population"]
        metadata = populations[key]
        decisions = int(entry["decisions"])
        mismatches = int(entry["own_victim_mismatches"])
        out.append({
            **red._identity(lineage), "iteration": FIRST, "policy": f"pi{FIRST}",
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


# --- derived tables -----------------------------------------------------------------------------


def _stats(values) -> dict[str, float]:
    """Mean, min, max (the diagnosis's `_summary`: all nan when any seed is
    nan) and the 95% t half-width over seeds (`gap.summarize`; nan then too)."""
    values = [float(value) for value in values]
    out = red._summary(values)
    finite = bool(values) and all(math.isfinite(value) for value in values)
    out["ci95_half"] = summarize(values)["ci95_half"] if finite else math.nan
    return out


def _put(row: dict, field: str, values) -> None:
    for statistic, value in _stats(values).items():
        row[f"{field}_{statistic}"] = value


def _per_horizon(results, h: float) -> dict[tuple, dict]:
    """The diagnosis's results layout for one horizon, so that its own
    derivations (`class_seed_rows`, `rate_seed_rows`) read our numbers."""
    return {lineage: {"populations": {FIRST: result["population"]["horizons"][h]}}
            for lineage, result in results.items()}


def _horizon(results, lineage: tuple, h: float) -> dict:
    return results[lineage]["population"]["horizons"][h]


def class_and_rate_tables(results) -> dict[str, list[dict]]:
    """Readings 1 and 2: the diagnosis's class and rate tables at every
    horizon, built by the diagnosis's own functions, each row with h, h* and
    `matched`; rate counts per horizon and at each cell's h*."""
    tables = {name: [] for name in ("classes_seeds", "classes", "rates_seeds", "rates",
                                    "rates_counts")}
    for h in HORIZONS_SECONDS:
        layout = _per_horizon(results, h)
        class_seeds = red.class_seed_rows(layout)
        rate_seeds = red.rate_seed_rows(layout)
        rates = red.rate_rows(rate_seeds)
        tables["classes_seeds"] += [_with_horizon(row, h) for row in class_seeds]
        tables["classes"] += [_with_horizon(row, h) for row in red.class_summary_rows(class_seeds)]
        tables["rates_seeds"] += [_with_horizon(row, h) for row in rate_seeds]
        tables["rates"] += [_with_horizon(row, h) for row in rates]
        tables["rates_counts"] += [{"horizon": f"{h:g}", "matched": False, **row}
                                   for row in red.rate_count_rows(rates)]
    matched = [row for row in tables["rates"] if row["matched"]]
    tables["rates_counts"] += [{"horizon": "matched", "matched": True, **row}
                               for row in red.rate_count_rows(matched)]
    return tables


def statistics_seed_rows(results) -> list[dict]:
    """m3 and m4 per lineage x horizon x victim source x subset."""
    out = []
    for lineage in red.lineages():
        for h in HORIZONS_SECONDS:
            statistics = _horizon(results, lineage, h)["statistics"]
            for source in VICTIM_SOURCES:
                for subset in errordiag.SUBSETS:
                    out.append({**_with_horizon(red._identity(lineage), h),
                                "victim_source": source, "subset": subset,
                                **statistics[source][subset]})
    return out


def concordance_seed_rows(results) -> list[dict]:
    return [{**_with_horizon(red._identity(lineage), h),
             **_horizon(results, lineage, h)["concordance"]}
            for lineage in red.lineages() for h in HORIZONS_SECONDS]


def _seed_index(rows: list[dict], *fields: str) -> dict[tuple, dict]:
    return {(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["target"], row["seed"])
            + tuple(row[field] for field in fields): row for row in rows}


def concordance_rows(seed_rows: list[dict]) -> list[dict]:
    """Reading 3 per trace x cell x target x horizon, over seeds."""
    index = _seed_index(seed_rows, "h")
    out = []
    for cell in red.grid_cells():
        for h in HORIZONS_SECONDS:
            seeds = [index[cell + (seed, h)] for seed in red.SEEDS]
            ranker = [seed["ranker"] for seed in seeds]
            recency = [seed["recency"] for seed in seeds]
            row = {**_with_horizon(red._cell_identity(cell), h), "seeds": len(seeds),
                   "decisions_mean": red._mean([seed["decisions"] for seed in seeds])}
            _put(row, "ranker", ranker)
            _put(row, "recency", recency)
            row["uniform"] = UNIFORM_CONCORDANCE
            for name, other in (("recency", recency),
                                ("uniform", [UNIFORM_CONCORDANCE] * len(seeds))):
                differences = [r - o for r, o in zip(ranker, other)]
                row.update({f"ranker_minus_{name}_mean": red._mean(differences),
                            f"ranker_minus_{name}_seed_signs": errordiag.seed_signs(differences),
                            f"ranker_minus_{name}_reading": sign_reading(differences)})
            out.append(row)
    return out


def concordance_reading_rows(seed_rows: list[dict], summary: list[dict]) -> list[dict]:
    """Reading 3's registered prediction per trace x cell x target: the
    ranker's five-seed mean concordance for the h*-bit against the 600-second
    bit. Registered for `next_use` where h* < 600 s; the other rows are
    reported without a reading."""
    index = _seed_index(seed_rows, "h")
    means = {(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["target"], row["h"]): row
             for row in summary}
    out = []
    for cell in red.grid_cells():
        trace, fraction, multiplier, target = cell
        star = matched_horizon(fraction, multiplier)
        at_star, at_600 = means[cell + (star,)], means[cell + (DIAGNOSIS_HORIZON_SECONDS,)]
        differences = [index[cell + (seed, star)]["ranker"]
                       - index[cell + (seed, DIAGNOSIS_HORIZON_SECONDS)]["ranker"]
                       for seed in red.SEEDS]
        registered = matcheddiag.prediction_registered(target, fraction, multiplier)
        below = matcheddiag.concordance_prediction_holds(at_star["ranker_mean"],
                                                         at_600["ranker_mean"])
        out.append({
            **red._cell_identity(cell), "h_star": star, "seeds": len(differences),
            "registered": registered,
            "ranker_at_h_star_mean": at_star["ranker_mean"],
            "ranker_at_600_mean": at_600["ranker_mean"],
            "h_star_minus_600_mean": red._mean(differences),
            "h_star_minus_600_seed_signs": errordiag.seed_signs(differences),
            "recency_at_h_star_mean": at_star["recency_mean"],
            "recency_at_600_mean": at_600["recency_mean"],
            "below_600": below,
            "prediction": ("holds" if below else "fails") if registered else "not_registered",
        })
    return out


def concordance_count_rows(rows: list[dict]) -> list[dict]:
    out = []
    for target in red.TARGETS:
        members = [row for row in rows if row["target"] == target]
        shorter = [row for row in members if row["h_star"] < DIAGNOSIS_HORIZON_SECONDS]
        out.append({
            "target": target, "cells": len(members), "cells_h_star_below_600": len(shorter),
            "registered": sum(row["registered"] for row in members),
            "holds": sum(row["prediction"] == "holds" for row in members),
            "fails": sum(row["prediction"] == "fails" for row in members),
            "below_600": sum(row["below_600"] for row in shorter),
        })
    return out


def composition_rows(classes: list[dict]) -> list[dict]:
    """Reading 1's prediction per trace x cell x target: the avoidable
    reusable eviction's share of the label excess of every decision at h*,
    against 0.99 on the five-seed mean (registered) and in every seed."""
    index = {(row["trace"], row["l1_fraction"], row["l2_multiplier"], row["target"], row["h"]): row
             for row in classes if row["subset"] == "all" and row["class"] == "avoidable_reusable"}
    out = []
    for cell in red.grid_cells():
        star = matched_horizon(cell[1], cell[2])
        row, at_600 = index[cell + (star,)], index[cell + (DIAGNOSIS_HORIZON_SECONDS,)]
        out.append({
            **red._cell_identity(cell), "h_star": star, "seeds": row["seeds"],
            "excess_share_mean": row["excess_share_mean"],
            "excess_share_min": row["excess_share_min"],
            "excess_share_max": row["excess_share_max"],
            "decision_share_mean": row["decision_share_mean"],
            "excess_share_at_600_mean": at_600["excess_share_mean"],
            "holds": matcheddiag.composition_holds(row["excess_share_mean"]),
            "holds_every_seed": matcheddiag.composition_holds(row["excess_share_min"]),
        })
    return out


def composition_count_rows(rows: list[dict]) -> list[dict]:
    out = []
    for target in tuple(red.TARGETS) + ("all",):
        members = [row for row in rows if target == "all" or row["target"] == target]
        out.append({"target": target, "cells": len(members),
                    "holds": sum(row["holds"] for row in members),
                    "holds_every_seed": sum(row["holds_every_seed"] for row in members)})
    return out


def bridge_rows(results, gaps: dict[tuple, dict]) -> list[dict]:
    """Reading 4 per trace x cell x target x horizon: the ranker's m4 by its
    logged victim over every decision and over the resident-victim decisions,
    beside the published U gaps of the trace x cell against that target's
    ranker's own replay (`RANKER_ARMS`)."""
    out = []
    for cell in red.grid_cells():
        trace, fraction, multiplier, target = cell
        ranker_arm = RANKER_ARMS[target]
        cell_gaps = gaps[(trace, fraction, multiplier, ranker_arm)]
        for h in HORIZONS_SECONDS:
            logged = [_horizon(results, cell + (seed,), h)["statistics"]["logged"]
                      for seed in red.SEEDS]
            row = {**_with_horizon(red._cell_identity(cell), h), "seeds": len(logged),
                   "ranker_arm": ranker_arm}
            for subset in M4_SUBSETS:
                _put(row, f"m4_{subset}", [entry[subset]["m4"] for entry in logged])
            row["resident_decisions_mean"] = red._mean([entry["resident"]["decisions"]
                                                        for entry in logged])
            source, arm = LABEL_BINARY_ARMS[h]
            binary = cell_gaps["label_binary"][h]
            for name, values in (("label", cell_gaps["label"]), ("label_binary", binary)):
                stats = (_stats(values) if values is not None
                         else {"mean": math.nan, "min": math.nan, "max": math.nan,
                               "ci95_half": math.nan})
                for statistic in ("mean", "min", "max"):
                    row[f"u_{name}_minus_ranker_{statistic}"] = stats[statistic]
            row["label_binary_arm"] = arm if binary is not None else ""
            row["label_binary_source"] = source if binary is not None else ""
            out.append(row)
    return out


def bridge_spearman_rows(rows: list[dict]) -> list[dict]:
    """Reading 4's description: Spearman's rank correlation over the trace x
    cells of each target, of each m4 against each U gap, at each cell's h*
    (`matched`, the registered one) and at each fixed horizon; nan unless every
    cell has both values."""
    out = []
    selections = [("matched", None)] + [(f"{h:g}", h) for h in HORIZONS_SECONDS]
    for target in red.TARGETS:
        for horizon, h in selections:
            members = [row for row in rows if row["target"] == target
                       and (row["matched"] if h is None else row["h"] == h)]
            for subset in M4_SUBSETS:
                for name in U_GAPS:
                    x = [row[f"m4_{subset}_mean"] for row in members]
                    y = [row[f"u_{name}_minus_ranker_mean"] for row in members]
                    points = sum(math.isfinite(a) and math.isfinite(b) for a, b in zip(x, y))
                    out.append({"target": target, "ranker_arm": RANKER_ARMS[target],
                                "horizon": horizon, "matched": h is None,
                                "m4_subset": subset, "u_gap": name, "cells": len(members),
                                "points": points,
                                "spearman": (matcheddiag.spearman(x, y) if points == len(members)
                                             else math.nan)})
    return out


def derive_tables(results, gaps) -> dict[str, list[dict]]:
    tables = class_and_rate_tables(results)
    concordance_seeds = concordance_seed_rows(results)
    concordance = concordance_rows(concordance_seeds)
    concordance_reading = concordance_reading_rows(concordance_seeds, concordance)
    composition = composition_rows(tables["classes"])
    bridge = bridge_rows(results, gaps)
    return {
        **tables,
        "statistics_seeds": statistics_seed_rows(results),
        "concordance_seeds": concordance_seeds,
        "concordance": concordance,
        "concordance_reading": concordance_reading,
        "concordance_counts": concordance_count_rows(concordance_reading),
        "composition_reading": composition,
        "composition_counts": composition_count_rows(composition),
        "bridge": bridge,
        "bridge_spearman": bridge_spearman_rows(bridge),
    }


# --- the reproduction at 600 s ------------------------------------------------------------------


def _as_written(rows: list[dict], fields: list[str]) -> list[dict[str, str]]:
    """The rows exactly as `csv.DictWriter` writes them, read back as strings."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    buffer.seek(0)
    return list(csv.DictReader(buffer))


def reproduction_rows(derived: dict[str, list[dict]], published: dict[str, dict]) -> list[dict]:
    """The h = 600 rows of each reproduced table against the published one:
    the same row keys, and every common column equal as written (exact float
    equality: `csv` writes a float's shortest round-trip repr). A published
    column absent here is a difference, not a skipped column."""
    out = []
    for name, key_columns in REPRODUCED.items():
        table = published[name]
        fields = columns_of(name)
        ours = [row for row in _as_written(derived[name], fields)
                if float(row["h"]) == DIAGNOSIS_HORIZON_SECONDS]
        common = [column for column in table["columns"] if column in fields]
        absent = [column for column in table["columns"] if column not in fields]
        index, duplicates = {}, 0
        for row in ours:
            key = tuple(row[column] for column in key_columns)
            duplicates += key in index
            index[key] = row
        published_keys, differing, compared = set(), 0, 0
        first = ""
        for row in table["rows"]:
            key = tuple(row.get(column) for column in key_columns)
            duplicates += key in published_keys
            published_keys.add(key)
            mine = index.get(key)
            if mine is None:
                continue
            for column in common:
                compared += 1
                if mine[column] != row[column]:
                    differing += 1
                    if not first:
                        first = (f"{'/'.join(key)}: {column} published {row[column]!r}, "
                                 f"here {mine[column]!r}")
        missing = len(published_keys - set(index))
        extra = len(set(index) - published_keys)
        out.append({
            "table": name, "published_path": red._display(table["path"]),
            "published_sha256": table["sha256"], "published_rows": len(table["rows"]),
            "rows_at_600": len(ours), "key_columns": ";".join(key_columns),
            "common_columns": len(common), "published_columns_absent": ";".join(absent),
            "missing_rows": missing, "extra_rows": extra, "duplicate_keys": duplicates,
            "compared_values": compared, "differing_values": differing,
            "first_difference": first,
            "equal": not (missing or extra or duplicates or differing or absent),
        })
    return out


# --- the tables and their columns ---------------------------------------------------------------

_CELL = red._CELL
_SEED = red._SEED
_H = [
    ("h", "reuse horizon h in seconds: label_h = -log1p(min(next_use_delta_s, h)); a candidate is "
          "reusable at h when its next use is at most h away"),
    ("h_star", "the cell's matched horizon h* in seconds (the plan's table, the same on both traces)"),
    ("matched", "h equals h*"),
]


def _parent_columns(name: str, seeded: bool) -> list[tuple[str, str]]:
    """A diagnosis table's columns with h, h* and `matched` after the identity."""
    columns = red.TABLES[name][1]
    head = _CELL + _SEED if seeded else _CELL
    if columns[:len(head)] != head:
        raise AssertionError(f"the diagnosis table {name} does not start with its identity")
    return head + _H + columns[len(head):]


def _stats_columns(field: str, text: str) -> list[tuple[str, str]]:
    return red._summary_columns(field, text) + [
        (f"{field}_ci95_half", f"95% t half-width over the seeds of {text} (nan when any seed is nan)")]


TABLES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "checks": (
        "Required checks, one row per pi0 population (trace x cell x target x seed); the "
        "diagnosis's check table.",
        red.TABLES["checks"][1],
    ),
    "classes_seeds": (
        "Reading 1 per seed and horizon: the four error kinds of the pi0 population's decisions by "
        "their logged victim, with labels and reuse at h, for every decision and separately for "
        "decisions whose victim is the arrival or a resident, built by the diagnosis's own "
        "functions (its summary at h = 600 is compared in reproduction.csv).",
        _parent_columns("classes_seeds", seeded=True),
    ),
    "classes": (
        "Reading 1 over seeds, per trace x cell x target x horizon x subset x class. The h = 600 "
        "rows equal the published diagnosis's `classes.csv` (reproduction.csv).",
        _parent_columns("classes", seeded=False),
    ),
    "rates_seeds": (
        "Reading 2 per seed and horizon, on the pi0 population, with reuse at h. The h = 600 rows "
        "equal the published diagnosis's `rates_seeds.csv` (reproduction.csv).",
        _parent_columns("rates_seeds", seeded=True),
    ),
    "rates": (
        "Reading 2 over seeds, per trace x cell x target x horizon x rate. The h = 600 rows equal "
        "the published diagnosis's `rates.csv` (reproduction.csv).",
        _parent_columns("rates", seeded=False),
    ),
    "rates_counts": (
        "Reading 2 counts, per horizon selection x target x rate, out of the trace x cells.",
        [("horizon", "`60`, `150`, `300` or `600` (every cell at that h), or `matched` (every "
                     "cell at its own h*)"),
         ("matched", "the row counts every cell at its own h*")] + red.TABLES["rates_counts"][1],
    ),
    "statistics_seeds": (
        "m3 and m4 at each horizon, per seed, for the logged victim and for the published pi0 "
        "model's recomputed victim (the first minimum of its score and the stored tie-break), "
        "over every decision and over the decisions whose victim (of that source) is the arrival "
        "or a resident.",
        _CELL + _SEED + _H + [
            ("victim_source", "`logged` or `scorer_pi0`"),
            ("subset", "`all`, `arrival` (the victim is the arrival) or `resident`"),
            ("decisions", "decisions in the subset"),
            ("excess_sum", "their summed label_h(victim) - min label_h"),
            ("m3", "excess_sum / decisions"),
            ("avoidable_count", "decisions whose victim is reusable at h while some candidate is not"),
            ("m4", "avoidable_count / decisions"),
        ],
    ),
    "concordance_seeds": (
        "Reading 3 per seed and horizon: the within-decision concordance of each chooser's key for "
        "the reuse bit at h, over the decisions with both a reusable and a non-reusable candidate.",
        _CELL + _SEED + _H + [
            ("decisions", "decisions with both a reusable-at-h and a non-reusable-at-h candidate"),
            ("ranker",
             "mean over those decisions (each weighing one) of the share of (reusable, "
             "non-reusable) candidate pairs in which the key (score of the published pi0 model, "
             "stored tie-break), compared lexicographically, is lower for the non-reusable "
             "candidate, i.e. would evict it first; an exact tie of the key counts one half"),
            ("recency", "the same with the stored tie-break alone as the key"),
            ("uniform", "a uniformly drawn victim: 0.5 exactly"),
        ],
    ),
    "concordance": (
        "Reading 3 over seeds, per trace x cell x target x horizon.",
        _CELL + _H + [("seeds", "seeds"),
                      ("decisions_mean", "mean decisions the concordance is taken over")]
        + _stats_columns("ranker", "the ranker's concordance")
        + _stats_columns("recency", "recency's concordance") + [
            ("uniform", "a uniformly drawn victim's concordance, 0.5"),
            ("ranker_minus_recency_mean", "mean of the seed-paired ranker - recency"),
            ("ranker_minus_recency_seed_signs", "signs of ranker - recency, seeds 0-4"),
            ("ranker_minus_recency_reading",
             "consistent_gain (the ranker's key separates better in every seed), consistent_loss "
             "or mixed; descriptive"),
            ("ranker_minus_uniform_mean", "mean of ranker - 0.5"),
            ("ranker_minus_uniform_seed_signs", "signs of ranker - 0.5, seeds 0-4"),
            ("ranker_minus_uniform_reading", "consistent_gain / consistent_loss / mixed; descriptive"),
        ],
    ),
    "concordance_reading": (
        "Reading 3's registered prediction (the fit-horizon hypothesis), per trace x cell x "
        "target: the ranker's concordance for the h*-bit against the 600-second bit on the same "
        "populations, five-seed means. Registered for `next_use` in the trace x cells with h* < "
        "600 s; the other rows are descriptive.",
        _CELL + [
            ("h_star", "the cell's matched horizon h* in seconds"),
            ("seeds", "seed-paired values"),
            ("registered", "the prediction is registered here (target next_use and h* < 600 s)"),
            ("ranker_at_h_star_mean", "five-seed mean of the ranker's concordance at h* "
                                      "(concordance.csv)"),
            ("ranker_at_600_mean", "five-seed mean of the ranker's concordance at 600 s"),
            ("h_star_minus_600_mean", "mean of the seed-paired concordance at h* - at 600 s"),
            ("h_star_minus_600_seed_signs", "signs of that difference, seeds 0-4"),
            ("recency_at_h_star_mean", "five-seed mean of recency's concordance at h*"),
            ("recency_at_600_mean", "five-seed mean of recency's concordance at 600 s"),
            ("below_600", "ranker_at_h_star_mean < ranker_at_600_mean (strictly; equal is not below)"),
            ("prediction", "`holds` (registered and below_600), `fails` (registered and not "
                           "below_600) or `not_registered`"),
        ],
    ),
    "concordance_counts": (
        "Reading 3 counts, per target.",
        [("target", "training target"), ("cells", "trace x cells (12)"),
         ("cells_h_star_below_600", "trace x cells with h* < 600 s (8)"),
         ("registered", "trace x cells where the prediction is registered (8 for next_use, 0 for "
                        "binary)"),
         ("holds", "registered trace x cells where it holds"),
         ("fails", "registered trace x cells where it fails"),
         ("below_600", "trace x cells with h* < 600 s where the h* concordance is below the "
                       "600-second one (for binary: descriptive)")],
    ),
    "composition_reading": (
        "Reading 1's prediction, per trace x cell x target: the avoidable reusable eviction's "
        "share of the label_h* excess of every decision (subset `all` of classes.csv at h*).",
        _CELL + [
            ("h_star", "the cell's matched horizon h* in seconds"),
            ("seeds", "seeds"),
            ("excess_share_mean", "five-seed mean of the share at h* (classes.csv)"),
            ("excess_share_min", "its minimum over seeds"),
            ("excess_share_max", "its maximum over seeds"),
            ("decision_share_mean", "five-seed mean of the class's share of decisions at h*"),
            ("excess_share_at_600_mean", "the same share at 600 s (the diagnosis's)"),
            ("holds", "excess_share_mean >= 0.99 (the registered prediction; nan does not hold)"),
            ("holds_every_seed", "excess_share_min >= 0.99 (every seed; descriptive)"),
        ],
    ),
    "composition_counts": (
        "Reading 1 counts, per target and over both (`all`, out of 24).",
        [("target", "next_use, binary, or all"), ("cells", "trace x cell x targets counted"),
         ("holds", "rows of composition_reading.csv where the prediction holds"),
         ("holds_every_seed", "rows where it holds in every seed")],
    ),
    "bridge": (
        "Reading 4, descriptive, per trace x cell x target x horizon: the ranker's m4 at h by its "
        "logged victim, beside the published utility gaps of the trace x cell against that "
        "target's ranker's own replay (all16; U is extra avoided prefill tokens over L1 alone in "
        "points of input tokens). The registered reading is the `matched` rows.",
        _CELL + _H + [("seeds", "seeds"),
                      ("ranker_arm", "the ranker's own published all16 replay in "
                                     "error_location_001 the gaps are taken against: `learned` "
                                     "(frozen pi0 next_use) for next_use, `pi0_binary` for binary")]
        + _stats_columns("m4_all", "m4 over every decision")
        + _stats_columns("m4_resident", "m4 over the decisions whose logged victim is a resident")
        + [("resident_decisions_mean", "mean number of resident-victim decisions")]
        + red._summary_columns("u_label_minus_ranker",
                               "U(label) - U(ranker_arm), seed by seed from the published "
                               "error_location_001 replay rows (each seed's token difference in "
                               "points of its input tokens)")
        + red._summary_columns("u_label_binary_minus_ranker",
                               "U(label_binary_h) - U(ranker_arm), the same way; nan where "
                               "label_binary_h was not replayed for the cell (150 s outside the "
                               "fill-in's cells)")
        + [("label_binary_arm", "the published arm read for U(label_binary_h), empty when none"),
           ("label_binary_source", "its published directory (horizon_control_001 or "
                                   "horizon_fill_001), empty when none")],
    ),
    "bridge_spearman": (
        "Reading 4's description: Spearman's rank correlation (Pearson's on average ranks, ties "
        "averaged) over the trace x cells of a target between a five-seed mean m4 and a "
        "five-seed mean U gap of bridge.csv. Twelve points, not a test.",
        [("target", "training target"),
         ("ranker_arm", "the replay the target's U gaps are taken against (as in bridge.csv)"),
         ("horizon", "`matched` (each cell at its own h*: the registered description) or a fixed "
                     "h for every cell"),
         ("matched", "horizon is `matched`"),
         ("m4_subset", "`all` (m4_all_mean) or `resident` (m4_resident_mean)"),
         ("u_gap", "`label` (U(label) - U(ranker_arm)) or `label_binary` (U(label_binary_h) - "
                   "U(ranker_arm) at the row's h)"),
         ("cells", "trace x cells of the target"),
         ("points", "of those, cells with both values finite"),
         ("spearman", "the rank correlation; nan unless points equals cells")],
    ),
    "reproduction": (
        "Required check: the h = 600 rows of classes.csv, rates.csv and rates_seeds.csv against "
        "the published ranker-error diagnosis tables, value for value as written (exact; no "
        "tolerance). The run publishes nothing unless every row is equal.",
        [("table", "the table compared"),
         ("published_path", "the published table read"),
         ("published_sha256", "its SHA-256"),
         ("published_rows", "its rows"),
         ("rows_at_600", "rows of this run's table at h = 600"),
         ("key_columns", "the columns identifying a row"),
         ("common_columns", "published columns that this table also has, all compared"),
         ("published_columns_absent", "published columns this table lacks (must be empty)"),
         ("missing_rows", "published rows without a row here"),
         ("extra_rows", "rows here without a published row"),
         ("duplicate_keys", "repeated row keys on either side"),
         ("compared_values", "values compared"),
         ("differing_values", "values not equal as written"),
         ("first_difference", "the first of them, empty when none"),
         ("equal", "no missing, extra, duplicate or differing value and no absent column")],
    ),
}
# The tables a complete run writes, in this order.
TABLE_NAMES = tuple(TABLES)


def columns_of(name: str) -> list[str]:
    return [column for column, _ in TABLES[name][1]]


README_HEAD = """# Matched-horizon diagnosis on saved decision logs

Pre-registration: `docs/matched-horizon-diagnosis-plan.md`. Logs: the held-out `pi0` decision populations of the published on-policy run, 2 traces x 6 cells x 2 targets x 5 seeds, each a reservoir of at most 40,000 whole decisions at least 600 s before the trace end, loaded by the SHA-256 the published table records; no `pi3` population is read. Scorer: the published `pi0` model of the lineage; its victim on a logged decision is the first minimum, in stored candidate order, of (score on the stored features, stored tie-break); no hybrid, protection or override. Horizons h in {60, 150, 300, 600} s: `label_h = -log1p(min(next_use_delta_s, h))`, and a candidate is reusable at h when its next use is at most h away. The matched horizon h* of a cell: 60 s at 0.25%x1, 150 s at 0.25%x4 and 1%x1, 300 s at 2%x1, 600 s at 1%x4 and 2%x4, the same on both traces. The error kinds, their precedence, `m3`, `m4` and the conditional rates are the ranker-error diagnosis's with h in place of 600, computed by its own code; at h = 600 the class, rate and per-seed rate tables equal its published ones value for value (`reproduction.csv`). The within-decision concordance of a key for the reuse bit at h is, over the decisions with both a reusable and a non-reusable candidate, the share of (reusable, non-reusable) candidate pairs in which the key is lower for the non-reusable one (it would be evicted first), an exact tie counting one half, averaged with each decision weighing one; the ranker's key is (score, stored tie-break), recency's the stored tie-break alone, and a uniformly drawn victim has 0.5. `U` is extra avoided prefill tokens (avoided minus L1-avoided) in points of input tokens, from the published `all16` replay rows. "Consistent" means the same sign in all five seed-paired values; seed signs list seeds 0-4 as +, -, 0, or n for nan.

Nothing was replayed or fitted. The populations are the frozen ranker's own store; a store kept by the exact bit would present other candidate sets. h* was read from replays of the same two traces after the fact. Concordance and rates on logged candidates are properties of a score on those sets, not utility. Both published rankers were fitted to 600-second targets.
"""

UNRESOLVED_HEAD = """# Matched-horizon diagnosis on saved decision logs: UNRESOLVED

Pre-registration: `docs/matched-horizon-diagnosis-plan.md`. The published `pi0` model applied to its own population did not reproduce the logged victim in more than 0.1% of the decisions of at least one population (`checks.csv`, `exceeds_stop_rule`), so the diagnosis stops as unresolved by the plan's rule and no reading is derived.
"""


def readme(names: tuple[str, ...], head: str) -> str:
    sections = [head]
    for name in names:
        description, columns = TABLES[name]
        lines = [f"## `{name}.csv`", "", description, ""]
        lines += [f"- `{column}`: {text}" for column, text in columns]
        sections.append("\n".join(lines) + "\n")
    sections.append("## `run_config.json`\n\nPlan and code commits, source manifest, the "
                    "SHA-256 of every population, model and published table read, the horizons "
                    "and the matched map, check counts, reading counts, timing and memory.\n")
    return "\n".join(sections)


def _write(path: Path, rows: list[dict], name: str) -> None:
    """A table with exactly its documented columns, in their documented order."""
    fields = columns_of(name)
    if len(fields) != len(set(fields)):
        raise AssertionError(f"{name}: a column is documented twice")
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


def _reading_summary(derived: dict[str, list[dict]]) -> dict[str, object]:
    return {
        "composition_counts": derived["composition_counts"],
        "rates_counts_matched": [row for row in derived["rates_counts"] if row["matched"]],
        "concordance_counts": derived["concordance_counts"],
        "bridge_spearman_matched": [row for row in derived["bridge_spearman"] if row["matched"]],
    }


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
    differing = sources_differing_from_head(args.paper_dir)
    if differing:
        raise SystemExit("execution source, plan or published input differs from HEAD: "
                         f"{differing}")
    head_start = _git("rev-parse", "HEAD")
    plan_commit = _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(REPOSITORY)))
    manifest_start = source_manifest()
    available_start = red._proc_mib("/proc/meminfo", "MemAvailable")

    # Checks on the published tables, the models and the references, before any
    # population is read.
    published, problems = red.load_published()
    tables = published["tables"]
    populations, found = population_index(tables[red.TEST_POPULATIONS])
    problems += found
    models, found = red.model_index(tables[red.CANONICAL_MODELS], tables[red.MODEL_LINKS],
                                    published["run_manifest"])
    problems += found
    models = {key: model for key, model in models.items() if key[5] == FIRST}
    rankers, model_hashes, found = red.load_rankers(models)
    problems += found
    reproduced, found = load_reproduced(args.paper_dir)
    problems += found
    replays, found = load_replays(args.paper_dir)
    problems += found
    if problems:
        _fail(problems)
    gaps = utility_gaps(replays)
    references = {red._display(path): {"sha256": sha256_path(path)}
                  for path in reference_paths(args.paper_dir).values()}
    print(f"published inputs checked: {len(populations)} pi0 populations, {len(model_hashes)} "
          f"models, {len(replays)} replay rows, {len(reproduced)} diagnosis tables", flush=True)

    SHARED.clear()
    SHARED.update(populations=populations, rankers=rankers)
    tasks = red.lineages()
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
    for relative, entry in {**published["hashes"], **references}.items():
        if sha256_path(REPOSITORY / relative) != entry["sha256"]:
            raise SystemExit(f"published input {relative} changed during the run")

    config = {
        "phase": "matched_horizon_diagnosis",
        "status": "unresolved" if exceeded else "complete",
        "plan": str(PLAN_PATH.relative_to(REPOSITORY)),
        "plan_sha256": sha256_path(PLAN_PATH),
        "plan_commit": plan_commit,
        "code_commit": head_start,
        "source_manifest": manifest_start,
        "published_inputs": {**published["hashes"], **references},
        "models": model_hashes,
        "populations": {red._name(key): {"path": red._display(populations[key]["population_path"]),
                                         "sha256": populations[key]["population_sha256"]}
                        for key in first_keys()},
        "traces": list(red.TRACES), "cells": [list(cell) for cell in red.CELLS],
        "targets": list(red.TARGETS), "seeds": list(red.SEEDS), "iterations": [FIRST],
        "horizons_seconds": list(HORIZONS_SECONDS),
        "matched_horizon_seconds": [[fraction, multiplier, h] for (fraction, multiplier), h
                                    in MATCHED_HORIZON_SECONDS.items()],
        "composition_share": COMPOSITION_SHARE,
        "uniform_concordance": UNIFORM_CONCORDANCE,
        "label_binary_arms": {f"{h:g}": list(source) for h, source in LABEL_BINARY_ARMS.items()},
        "checks": {
            "published_tables_matching_recorded_sha256": len(red.published_tables()),
            "populations_hashed": len(checks),
            "populations_matching_published_sha256": sum(row["sha256_matches"] for row in checks),
            "models_matching_published_sha256": len(model_hashes),
            "replay_rows_read": len(replays),
            "own_victim_mismatches_total": sum(row["own_victim_mismatches"] for row in checks),
            "own_victim_mismatch_share_max": max(row["own_victim_mismatch_share"]
                                                 for row in checks),
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
                   "children_peak_rss_mib": red._peak_rss_mib(resource.RUSAGE_CHILDREN),
                   "parent_peak_rss_mib": red._peak_rss_mib(resource.RUSAGE_SELF),
                   "mem_available_mib_at_start": available_start},
        "note": "Arithmetic on saved held-out decision logs and published tables. Nothing is "
                "replayed, fitted or proposed.",
    }
    if exceeded:
        publish(args.output_dir, {"checks": checks}, config, ("checks",), UNRESOLVED_HEAD)
        raise SystemExit(f"unresolved: {len(exceeded)} populations exceed the 0.1% stop rule; "
                         f"checks written to {args.output_dir}")
    derived = derive_tables(results, gaps)
    reproduction = reproduction_rows(derived, reproduced)
    failed = [row for row in reproduction if not row["equal"]]
    for row in reproduction:
        print(f"  reproduction at 600 s, {row['table']}: {row['compared_values']} values, "
              f"{row['differing_values']} differing, {row['missing_rows']} missing, "
              f"{row['extra_rows']} extra rows", flush=True)
    if failed:
        for row in failed:
            print(f"    CHECK {row['table']}: {row['first_difference'] or row}", flush=True)
        raise SystemExit(f"the h = 600 tables do not reproduce the published diagnosis in "
                         f"{len(failed)} tables; nothing published")
    config["checks"]["reproduction"] = {row["table"]: {"compared_values": row["compared_values"],
                                                       "differing_values": row["differing_values"]}
                                        for row in reproduction}
    config["readings"] = _reading_summary(derived)
    config["timing"]["wall_seconds"] = time.time() - clock
    for row in derived["concordance_counts"]:
        print(f"  reading 3 {row['target']}: prediction holds {row['holds']} of "
              f"{row['registered']} registered", flush=True)
    publish(args.output_dir, {"checks": checks, "reproduction": reproduction, **derived}, config,
            TABLE_NAMES, README_HEAD)
    print(f"written to {args.output_dir}; done in {time.time() - clock:.0f}s", flush=True)


if __name__ == "__main__":
    main()
