"""Exploratory postprocessing of frozen on-policy utility and decision logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from persistent_kv_admission.decisionpop import state_indices, target_column
from persistent_kv_admission.onpolicy import deserialize_ranker, sequential_ranker_score
from persistent_kv_admission.trace import load_mooncake_trace


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/paper/onpolicy_learning"
OUTPUT = ROOT / "results/paper/retention_diagnostics"
KEY = ("trace", "cell", "target", "seed")
ITERATIONS = (0, 1, 2, 3)
DECISION_ITERATIONS = (0, 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise ValueError(f"inconsistent columns in {path}")
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def identity(row: dict) -> tuple[str, str, str, int]:
    return str(row["trace"]), str(row["cell"]), str(row["target"]), int(row["seed"])


def verify_published_csv(name: str, config: dict) -> Path:
    path = SOURCE / name
    expected = config["output_artifacts"][name]["sha256"]
    if sha256_file(path) != expected:
        raise ValueError(f"published CSV hash mismatch: {path}")
    return path


def utility_rows(source: list[dict], ranking: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(source) != 480:
        raise ValueError("expected 480 seed utility rows")
    indexed = {(identity(row), int(row["iteration"])): row for row in source}
    if len(indexed) != 480:
        raise ValueError("duplicate utility identity")
    terminal = {identity(row): float(row["terminal_common_population_ranking_delta"])
                for row in ranking}
    if len(terminal) != 120:
        raise ValueError("expected 120 terminal ranking rows")
    output = []
    for key in sorted({key for key, _ in indexed}):
        base = indexed.get((key, 0))
        if base is None or key not in terminal:
            raise ValueError(f"missing pi0/ranking row {key}")
        for iteration in ITERATIONS:
            row = indexed.get((key, iteration))
            if row is None:
                raise ValueError(f"missing iteration {iteration} for {key}")
            for window, prefix in (("full", ""), ("label", "label_window_")):
                requests = int(row[f"{prefix}requested_tokens"])
                base_requests = int(base[f"{prefix}requested_tokens"])
                if requests != base_requests or requests <= 0:
                    raise ValueError(f"request denominator mismatch {key} {window}")
                avoided = int(row[f"{prefix}avoided_prefill_tokens"])
                base_avoided = int(base[f"{prefix}avoided_prefill_tokens"])
                delta = avoided - base_avoided
                points = 100.0 * delta / requests
                if window == "full" and not np.isclose(
                    points, float(row["input_token_points_vs_pi0"]), atol=1e-12
                ):
                    raise ValueError(f"published full utility mismatch {key}")
                rank_delta = terminal[key]
                output.append({
                    "trace": key[0], "cell": key[1], "target": key[2], "seed": key[3],
                    "iteration": iteration, "policy": f"pi{iteration}", "window": window,
                    "requested_tokens": requests, "avoided_tokens": avoided,
                    "pi0_avoided_tokens": base_avoided, "delta_tokens_vs_pi0": delta,
                    "delta_input_token_points_vs_pi0": points,
                    "terminal_common_population_ranking_delta": rank_delta if iteration == 3 else "",
                    "ranking_utility_sign_agreement": (
                        "not_applicable" if iteration != 3 else
                        "same" if delta * rank_delta > 0 else
                        "opposite" if delta * rank_delta < 0 else "zero"
                    ),
                })
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in output:
        group_key = (row["trace"], row["cell"], row["target"], row["iteration"], row["window"])
        grouped[group_key].append(row)
    aggregate = []
    for group_key, rows in sorted(grouped.items()):
        if sorted(row["seed"] for row in rows) != list(range(5)):
            raise ValueError(f"incomplete seed group {group_key}")
        values = np.array([row["delta_input_token_points_vs_pi0"] for row in rows])
        raw = np.array([row["delta_tokens_vs_pi0"] for row in rows])
        aggregate.append({
            "trace": group_key[0], "cell": group_key[1], "target": group_key[2],
            "iteration": group_key[3], "window": group_key[4],
            "mean_delta_input_token_points": float(values.mean()),
            "min_seed_points": float(values.min()), "max_seed_points": float(values.max()),
            "mean_delta_tokens": float(raw.mean()),
            "positive_seeds": int((values > 0).sum()),
            "negative_seeds": int((values < 0).sum()),
            "zero_seeds": int((values == 0).sum()),
            "same_ranking_sign_seeds": sum(row["ranking_utility_sign_agreement"] == "same" for row in rows),
            "opposite_ranking_sign_seeds": sum(row["ranking_utility_sign_agreement"] == "opposite" for row in rows),
        })
    return output, aggregate


def group_boundaries(group: np.ndarray, expected: int) -> tuple[np.ndarray, np.ndarray]:
    if group.ndim != 1 or not len(group):
        raise ValueError("empty or malformed groups")
    starts = np.r_[0, np.flatnonzero(np.diff(group)) + 1]
    if len(starts) != expected or not np.array_equal(group[starts], np.arange(expected)):
        raise ValueError("groups are not complete, contiguous, dense decisions")
    widths = np.diff(np.r_[starts, len(group)])
    if (widths <= 0).any():
        raise ValueError("empty group")
    return starts, widths


def first_tuple_argmin(primary: np.ndarray, secondary: np.ndarray,
                       starts: np.ndarray, widths: np.ndarray) -> np.ndarray:
    if not (len(primary) == len(secondary) == int(widths.sum())):
        raise ValueError("score/group length mismatch")
    minimum = np.minimum.reduceat(primary, starts)
    first_mask = primary == np.repeat(minimum, widths)
    tie_minimum = np.minimum.reduceat(np.where(first_mask, secondary, np.inf), starts)
    eligible = first_mask & (secondary == np.repeat(tie_minimum, widths))
    chosen = np.minimum.reduceat(np.where(eligible, np.arange(len(primary)), len(primary)), starts)
    if (chosen >= len(primary)).any():
        raise ValueError("missing tuple minimum")
    return chosen


def candidate_measure(values: np.ndarray, chosen: np.ndarray,
                      starts: np.ndarray, widths: np.ndarray) -> dict[str, np.ndarray]:
    minimum = np.minimum.reduceat(values, starts)
    maximum = np.maximum.reduceat(values, starts)
    actual = values[chosen]
    tie_count = np.add.reduceat(values == np.repeat(minimum, widths), starts)
    repeated_actual = np.repeat(actual, widths)
    lower = np.add.reduceat(values < repeated_actual, starts)
    equal = np.add.reduceat(values == repeated_actual, starts)
    rank = np.divide(lower + (equal - 1) / 2, widths - 1,
                     out=np.full(len(widths), np.nan), where=widths > 1)
    gap = actual - minimum
    if (gap < -1e-10).any():
        raise ValueError("candidate-relative gap was negative")
    return {
        "minimum": minimum, "maximum": maximum, "actual": actual,
        "gap": gap, "correct": actual == minimum,
        "random_correct": tie_count / widths,
        "discriminative": minimum < maximum, "rank_fraction": rank,
    }


def concentration(gaps: np.ndarray, fraction: float) -> float:
    total = float(gaps.sum())
    if total == 0 or not len(gaps):
        return math.nan
    count = math.ceil(fraction * len(gaps))
    return float(np.partition(gaps, len(gaps) - count)[-count:].sum() / total)


def summarize_decisions(native: dict, counts: dict, proxy: dict,
                        category: np.ndarray, selected: str) -> dict:
    mask = category == selected if selected != "all" else np.ones(len(category), dtype=bool)
    n = int(mask.sum())
    if not n:
        raise ValueError(f"empty decision category {selected}")
    discriminatory = native["discriminative"][mask]
    d = int(discriminatory.sum())
    p = proxy["gap"][mask]
    rank = native["rank_fraction"][mask]
    return {
        "decision_type": selected, "decisions": n,
        "discriminative_decisions": d,
        "constant_label_decisions": n - d,
        "native_min_correct_rate_all": float(native["correct"][mask].mean()),
        "native_uniform_expected_min_correct_rate_all": float(native["random_correct"][mask].mean()),
        "native_min_correct_rate_discriminative": (
            float(native["correct"][mask][discriminatory].mean()) if d else math.nan
        ),
        "native_uniform_expected_min_correct_rate_discriminative": (
            float(native["random_correct"][mask][discriminatory].mean()) if d else math.nan
        ),
        "native_rank_fraction_mean": float(np.nanmean(rank)) if np.isfinite(rank).any() else math.nan,
        "native_gap_positive_rate": float((native["gap"][mask] > 0).mean()),
        "native_gap_mean": float(native["gap"][mask].mean()),
        "count_min_positive_rate": float((counts["minimum"][mask] > 0).mean()),
        "count_gap_positive_rate": float((counts["gap"][mask] > 0).mean()),
        "count_gap_mean": float(counts["gap"][mask].mean()),
        "proxy_min_positive_rate": float((proxy["minimum"][mask] > 0).mean()),
        "proxy_gap_positive_rate": float((p > 0).mean()),
        "proxy_gap_mean_tokens": float(p.mean()),
        "proxy_gap_p50_tokens": float(np.median(p)),
        "proxy_gap_p90_tokens": float(np.quantile(p, 0.9)),
        "proxy_gap_p99_tokens": float(np.quantile(p, 0.99)),
        "proxy_gap_sum_sampled_tokens": float(p.sum()),
        "proxy_gap_top1pct_share_all_decisions": concentration(p, 0.01),
        "proxy_gap_top10pct_share_all_decisions": concentration(p, 0.10),
    }


def verify_manifest(rows: list[dict]) -> list[dict]:
    selected = [row for row in rows if int(row["iteration"]) in DECISION_ITERATIONS]
    if len(rows) != 480 or len(selected) != 240:
        raise ValueError("unexpected test population manifest size")
    expected = {(identity(row), iteration) for row in rows for iteration in DECISION_ITERATIONS}
    observed = {(identity(row), int(row["iteration"])) for row in selected}
    if len(observed) != 240 or observed != expected:
        raise ValueError("incomplete or duplicate selected test populations")
    return sorted(selected, key=lambda r: (*identity(r), int(r["iteration"])))


def load_model_index(rows: list[dict]) -> dict[tuple, dict]:
    selected = {(identity(row), int(row["iteration"])): row for row in rows
                if int(row["iteration"]) in DECISION_ITERATIONS}
    if len(selected) != 240:
        raise ValueError("missing selected model links")
    return selected


def analyze_population(row: dict, block_tokens: np.ndarray,
                       models: dict[tuple, dict]) -> tuple[list[dict], list[dict], dict]:
    key = identity(row)
    iteration = int(row["iteration"])
    path = Path(row["population_path"])
    if path.resolve().parent.parent != (ROOT / "results/onpolicy_full_feb30eb_001/populations/test").resolve():
        raise ValueError(f"population escaped frozen test run: {path}")
    digest = sha256_file(path)
    if digest != row["population_sha256"]:
        raise ValueError(f"population hash mismatch: {path}")
    needed = ["group", "victim", "arriving", "state_index", "next_use_delta_ms",
              "count_within_h", "arm_score", "arm_tiebreak", "horizon_seconds", "window", "seed"]
    if iteration == 3:
        needed.append("features")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in needed}
    if str(arrays["window"]) != "test" or int(arrays["seed"]) != key[3]:
        raise ValueError("population window/seed mismatch")
    horizon = float(arrays["horizon_seconds"])
    if horizon != 600:
        raise ValueError("unexpected label horizon")
    group = arrays["group"]
    expected_decisions = int(row["decisions_kept"])
    starts, widths = group_boundaries(group, expected_decisions)
    if expected_decisions != 40000:
        raise ValueError("fixed sample decision cap not met")
    victim = arrays["victim"]
    arrival = arrays["arriving"]
    if not (np.isin(victim, (0, 1)).all() and np.isin(arrival, (0, 1)).all()):
        raise ValueError("nonbinary victim or arrival flags")
    if not np.all(np.add.reduceat(victim, starts) == 1):
        raise ValueError("expected one actual victim per group")
    if np.any(np.add.reduceat(arrival, starts) > 1):
        raise ValueError("multiple arriving candidates")
    actual = np.flatnonzero(victim)
    if len(actual) != expected_decisions:
        raise ValueError("wrong number of actual victims")
    recorded = first_tuple_argmin(arrays["arm_score"], arrays["arm_tiebreak"], starts, widths)
    if not np.array_equal(actual, recorded):
        raise ValueError("recorded tuple argmin disagrees with actual victim")
    arrival_row = np.minimum.reduceat(np.where(arrival, np.arange(len(arrival)), len(arrival)), starts)
    category = np.where(actual == arrival_row, "arrival_rejection", "resident_eviction")
    state = arrays["state_index"]
    if state.min() < 0 or state.max() >= len(block_tokens):
        raise ValueError("state index outside trace map")
    counts = arrays["count_within_h"]
    if (counts < 0).any() or not np.all(counts == np.floor(counts)):
        raise ValueError("invalid future reuse counts")
    values = target_column(arrays["next_use_delta_ms"], counts, key[2], horizon)
    proxy_values = counts * block_tokens[state]
    native = candidate_measure(values, actual, starts, widths)
    count_measure = candidate_measure(counts, actual, starts, widths)
    proxy = candidate_measure(proxy_values, actual, starts, widths)
    common = {"trace": key[0], "cell": key[1], "target": key[2], "seed": key[3],
              "population_iteration": iteration, "policy": f"pi{iteration}"}
    summaries = [{**common, **summarize_decisions(native, count_measure, proxy, category, part)}
                 for part in ("all", "arrival_rejection", "resident_eviction")
                 if part == "all" or np.any(category == part)]
    comparisons: list[dict] = []
    if iteration == 3:
        for scoring_iteration in DECISION_ITERATIONS:
            model_row = models[(key, scoring_iteration)]
            ranker = deserialize_ranker(model_row["model_path"], model_row["model_sha256"])
            primary = sequential_ranker_score(ranker, arrays["features"])
            proposed = first_tuple_argmin(primary, arrays["arm_tiebreak"], starts, widths)
            if scoring_iteration == 3 and not np.array_equal(proposed, actual):
                raise ValueError("pi3 rescoring does not reproduce actual victim")
            native_proposed = candidate_measure(values, proposed, starts, widths)
            count_proposed = candidate_measure(counts, proposed, starts, widths)
            proxy_proposed = candidate_measure(proxy_values, proposed, starts, widths)
            proposed_category = np.where(proposed == arrival_row, "arrival_rejection", "resident_eviction")
            for part in ("all", "arrival_rejection", "resident_eviction"):
                if part != "all" and not np.any(proposed_category == part):
                    continue
                summary = summarize_decisions(native_proposed, count_proposed, proxy_proposed, proposed_category, part)
                mask = proposed_category == part if part != "all" else np.ones(len(category), dtype=bool)
                comparisons.append({**common, "scoring_iteration": scoring_iteration,
                                    "scoring_policy": f"pi{scoring_iteration}",
                                    "proposed_actual_agreement_rate": float((proposed[mask] == actual[mask]).mean()),
                                    **summary})
    audit = {**common, "population_path": str(path), "population_sha256": digest,
             "decisions_verified": expected_decisions, "rows_verified": len(group),
             "recorded_argmin_mismatches": 0,
             "rejections": int((category == "arrival_rejection").sum()),
             "evictions": int((category == "resident_eviction").sum())}
    return summaries, comparisons, audit


def figures(aggregate: list[dict], decisions: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = ["l1=0.0025,l2x1", "l1=0.0025,l2x4", "l1=0.01,l2x1",
             "l1=0.01,l2x4", "l1=0.02,l2x1", "l1=0.02,l2x4"]
    traces = ["conversation_trace", "toolagent_trace"]
    targets = ["next_use", "binary"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for ti, target in enumerate(targets):
        for wi, window in enumerate(("full", "label")):
            axis = axes[ti, wi]
            for j, trace in enumerate(traces):
                vals = [next(float(r["mean_delta_input_token_points"]) for r in aggregate
                             if r["trace"] == trace and r["target"] == target and
                             r["cell"] == cell and r["iteration"] == 3 and r["window"] == window)
                        for cell in cells]
                axis.plot(np.arange(6), vals, marker="o", label=trace.replace("_trace", ""))
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set_xticks(range(6), [".25x1", ".25x4", "1x1", "1x4", "2x1", "2x4"])
            axis.set_title(f"{target}: {window} window")
            axis.set_ylabel("pi3 - pi0 (input-token points)")
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "window_sensitivity.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for axis, trace in zip(axes, traces):
        for iteration, offset, label in ((0, -0.18, "pi0"), (3, 0.18, "pi3")):
            vals = []
            for cell in cells:
                chosen = [float(r["proxy_gap_positive_rate"]) for r in decisions
                          if r["trace"] == trace and r["target"] == "next_use" and
                          r["cell"] == cell and r["population_iteration"] == iteration and
                          r["decision_type"] == "all"]
                vals.append(float(np.mean(chosen)))
            axis.bar(np.arange(6) + offset, vals, width=0.34, label=label)
        axis.set_xticks(range(6), [".25x1", ".25x4", "1x1", "1x4", "2x1", "2x4"])
        axis.set_title(trace.replace("_trace", ""))
        axis.set_ylabel("Positive candidate-relative exposure gap rate")
        axis.legend()
    fig.tight_layout()
    fig.savefig(out / "victim_proxy_positive_rate.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_path = SOURCE / "onpolicy_learning_config.json"
    config = json.loads(config_path.read_text())
    inputs = {}
    for name in ("onpolicy_seed_utility.csv", "onpolicy_terminal_ranking.csv",
                 "onpolicy_test_populations.csv", "onpolicy_model_links.csv"):
        path = verify_published_csv(name, config)
        inputs[name] = sha256_file(path)
    seed_utility = read_csv(SOURCE / "onpolicy_seed_utility.csv")
    terminal_ranking = read_csv(SOURCE / "onpolicy_terminal_ranking.csv")
    utility, aggregate = utility_rows(seed_utility, terminal_ranking)
    write_csv(out / "window_utility_seeds.csv", utility)
    write_csv(out / "window_utility_aggregate.csv", aggregate)
    population_rows = verify_manifest(read_csv(SOURCE / "onpolicy_test_populations.csv"))
    model_index = load_model_index(read_csv(SOURCE / "onpolicy_model_links.csv"))
    tokens = {}
    for name, trace_config in sorted(config["trace_files"].items()):
        path = Path(trace_config["path"])
        if sha256_file(path) != trace_config["sha256"]:
            raise ValueError(f"trace hash mismatch: {path}")
        trace = load_mooncake_trace(path)
        mapping = state_indices(trace)
        if list(mapping.values()) != list(range(len(mapping))):
            raise ValueError("state index order mismatch")
        tokens[name] = np.array([trace.states[state].block_tokens for state in mapping], dtype=np.int32)
        del trace, mapping
    decisions, comparisons, audits = [], [], []
    for number, row in enumerate(population_rows, 1):
        summaries, cross, audit = analyze_population(row, tokens[row["trace"]], model_index)
        decisions.extend(summaries)
        comparisons.extend(cross)
        audits.append(audit)
        if number % 20 == 0:
            print(f"verified {number}/{len(population_rows)} population files", flush=True)
    write_csv(out / "victim_quality_seeds.csv", decisions)
    write_csv(out / "terminal_fixed_population_choices.csv", comparisons)
    write_csv(out / "population_integrity.csv", audits)
    figures(aggregate, decisions, out)
    output_names = ["window_utility_seeds.csv", "window_utility_aggregate.csv",
                    "victim_quality_seeds.csv", "terminal_fixed_population_choices.csv",
                    "population_integrity.csv", "window_sensitivity.png",
                    "victim_proxy_positive_rate.png"]
    audit_config = {
        "analysis": "posthoc_retention_decisions", "source_run_code_commit": config["code_commit"],
        "analysis_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_config_sha256": sha256_file(config_path), "source_csv_sha256": inputs,
        "script_sha256": sha256_file(Path(__file__)),
        "source_module_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in (
                ROOT / "src/persistent_kv_admission/onpolicy.py",
                ROOT / "src/persistent_kv_admission/decisionpop.py",
                ROOT / "src/persistent_kv_admission/trace.py",
            )
        },
        "trace_sha256": {name: entry["sha256"] for name, entry in config["trace_files"].items()},
        "population_iterations": list(DECISION_ITERATIONS), "population_files_verified": len(audits),
        "sampled_decisions_verified": sum(row["decisions_verified"] for row in audits),
        "workers": 1, "new_fit": False, "new_replay": False,
        "outputs_sha256": {name: sha256_file(out / name) for name in output_names},
    }
    (out / "analysis_config.json").write_text(json.dumps(audit_config, indent=2, sort_keys=True) + "\n")
    print(f"completed {len(audits)} verified population files; outputs: {out}")


if __name__ == "__main__":
    main()
