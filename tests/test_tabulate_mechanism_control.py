"""Tests for the read-only mechanism-control tabulation (scripts/tabulate_mechanism_control.py).

Every number of the findings document comes from this script, so its
arithmetic is checked on hand-made rows shaped like the published CSVs: the
recovery ratios, the seed pairing and sign counts of the term changes, the
literal pre-registered inversion reading in all three outcomes, and the match
of the heap comparator's present-but-unusable tokens (and its absence). One
end-to-end run on a synthetic directory checks that `main` reads and prints
without writing; one on the published directory, when present, checks that it
runs cleanly on the real tables.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import math
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
PAPER_DIR = REPOSITORY / "results/paper/mechanism_control_001"
REFERENCES = REPOSITORY / "results/paper/decision_population_replay_seeds.csv"


def _script():
    path = REPOSITORY / "scripts" / "tabulate_mechanism_control.py"
    spec = importlib.util.spec_from_file_location("_tabulate_mechanism_control_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tabulate = _script()
CONV, TOOL = "conversation_trace", "toolagent_trace"


# --- row builders shaped like the published CSVs (every value a string) ----------------


def _cell(trace, fraction, multiplier) -> dict:
    return {"trace": trace, "l1_fraction": str(fraction), "l2_multiplier": str(multiplier),
            "cell": f"l1={fraction:g},l2x{multiplier:g}"}


def decomposition_row(trace, fraction, multiplier, mechanism, h_off, h_lru, utilities,
                      dominant="signal_bound", seed_labels=("signal_bound",) * 5) -> dict:
    lru, learned, label, offline = (utilities[rung] for rung in tabulate.RUNGS)
    gaps = {"candidate_search": h_off - offline, "objective": offline - label,
            "signal": label - learned, "achieved": learned - lru}
    total = h_off - lru
    row = {**_cell(trace, fraction, multiplier), "mechanism": mechanism, "seeds": "5"}
    levels = {"H_off": h_off, "H_lru": h_lru, "U_lru": lru, "U_learned": learned,
              "U_label": label, "U_offline": offline, "T": total}
    for level, value in levels.items():
        row[f"{level}_points_mean"] = str(value)
    for gap, value in gaps.items():
        row[f"{gap}_points_mean"] = str(value)
        row[f"{gap}_share_of_mean_T"] = str(value / total) if total else "nan"
        row.update({f"{gap}_seeds_pos": "5", f"{gap}_seeds_zero": "0", f"{gap}_seeds_neg": "0"})
    row.update(dominant=dominant, seed_dominant="|".join(seed_labels),
               identity_exact_all_seeds="True")
    return row


def seed_row(trace, fraction, multiplier, mechanism, seed, **points) -> dict:
    row = {**_cell(trace, fraction, multiplier), "mechanism": mechanism, "seed": str(seed)}
    for column in tabulate.GAPS + ("H_off", "U_lru", "U_learned", "U_label", "U_offline"):
        row[f"{column}_points"] = str(points.get(column, 0.0))
    return row


def ladder_row(trace, fraction, multiplier, mechanism, ordered=True) -> dict:
    return {**_cell(trace, fraction, multiplier), "mechanism": mechanism,
            "ordered": str(bool(ordered))}


def inversion_row(trace, fraction, multiplier, mechanism, pair, inverted=5, mean=-0.1) -> dict:
    return {**_cell(trace, fraction, multiplier), "mechanism": mechanism, "pair": pair,
            "seeds_inverted": str(inverted), "seeds": "5", "mean_diff_points": str(mean)}


def reference_row(trace, fraction, multiplier, unusable, requested, kind="heap",
                  policy="offline_next_use") -> dict:
    return {"trace": trace, "l1_fraction": fraction, "l2_multiplier": multiplier, "kind": kind,
            "l2_policy": policy, "l2_present_unusable_tokens": str(unusable),
            "requested_tokens": str(requested)}


def write_csv(path: Path, rows, fieldnames=None) -> None:
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_main(argv) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = tabulate.main([str(argument) for argument in argv])
    return code, out.getvalue()


# --- T1-T3 ---------------------------------------------------------------------------------


class DecompositionTests(unittest.TestCase):
    UTILITIES = {"lru": 1.0, "learned": 3.0, "label": 6.0, "offline": 8.0}

    def test_recovery_ratios_and_differences(self):
        rows = [decomposition_row(CONV, 0.01, 1.0, "all16", 10.0, 0.5, self.UTILITIES)]
        (entry,) = tabulate.recovery_ratios(rows)
        self.assertAlmostEqual(entry["label_over_H_off"], 0.6)
        self.assertAlmostEqual(entry["offline_over_H_off"], 0.8)
        self.assertAlmostEqual(entry["learned_share_of_label"], (3.0 - 1.0) / (6.0 - 1.0))
        self.assertAlmostEqual(entry["U_lru_minus_H_lru"], 0.5)
        self.assertAlmostEqual(entry["U_learned_minus_H_lru"], 2.5)

    def test_zero_denominators_give_nan(self):
        flat = {"lru": 2.0, "learned": 2.0, "label": 2.0, "offline": 2.0}
        (entry,) = tabulate.recovery_ratios(
            [decomposition_row(CONV, 0.01, 1.0, "all16", 0.0, 0.0, flat)])
        self.assertTrue(math.isnan(entry["label_over_H_off"]))
        self.assertTrue(math.isnan(entry["offline_over_H_off"]))
        self.assertTrue(math.isnan(entry["learned_share_of_label"]))

    def test_rows_come_out_in_trace_cell_mechanism_order(self):
        rows = [decomposition_row(TOOL, 0.0025, 1.0, "all16", 10.0, 0.5, self.UTILITIES),
                decomposition_row(CONV, 0.01, 1.0, "leaf64", 10.0, 0.5, self.UTILITIES),
                decomposition_row(CONV, 0.01, 1.0, "all16", 10.0, 0.5, self.UTILITIES),
                decomposition_row(CONV, 0.0025, 4.0, "all64", 10.0, 0.5, self.UTILITIES),
                decomposition_row(CONV, 0.0025, 1.0, "leaf16", 10.0, 0.5, self.UTILITIES)]
        order = [(entry["trace"], entry["l1_fraction"], entry["l2_multiplier"],
                  entry["mechanism"]) for entry in tabulate.decomposition_table(rows)]
        self.assertEqual(order, [(CONV, 0.0025, 1.0, "leaf16"), (CONV, 0.0025, 4.0, "all64"),
                                 (CONV, 0.01, 1.0, "all16"), (CONV, 0.01, 1.0, "leaf64"),
                                 (TOOL, 0.0025, 1.0, "all16")])

    def test_decomposition_terms_shares_and_seed_labels(self):
        split = ("signal_bound",) * 4 + ("mixed",)
        rows = [decomposition_row(CONV, 0.01, 1.0, "all16", 10.0, 0.5, self.UTILITIES,
                                  dominant="mixed", seed_labels=split)]
        (entry,) = tabulate.decomposition_table(rows)
        self.assertAlmostEqual(entry["candidate_search"], 2.0)
        self.assertAlmostEqual(entry["signal_share"], 3.0 / 9.0)
        self.assertEqual(entry["signal_signs"], (5, 0, 0))
        self.assertFalse(entry["seed_unanimous"])
        self.assertEqual(entry["seed_label"], "split")
        self.assertTrue(entry["identity_exact_all_seeds"])

    def test_dominant_counts(self):
        rows = [decomposition_row(CONV, 0.01, 1.0, "all16", 10.0, 0.5, self.UTILITIES),
                decomposition_row(TOOL, 0.01, 1.0, "all16", 10.0, 0.5, self.UTILITIES,
                                  dominant="mixed"),
                decomposition_row(CONV, 0.01, 1.0, "leaf16", 10.0, 0.5, self.UTILITIES,
                                  seed_labels=("signal_bound",) * 4 + ("achieved",))]
        by_mechanism = {entry["mechanism"]: entry for entry in tabulate.dominant_counts(rows)}
        self.assertEqual(list(by_mechanism), ["all16", "leaf16"])
        all16 = by_mechanism["all16"]
        self.assertEqual((all16["rows"], all16["signal_bound"], all16["mixed"]), (2, 1, 1))
        # Both all16 rows are seed-unanimous; only one agrees with its mean label.
        self.assertEqual((all16["seed_unanimous"], all16["seed_unanimous_as_mean"]), (2, 1))
        self.assertEqual(by_mechanism["leaf16"]["seed_unanimous"], 0)


# --- T5 --------------------------------------------------------------------------------------


class TermChangeTests(unittest.TestCase):
    def setUp(self):
        base = [1.0, 2.0, 3.0, 4.0, 5.0]
        leaf16 = [2.0, 2.0, 2.5, 4.0, 6.0]     # differences 1, 0, -0.5, 0, 1
        self.rows = [seed_row(CONV, 0.01, 1.0, "all16", seed, signal=value)
                     for seed, value in enumerate(base)]
        # Written in reverse seed order: the pairing is by seed, not by position.
        self.rows += [seed_row(CONV, 0.01, 1.0, "leaf16", seed, signal=leaf16[seed])
                      for seed in reversed(range(5))]
        # all64 has three seeds only; leaf64 has none.
        self.rows += [seed_row(CONV, 0.01, 1.0, "all64", seed, signal=value)
                      for seed, value in zip((2, 0, 1), (3.0, 1.5, 2.0))]

    def _entry(self, term, mechanism):
        (entry,) = [row for row in tabulate.term_changes(self.rows)
                    if row["term"] == term and row["mechanism"] == mechanism]
        return entry

    def test_seed_paired_difference_and_signs(self):
        entry = self._entry("signal", "leaf16")
        self.assertEqual(entry["seeds"], 5)
        self.assertAlmostEqual(entry["diff_mean"], (1.0 + 0.0 - 0.5 + 0.0 + 1.0) / 5)
        self.assertEqual(entry["signs"], (2, 2, 1))
        self.assertAlmostEqual(entry["base"], 3.0)
        self.assertEqual(entry["base_seeds"], 5)

    def test_pairing_uses_only_the_shared_seeds(self):
        entry = self._entry("signal", "all64")
        self.assertEqual(entry["seeds"], 3)
        self.assertAlmostEqual(entry["diff_mean"], (0.5 + 0.0 + 0.0) / 3)
        self.assertEqual(entry["signs"], (1, 2, 0))
        self.assertAlmostEqual(entry["base"], 3.0)      # the base keeps its five seeds
        missing = self._entry("signal", "leaf64")
        self.assertEqual((missing["seeds"], missing["signs"]), (0, (0, 0, 0)))
        self.assertTrue(math.isnan(missing["diff_mean"]))

    def test_every_term_and_mechanism_is_reported(self):
        rows = tabulate.term_changes(self.rows)
        self.assertEqual(len(rows), len(tabulate.GAPS) * 3)
        self.assertEqual(self._entry("objective", "leaf16")["signs"], (0, 5, 0))


# --- T6 --------------------------------------------------------------------------------------


A, B = (0.0025, 1.0), (0.01, 4.0)


def _ladder_fixture():
    inversions = []
    # Cell A, label_le_offline: under all four mechanisms on both traces.
    for trace in (CONV, TOOL):
        for mechanism in tabulate.MECHANISMS:
            inversions.append(inversion_row(trace, *A, mechanism, "label_le_offline"))
    # Cell A, lru_le_learned: all16 on both traces; leaf64 removes it on the
    # conversation trace, leaf16 (and all64) on the tool-agent trace.
    for mechanism in ("all16", "leaf16", "all64"):
        inversions.append(inversion_row(CONV, *A, mechanism, "lru_le_learned", 3, 0.25))
    for mechanism in ("all16", "leaf64"):
        inversions.append(inversion_row(TOOL, *A, mechanism, "lru_le_learned", 2, -0.5))
    # Cell B, label_le_offline: all16 on the conversation trace only.
    inversions.append(inversion_row(CONV, *B, "all16", "label_le_offline", 1, 0.02))
    # Cell B, lru_le_learned: all16 and both leaf mechanisms on both traces, not all64.
    for trace in (CONV, TOOL):
        for mechanism in ("all16", "leaf16", "leaf64"):
            inversions.append(inversion_row(trace, *B, mechanism, "lru_le_learned"))
    # Cell B, learned_le_label: all four mechanisms, but on one trace only.
    for mechanism in tabulate.MECHANISMS:
        inversions.append(inversion_row(CONV, *B, mechanism, "learned_le_label"))
    inverted = {(row["trace"], float(row["l1_fraction"]), float(row["l2_multiplier"]),
                 row["mechanism"]) for row in inversions}
    ladder = [ladder_row(trace, *cell, mechanism,
                         ordered=(trace, *cell, mechanism) not in inverted)
              for trace in (TOOL, CONV) for cell in (B, A) for mechanism in tabulate.MECHANISMS]
    return ladder, inversions


class LadderTests(unittest.TestCase):
    def setUp(self):
        self.ladder, self.inversions = _ladder_fixture()
        self.presence = tabulate.inversion_presence(self.ladder, self.inversions)
        self.classified = {(row["l1_fraction"], row["l2_multiplier"], row["pair"]): row
                           for row in tabulate.classify_inversions(self.presence)}

    def test_presence_lists_every_cell_pair_and_trace(self):
        self.assertEqual(len(self.presence), 2 * len(tabulate.PAIRS) * 2)
        first = self.presence[0]
        self.assertEqual((first["l1_fraction"], first["pair"], first["trace"]),
                         (0.0025, "lru_le_learned", CONV))
        self.assertEqual(first["present"], ("all16", "leaf16", "all64"))
        self.assertEqual(first["all16"], {"seeds_inverted": 3, "seeds": 5,
                                          "mean_diff_points": 0.25})
        self.assertIsNone(first["leaf64"])

    def test_score_property(self):
        row = self.classified[A + ("label_le_offline",)]
        self.assertEqual(row["classification"], "score_property")
        self.assertEqual(row["traces_inverted"], 2)

    def test_mechanism_borne_names_where_it_is_absent(self):
        row = self.classified[A + ("lru_le_learned",)]
        self.assertEqual(row["classification"], "mechanism_borne")
        self.assertEqual(row["absent"], {CONV: ("leaf64",), TOOL: ("leaf16", "all64")})
        self.assertEqual(row["present"][TOOL], ("all16", "leaf64"))

    def test_unclassified_outcomes(self):
        # No inversion anywhere.
        none = self.classified[A + ("learned_le_label",)]
        self.assertEqual((none["classification"], none["traces_inverted"]), ("unclassified", 0))
        # Present under all16 and absent under both leaf mechanisms, but on one trace only.
        one_trace = self.classified[B + ("label_le_offline",)]
        self.assertEqual((one_trace["classification"], one_trace["traces_inverted"]),
                         ("unclassified", 1))
        # Under all four on one trace only: not a score property either.
        self.assertEqual(self.classified[B + ("learned_le_label",)]["classification"],
                         "unclassified")
        # On both traces, survives both leaf mechanisms but not all64: neither rule.
        self.assertEqual(self.classified[B + ("lru_le_learned",)]["classification"],
                         "unclassified")

    def test_order_and_counts(self):
        by_mechanism = {row["mechanism"]: row for row in tabulate.ladder_order(self.ladder)}
        self.assertEqual(list(by_mechanism), list(tabulate.MECHANISMS))
        # all64: every cell inverted except cell B on the tool-agent trace.
        self.assertEqual(by_mechanism["all64"]["ordered"], 1)
        self.assertEqual(by_mechanism["all64"]["ordered_cells"], [f"{TOOL}/l1=0.01,l2x4"])
        self.assertEqual(by_mechanism["all16"]["rows"], 4)
        counts = {row["pair"]: row for row in tabulate.inversion_counts(self.inversions)}
        self.assertEqual(list(counts), list(tabulate.PAIRS))
        self.assertEqual(counts["lru_le_learned"]["all16"], 4)
        self.assertEqual(counts["lru_le_learned"]["all64"], 1)
        self.assertEqual(counts["label_le_offline"]["total"], 9)


# --- T9 --------------------------------------------------------------------------------------


class ControlledVersusHeapTests(unittest.TestCase):
    def setUp(self):
        self.rows = [seed_row(CONV, 0.01, 1.0, "leaf64", seed, H_off=10.0, U_offline=offline,
                              U_label=9.0)
                     for seed, offline in zip((2, 0, 1), (9.5, 10.5, 10.0))]
        self.rows += [seed_row(CONV, 0.01, 1.0, "all16", seed, H_off=10.0, U_offline=1.0,
                               U_label=1.0) for seed in range(3)]
        self.rows += [seed_row(TOOL, 0.02, 4.0, "leaf64", 0, H_off=5.0, U_offline=5.0,
                               U_label=4.0)]
        self.references = [
            # Matched although written differently: the fractions compare as floats.
            reference_row(CONV, "0.010", "1", 50, 1000),
            reference_row(CONV, "0.01", "1.0", 100, 1000),
            # Not the heap offline comparator, or not this cell.
            reference_row(CONV, "0.01", "1.0", 999, 1000, policy="lru"),
            reference_row(CONV, "0.01", "1.0", 999, 1000, kind="sampled"),
            reference_row(CONV, "0.01", "4.0", 999, 1000),
            reference_row(TOOL, "0.01", "1.0", 999, 1000),
        ]

    def test_paired_differences_and_matched_heap_rows(self):
        conversation, tool = tabulate.controlled_versus_heap(self.rows, self.references)
        self.assertEqual((conversation["trace"], conversation["seeds"]), (CONV, 3))
        self.assertAlmostEqual(conversation["offline_minus_H_off"], 0.0)
        self.assertEqual(conversation["offline_minus_H_off_signs"], (1, 1, 1))
        self.assertAlmostEqual(conversation["label_minus_H_off"], -1.0)
        self.assertEqual(conversation["label_minus_H_off_signs"], (0, 0, 3))
        self.assertTrue(conversation["heap_available"])
        self.assertEqual(conversation["heap_rows"], 2)
        self.assertAlmostEqual(conversation["heap_present_unusable_mean"], 7.5)
        self.assertAlmostEqual(conversation["heap_present_unusable_min"], 5.0)
        self.assertAlmostEqual(conversation["heap_present_unusable_max"], 10.0)
        # A cell without a published heap row is reported as unmatched, not dropped.
        self.assertEqual((tool["trace"], tool["heap_rows"]), (TOOL, 0))
        self.assertTrue(math.isnan(tool["heap_present_unusable_mean"]))

    def test_unavailable_references(self):
        for entry in tabulate.controlled_versus_heap(self.rows, None):
            self.assertFalse(entry["heap_available"])
            self.assertEqual(entry["heap_rows"], 0)
            self.assertTrue(math.isnan(entry["heap_present_unusable_mean"]))
        rows, note = tabulate.load_references(None)
        self.assertIsNone(rows)
        self.assertIn("unavailable", note)
        with tempfile.TemporaryDirectory() as directory:
            rows, note = tabulate.load_references(Path(directory) / "missing.csv")
        self.assertIsNone(rows)
        self.assertIn("does not exist", note)


# --- main ------------------------------------------------------------------------------------


def write_synthetic_paper_dir(directory: Path) -> None:
    utilities = {"lru": 1.0, "learned": 3.0, "label": 6.0, "offline": 8.0}
    cells = [(trace, *cell) for trace in (CONV, TOOL) for cell in (A, B)]
    decomposition, seeds, effects, capacity, replay = [], [], [], [], []
    for trace, fraction, multiplier in cells:
        for mechanism in tabulate.MECHANISMS:
            decomposition.append(decomposition_row(trace, fraction, multiplier, mechanism,
                                                   10.0, 0.5, utilities))
            seeds += [seed_row(trace, fraction, multiplier, mechanism, seed, H_off=10.0,
                               U_offline=8.0, U_label=6.0, signal=3.0) for seed in range(5)]
            capacity.append({**_cell(trace, fraction, multiplier), "mechanism": mechanism,
                             "learned_minus_lru_mean_points": "2.0",
                             "learned_minus_lru_reading": "consistent_gain",
                             "label_minus_lru_mean_points": "5.0",
                             "label_minus_lru_reading": "consistent_gain"})
            for rung in tabulate.RUNGS:
                replay.append({**_cell(trace, fraction, multiplier), "mechanism": mechanism,
                               "rung": rung, "seeds": "5",
                               **{column: "1.0" for column in tabulate.REPLAY_COUNTERS}})
        for contrast in tabulate.CONTRASTS:
            for rung in tabulate.RUNGS:
                effects.append({**_cell(trace, fraction, multiplier), "rung": rung,
                                "contrast": contrast, "diff_points_mean": "0.5",
                                "reading": "mixed" if rung == "lru" else "consistent_gain"})
    ladder, inversions = _ladder_fixture()
    write_csv(directory / "decomposition.csv", decomposition)
    write_csv(directory / "decomposition_seeds.csv", seeds)
    write_csv(directory / "mechanism_effects.csv", effects)
    write_csv(directory / "ladder.csv", ladder)
    write_csv(directory / "ladder_inversions.csv", inversions)
    write_csv(directory / "capacity_pattern.csv", capacity)
    write_csv(directory / "replay.csv", replay)


def _snapshot(directory: Path) -> dict:
    return {path.name: (path.stat().st_mtime_ns, path.read_bytes())
            for path in sorted(directory.iterdir())}


class MainTests(unittest.TestCase):
    def test_synthetic_directory_without_references(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            write_synthetic_paper_dir(directory)
            before = _snapshot(directory)
            code, output = run_main([directory, "--references", directory / "missing.csv"])
            self.assertEqual(_snapshot(directory), before)      # read only
        self.assertEqual(code, 0)
        for section in ("T1a", "T1b", "T2", "T3", "T4", "T5", "T6a", "T6b", "T6c", "T6d",
                        "T7", "T8", "T9"):
            self.assertIn(f"### {section} ", output)
        t9 = output[output.index("### T9"):]
        self.assertIn("does not exist; heap column unavailable", t9)
        self.assertIn("| unavailable", t9)
        self.assertIn("~0.500", output)              # mixed readings are marked in T4

    @unittest.skipUnless(PAPER_DIR.is_dir(), "published mechanism-control tables not present")
    def test_published_directory(self):
        argv = [PAPER_DIR] + (["--references", REFERENCES] if REFERENCES.is_file() else [])
        code, output = run_main(argv)
        self.assertEqual(code, 0)
        rows = tabulate.dominant_counts(tabulate.read_csv(PAPER_DIR / "decomposition.csv"))
        self.assertEqual([row["mechanism"] for row in rows], list(tabulate.MECHANISMS))
        self.assertTrue(all(row["rows"] == 12 for row in rows))
        t2 = output[output.index("### T2 "):output.index("### T3 ")]
        printed = {line.split("|")[1].strip(): line.split("|")[2].strip()
                   for line in t2.splitlines()
                   if line.startswith("| ") and line.split("|")[1].strip() in tabulate.MECHANISMS}
        self.assertEqual(printed, {mechanism: "12" for mechanism in tabulate.MECHANISMS})


if __name__ == "__main__":
    unittest.main()
