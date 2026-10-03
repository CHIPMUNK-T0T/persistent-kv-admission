"""Tests for the read-only closure-cost tabulation (scripts/tabulate_bailian_closure_cost.py).

Everything runs on hand-made rows shaped like the published CSVs; no published
result is read. What is checked: the per trace x cell quantities on a
hand-computed two-seed example under both mechanisms (Bailian on W, Mooncake
on the full window), the evaluable boundary per mechanism, the Pearson helper
(including the degenerate inputs that return nan), the summary populations,
the residual against the accounting identity P = U + pu (hand-computed, with a
leaf16 pu gap, and failing loudly on doctored rows), the per-trace table with
a degenerate trace, the consistency checks against the published tables
failing loudly, the Mooncake conflict detection across directories, and the
errors for a missing arm, seed, column or file. End-to-end runs on a synthetic
directory check that `main` prints without writing and exits non-zero on a
mismatch or a broken identity.
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

from persistent_kv_admission import bailiancheck as bc

REPOSITORY = Path(__file__).resolve().parents[1]


def _script():
    path = REPOSITORY / "scripts" / "tabulate_bailian_closure_cost.py"
    spec = importlib.util.spec_from_file_location("_tabulate_bailian_closure_cost_under_test",
                                                  path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tabulate = _script()
TRACE = "bailian_toc_trace"
CELL = (0.0025, 1.0)                      # h* = 60 s
BIT, RAND = "label_binary_60", "label_binary_random_60"
SEEDS = (0, 1)


def bailian_row(mechanism, arm, seed, pu_tokens, u_points, trace=TRACE, cell=CELL,
                requested=1000, l2_avoided=None):
    """One replay_seeds.csv row as the CSV reader returns it (strings). The
    L2-avoided tokens are U's tokens unless `l2_avoided` doctors them."""
    tokens = u_points * requested / 100
    assert abs(tokens - round(tokens)) < 1e-6, "U must be a whole number of tokens"
    return {"trace": trace, "l1_fraction": str(cell[0]), "l2_multiplier": str(cell[1]),
            "mechanism": mechanism, "arm": arm, "seed": str(seed),
            "W_l2_present_unusable_tokens": str(pu_tokens),
            "W_l2_avoided_tokens": str(round(tokens) if l2_avoided is None else l2_avoided),
            "W_requested_tokens": str(requested), "U_W_points": repr(float(u_points))}


def mooncake_row(mechanism, arm, seed, pu_tokens, extra_tokens, trace="conversation_trace",
                 cell=CELL, requested=2000, variant="main", l2_avoided=None):
    return {"trace": trace, "l1_fraction": str(cell[0]), "l2_multiplier": str(cell[1]),
            "mechanism": mechanism, "arm": arm, "seed": str(seed), "variant": variant,
            "l2_present_unusable_tokens": str(pu_tokens),
            "l2_avoided_tokens": str(extra_tokens if l2_avoided is None else l2_avoided),
            "requested_tokens": str(requested), "extra_avoided_tokens": str(extra_tokens)}


# The hand-computed Bailian example (requested 1,000 tokens: pu points = tokens / 10).
#   all16  lru   U 1.0 1.0                        mean U 1.0
#          label pu 20 26 -> 2.0 2.6 (2.3)        U 3.0 4.0 (3.5)
#          bit   pu 4 6   -> 0.4 0.6 (0.5)        U 2.5 3.5 (3.0)
#          rand  pu 10 10 -> 1.0                  U 2.0 2.4 (2.2)
#   leaf16 lru   U 1.0 2.0 (1.5); label U 5.0 6.0 (5.5); bit U 3.0 3.0 (3.0);
#          label_binary_600 pu 0 2 -> 0.0 0.2 (an arm read only by the leaf16 maximum)
# P = U + pu: all16 label 5.8, bit 3.5, rand 3.2; leaf16 label 5.5, bit 3.0. So
# residual = separation - pu_gap = 2.0 - 1.8 = 0.2 = (5.5 - 5.8) - (3.0 - 3.5),
# residual_rand = D_rand + pu_gap_rand = -0.8 + 0.5 = -0.3 = 3.2 - 3.5.
EXAMPLE = {
    ("all16", "lru"): ((0, 0), (1.0, 1.0)),
    ("all16", "label"): ((20, 26), (3.0, 4.0)),
    ("all16", BIT): ((4, 6), (2.5, 3.5)),
    ("all16", RAND): ((10, 10), (2.0, 2.4)),
    ("leaf16", "lru"): ((0, 0), (1.0, 2.0)),
    ("leaf16", "label"): ((0, 0), (5.0, 6.0)),
    ("leaf16", BIT): ((0, 0), (3.0, 3.0)),
    ("leaf16", "label_binary_600"): ((0, 2), (2.0, 2.0)),
}


def example_rows(skip=(), values=None, requested=1000) -> list[dict]:
    rows = []
    for (mechanism, arm), (pu, u) in (values or EXAMPLE).items():
        for seed in SEEDS:
            if (mechanism, arm, seed) not in skip and (mechanism, arm) not in skip:
                rows.append(bailian_row(mechanism, arm, seed, pu[seed], u[seed],
                                        requested=requested))
    return rows


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ReplayIndexTests(unittest.TestCase):
    def test_bailian_points_and_heap_rows(self):
        rows = example_rows() + [{**bailian_row("heap", "H_off", 0, 0, 9.0), "seed": ""}]
        replays = tabulate.bailian_replays(rows)
        self.assertNotIn((TRACE,) + CELL + ("heap", "H_off"), replays)
        # (pu, U, P) in points: P = (L2-avoided + unusable tokens) / 10.
        self.assertEqual(replays[(TRACE,) + CELL + ("all16", "label")],
                         {0: (2.0, 3.0, 5.0), 1: (2.6, 4.0, 6.6)})

    def test_bailian_duplicate_and_bad_values(self):
        rows = example_rows()
        with self.assertRaisesRegex(ValueError, "listed twice"):
            tabulate.bailian_replays(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "not an integer"):
            tabulate.bailian_replays([{**rows[0], "W_requested_tokens": "1e3"}])
        with self.assertRaisesRegex(ValueError, "not a finite number"):
            tabulate.bailian_replays([{**rows[0], "U_W_points": "nan"}])

    def test_missing_columns(self):
        row = example_rows()[0]
        del row["W_l2_present_unusable_tokens"]
        with self.assertRaisesRegex(ValueError, "missing column.*W_l2_present_unusable_tokens"):
            tabulate.bailian_replays([row])
        row = example_rows()[0]
        del row["W_l2_avoided_tokens"]
        with self.assertRaisesRegex(ValueError, "missing column.*W_l2_avoided_tokens"):
            tabulate.bailian_replays([row])
        with self.assertRaisesRegex(ValueError, "no rows"):
            tabulate.bailian_replays([])
        bare = mooncake_row("all16", "label", 0, 0, 0)
        del bare["extra_avoided_tokens"]
        with self.assertRaisesRegex(ValueError, "src: missing column.*extra_avoided_tokens"):
            tabulate.mooncake_replays({"src": [bare]})

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not found"):
                tabulate.read_csv(Path(directory) / "replay_seeds.csv")


class CellQuantityTests(unittest.TestCase):
    def test_bailian_example(self):
        row = tabulate.cell_quantities(tabulate.bailian_replays(example_rows()), TRACE, *CELL,
                                       SEEDS)
        expected = {"h_star": 60.0, "evaluable_all16": True, "evaluable_leaf16": True,
                    "pu_all16_label": 2.3, "pu_all16_bit": 0.5, "pu_all16_rand": 1.0,
                    "pu_leaf16_max": 0.2, "gain_label": 2.0, "gain_bit": 0.0,
                    "Delta_all16": 0.5, "Delta_leaf16": 2.5, "separation": 2.0, "pu_gap": 1.8,
                    "pu_gap_leaf16": 0.0, "residual": 0.2, "residual_from_P": 0.2,
                    "D_rand": -0.8, "pu_gap_rand": 0.5, "residual_rand": -0.3,
                    "residual_rand_from_P": -0.3}
        self.assertEqual(row["cell"], "l1=0.0025,l2x1")
        self.assertEqual(set(expected) - set(row), set())
        for key, value in expected.items():
            if isinstance(value, bool):
                self.assertIs(row[key], value, key)
            else:
                self.assertAlmostEqual(row[key], value, places=12, msg=key)
        # separation is gain(label) - gain(bit) as well as Delta_leaf16 - Delta_all16.
        self.assertAlmostEqual(row["separation"], row["gain_label"] - row["gain_bit"], places=12)

    def test_mooncake_example(self):
        # requested 2,000 tokens: points = tokens / 20.
        sources = {
            "error_location_001/replay_seeds.csv": [
                mooncake_row("all16", "label", 0, 60, 100), mooncake_row("all16", "label", 1, 50, 120)],
            "horizon_control_001/replay_seeds.csv": [
                mooncake_row("all16", BIT, 0, 20, 110), mooncake_row("all16", BIT, 1, 20, 130)],
            "leaf_matched_horizon_001/replay_seeds.csv": [
                mooncake_row("leaf16", "label", 0, 0, 160), mooncake_row("leaf16", "label", 1, 0, 160),
                mooncake_row("leaf16", BIT, 0, 0, 130), mooncake_row("leaf16", BIT, 1, 0, 150)]}
        row = tabulate.cell_quantities(tabulate.mooncake_replays(sources), "conversation_trace",
                                       *CELL, SEEDS, bailian=False)
        # P: all16 label 5.5 + 2.75, bit 6.0 + 1.0; leaf16 label 8.0, bit 7.0.
        expected = {"pu_all16_label": 2.75, "pu_all16_bit": 1.0, "pu_leaf16_max": 0.0,
                    "gain_label": 2.5, "gain_bit": 1.0, "Delta_all16": -0.5, "Delta_leaf16": 1.0,
                    "separation": 1.5, "pu_gap": 1.75, "residual": -0.25,
                    "residual_from_P": -0.25}
        for key, value in expected.items():
            self.assertAlmostEqual(row[key], value, places=12, msg=key)
        for absent in ("evaluable_all16", "evaluable_leaf16", "pu_all16_rand", "D_rand",
                       "pu_gap_rand", "residual_rand", "residual_rand_from_P"):
            self.assertNotIn(absent, row)

    def test_evaluable_boundary_per_mechanism(self):
        def flags(label_all16, label_leaf16):
            values = dict(EXAMPLE)
            values["all16", "label"] = (values["all16", "label"][0], label_all16)
            values["leaf16", "label"] = (values["leaf16", "label"][0], label_leaf16)
            # 100,000 requested tokens, so 1.99 points is a whole number of tokens.
            rows = example_rows(values=values, requested=100000)
            row = tabulate.cell_quantities(tabulate.bailian_replays(rows), TRACE, *CELL, SEEDS)
            return row["evaluable_all16"], row["evaluable_leaf16"]

        # lru means: all16 1.0, leaf16 1.5. Exactly 1.0 point of headroom is evaluable.
        self.assertEqual(flags((2.0, 2.0), (2.5, 2.5)), (True, True))
        self.assertEqual(flags((1.99, 1.99), (2.5, 2.5)), (False, True))
        self.assertEqual(flags((2.0, 2.0), (2.49, 2.49)), (True, False))

    def test_missing_arm_and_seed(self):
        replays = tabulate.bailian_replays(example_rows(skip={("leaf16", BIT)}))
        with self.assertRaisesRegex(ValueError, "missing arm: .* leaf16 label_binary_60"):
            tabulate.cell_quantities(replays, TRACE, *CELL, SEEDS)
        replays = tabulate.bailian_replays(example_rows(skip={("all16", RAND)}))
        with self.assertRaisesRegex(ValueError, "missing arm: .* all16 label_binary_random_60"):
            tabulate.cell_quantities(replays, TRACE, *CELL, SEEDS)
        # Mooncake needs neither lru nor the random arm.
        row = tabulate.cell_quantities(replays, TRACE, *CELL, SEEDS, bailian=False)
        self.assertAlmostEqual(row["separation"], 2.0, places=12)
        replays = tabulate.bailian_replays(example_rows(skip={("all16", "label", 1)}))
        with self.assertRaisesRegex(ValueError, r"all16 label: seeds \[0\], expected \[0, 1\]"):
            tabulate.cell_quantities(replays, TRACE, *CELL, SEEDS)
        with self.assertRaisesRegex(ValueError, "no h"):
            tabulate.cell_quantities(replays, TRACE, 0.5, 1.0, SEEDS)


class MooncakeConflictTests(unittest.TestCase):
    def test_agreeing_overlap_and_other_variants(self):
        rung = mooncake_row("all16", BIT, 0, 20, 110)
        replays = tabulate.mooncake_replays({
            "horizon_control_001": [rung],
            "horizon_fill_001": [dict(rung),
                                 mooncake_row("all16", BIT, 0, 99, 999, variant="other")]})
        self.assertEqual(replays[("conversation_trace",) + CELL + ("all16", BIT)],
                         {0: (1.0, 5.5, 6.5)})

    def test_conflict_is_named(self):
        sources = {"horizon_control_001": [mooncake_row("all16", BIT, 0, 20, 110),
                                           mooncake_row("all16", BIT, 1, 20, 130)],
                   "horizon_fill_001": [mooncake_row("all16", BIT, 0, 20, 111),
                                        mooncake_row("all16", BIT, 1, 21, 130)]}
        with self.assertRaises(ValueError) as caught:
            tabulate.mooncake_replays(sources)
        message = str(caught.exception)
        self.assertIn("conflicting Mooncake rows", message)
        self.assertIn("seed 0", message)       # utility differs
        self.assertIn("seed 1", message)       # present-but-unusable differs
        self.assertIn("horizon_control_001", message)
        self.assertIn("horizon_fill_001", message)

    def test_a_conflict_in_l2_avoided_alone_is_named(self):
        sources = {"horizon_control_001": [mooncake_row("all16", BIT, 0, 20, 110)],
                   "horizon_fill_001": [mooncake_row("all16", BIT, 0, 20, 110, l2_avoided=111)]}
        with self.assertRaisesRegex(ValueError, "conflicting Mooncake rows"):
            tabulate.mooncake_replays(sources)


class AccountingTests(unittest.TestCase):
    """residual = separation - pu_gap against its right-hand side from P."""

    def _row(self, values=None, rows=None):
        return tabulate.cell_quantities(
            tabulate.bailian_replays(rows or example_rows(values=values)), TRACE, *CELL, SEEDS)

    def test_the_identity_holds_on_the_example(self):
        row = self._row()
        tabulate.check_identities([row])
        self.assertAlmostEqual(row["residual"], 0.2, places=12)
        self.assertAlmostEqual(row["residual_rand"], -0.3, places=12)

    def test_a_leaf16_pu_gap_enters_the_right_hand_side(self):
        # leaf16 label leaves 0.5 point unusable: separation and pu_gap do not
        # move, P_leaf16(label) rises to 6.0, and the identity still holds only
        # with the measured pu_leaf16(label) - pu_leaf16(bit) = 0.5 subtracted.
        values = dict(EXAMPLE)
        values["leaf16", "label"] = ((5, 5), values["leaf16", "label"][1])
        row = self._row(values=values)
        tabulate.check_identities([row])
        self.assertAlmostEqual(row["pu_gap_leaf16"], 0.5, places=12)
        self.assertAlmostEqual(row["residual"], 0.2, places=12)
        self.assertAlmostEqual(row["residual_from_P"], 0.2, places=12)
        self.assertAlmostEqual(row["residual_from_P"] + row["pu_gap_leaf16"], 0.7, places=12)

    def test_doctored_rows_fail_loudly(self):
        def doctored(mechanism, arm, seed, tokens):
            rows = example_rows()
            target = next(row for row in rows if (row["mechanism"], row["arm"], row["seed"])
                          == (mechanism, arm, str(seed)))
            target["W_l2_avoided_tokens"] = str(tokens)
            return self._row(rows=rows)

        # all16 bit, seed 0: 25 L2-avoided tokens become 27 (U unchanged), so
        # P_all16(bit) and both right-hand sides move by 0.1 point.
        row = doctored("all16", BIT, 0, 27)
        self.assertAlmostEqual(row["residual_from_P"] - row["residual"], 0.1, places=12)
        with self.assertRaises(ValueError) as caught:
            tabulate.check_identities([row])
        message = str(caught.exception)
        self.assertIn("accounting identity", message)
        self.assertIn("bailian_toc_trace l1=0.0025,l2x1 residual ", message)
        self.assertIn("residual_rand", message)
        # The random arm moves residual_rand only; leaf16 label moves residual only.
        for (mechanism, arm, seed, tokens), name in (((("all16", RAND, 1, 25)), "residual_rand"),
                                                     ((("leaf16", "label", 0, 51)), "residual")):
            with self.assertRaises(ValueError) as caught:
                tabulate.check_identities([doctored(mechanism, arm, seed, tokens)])
            problems = str(caught.exception).splitlines()[1:]
            self.assertEqual(len(problems), 1, problems)
            self.assertIn(f"l2x1 {name} ", problems[0])

    def test_mooncake_rows_are_checked_too(self):
        sources = {"a": [mooncake_row("all16", "label", s, 60, 100) for s in SEEDS]
                   + [mooncake_row("all16", BIT, s, 20, 110) for s in SEEDS]
                   + [mooncake_row("leaf16", "label", s, 0, 160) for s in SEEDS]
                   + [mooncake_row("leaf16", BIT, 0, 0, 130),
                      mooncake_row("leaf16", BIT, 1, 0, 130, l2_avoided=131)]}
        row = tabulate.cell_quantities(tabulate.mooncake_replays(sources), "conversation_trace",
                                       *CELL, SEEDS, bailian=False)
        with self.assertRaisesRegex(ValueError, "conversation_trace l1=0.0025,l2x1 residual"):
            tabulate.check_identities([row])

    def test_display_hides_the_right_hand_sides(self):
        shown = tabulate.display([self._row()])[0]
        for hidden in ("residual_from_P", "residual_rand_from_P", "pu_gap_leaf16",
                       "l1_fraction", "pu_gap_rand"):
            self.assertNotIn(hidden, shown)
        columns = list(shown)
        self.assertEqual(columns[columns.index("pu_gap"):],
                         ["pu_gap", "residual", "D_rand", "-pu_gap_rand", "residual_rand"])
        self.assertAlmostEqual(shown["-pu_gap_rand"], -0.5, places=12)
        self.assertEqual(shown["h_star"], "60")


class PearsonAndSummaryTests(unittest.TestCase):
    def test_pearson(self):
        self.assertAlmostEqual(tabulate.pearson([1, 2, 3], [2, 4, 6]), 1.0, places=12)
        self.assertAlmostEqual(tabulate.pearson([1, 2, 3], [3, 2, 1]), -1.0, places=12)
        # x = (1, 2, 3, 4), y = (1, 3, 2, 4): sxy = 4, sxx = syy = 5.
        self.assertAlmostEqual(tabulate.pearson([1, 2, 3, 4], [1, 3, 2, 4]), 0.8, places=12)

    def test_pearson_degenerate(self):
        self.assertTrue(math.isnan(tabulate.pearson([0.1, 0.1, 0.1], [1, 2, 3])))
        self.assertTrue(math.isnan(tabulate.pearson([1, 2, 3], [0.7, 0.7, 0.7])))
        self.assertTrue(math.isnan(tabulate.pearson([1.0], [2.0])))
        self.assertTrue(math.isnan(tabulate.pearson([], [])))
        with self.assertRaises(ValueError):
            tabulate.pearson([1, 2], [1, 2, 3])

    @staticmethod
    def _row(separation, pu_gap, d_rand, pu_gap_rand, all16=True, leaf16=True, leaf_max=0.0,
             trace="t", cell="c"):
        return {"trace": trace, "cell": cell, "evaluable_all16": all16,
                "evaluable_leaf16": leaf16, "separation": separation, "pu_gap": pu_gap,
                "residual": separation - pu_gap, "D_rand": d_rand, "pu_gap_rand": pu_gap_rand,
                "residual_rand": d_rand + pu_gap_rand, "pu_leaf16_max": leaf_max}

    def test_summary_populations(self):
        rows = [self._row(1.0, 2.0, -1.0, 1.0, cell="c0"),
                self._row(0.4, 0.5, -0.5, 0.5, cell="c1"),
                self._row(0.1, 0.49, -0.2, 0.1, leaf_max=0.3, cell="c2"),
                self._row(9.0, 1.0, -9.0, 9.0, leaf16=False, cell="c3"),  # out of the separation pool
                self._row(5.0, 5.0, 5.0, 5.0, all16=False, cell="c4")]    # out of both pools
        lines = {line["quantity"]: line for line in tabulate.summary(rows)}
        self.assertEqual(lines["trace x cell"]["cells"], 5)
        pool = rows[:3]
        self.assertEqual(lines["corr(separation, pu_gap)"]["cells"], 3)
        self.assertAlmostEqual(lines["corr(separation, pu_gap)"]["value"],
                               tabulate.pearson([r["separation"] for r in pool],
                                                [r["pu_gap"] for r in pool]), places=12)
        # pu_gap >= 0.5 keeps 2.0 and exactly 0.5, drops 0.49: ratios 0.5 and 0.8.
        self.assertEqual(lines["min separation / pu_gap"]["cells"], 2)
        self.assertAlmostEqual(lines["min separation / pu_gap"]["value"], 0.5, places=12)
        self.assertAlmostEqual(lines["max separation / pu_gap"]["value"], 0.8, places=12)
        self.assertEqual((lines["min separation / pu_gap"]["at"],
                          lines["max separation / pu_gap"]["at"]), ("t c0", "t c1"))
        # residuals over the pool: -1.0, -0.1, -0.39.
        self.assertEqual(lines["mean residual"]["cells"], 3)
        self.assertAlmostEqual(lines["mean residual"]["value"], -1.49 / 3, places=12)
        self.assertAlmostEqual(lines["mean |residual|"]["value"], 1.49 / 3, places=12)
        self.assertAlmostEqual(lines["max |residual|"]["value"], 1.0, places=12)
        self.assertEqual(lines["max |residual|"]["at"], "t c0")
        # residual_rand over the cells evaluable under all16: 0, 0, -0.1, 0.
        self.assertEqual(lines["mean residual_rand"]["cells"], 4)
        self.assertAlmostEqual(lines["mean residual_rand"]["value"], -0.025, places=12)
        self.assertAlmostEqual(lines["mean |residual_rand|"]["value"], 0.025, places=12)
        self.assertAlmostEqual(lines["max |residual_rand|"]["value"], 0.1, places=12)
        self.assertEqual(lines["max |residual_rand|"]["at"], "t c2")
        random_pool = rows[:4]
        self.assertEqual(lines["corr(D_rand, -pu_gap_rand)"]["cells"], 4)
        self.assertAlmostEqual(lines["corr(D_rand, -pu_gap_rand)"]["value"],
                               tabulate.pearson([r["D_rand"] for r in random_pool],
                                                [-r["pu_gap_rand"] for r in random_pool]),
                               places=12)
        self.assertEqual(lines["max pu over every leaf16 replay"]["value"], 0.3)

    def test_mooncake_summary_has_no_evaluable_rule(self):
        rows = [{"trace": "t", "cell": f"c{index}", "separation": s, "pu_gap": g,
                 "residual": s - g, "pu_leaf16_max": 0.0}
                for index, (s, g) in enumerate(((1.0, 1.2), (0.0, 0.1), (2.0, 2.4)))]
        lines = {line["quantity"]: line for line in tabulate.summary(rows, bailian=False)}
        self.assertEqual(lines["corr(separation, pu_gap)"]["cells"], 3)
        self.assertNotIn("corr(D_rand, -pu_gap_rand)", lines)
        self.assertNotIn("mean residual_rand", lines)
        self.assertIn("no evaluable rule", lines["corr(separation, pu_gap)"]["over"])
        self.assertEqual(lines["min separation / pu_gap"]["cells"], 2)
        self.assertEqual(lines["mean residual"]["cells"], 3)
        self.assertAlmostEqual(lines["max |residual|"]["value"], 0.4, places=12)
        self.assertEqual(lines["max |residual|"]["at"], "t c2")

    def test_per_trace(self):
        rows = [self._row(1.0, 1.1, 0, 0, trace="a"), self._row(2.0, 2.4, 0, 0, trace="a"),
                self._row(3.5, 3.4, 0, 0, trace="a"),
                # b: a constant pu_gap, so no correlation; the evaluable flags
                # keep its third cell out.
                self._row(0.1, 0.3, 0, 0, trace="b"), self._row(0.2, 0.3, 0, 0, trace="b"),
                self._row(9.0, 0.0, 0, 0, trace="b", leaf16=False),
                # c: no cell in the population.
                self._row(1.0, 1.0, 0, 0, trace="c", all16=False)]
        table = {entry["trace"]: entry for entry in tabulate.per_trace(rows)}
        self.assertEqual(list(table), ["a", "b", "c"])
        a, b, c = table["a"], table["b"], table["c"]
        self.assertEqual(a["cells"], 3)
        self.assertAlmostEqual(a["corr_separation_pu_gap"],
                               tabulate.pearson([1.0, 2.0, 3.5], [1.1, 2.4, 3.4]), places=12)
        # a's residuals: -0.1, -0.4, +0.1.
        self.assertAlmostEqual(a["mean_residual"], -0.4 / 3, places=12)
        self.assertAlmostEqual(a["mean_abs_residual"], 0.6 / 3, places=12)
        self.assertAlmostEqual(a["max_abs_residual"], 0.4, places=12)
        self.assertEqual((a["pu_gap_min"], a["pu_gap_max"]), (1.1, 3.4))
        self.assertEqual(b["cells"], 2)
        self.assertTrue(math.isnan(b["corr_separation_pu_gap"]))
        self.assertAlmostEqual(b["max_abs_residual"], 0.2, places=12)
        self.assertEqual((b["pu_gap_min"], b["pu_gap_max"]), (0.3, 0.3))
        self.assertEqual(c["cells"], 0)
        for key in ("corr_separation_pu_gap", "mean_residual", "mean_abs_residual",
                    "max_abs_residual", "pu_gap_min", "pu_gap_max"):
            self.assertTrue(math.isnan(c[key]), key)
        mooncake = tabulate.per_trace(rows, bailian=False)
        self.assertEqual([entry["cells"] for entry in mooncake], [3, 3, 1])


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.row = tabulate.cell_quantities(tabulate.bailian_replays(example_rows()), TRACE,
                                            *CELL, SEEDS)

    def _published(self, role="h_star", **overrides):
        entry = {"trace": TRACE, "l1_fraction": "0.0025", "l2_multiplier": "1.0", "role": role,
                 "evaluable_all16": "True", "evaluable_leaf16": "True",
                 "Delta_all16_points_mean_W": repr(0.5), "Delta_leaf16_points_mean_W": repr(2.5)}
        entry.update(overrides)
        return entry

    def test_matching_tables_pass(self):
        decoy = self._published(role="600", Delta_all16_points_mean_W="7.0")
        tabulate.check_order([self.row], [self._published(), decoy], "W")
        tabulate.check_order([self.row], [self._published(
            Delta_leaf16_points_mean_W=repr(2.5 + 1e-12))], "W")
        tabulate.check_random([self.row], [
            {"trace": TRACE, "l1_fraction": "0.0025", "l2_multiplier": "1.0", "mechanism": "all16",
             "D_rand_points_mean_W": repr(-0.8)},
            {"trace": TRACE, "l1_fraction": "0.0025", "l2_multiplier": "1.0", "mechanism": "leaf16",
             "D_rand_points_mean_W": "5.0"}])

    def test_mismatches_fail(self):
        with self.assertRaisesRegex(ValueError, "Delta_all16 computed"):
            tabulate.check_order([self.row], [self._published(
                Delta_all16_points_mean_W=repr(0.5 + 1e-6))], "W")
        with self.assertRaisesRegex(ValueError, "Delta_leaf16 computed .* published ''"):
            tabulate.check_order([self.row], [self._published(Delta_leaf16_points_mean_W="")],
                                 "W")
        with self.assertRaisesRegex(ValueError, "evaluable_leaf16 computed True"):
            tabulate.check_order([self.row], [self._published(evaluable_leaf16="False")], "W")
        with self.assertRaisesRegex(ValueError, "0 rows for"):
            tabulate.check_order([self.row], [self._published(role="600")], "W")
        with self.assertRaisesRegex(ValueError, "2 rows for"):
            tabulate.check_order([self.row], [self._published(), self._published()], "W")
        with self.assertRaisesRegex(ValueError, "not in the computed table"):
            tabulate.check_order([self.row], [self._published(),
                                              self._published(l2_multiplier="4.0")], "W")
        with self.assertRaisesRegex(ValueError, "missing column.*Delta_all16_points_mean_full"):
            tabulate.check_order([self.row], [self._published()], "full", flags=False)
        with self.assertRaisesRegex(ValueError, "D_rand computed"):
            tabulate.check_random([self.row], [
                {"trace": TRACE, "l1_fraction": "0.0025", "l2_multiplier": "1.0",
                 "mechanism": "all16", "D_rand_points_mean_W": "-0.7"}])

    def test_mooncake_order_without_flags(self):
        row = {key: value for key, value in self.row.items() if not key.startswith("evaluable")}
        published = {"workload": "mooncake", "trace": TRACE, "l1_fraction": "0.0025",
                     "l2_multiplier": "1.0", "role": "h_star",
                     "Delta_all16_points_mean_full": repr(0.5),
                     "Delta_leaf16_points_mean_full": repr(2.5)}
        tabulate.check_order([row], [published], "full", flags=False)
        with self.assertRaisesRegex(ValueError, "Delta_leaf16"):
            tabulate.check_order([row], [{**published, "Delta_leaf16_points_mean_full": "2.6"}],
                                 "full", flags=False)


# --- end to end on a synthetic directory (the registered traces, cells and seeds) ---------------


def _u(mechanism: str, role: str, seed: int) -> float:
    base = {("all16", "lru"): 1.0, ("all16", "label"): 3.0, ("all16", "bit"): 2.5,
            ("all16", "rand"): 2.0, ("leaf16", "lru"): 1.0, ("leaf16", "label"): 4.0,
            ("leaf16", "bit"): 3.0, ("leaf16", "rand"): 2.0}[mechanism, role]
    return base + 0.125 * seed


def synthetic_paper(directory: Path) -> None:
    """replay_seeds.csv, order_beyond_bit.csv and random.csv for every
    registered trace x cell: Delta_all16 = 0.5, Delta_leaf16 = 1.0,
    D_rand = -0.5, every cell evaluable."""
    replays, order, random_rows = [], [], []
    for trace in bc.TRACES:
        for cell in bc.CELLS:
            arms = {"lru": bc.LRU, "label": bc.LABEL, "bit": bc.transplant_arm(*cell),
                    "rand": bc.cell_random_arm(*cell)}
            for mechanism in ("all16", "leaf16"):
                for role, arm in arms.items():
                    for seed in bc.SEEDS:
                        pu = 0 if mechanism == "leaf16" else {"label": 30, "bit": 10}.get(role, 0)
                        replays.append(bailian_row(mechanism, arm, seed, pu,
                                                   _u(mechanism, role, seed), trace, cell,
                                                   requested=8000))
            where = {"trace": trace, "l1_fraction": str(cell[0]), "l2_multiplier": str(cell[1])}
            for role in ("h_star", "600"):
                order.append({**where, "role": role, "evaluable_all16": "True",
                              "evaluable_leaf16": "True", "Delta_all16_points_mean_W": "0.5",
                              "Delta_leaf16_points_mean_W": "1.0"})
            for mechanism in ("all16", "leaf16"):
                random_rows.append({**where, "mechanism": mechanism,
                                    "D_rand_points_mean_W": "-0.5"})
    replays.append({**bailian_row("heap", "H_off", 0, 0, 0.0), "seed": ""})
    write_csv(directory / "replay_seeds.csv", replays)
    write_csv(directory / "order_beyond_bit.csv", order)
    write_csv(directory / "random.csv", random_rows)


def synthetic_mooncake(root: Path, paper: Path, conflict: bool = False) -> None:
    """The four published directories (all16 label in error_location_001, the
    all16 rung in horizon_control_001 and again in horizon_fill_001, leaf16 in
    leaf_matched_horizon_001) and mooncake_order_beyond_bit.csv."""
    tables = {name: [] for name in tabulate.MOONCAKE_DIRS}
    order = []
    for trace in tabulate.MOONCAKE_TRACES:
        for cell in bc.CELLS:
            bit = bc.transplant_arm(*cell)
            for seed in bc.SEEDS:
                tables["error_location_001"].append(
                    mooncake_row("all16", "label", seed, 40, 100 + seed, trace, cell))
                tables["horizon_control_001"].append(
                    mooncake_row("all16", bit, seed, 20, 120 + seed, trace, cell))
                tables["horizon_fill_001"].append(
                    mooncake_row("all16", bit, seed, 20, 120 + seed + (1 if conflict else 0),
                                 trace, cell))
                tables["leaf_matched_horizon_001"] += [
                    mooncake_row("leaf16", "label", seed, 0, 160 + seed, trace, cell),
                    mooncake_row("leaf16", bit, seed, 0, 140 + seed, trace, cell)]
            order.append({"workload": "mooncake", "trace": trace, "l1_fraction": str(cell[0]),
                          "l2_multiplier": str(cell[1]), "role": "h_star",
                          "Delta_all16_points_mean_full": "-1.0",
                          "Delta_leaf16_points_mean_full": "1.0"})
    for name, rows in tables.items():
        write_csv(root / name / "replay_seeds.csv", rows)
    write_csv(paper / "mooncake_order_beyond_bit.csv", order)


class MainTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.paper = self.root / "bailian_external_check_001"
        synthetic_paper(self.paper)

    def tearDown(self):
        self._directory.cleanup()

    def _files(self):
        return sorted((str(path), path.stat().st_mtime_ns) for path in self.root.rglob("*"))

    def _run(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = tabulate.main([str(self.paper), *extra])
        return code, out.getvalue()

    def test_prints_without_writing(self):
        synthetic_mooncake(self.root, self.paper)
        before = self._files()
        code, text = self._run("--mooncake-root", str(self.root))
        self.assertEqual(code, 0)
        self.assertEqual(self._files(), before)
        for title in ("B1 Bailian", "B2 Bailian summary", "B3 Bailian per trace", "M1 Mooncake",
                      "M2 Mooncake summary", "M3 Mooncake per trace"):
            self.assertIn(title, text)
        self.assertRegex(text, r"\| corr\(separation, pu_gap\)\s+\| evaluable under all16 and "
                               r"leaf16\s+\| 24\s+\|")
        code, text = self._run()
        self.assertEqual(code, 0)
        self.assertNotIn("M1 Mooncake", text)
        self.assertIn("not read (no --mooncake-root)", text)

    def test_mismatch_exits_loudly(self):
        rows = tabulate.read_csv(self.paper / "order_beyond_bit.csv")
        rows[0]["Delta_leaf16_points_mean_W"] = "1.001"
        write_csv(self.paper / "order_beyond_bit.csv", rows)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as caught:
            tabulate.main([str(self.paper)])
        self.assertIn("Delta_leaf16 computed", str(caught.exception.code))
        self.assertEqual(out.getvalue(), "")          # nothing printed before the check

    def test_missing_file_column_and_arm(self):
        (self.paper / "random.csv").unlink()
        with self.assertRaises(SystemExit) as caught:
            self._run()
        self.assertRegex(str(caught.exception.code), "random.csv not found")
        synthetic_paper(self.paper)
        rows = tabulate.read_csv(self.paper / "order_beyond_bit.csv")
        write_csv(self.paper / "order_beyond_bit.csv",
                  [{key: value for key, value in row.items() if key != "evaluable_all16"}
                   for row in rows])
        with self.assertRaises(SystemExit) as caught:
            self._run()
        self.assertRegex(str(caught.exception.code), "missing column.*evaluable_all16")
        synthetic_paper(self.paper)
        rows = [row for row in tabulate.read_csv(self.paper / "replay_seeds.csv")
                if not (row["trace"] == "bailian_coder_trace" and row["arm"] == "label_binary_600")]
        write_csv(self.paper / "replay_seeds.csv", rows)
        with self.assertRaises(SystemExit) as caught:
            self._run()
        self.assertRegex(str(caught.exception.code), "missing arm: bailian_coder_trace")

    def test_a_broken_identity_exits_loudly(self):
        rows = tabulate.read_csv(self.paper / "replay_seeds.csv")
        target = next(row for row in rows if row["trace"] == "bailian_coder_trace"
                      and row["mechanism"] == "leaf16" and row["arm"] == "label_binary_300")
        target["W_l2_avoided_tokens"] = str(int(target["W_l2_avoided_tokens"]) + 1)
        write_csv(self.paper / "replay_seeds.csv", rows)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as caught:
            tabulate.main([str(self.paper)])
        self.assertIn("accounting identity", str(caught.exception.code))
        self.assertIn("bailian_coder_trace l1=0.02,l2x1 residual", str(caught.exception.code))
        self.assertEqual(out.getvalue(), "")

    def test_mooncake_conflict_exits_loudly(self):
        synthetic_mooncake(self.root, self.paper, conflict=True)
        with self.assertRaises(SystemExit) as caught:
            self._run("--mooncake-root", str(self.root))
        self.assertIn("conflicting Mooncake rows", str(caught.exception.code))


if __name__ == "__main__":
    unittest.main()
