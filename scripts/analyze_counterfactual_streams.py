#!/usr/bin/env python3
"""Reanalyze published counterfactual CSVs without replaying experiments.

Reads only small published CSV/config/source files. Writes a summary to a
new, explicitly chosen directory. Never imports or runs experiment code.
"""

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def display_path(path):
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values):
    values = list(values)
    assert values
    return sum(values) / len(values)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [None] * len(values)
    pos = 0
    while pos < len(values):
        end = pos + 1
        while end < len(values) and values[order[end]] == values[order[pos]]:
            end += 1
        rank = (pos + end - 1) / 2
        for i in order[pos:end]:
            result[i] = rank
        pos = end
    return result


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = mean(ra), mean(rb)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra)
    db = sum((y - mb) ** 2 for y in rb)
    return None if da == 0 or db == 0 else numerator / math.sqrt(da * db)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path,
                        default=REPO / 'results/paper/counterfactual_action_001',
                        help='directory with the published counterfactual CSVs and run_config.json')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='new directory for summary.json; an existing path is refused')
    args = parser.parse_args()
    data = args.paper_dir.resolve()
    out = args.output_dir.resolve()
    if out.exists():
        parser.error(f'output directory already exists: {out}')
    inputs = [
        data / 'selected_decisions.csv',
        data / 'branch_actions.csv',
        data / 'decision_regrets.csv',
        data / 'sensitivity.csv',
        data / 'run_config.json',
        REPO / 'scripts/run_counterfactual.py',
        REPO / 'src/persistent_kv_admission/counterfactual.py',
    ]
    selected = read_csv(data / 'selected_decisions.csv')
    branches = read_csv(data / 'branch_actions.csv')
    regrets = read_csv(data / 'decision_regrets.csv')
    sensitivity = read_csv(data / 'sensitivity.csv')
    config = json.loads((data / 'run_config.json').read_text())
    for name in ('selected_decisions.csv', 'branch_actions.csv', 'decision_regrets.csv', 'sensitivity.csv'):
        assert sha256(data / name) == config['files_sha256'][name]
    for name in ('scripts/run_counterfactual.py', 'src/persistent_kv_admission/counterfactual.py'):
        assert sha256(REPO / name) == config['source_manifest_end']['files'][name]
    assert len(selected) == 320
    ids = {(r['lineage'], r['selection_hash']) for r in selected if int(r['hash_rank']) < 2}
    assert len(ids) == 80
    assert Counter(lineage for lineage, _ in ids) == Counter({r['lineage']: 2 for r in selected})
    assert all(int(r['hash_rank']) < 2 for r in selected if (r['lineage'], r['selection_hash']) in ids)

    grouped = defaultdict(lambda: defaultdict(dict))
    for row in branches:
        state = row['lineage'], row['selection_hash']
        if state not in ids:
            continue
        window, stream, action = row['window'], int(row['replicate']), int(row['action_index'])
        assert action not in grouped[(state, window)][stream]
        grouped[(state, window)][stream][action] = row

    regret_index = {}
    for row in regrets:
        state = row['lineage'], row['selection_hash']
        if state in ids and row['selector'] in {'count', 'next_use', 'learned'}:
            key = state, row['window'], int(row['replicate']), row['selector']
            assert key not in regret_index
            regret_index[key] = row

    metrics = {}
    for window in ('600', 'end'):
        fixed = []
        hindsight_tie = []
        actual = []
        heldout_tie = []
        heldout_all = []
        heldout_tie_high_index = []
        heldout_all_high_index = []
        correlations = []
        q_tie_counts = Counter()
        tie_size = []
        min_counts = []
        per_state = {}
        for state in sorted(ids):
            streams = grouped[(state, window)]
            assert set(streams) == {0, 1, 2}
            action_ids = set(streams[0])
            assert action_ids == set(range(len(action_ids)))
            assert len(action_ids) in (16, 17)
            assert all(set(streams[s]) == action_ids for s in streams)
            assert all(streams[s][a]['candidate_state_index'] == streams[0][a]['candidate_state_index']
                       and streams[s][a]['count_within_h'] == streams[0][a]['count_within_h']
                       and streams[s][a]['last_group'] == streams[0][a]['last_group']
                       for s in (1, 2) for a in action_ids)
            counts = {a: int(streams[0][a]['count_within_h']) for a in action_ids}
            min_counts.append(min(counts.values()))
            ties = [a for a in sorted(action_ids) if counts[a] == min(counts.values())]
            fixed_action = min(ties, key=lambda a: (int(streams[0][a]['last_group']), a))
            actual_action = next(a for a in action_ids if streams[0][a]['is_actual'] == 'True')
            assert all(streams[s][actual_action]['is_actual'] == 'True' for s in (1, 2))
            tie_size.append(len(ties))
            q = {s: {a: int(streams[s][a]['q_tokens']) for a in action_ids} for s in (0, 1, 2)}
            best = {s: max(q[s].values()) for s in (0, 1, 2)}
            per_state[state] = {'fixed': [], 'hindsight_tie': [], 'actual': []}
            for s in (0, 1, 2):
                fixed_regret = best[s] - q[s][fixed_action]
                tie_regret = best[s] - max(q[s][a] for a in ties)
                actual_regret = best[s] - q[s][actual_action]
                fixed.append(fixed_regret)
                hindsight_tie.append(tie_regret)
                actual.append(actual_regret)
                per_state[state]['fixed'].append(fixed_regret)
                per_state[state]['hindsight_tie'].append(tie_regret)
                per_state[state]['actual'].append(actual_regret)
                for selector, chosen, value in (
                    ('count', fixed_action, fixed_regret),
                    ('next_use', fixed_action, fixed_regret),
                    ('learned', actual_action, actual_regret),
                ):
                    row = regret_index[(state, window, s, selector)]
                    assert int(row['chosen_index']) == chosen
                    assert int(row['regret_tokens']) == value
                    if selector == 'count':
                        assert int(row['label_min_tie_count']) == len(ties)
                        assert int(row['label_tie_best_regret_tokens']) == tie_regret
            for left, right in ((0, 1), (0, 2), (1, 2)):
                value = spearman([q[left][a] for a in sorted(action_ids)],
                                 [q[right][a] for a in sorted(action_ids)])
                assert value is not None
                correlations.append(value)
            for source in (0, 1, 2):
                source_best_all = max(q[source].values())
                source_best_tie = max(q[source][a] for a in ties)
                winners_all = [a for a in sorted(action_ids) if q[source][a] == source_best_all]
                winners_tie = [a for a in ties if q[source][a] == source_best_tie]
                q_tie_counts['all_multiple' if len(winners_all) > 1 else 'all_unique'] += 1
                q_tie_counts['within_tie_multiple' if len(winners_tie) > 1 else 'within_tie_unique'] += 1
                for target in (0, 1, 2):
                    if source == target:
                        continue
                    heldout_all.append(best[target] - q[target][winners_all[0]])
                    heldout_tie.append(best[target] - q[target][winners_tie[0]])
                    heldout_all_high_index.append(best[target] - q[target][winners_all[-1]])
                    heldout_tie_high_index.append(best[target] - q[target][winners_tie[-1]])
        assert len(fixed) == len(hindsight_tie) == len(actual) == 240
        assert len(heldout_tie) == len(heldout_all) == 480
        assert len(correlations) == 240
        assert set(min_counts) == {0}
        metrics[window] = {
            'states': len(ids), 'state_streams': len(fixed), 'ordered_heldout_directions': len(heldout_tie),
            'unordered_stream_pairs': len(correlations),
            'mean_regret_tokens': {
                'fixed_exact': mean(fixed),
                'within_stream_hindsight_best_tie': mean(hindsight_tie),
                'heldout_best_tie': mean(heldout_tie),
                'heldout_best_all': mean(heldout_all),
                'actual_learned': mean(actual),
            },
            'mean_cross_stream_q_spearman': mean(correlations),
            'q_max_tie_counts': dict(q_tie_counts),
            'heldout_mean_with_max_action_index_on_source_q_tie': {
                'best_tie': mean(heldout_tie_high_index),
                'best_all': mean(heldout_all_high_index),
            },
            'heldout_directions_changed_by_max_index_tiebreak': {
                'best_tie': sum(a != b for a, b in zip(heldout_tie, heldout_tie_high_index)),
                'best_all': sum(a != b for a, b in zip(heldout_all, heldout_all_high_index)),
            },
            'mean_min_count_tie_size': mean(tie_size),
            'all_state_min_counts_are_zero': True,
        }
        if window == '600':
            expected = {'fixed_exact': 34419, 'within_stream_hindsight_best_tie': 4114,
                        'heldout_best_tie': 35613, 'heldout_best_all': 37360,
                        'actual_learned': 35107}
            metrics[window]['rounded_expected_comparison'] = {
                name: {'claimed': value, 'observed_round': round(metrics[window]['mean_regret_tokens'][name]),
                       'matches': round(metrics[window]['mean_regret_tokens'][name]) == value}
                for name, value in expected.items()
            }

    # Independent arithmetic check against the published three-stream mean table.
    for row in sensitivity:
        state = row['lineage'], row['selection_hash']
        if state not in ids or row['selector'] not in {'count', 'learned'}:
            continue
        window = row['window']
        stream_regrets = []
        for s in (0, 1, 2):
            stream_regrets.append(int(regret_index[(state, window, s, row['selector'])]['regret_tokens']))
        assert math.isclose(float(row['three_stream_mean_regret_tokens']), mean(stream_regrets), abs_tol=1e-8)

    summary = {
        'analysis_checkout_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'analysis_script_sha256': sha256(Path(__file__).resolve()),
        'input_sha256': {display_path(p): sha256(p) for p in inputs},
        'published_source_manifest_sha256': config.get('source_manifest_end', {}).get('sha256'),
        'published_plan_commit': config.get('plan_commit'),
        'definition': {
            'subset': 'hash_rank 0 or 1 per each of 40 lineages',
            'label_tie': 'minimum count_within_h (600-second exact count label)',
            'fixed_exact_tiebreak': 'minimum last_group, then minimum action_index',
            'heldout_q_tiebreak': 'minimum action_index among source-stream Q maxima',
            'regret': 'target-stream max Q over all legal actions minus target-stream Q of chosen action',
            'spearman': 'mean of per-state Spearman rank correlations over unordered stream pairs; average ranks for Q ties',
        },
        'metrics': metrics,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary['metrics'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
