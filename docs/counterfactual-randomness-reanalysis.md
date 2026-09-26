# 旧 counterfactual 3-stream CSV の読み取り専用再集計

対象はコミット済み `results/paper/counterfactual_action_001/` の小さな CSV と run config のみ。実験、テスト、trace/NPZ 走査、新しい randomness run の読み取りは行っていない。計算手順は [`scripts/analyze_counterfactual_streams.py`](../scripts/analyze_counterfactual_streams.py)、機械可読な全値は [`summary.json`](../results/paper/counterfactual_stream_reanalysis/summary.json) に記録した。解析checkoutのHEADは `3362c7b1ec6ab60f277e65009c260667b81c6dd5`。このスクリプトはそのHEADには未収録のため、下記の実内容SHA-256で別途固定する。新しいfresh-stream実験の成果を確認したものではない。

## 再現条件と分母

- `selected_decisions.csv` の 40 lineage 各 8 状態から `hash_rank` **0 または 1** を選択し、計 80 状態。各状態に同じ候補 16 または 17 個、replicate 0/1/2 の 3 stream と 600 秒/trace-end の 2 window がある。
- exact/count の同点集合は `count_within_h` 最小の全候補。80 状態すべてで最小 count は 0。固定 exact は集合内で `last_group` 最小、さらに同点なら `action_index` 最小を選ぶ。`decision_regrets.csv` の count と next_use の選択番号・regret に全 stream/window で一致した。actual/learned は `is_actual=True` の action を使用し、公表 regret と一致した。
- stream `s` で選ぶ `best tie` は同点集合内で `Q_s` 最大、`best all` は全候補で `Q_s` 最大。同じ最大 Q が複数なら **`action_index` 最小**。別 stream `s'` の regret は `max_a Q_s'(a) - Q_s'(chosen)`。80 状態 × 3 選択 stream × 2 評価 stream = **480 有向評価**。固定 exact、within-stream hindsight best tie、actual は 80 × 3 = **240 状態×stream** の平均。相関は各状態の 3 つの無向 stream 組、計 **240 相関係数** の平均で、候補 Q の Spearman 順位には同順位の平均ランクを用いた。
- 元 CSV の候補番号、候補 identity、count、last_group は 3 stream で一致することを検査した。`decision_regrets.csv` の関連行および `sensitivity.csv` の count/learned の 3-stream 平均とも一致した。

## 検算値

単位は候補集合内の最良 Q に対する regret（avoided-prefill tokens）。表の右端は提示された整数に丸めた値。

| 指標 | 600 秒平均（未丸め） | 整数 |
| --- | ---: | ---: |
| 固定 exact/count | 34,419.2750 | 34,419 |
| 同じ stream 内の hindsight best tie | 4,114.1458 | 4,114 |
| 別 stream 評価の best tie | 35,613.2542 | 35,613 |
| 別 stream 評価の best all | 37,359.7896 | 37,360 |
| actual/learned | 35,106.8875 | 35,107 |

5 件とも提示値に一致した。平均 Q 順位 Spearman は 600 秒 **0.0711161**、trace end **0.0319674**。trace end の regret 平均もスクリプトの `summary.json` に記録したが、今回提示された比較対象はない。

`action_index` 最小の規則は held-out 2 値の再現条件である。600 秒の 240 source-state/stream のうち best-tie の最大 Q は 30 件、best-all の最大 Q は 17 件で複数候補同点だった。最大候補番号を選ぶと held-out best tie は 35,659.8813、best all は 37,413.4292 となる。したがって同点規則を省いた数値は一意に定まらない。

## 読める範囲

この再集計は、**既存 CSV の算術、対象行、同点規則**を検証したもの。branch replay の正しさ、未公開の新実験、trace 全体や他 workload への外挿は検証していない。`within-stream hindsight best tie` は評価 stream 自身の Q を見て選んだ診断下界で、実装可能な選択器の性能ではない。held-out もわずか 3 stream の同じ 80 状態を再利用した有向比較であり、480 件を独立標本とは扱えない。

Spearman の 0.071/0.032 は **候補内 Q 順位の stream 間平均相関**である。これを ICC、分散成分、`93% noise`、または `sqrt(Spearman)` による達成可能順位相関の上限に読み替える根拠は今回の集計からは得られない。そのような推定は実施していない。

## 入力と解析の SHA-256

公開 `run_config.json` の `files_sha256` と CSV 4 件が一致し、同 config の `source_manifest_end.files` と source 2 件が一致した。公開 source manifest 自体の SHA-256 は `cbc133ff3904d623ac7dfd374e09671b5422c0962e3bf04ae66ac438dfb3540a`。

| ファイル | SHA-256 |
| --- | --- |
| `selected_decisions.csv` | `331c0ea0fb2d68501d755b4fbaf8ff7095037e7f6616f981eafce232fc187d11` |
| `branch_actions.csv` | `c184880b0d0093da743c9a8d545600cd0f01b693987ad9c55926d8c9613ed88f` |
| `decision_regrets.csv` | `26ba84ec9c0133af59597527cd6679683b1b786659596b01721ec7f5c6c41c37` |
| `sensitivity.csv` | `121d38d763fe410cd3adbffca25f89f19104f6e11da14f80444924cc2a67baec` |
| `run_config.json` | `175970ef039c76e930398745c633598b3463109b8205191a6c90ce0f66ebbb7a` |
| `scripts/run_counterfactual.py` | `5cf06f832c57904e8c5e9a4a32ad76279e2f08919a54814bee50d055cedab170` |
| `src/persistent_kv_admission/counterfactual.py` | `3f7b976e28a67a11224ddfeae22ab5d9ef9a5ad5da8b62f407ffc1c031b2b88d` |
| `scripts/analyze_counterfactual_streams.py` | `334322538082daac40577df77f364043b9534fee59c4d29ae7539c6973d80d68` |
