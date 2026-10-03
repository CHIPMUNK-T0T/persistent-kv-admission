# 研究の現在地

この文書はREADMEから参照する現状の地図である。観測値と解釈を分ける。
先行研究との重複、新規性候補、次の戦略相談は
[2026-09-27の引継ぎ](strategy-handoff-20260927.md)に分けて記録した。
そこに挙げた候補は未決であり、新しい実験の事前登録ではない。
最新結果を踏まえた次の依頼範囲は[2026-10-03の引継ぎ](next-phase-handoff-20261003.md)を参照する。
こちらも計画依頼の文書であり、新規実験の事前登録ではない。
fresh-stream実験は[事前登録](counterfactual-randomness-plan.md)に沿って完了し、
[結果](counterfactual-randomness-findings.md)と[出力](../results/paper/counterfactual_randomness_001/README.md)を収録した。
その後、同じ固定状態・乱数列での[保持時間診断](counterfactual-residence-findings.md)も完了した。
引継ぎで未実施としていたsampling機構の対照は、[事前登録](mechanism-control-plan.md)に沿って完了し、
[結果](mechanism-control-findings.md)を収録した。
容量によるsign反転がworking set比で位置づくかの検査も、[事前登録](working-set-ratio-plan.md)に沿って完了し、
[結果](working-set-ratio-findings.md)を収録した。
凍結rankerとそのexact labelの差がどの判断にあるかを調べる誤り位置の対照も、[事前登録](error-location-plan.md)に沿って完了し、
[結果](error-location-findings.md)を収録した。
保存済みの判断ログだけを使う実rankerの誤り診断も、[事前登録](ranker-error-diagnosis-plan.md)に沿って完了し、
[結果](ranker-error-diagnosis-findings.md)を収録した。
exactな再利用labelの時間範囲（horizon）とclass内順序の対照も、[事前登録](horizon-control-plan.md)と、
その結果を見てから登録した[grid補間](horizon-fill-plan.md)に沿って完了し、[結果](horizon-control-findings.md)を収録した。
続いて、保存ログを容量に合うhorizonで再採点する[診断](matched-horizon-diagnosis-plan.md)と、合うhorizonの
classをresidentに与える[class-order対照](matched-class-order-plan.md)（実行前のaddendumで新規cellを8と訂正）を完了し、
[結果](matched-horizon-findings.md)を収録した。
その読みの三つの確認（評価窓の末尾600秒を除いた再集計、class内順序の混合と乱数順序、leaf適格での再実行）も、
三つの[事前登録](tail-window-check-plan.md)（[混合](class-order-mix-plan.md)、[leaf](leaf-matched-horizon-plan.md)、
leafは実行前のaddendumで再現検査の内容を訂正）に沿って完了し、[結果](matched-horizon-checks-findings.md)を収録した。
独立workloadでの確認に向けて、Qwen-Bailian 4 traceの[入力監査](bailian-input-audit.md)（138/138検査合格、replayなし）を終え、
[外部workload確認の事前登録案](bailian-external-check-plan.md)をレビュー用の下書きとして置いた。案は未合意で、実装・replay・fitは行っていない。
対象は固定Mooncake FAST'25トレースの正確なprefix再利用であり、報酬は回避した
prefill token数である。GPU時間や実運用上の速度改善は測っていない。

## 現在の問いと当初の目的（2026-10-03）

[RULES.md](../RULES.md)の目的は「有限容量のPersistent KV Cacheにおいて、将来再利用価値の高いstateをどう選び、限られた容量を最も有効に使うか」である。対象はL1から追い出されたstateを受け取る有限L2、主指標はL1単独に対する追加の回避prefill tokenで、目的は変えていない。

現在の問いは、**その効用を増やすために、どの保持判断について、どの時間範囲の再利用を見分け、その情報をどの退去候補・選択規則に渡す必要があるか**である。構造特徴や履歴予測はそのための手段であり、現段階の成果は予測・選択問題の具体化である。過去だけを使う優れた実用policyの完成とは区別する。

### 現在の証拠を四つの問いで読む

| 問い | 確認したこと | 適用範囲・残る問い |
|---|---|---|
| 選ぶ価値の差はあるか | 将来を読む比較器は有限L2でも汎用policyを上回る。公表機構の凍結rankerは、正確な訓練ラベルがsampled LRUに対して得る利得の11.1–27.9%を得る。 | 同じ機構で使える将来情報の効用を測った。比較器はgreedyで、最適値の上界でも過去だけで実行できるpolicyでもない。[機構対照](mechanism-control-findings.md)。 |
| どの判断を改善すると効用が増えるか | all16ではresidentの退去選択だけをexact labelへ置換すると、凍結rankerとlabelの差の65–100%が回復する。leaf16でも当該gapはeviction側という判定が12/12で保たれた。 | 一つの凍結ranker・相対的なarrival拒否規則への介入。admissionとの交互作用があり、admission一般が不要という結果ではない。[誤り位置](error-location-findings.md)、[horizon対照](horizon-control-findings.md)。 |
| どの再利用情報が役立つか | all16では、事後に選んだhorizonの正確な再利用bitとrecencyの組がgreedy exact next-use比較器のLRU比利得の90%以上に届く。登録gridで8/12、結果を見てから登録した補間で残り4/4（合算しない）。 | bit単独の十分性ではない。horizonを事前に決める方法や、他の機構・traceへの移植は未確定。[horizon対照](horizon-control-findings.md)。 |
| 同じclass内では何を優先するか | all16では学習順序がrecencyに8/12で全seed負けるが、leaf16では11/12で全seed勝つ。all16の損は、h以内に再利用されないclassだけに学習順序を使う介入でも現れる。 | 選択規則の効用は退去候補の適格性に依存する。なぜ逆転するか、過去の情報から必要なclassを得られるかは未解決。[三つの確認](matched-horizon-checks-findings.md)。 |

構造特徴の追加効果が小さかったこと、globalな予測指標だけではretention効用を評価できなかったことは、使う情報と評価対象を絞る根拠として残る。これらは履歴一般の限界やsemantic signalの必要性を証明しない。counterfactualの実現regretとfresh-stream検証は、単一実現の事後最大を安定した選択利得と読めないことを示した診断として位置づけ、現在の主題をexpected-Qの解明へ変更しない。

### 最新の三つの確認で更新した読み

- **評価窓。** all16のfull窓から末尾600秒を除いたhead窓でも、bit＋recencyは比較器のLRU比利得の90%以上に12/12で届く。学習順序を残したclass介入の`R ≥ 0.9`は10/12から9/12へ変わる（会話0.25%×4、0.903→0.859）。class内の学習順序とrecencyの効用差の符号は12/12で維持された。主要な読みは末尾600秒だけには依存しないが、全判定が不変とは言わない。
- **class内順序。** all16で学習順序をh以内に再利用されるclassだけに使うと、6/12で全seed改善、2/12でrecency armとdecision digestまで完全一致、残り4/12は符号混在。一つの平均は微小な負（tool-agent 1%×4、−0.00216 input-token points）なので「12/12でrecency以上」ではない。h以内に再利用されないclassだけに使うと9/12で全seed悪化し、元の8つの一貫した損をすべて含む。ただし2%×4では両traceとも改善する。両class内の一様乱数順序はrecencyに12/12で全seed負けるため、bitに加える選択規則にも効用がある。
- **leaf適格性。** all16で選んだh*を持ち越したleaf16では、学習順序を残したclass介入の`R ≥ 0.9`は同じ10/12だが、bit＋recencyの`S ≤ 0.10`は9/12。class内の学習順序は11/12で全seedにおいてrecencyに勝ち、recency順序の`R ≥ 0.9`は5/12である。これは比較器に対する相対的な不足で、絶対効用の低下ではない。例えば会話0.25%×1のbit＋recencyはall16の6.27点からleaf16の7.30点へ増え、label比較器は6.44点から9.22点へさらに増えた。leaf16で最良のhorizonを探した結果ではない。

ここで、`U`はL1単独に対する追加回避prefillのinput-token pointsである。`R = (U(class arm) − U(learned)) / (U(evict_label) − U(learned))`は、**learnedのadmissionを維持してresident選択だけをexact next-useにした改善**を分母とし、全headroomの回収率ではない。`S = (U(label) − U(bit＋recency)) / (U(label) − U(lru))`も、各機構内のgreedy比較器とsampled LRUを基準にする。機構が違えば分母も違うため、絶対Uと併記する。

「h以内に再利用されない」は、hより後の再利用とtrace内で再利用されないstateを含み、永久に不要という意味ではない。class混合の介入は後続storeや候補集合も変えるので、二つのclassの差を足して損失の因果的な寄与率とはしない。leaf化はresidentだけでなくarrivalの適格性も変える。内部node退去やpresent-but-unusable tokenのcounterだけでは、順序の逆転を説明も反証もできていない。

以前の「sampling機構は凍結rankerとlabelのgap全体の主因ではない」と、今回の「class内の選択規則の優劣が適格性で変わる」は両立する。前者は全体の不足の大きさ、後者は同じ情報を使う規則と機構の相互作用を測っている。

### 未達と次の判断

当初の目的に対して、選択改善余地と、そのために役立つ将来情報・resident選択・適格性の条件は具体化できた。一方、**過去だけからその情報を得て、限られた容量を有効に使うpolicyにする部分は未達**である。

残る中心的な問いは、(1) 別workloadで関係が再現するか、(2) horizonと機構を結果を見る前に固定できるか、(3) その条件で必要な再利用classを履歴から学習し、実際の回避token増加へつなげられるか、の三つとする。約59分のMooncake 2 traceから時間・日単位のpersistent reuseや実機TTFTは主張しない。I/O最適化へ範囲を広げない。

Bailianによる独立workload確認は、[入力監査](bailian-input-audit.md)を終え、[事前登録案](bailian-external-check-plan.md)をレビュー用に置いた段階である。案はMooncakeのh*表の移植（主解析）と固定grid内の存在（副解析）を分け、all16とleaf16を各機構の参照で評価し、学習モデルは使わず乱数のclass内順序を対照に加える。未決事項（gridの6秒と1,200秒、主窓、予測1の閾値、モデル移植の時期、評価可能cellの規則）は案の末尾に列挙し、合意後に計画だけを単独でcommitしてから実装へ進む。履歴fitは別の計画とする。[Claudeへの引継ぎ](next-phase-handoff-20261003.md)に範囲と停止点を記した。引継ぎも案も事前登録ではなく、新規fit・replayはまだ実施していない。追加実験は「何を予測し、どの候補・選択規則へ渡すべきかの判断が変わるか」で選び、Mooncake内の診断を無制限に増やさない。

### 論文の中心命題と新規性の位置づけ（2026-10-03、Bailian確認前）

三つの確認までの結果をAstraが論文の観点で整理し、Fableが表中の数値を公開CSVから再計算して照合した（[台帳](#仮説と証拠の台帳)の該当行と同じ値）。対象はMooncake実trace 2本（各3,537秒）、L1 victimを有限L2が受け取る設定である。比較器はいずれも将来を読むgreedyな参照で最適解ではなく、「90%」はそれぞれ指定した比較器との差に対する割合であって、全改善余地の回収率ではない。

| 確認できたこと | 主な結果 | 言えること |
|---|---|---|
| 選択改善の余地がある | 凍結rankerは、正確な訓練ラベルがsampled LRUに対して得る利得の11.1–27.9%しか回収していない | 今の機構でも、有用な将来情報を選択へ渡せれば改善余地がある。[機構対照](mechanism-control-findings.md) |
| 改善箇所を絞れた | residentの退去選択だけを正確なラベルにすると、ranker–label差の65–100%が回復する（12/12） | この凍結rankerでは、保存済みstateの選び直しが主要な改善箇所。[誤り位置](error-location-findings.md) |
| 再利用情報の時間範囲が重要 | all16では、事後に選んだhorizonの正確なbit＋recencyがgreedy `next_use` 比較器のLRU比利得の90%以上に届く（登録grid 8/12、事後登録の補間4/4、合算しない） | 次回利用時刻だけが有効な表現ではない。ただしhorizonは事後選択。[horizon対照](horizon-control-findings.md) |
| 正確なclassだけでは選択は終わらない | 両class内を一様乱数で選ぶとrecencyに12/12で全seed負ける | 同じclassに入った候補の選び方にも効用がある。[三つの確認](matched-horizon-checks-findings.md) |
| 順位情報の価値が機構で逆転する | class内の学習順序は、all16ではrecencyに8/12で全seed負け、h*を持ち越したleaf16では11/12で全seed勝つ | 同じ学習scoreの有用性は、退去候補の制約から独立には評価できない。同上 |
| 主要な結果は末尾だけの現象ではない | 末尾600秒を除いてもbit＋recencyの`S ≤ 0.10`は12/12、順序差の符号も12/12で維持（classの読みは10/12→9/12） | trace終端の情報だけでは主要な読みを説明できない。同上 |

**中心命題（現時点の案）。** 有限容量の階層KV cacheでは、再利用予測の価値は、その時間範囲と、予測を使う選択規則・退去候補の制約によって変わる。正確な再利用classを与えてもclass内選択には効用差が残り、同じ学習順位情報の追加価値が候補適格性によって逆転する。したがって下位tierでは、何を予測するかと、その情報をどの候補・選択規則に使うかを一体で評価する必要がある。新policyは未完成だが、良いKVを選ぶために必要な条件を、実際の回避prefill tokenへの介入で具体化した。

**新規性として押す三点。**

1. 正確な再利用classを与えても、追加の順位情報の価値が候補適格性で逆転する。介入効果として測れたが、内部原因は未説明で、leaf化はarrivalの適格性と後続cache状態も変える。
2. 学習順序の損が、予想と反対の非再利用classに現れる。再利用classだけに学習順序を使うと6/12で全seed改善、h以内に再利用されないclassだけでは9/12で全seed悪化し、後者は元の8つの一貫した損をすべて含む。非再利用classは永久に不要なstateではなく、容量占有や後続再利用への即断はしない。
3. 下位tierで、情報の価値を実際の判断（到着拒否、resident選択、classの正確化、class内順序、適格性）への介入で測り、「どの情報が当たるか」から「どこへ渡すと再計算が減るか」まで接続した。実験数ではなく、改善箇所と適用条件を区別できたことに価値を置く。

**先行研究との対応。** [2026-09-27の引継ぎ](strategy-handoff-20260927.md)の表に加えて、以下を照合した（リンク先は2026-10-03にFableが確認）。

| 既に先行研究にあること | こちらが追加できる部分 |
|---|---|
| 再利用境界（Belady boundary）で厳密な最遠next-use予測を緩和する。[LRB, NSDI'20](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) | 境界を与えた後にも残るclass内選択の価値と、それが候補適格性で逆転する条件 |
| 会話継続の予測と最終アクセス時刻を組み合わせてLLM prefix cacheを退去する。[LPC, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/414f642a1ea9350006669774cba9bcd4-Paper-Conference.pdf) | 二層のresident判断で、情報・horizon・class内順序を分けて効用を測る介入。最も近い先行研究で、同条件の比較は未実施 |
| reuse working setと容量でadmissionの効果が変わる。[EfficientAgent, arXiv 2609.33762](https://arxiv.org/abs/2609.33762) | 容量依存の符号反転自体ではなく（[working set比](working-set-ratio-findings.md)で既存論に帰着済み）、固定rankerの不足がどの判断・classに現れるか |
| session reuseとstructural reuseを統合した退去、agent workloadのround-robin tail eviction。[UniCache, SIGMETRICS 2026](https://jxing.me/pdf/unicache-sigmetrics26.pdf)、[RR-Evict, arXiv 2609.32278](https://arxiv.org/abs/2609.32278) | policyの提案ではなく、同じ予測情報の追加価値が退去候補の制約で変わることの対照実験 |

「1 bitを使う」「LRUに改善余地がある」「容量で良いpolicyが変わる」だけを中心に据えない。主要文献を照合した範囲では、再利用情報・class内順序・候補適格性の相互作用を下位KV tierの効用への介入として示した部分が最も差別化しやすい。世界初と断定する段階ではない。

**足りないもの。** 外的妥当性（同一providerの短い2 traceを越えて再現するか。[Bailian計画案](bailian-external-check-plan.md)がこれに当たる）、逆転の説明（なぜall16とleaf16で学習順序の価値が変わるか）、実現可能性（必要なclassを過去だけから予測し、実際のpolicy改善につなげられるか）、事前選択（horizonをtest結果を見る前に決められるか）。characterizationとしての論文を強める優先事項は外部traceでの確認であり、実用policyの完成は別の到達点とする。

## 仮説と証拠の台帳

| 仮説・論点 | 現在の読み | 主要な証拠と境界 |
|---|---|---|
| finite容量には選択余地がある | greedy offline next-useは全検証済みtrace・budget・容量proxyでLRUより上で、当初の5% stop gateを通過した。ただし最適値の上界ではない。 | [構造特性](characterization-findings.md)、[時間予測](temporal-prediction-findings.md)。後の二層比較ではL1 victim stream上でも汎用L2との間に余地がある。 |
| 2-hitとstate-size proxyの感度 | 2-hitはsyntheticの全検証容量で改善したが、実traceでは容量により符号が変わる。packed対fixed-block充填の変更は21 trace×budget組のwinnerを変えず、全順位は19/21で同じだった。 | [構造特性](characterization-findings.md)。同一絶対byte容量での比較であり、他のtraceへの不変性ではない。 |
| H0: 構造・時間的局所性を使えば汎用policyを上回れる | 元の形では支持されない。構造信号は存在するが小さく、時間的履歴からの予測精度がretention gainに直結しない。 | [構造特性](characterization-findings.md)、[時間予測](temporal-prediction-findings.md)、[研究設計](research-design.md)。観測stateの予測と、eviction候補の選択は別問題。 |
| 同じ23履歴特徴でモデルを大きくすれば候補の二値ラベル順位が改善する | 当該候補ログ上ではgradient boostingによる一貫した大幅な改善は確認されなかった。小さな差や同等のcellはあり、これだけで全履歴特徴や別目標の限界とは言えない。 | [モデル容量診断](predictability-retention-gap.md)。固定23特徴・候補母集団・ラベルでの比較。 |
| 「600秒内に再利用されるか」という固定二値目標が十分 | 小容量では不十分と示唆された。reuse-count/next-useなどへの変更は一部改善したが、ギャップは閉じない。大容量で「目標が正しい」とは断定できない。 | [oracle分解](predictability-retention-gap.md)、[目標変更](target-change-findings.md)。oracleは既知ラベルに対する比較で、行動価値の最適性証明ではない。 |
| prefix tree固有の容量配分が主要な損失源 | Phase 0.95の**heap LRU/LFU/2-hit**ではmissing-ancestor hit損失がゼロだったため、事前gateは通らず中心課題には採用しなかった。ただし任意のtree-aware配置の余地は否定できない。 | [L1 victim調査](two-tier-victim-findings.md)。独立block controlは同じ内容物でのhit計算差である。後のsampled L2ではpresent-but-unusable KVが発生する。 |
| 決定母集団で訓練すれば予測をretentionへ変換できる | 当該固定設計では否定。汎用ログ上のoff-policy順位と、学習policy自身が作るon-policy候補集合の順位が食い違う。history表現の一般的な限界は未判定。 | [Phase 0.97/0.98/0.98b](decision-population-findings.md)。最良学習armのclosure 0.13–0.27は、このphase固有の分母（床はsampled L2-LRU）による。同じ尺度でheap LRUはcellにより0.00–0.19、2%×4では0.15–0.19にある。 |
| L2内に直接の子を持つ到着stateの拒否を防げば改善する | 弱いB armの2%×4では局所的に改善。ただし全到着stateの保護も近い改善を出し、祖先選択固有の追加利得は未立証。小容量では害。 | [Phase 1介入](phase1-intervention-findings.md)。既存policyへの限定介入であり、汎用の新best policyではない。 |
| per-block attributionで損失原因を特定できる | 失われたblockを最後の除去へ帰属させる会計はL2-hit差をtoken単位で閉じた。ただし一つの判断を変えたときの因果的回収量ではない。 | [Phase 0.98b](decision-population-findings.md)。rejection/eviction/compulsoryの内訳と介入効果は区別する。 |
| 自分の決定を学習する反復で改善する | `next_use` の固定3更新で、完全な後半40%のrequest windowの2%×1・2%×4は両実トレースでpi3がpi0を平均上回った。一方、事前登録した共通終端母集団の順位改善+0.05は全cellで未達。 | [on-policy findings](onpolicy-learning-findings.md)。40,000決定のreservoir cap、既使用test window、3更新での打切り、短いlabel-observable windowとの差を考慮する。 |
| 正確な次回利用ラベルなら1回の捨て方の価値を順序づけられる | 固定320状態の単一将来実現では、exactラベルの固定tie-breakにも大きな事後regretがある。しかしexpected `Q`の順位失敗とはまだ言えない。 | [反実仮想findings](counterfactual-action-value-findings.md)。最小ラベルtie内の事後最良は、将来の実現値を見て選んだ下界であり実装可能なselectorではない。 |
| 事後最大との差は継続サンプリングに対して安定 | 固定40状態・16本の新しいstreamの交差評価では、全候補選択の学習側平均+15,003.0 tokenに対しheld-outは−267.8、exact tie内の選択は+13,567.2に対し+90.3。層と状態で符号が混在し、旧実現値での事後最大を安定した選択利得と読む根拠は弱まった。ただしexpected `Q`の等価性は示さない。 | [事前登録](counterfactual-randomness-plan.md)、[fresh-stream結果](counterfactual-randomness-findings.md)。旧[3 stream監査](counterfactual-randomness-reanalysis.md)は別の事後計算。 |
| 学習armの残りheadroomはsampling機構が失わせている | 当該梯子では否定。scoreを固定して機構だけを変えると、公表機構（到着+16 sampled、誰でも退去可）でも凍結rankerの訓練ラベルそのものはheap offline参照の76.5–98.2%に届き、凍結rankerはそのラベル段のsampled LRU比利得の11.1–27.9%に留まる。rankerとラベルの差（signal gap）が12/12 real cell・全seed・4機構すべてで`T`（heap offline参照 − sampled LRU）の半分以上。機構由来の損失は公表機構で`T`の0–28.1%、leaf限定・幅64ではほぼ消える。 | [事前登録](mechanism-control-plan.md)、[結果](mechanism-control-findings.md)。960 replay。ラベル段は将来を読む比較対象で実装可能ではない。線形ranker 1つ・target 1つ・2 traceであり、他の因果的情報の限界は示さない。pi3更新・binary target・arrival protectionは未実行。leaf限定は対照であり提案policyではない。 |
| 容量によるsign反転はこのrepo固有の発見である | 否定。L1 victim流から計算した再利用working set（EfficientAgentの比、計算前に定義と閾値1を固定）をL2容量で割った比は、2-hit admission対LRUのsign反転をheap・sampledとも実trace×cellの12/12で位置づけた。到着保護は公表済み6 cell中4で一致し、外れた2 cellは比1.005と1.179でseedが混在。反転は既存のworking set論で説明でき、独立の貢献としない。比≤1のcellは各traceで2%×4の1つだけで、交差点は比0.553〜1.005の間としか言えない。 |
| 凍結rankerとexact labelの差は、到着を拒否するかどうかの判断にある | 否定。公表機構のまま、residentのどれを退去させるかだけをexact labelに置き換えると差の65–100%が回復し、実trace×cellの12/12で「eviction側」と読めた。到着の拒否判断だけを置き換えても回復しない（平均は常に0以下）。ただしL1 0.25%では差の33–42%が両方を置き換えたときにしか得られず、二つは足し算にならない。hybridは将来を読む比較器で、因果的な予測器が同じものを供給できるとは示していない。 |
| victimが参照と一致する割合で予測器の良し悪しを評価できる | 否定。一致率を同じ（0.75、0.50）に固定しても、誤ったvictimを2位の候補にするか一様に選ぶかで効用が4–27点変わる。4つの判断統計はどれも18 armの効用順をρ≥0.9で再現しなかった（最大2/12 cell）。実rankerのpi0→pi3については、victimのlabel超過の平均が効用変化の符号と24/24で一致し、pairwise一致率は4 cellで逆に動いた。2組のranker対だけの符号一致で、評価指標として検証済みではない。 |
| victimのlabel超過の変化は、rankerの選び方が良くなったことを表す | 限定的。各policy自身の判断ログでは、label超過の変化はlabel windowの効用変化と22/24で符号が一致し、一貫した効用変化と逆に動いた例はない。しかし候補集合を固定してpi0とpi3で選び直すと、符号が母集団によって変わるcellが`next_use`で5/12、`binary`で7/12ある。対応は選び方だけのものではなく、policyが作る候補集合の変化を含む。 |
| 凍結rankerの誤りは、再利用されるstateどうしの順序の誤りである | 否定（自身の判断ログ上では）。label超過の99.3%以上は、候補に再利用されないstateがあるのに600秒以内に再利用されるstateを退去させた判断から来る。全候補が再利用され順序が問題になる判断は0.3%以下。判断の69–87%は参照とtie-breakだけが違う。同じ候補集合で比べると、この種の退去をrecencyより減らせているのは9/12 cellで、小容量の会話traceの3 cellではrecencyより多い。exact labelが保つstore上では、horizon対照により、事後に選んだ容量に合うhorizonの1 bitとrecency tie-breakの組がgreedy exact `next_use` labelに届くことが12 cellすべてで示された（うち4 cellは補間gridで）。recencyという順序は使っており、比較先は最適解ではない。 |
| 二値（600秒以内に再利用されるか）が正確に分かれば十分 | 600秒という境界に依る。exactな二値labelは1%×4と2%×4では`next_use` labelと同じ効用に届くが、残り8 cellでは下回り、最小cellではlabelのLRUに対する利得の82%を失う。この不足はworking set比と単調に対応するが、比との照合は両方の結果を見た後に行った。 |
| all16で、事後に選んだhの正確な再利用bit＋recencyは`next_use` labelに届く | 支持（all16・exact情報・recencyとの組の範囲で）。登録した5点のgrid {6, 15, 60, 300, 600}秒では、labelのLRUに対する利得の90%以上に届くhがあるcellが8/12。届かなかった4 cell（L2容量1%）は、結果を見てから登録した補間（90–240秒、予測「4/4で届く」）で4/4、最小は4 cellとも150秒。合う horizon はL2容量とともに伸びる（60、150、300、600秒）。隣のgrid点では利得の6–85%を失う。6 cellではbit＋recencyがgreedy `next_use` labelを全seedで上回る（0.29–0.59点）。 | [事前登録](horizon-control-plan.md)、[補間の事前登録](horizon-fill-plan.md)、[結果](horizon-control-findings.md)。どのhも実行後にgridから選んだもので、容量から事前に決められることや学習器が当てられることは示していない。8/12と4/4は合算しない。 |
| 凍結rankerの不足は、再利用classの見分けにあり、class内の順序ではない | 容量による。residentにexactな600秒の再利用classを与えると、その境界が合う6 cellではexact labelをevictionに使った回復分の94–104%、合わない6 cellでは33–61%を回復する。合うhorizonのclassを与えると、新規8 cellのうち6で90%以上（1%の4 cellで90–98%、2%×1で103%）、0.25%×1では73–81%で予測「8/8」は外れた。0.25%×1の残りはclass内順序で、recencyの順序なら103–106%。合うclassの中ではrankerの順序はrecencyより一貫して悪いcellが8/12（600秒classでは良いcellが9/12）。all16の損はh以内に再利用されないclassへの介入でも現れる（h以内に再利用されるclassだけをrankerが順序付けると6/12で全seed改善、2/12で完全一致、4/12で符号混在（tool-agent 1%×4の平均は−0.00216点）。h以内に再利用されないclassだけでは9/12で全seed悪化、2%×4の2 cellでは改善。予測「再利用classの側」は0/8で外れ）。leaf16では逆転し、rankerの順序が11/12で一貫して勝つ。 | 同上、[合うhorizonの対照](matched-horizon-findings.md)、[三つの確認](matched-horizon-checks-findings.md)。h*は同じtraceのgridから事後に選んだもの。新規8と再現4は合算しない。非再利用classでrankerの順序が劣る理由、leaf16で逆転する理由は未検討。 |
| 合うhorizonの読みは、末尾600秒でexact labelがtraceの終端を知っていることに依る | 否定。末尾600秒を除いた窓（入力tokenの56–57%）でも、bit＋recencyはlabelのLRU比利得の90%以上に12/12で届き（S ≤ 0.10）、classの読みは閾値上の1 cell（会話0.25%×4、0.903→0.859）だけ動き、class内順序の損の符号は12/12で同じでむしろ大きく、admissionの差の符号もゼロでないcellでは同じ。末尾600秒は差`label−learned`の42–47%を持ち、入力tokenの割合と同程度。 | [事前登録](tail-window-check-plan.md)、[結果](matched-horizon-checks-findings.md)。420 replayはすべて公開行をdigestまで再現。予測4つのうち2つ（classの読み10/12、admissionの符号12/12）は閾値上の1 cellとゼロ値の2 cellで外れた。 |
| class内のrecency順序は1 bitに情報を足していない（bitだけで足りる） | 否定。両class内を一様乱数で順序付けると、recency順序に12/12で一貫して負ける（−0.1〜−2.9点、1%×1で最大）。乱数は学習順序にも8/12で負け、会話0.25%×1でだけ勝つ。 | [事前登録](class-order-mix-plan.md)、[結果](matched-horizon-checks-findings.md)。乱数orderは1 seedに1 streamで、seed区間は標本抽出と乱数の両方の変動を含む。 |
| all16のh*とclass内順序の読みはleaf16へそのまま移る | 一部は維持されるが、順序の読みは逆転する。h*をall16から持ち越したleaf16では、classの読み（R ≥ 0.9）は同じ10/12で保つが、1 bit＋recencyがlabelに届くcellは9/12（0.25%×1で0.20–0.21、会話2%×1で0.11不足。h* < 600秒の全cellでSが0.11–0.20増加。機構ごとに比較器・分母が異なり、絶対効用の低下を意味しない）、class内順序は逆転してrankerの順序がrecencyに11/12で一貫して勝ち、class内recencyは7/12でlabelの回復の0.9未満。admissionの差は0.05点以下（到着が候補になる判断は2.5–10%）。 | [事前登録](leaf-matched-horizon-plan.md)（実行前のaddendumで、公開行にdigestがないため再現検査を全公開counter列87列の一致に変更）、[結果](matched-horizon-checks-findings.md)。h*はall16の最良で、leaf16の最良horizonは探していない。leaf16は機構の対照であり提案ではない。 |
| 凍結rankerは、容量に合う境界での再利用classをrecencyより見分けられない | 容量による（保存ログ上）。合う境界での「再利用されるstateを退去させ非再利用候補を残す」判断は、0.25%×1（60秒）ではrecencyの0.64倍、1%の4 cell（150秒）では2.1–2.3倍、2%×1（300秒）では1.04倍と0.90倍。600秒fitのscoreの判断内分離は、h*のbitで600秒のbitより低いcellが8中6（登録予測）、0.25%×1では600秒のbitの分離が偶然並み（0.50）で60秒のbitが上回る。 | [事前登録](matched-horizon-diagnosis-plan.md)、[結果](matched-horizon-findings.md)。凍結rankerの自身のstoreのログで、exact classが保つstoreの候補集合ではない。自身のstore上の統計は、class-kept store上の順序の価値を予測しない（0.25%×1で逆向き）。 |
| 差がeviction側にあるのはall16の到着条件の産物である | 否定。leaf16でもeviction-locatedが12/12で、label evictionだけで差の97–100%を回復する。ただしleaf16では到着が候補になる判断が少数（0.25%×1で約半分、他は3–18%）で、「admissionは重要でない」とは区別できない。合うhorizonでのclass介入は、all16からh*を持ち越したleaf16の同じ10/12でR ≥ 0.9（上の行）。 | 同上、[三つの確認](matched-horizon-checks-findings.md)。 |
| 自己再利用ゼロ候補の価値差は保持byte-secondsだけで説明できる | 同一容量1 MiBの2候補を固定して比較すると、保持byte-seconds差と実現`ΔQ600`の単純な相関はほぼゼロ。629/640ペアで後続victim差が最初のhit差より先、両候補が600秒前に除去された497ペア中491ペアで報酬差がその後も続く。単一の保持時間だけでは実現値を要約しにくい。ただし媒介効果やexpected `Q`の差を証明しない。 | [保持時間の事前登録](counterfactual-residence-plan.md)、[結果](counterfactual-residence-findings.md)。旧16 streamを再計測した機構診断であり独立sampleではない。 |

## 数値を読むための分母

Phase 0.97の「最良学習L2がbest generic heapを上回る5/12 real cells」は、同じcellで
見た**事後選択した**arm/targetの5 seed平均の記述である。最大差は+0.7 input-token
pointsで、選択を含めた有意な優位の検定ではない。これを、固定pi3とpi0を比べる
on-policy実験の事前登録された結果と混同しない。前者の
`HeadroomClosure = (arm − sampled L2-LRU)/(heap offline L2 − sampled L2-LRU)`
は、その比較系の尺度であって到達可能な最適効用の割合ではない。初期single-tierの
closureは同じseedのsampled LRUとgreedy offline next-useを基準とする別の尺度である（床はheap LRUではない）。いずれの
future-aware comparatorもgreedyで、一般に最適とは証明されていない。

on-policy `next_use` 2%×4のpi3−pi0は、完全な後半40%ではconversation +0.538、
tool-agent +0.483 input-token pointsで両方5/5 seedが正。一方、600秒ラベルを
観測できる短いsubwindowでは+0.143（3/5）、+0.198（5/5）。後者は探索的な
時間窓診断であり、前者の登録済みutility判定を置換しない。順位指標は同じ保存済み
`D_test(pi3)`の候補を両モデルで再採点するため、どちらのrequest-level utilityとも
異なる推定対象である。[on-policy結果](onpolicy-learning-findings.md)と
[後付け診断](retention-decision-diagnostics.md)を参照。

[反実仮想実験](counterfactual-action-value-findings.md)は40 lineageから8決定ずつ、
合計320状態・8,143 action/stream継続を取得した。1決定で既に抽出された16または
17候補のうち1つだけを強制的に捨て、その後は元のpi0/pi3 policyを続ける。主報酬は
`(t,t+600秒]`の回避prefill token数。実現した同一streamでの候補最大`Q`を基準と
するregretには、選択後の情報が入る。320状態のexact selector平均regretは32,223.6、
同じexact最小ラベルtie内で**事後**に最良を選んだregretは3,660.8 tokenであった。
318/320決定に複数の最小ラベル候補があり、98/320ではそのtie内に実現`Q`最大候補が
いなかった。これは固定ラベルと実現行動価値の食い違いを示すが、将来を見ない
selectorの利得、期待`Q`の順位失敗、全トレースの回収可能損失を定量化しない。
pi0/pi3はそれぞれ異なるstateを作るため、両者のstratum平均を因果的に引かない。

旧80状態・3 streamの[公開sensitivity](counterfactual-action-value-findings.md)では、
600秒の平均regretはexact 34,419、learned 35,107 tokenだった。**事後再集計（公開CSVの対象行・算術・同点規則は独立検算済み）**では、
exact最小ラベルtie内での
同一stream事後最良4,114に対し、streamを分けて選択・評価した値は35,613、
別streamの全候補から選択したactionのheld-out regretは37,360 token、
stream間Spearmanは0.071である。
同一stream内の事後最大と、独立streamで評価する固定選択の期待価値を等号で
結ばない。これは40 lineageのhash順位0/1、80状態に限定される。別stream評価は
80状態×3選択stream×2評価streamの480有向比較で、独立な480標本ではない。
Q最大の同点は最小action indexを選ぶと提示値に一致した。再計算手順と入力hashは
[公開監査](counterfactual-randomness-reanalysis.md)と
[機械可読summary](../results/paper/counterfactual_stream_reanalysis/summary.json)に記録した。
監査対象は既存CSVの算術で、
branch replayの正しさや新しい実験結果は検証していない。3 streamから安定した
expected `Q`差や「regretの93%はrandomness」といった割合を確定できない。

## fresh-stream実験の結果と境界

[事前登録](counterfactual-randomness-plan.md)どおり、40 lineageからhash順位0の
40状態・678候補を固定した。各状態の16本の新しいpost-draw L2 sampling streamで
全候補を評価した10,848 branchと、推定に使わないcaptured元action確認40 branchの
全10,888 branchが完了し、失敗lineageはない。15個の公開出力のSHA256は
[run config](../results/paper/counterfactual_randomness_001/run_config.json)と一致する。
旧captured/旧sensitivity streamはこの推定に含めていない。

stream 1–8をA、9–16をBとして、片方の平均`Q600`で選び、もう片方で元actionと
paired比較した。折り返して得た40状態等重みの記述平均は、全候補で学習側
+15,003.0 token、held-out側−267.8（正17・負23状態）。exact next-use/countの
最小ラベルtieは40状態すべてで同じ候補集合・選択となり、学習側+13,567.2、
held-out側+90.3（正20・ゼロ1・負19）であった。固定exact actionは同じ640
state/streamペアで元actionより平均+2,350.1 tokenだが、未来の利用を知る比較器で
あり実装可能なpolicyではない。8 stratum別の値と符号は
[findings](counterfactual-randomness-findings.md)に示す。

この結果は、旧単一実現の事後最大を転用可能な利得と読む根拠を弱める。一方、
固定状態・固定将来requestにおける8 stream選択手続きの条件付き評価であり、
expected `Q`が等しい証明ではない。各方向の8 stream区間や共有requestを持つ
5 seedから、独立workloadへの有意差やtrace-wide回収量は推論しない。
600秒で選んだ同じactionのtrace-end報酬は副次評価である。
旧[3 stream公開監査](counterfactual-randomness-reanalysis.md)は別の事後計算であり、
「regretの93%はrandomness」といった割合を確定しない。

## 自己再利用ゼロ候補の保持時間と後続trajectory

[別途事前登録した診断](counterfactual-residence-plan.md)では、上の40状態・16 streamを
そのまま用い、600秒内の自己再利用数がともにゼロの2候補を結果非依存で固定した。
Eは旧exact-next-useの同点内固定選択、Zはlive `last_group`が最大の逆側候補である。
計1,280 instrumented branchの報酬とdigestは旧branchと全件一致した。

両候補は全640ペアで同じ1 MiBを占め、初回overflow round数も同じだった。
`ΔQ600=Q(Z)−Q(E)`の40状態等重み平均は+305.3 tokenだが、状態平均の符号は
正20・負20、640 streamでも正328・ゼロ2・負310と混在する。候補の保持
byte-seconds差と`ΔQ`のSpearmanは640行で−0.029、40状態平均で−0.090。
これは単一の保持時間による**実現値の記述**が弱いことを示すが、保持時間は
行動後変数なので因果的な媒介を否定しない。

629/640ペアで後続victim/rejectionの差が最初のL2 hit差より先に現れた。
両候補が600秒前に除去された497ペアのうち491ペアでは、その後もrequest単位の
報酬差が出る。従って違いは候補自身への直接hitではなく、後続cache状態に伝わる。
ただし差がどう形成されたかの寄与率、安定した期待行動価値、cache状態とRNGを
含めた再合流は未測定である。詳細な表・図と限界は[findings](counterfactual-residence-findings.md)。
sampling機構の対照、label表現の再検討、意味信号、新policyはこの診断に含まない。

## 実験モデルの境界

初期single-tierはprefix-closedな状態集合を保ち、葉だけをevictする。online scored
armは16葉をsampleして比較し、heap版は参照である。two-tierの主モデルではL1が
heap LRU/LFUで全到着blockを保持し、葉をevictして固定victim streamを作る。L2は
そのvictimを受けるexclusiveなunion storeで、L2単体のprefix closureは要求しない。
request時のprefix chainが`L1 ∪ L2`に揃った範囲だけhitになり、L2から読んだblockは
L1へmaterialiseしてL2から外れる。後の学習L2では到着victimと16 sampled residentsの
中から1つを捨てる。samplingにより祖先だけが消え、L2に存在しても使えない子孫が
生じる。Phase 0.95のheap汎用policyでmissing-ancestor損失ゼロという結果を、
sampled L2やtree-aware配置一般に外挿しない。[二層の厳密な設定](two-tier-victim-findings.md)、
[決定母集団](decision-population-findings.md)。
