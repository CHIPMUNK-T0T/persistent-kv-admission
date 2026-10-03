# 研究の現在地

この文書はREADMEから参照する現状の地図である。観測値と解釈を分ける。
先行研究との重複、新規性候補、次の戦略相談は
[2026-09-27の引継ぎ](strategy-handoff-20260927.md)に分けて記録した。
そこに挙げた候補は未決であり、新しい実験の事前登録ではない。
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
対象は固定Mooncake FAST'25トレースの正確なprefix再利用であり、報酬は回避した
prefill token数である。GPU時間や実運用上の速度改善は測っていない。

## 現在の問い

有限容量のpersistent KV tierがL1から追い出されたstateを受け取るとき、到着stateと
既存L2 residentのどれを捨てるべきか。過去の再利用履歴から作ったスコア、600秒内の
正確な次回利用ラベル、そして「その1回の捨て方を変え、その後は元のpolicyを続ける」
場合の将来回避token価値 `Q` は、どの条件で同じ選択をするのか。直近の実験では、
単一の将来サンプリング実現で見た`Q`差が、別の将来サンプリング列でも保たれるかを
調べ、さらに自己再利用ゼロの2候補について保持時間と後続decisionの時系列を測った。
将来のリクエスト列そのものは固定する。研究目的は一貫して有限容量でのretention価値の選択であり、
現在は目標を変える段階ではなく、既存の選択と説明の検証を深めている。

2026-10-03時点の整理。問いは「有限容量の階層KV cacheで再利用を増やすために、どの保持判断について、どの時間範囲の将来利用を、どの粒度で予測する必要があるか」に絞り、exactな将来情報を与えたときの比較性能を測る対照を重ねている（sampled-16のgreedyであり最適値の上界ではない）。判断はresident evictionの側にある（all16で12/12、leaf16でも12/12）。粒度は、事後にgridから選んだ容量に合うhorizonの1 bit（h秒以内に再利用されるか）とrecency tie-breakの組で、sampled-16のgreedy exact `next_use` labelに届く（登録gridで8/12、結果を見てから登録した補間gridで残り4/4、合算しない。6 cellでは上回る）。recencyという順序は使っており、比較先は最適解ではない。時間範囲はL2容量とともに伸びる（60、150、300、600秒。Phase 0.9の単層oracle sweepと同じ向き）。凍結rankerの不足については、600秒境界での誤りの99%超が再利用classの見分けであること、600秒のclassを与えると境界が合う6 cellで94–104%回復することまでが分かっている。容量に合う境界のclassをresidentに与える対照（[結果](matched-horizon-findings.md)）では、1%の4 cellで不足の90–98%が回復し（600秒classでは45–61%）、0.25%×1では73–81%にとどまって残りはrankerのclass内順序（recencyの順序なら103–106%）、2%以上では全回復だった。合うclassの中ではrankerの順序はrecencyより悪く（8/12で一貫して損。600秒classでは9/12で得）、rankerの拒否規則はlabelの拒否より最大4.1点劣る。その損は再利用されないclassの側にあり（rankerに再利用classだけを順序付けさせると12/12でrecency以上、非再利用classだけを順序付けさせると9/12で一貫して損。予測は逆で0/8）、class内の乱数順序はrecencyに12/12で負ける。末尾600秒を除いても読みは変わらない（1 bitは12/12で届き、classの読みは閾値上の1 cellだけ動き、順序の損の符号は12/12で同じ）。ただしleaf適格（候補はleafだけ）では、classの読みは同じ10/12で保つが、1 bitは9/12にとどまり（0.25%×1で0.20–0.21不足）、class内順序は逆転してrankerの順序がrecencyに11/12で一貫して勝つ（recencyのclass内順序は7/12でlabelの回復の0.9を下回る）。class内順序の読みは候補集合（機構）に依り、その理由は未検討である。保存ログ上では、合う境界での誤りはrecencyより0.25%×1で少なく（0.64倍）、1%の4 cellで多い（2.1–2.3倍）。未測定は、学習器がその1 bitをどれだけ当てられるか、horizonを事前に決められるか、別traceでの再現である。

### 当初の目的との照合（2026-10-03）

RULES.mdの目的「有限容量のPersistent KV Cacheにおいて、将来再利用価値の高いstateをどう選び、限られた容量を最も有効に使うか」は変えていない。変わったのは手段の仮説（構造特徴や履歴予測が効くか）と成果の重心で、現在の成果は「優れた実用policy」ではなく「良いpolicyが解くべき予測・選択問題の具体化」である。両者は分けて扱う。

| 当初知りたかったこと | 現在の回答 | 根拠 |
|---|---|---|
| stateごとに残す価値の差があるか | ある。将来を読む比較器は、L1 victimを受ける有限L2でも汎用policyを大きく上回る。ただしその比較器はsampled-16のgreedyで、最適解でも実行可能なpolicyでもない。 | [機構対照](mechanism-control-findings.md)、[誤り位置](error-location-findings.md) |
| 構造特徴を使えば良いstateを選べるか | 今回の特徴では追加効果が小さく、中心に置く根拠は弱い。 | [構造特性](characterization-findings.md) |
| reuse予測の精度が高ければ十分か | 全体の予測指標や決定統計だけでは判断できない。実際の候補集合と選択規則で確認が要る。 | [時間予測](temporal-prediction-findings.md)、[誤り位置](error-location-findings.md)、[合うhorizonの追試](matched-horizon-findings.md) |
| 何を正しく見分ける必要があるか | 改善余地の大部分はresidentの退去選択にある（差の65–100%）。容量に合うhorizonの再利用classが有力で、class内の選び方も条件次第で重要（all16では合うclassの中で学習scoreの順序がrecencyより悪い8/12で、損は非再利用classの側。leaf16では逆に学習scoreの順序が11/12で勝つ）。 | [誤り位置](error-location-findings.md)、[horizon対照](horizon-control-findings.md)、[合うhorizonの追試](matched-horizon-findings.md)、[三つの確認](matched-horizon-checks-findings.md) |
| その情報を過去の履歴だけから得られるか | **未解決。** 600秒fitの凍結scoreの再採点では、合う境界の判断内分離は0.55–0.64で偶然より上、exactには遠い。合う境界を目標に学習した場合の成否は測っていない（Phase 0.9の単層では小さな改善）。 | [合うhorizonの追試](matched-horizon-findings.md)、[目標変更](target-change-findings.md) |
| 実用的に良いretention policyができたか | **まだできていない。** | — |

残る中心的な穴は三つ。(a) 必要な情報の予測可能性（合う境界で実際のresident候補を履歴から見分けられるか）。(b) 条件の事前決定と再現性（合うhorizonは結果を見て選んでおり、容量から事前に決める方法と、別traceでの再現が未確認）。(c) persistent tierへの適用範囲（約59分のtraceでは時間・日単位の長期保持は示せず、実機の速度も未測定。これを補うためにI/O最適化へ広げることはしない）。

以後の追加実験の採否基準は「その結果で、何を予測し、どの選択規則に渡すべきかの判断が変わるか」とし、診断そのものを目的にしない。この基準で行った三つの確認（[結果](matched-horizon-checks-findings.md)）の位置づけは次の通り。評価窓の確認（[計画](tail-window-check-plan.md)）は中心結論の交絡の確認で、末尾600秒を除いても読みは変わらなかった。class内順序の混合（[計画](class-order-mix-plan.md)）は「予測を渡す選択規則」の確定で、学習順序の損は非再利用classの側にあり（予測と逆）、class内のrecencyは乱数より情報を持つ。leaf適格の確認（[計画](leaf-matched-horizon-plan.md)）は条件が機構の産物でないことの確認で、classの読みは保ち、1 bitの読みは弱まり（9/12）、class内順序の読みは逆転した。したがって「合うhorizonのclassをresidentに与える」は候補集合によらず有効だが、「class内はrecency」という選択規則はall16の候補集合に特有で、渡す規則は機構ごとに確かめる必要がある。その後の外部trace（Bailian）は再現性、合う境界を目標にした履歴からの学習は穴(a)への接続点であり、どちらも別の事前登録で行う。

現在の成果を当初の目的に沿って一文で言えば、「有限容量の下位KV cacheには大きな選択改善余地があり、今回の条件ではresidentの退去選択が主要な改善箇所で、役立つ再利用の時間範囲は容量によって変わり、class内の選択方法は容量と候補集合によって変わる。ただし、その情報を履歴から取得して実用policyにする方法は未解決」である。

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
| 「h秒以内に再利用されるか」の1 bitは、hが容量に合えば`next_use` labelに届く | 支持（exact情報の範囲で）。登録した5点のgrid {6, 15, 60, 300, 600}秒では、labelのLRUに対する利得の90%以上に届くhがあるcellが8/12。届かなかった4 cell（L2容量1%）は、結果を見てから登録した補間（90–240秒、予測「4/4で届く」）で4/4、最小は4 cellとも150秒。合う horizon はL2容量とともに伸びる（60、150、300、600秒）。隣のgrid点では利得の6–85%を失う。6 cellでは1 bitが`next_use` labelを全seedで上回る（0.29–0.59点）。 | [事前登録](horizon-control-plan.md)、[補間の事前登録](horizon-fill-plan.md)、[結果](horizon-control-findings.md)。どのhも実行後にgridから選んだもので、容量から事前に決められることや学習器が当てられることは示していない。8/12と4/4は合算しない。 |
| 凍結rankerの不足は、再利用classの見分けにあり、class内の順序ではない | 容量による。residentにexactな600秒の再利用classを与えると、その境界が合う6 cellではexact labelをevictionに使った回復分の94–104%、合わない6 cellでは33–61%を回復する。合うhorizonのclassを与えると、新規8 cellのうち6で90%以上（1%の4 cellで90–98%、2%×1で103%）、0.25%×1では73–81%で予測「8/8」は外れた。0.25%×1の残りはclass内順序で、recencyの順序なら103–106%。合うclassの中ではrankerの順序はrecencyより一貫して悪いcellが8/12（600秒classでは良いcellが9/12）。この損は非再利用classの側にある（再利用classだけをrankerが順序付けると12/12でrecency以上、6/12で一貫して得。非再利用classだけでは9/12で一貫して損。予測「再利用classの側」は0/8で外れ）。leaf16では逆転し、rankerの順序が11/12で一貫して勝つ。 | 同上、[合うhorizonの対照](matched-horizon-findings.md)、[三つの確認](matched-horizon-checks-findings.md)。h*は同じtraceのgridから事後に選んだもの。新規8と再現4は合算しない。非再利用classでrankerの順序が劣る理由、leaf16で逆転する理由は未検討。 |
| 合うhorizonの読みは、末尾600秒でexact labelがtraceの終端を知っていることに依る | 否定。末尾600秒を除いた窓（入力tokenの56–57%）でも、1 bitはlabelに12/12で届き（S ≤ 0.10）、classの読みは閾値上の1 cell（会話0.25%×4、0.903→0.859）だけ動き、class内順序の損の符号は12/12で同じでむしろ大きく、admissionの差の符号もゼロでないcellでは同じ。末尾600秒は差`label−learned`の42–47%を持ち、入力tokenの割合と同程度。 | [事前登録](tail-window-check-plan.md)、[結果](matched-horizon-checks-findings.md)。420 replayはすべて公開行をdigestまで再現。予測4つのうち2つ（classの読み10/12、admissionの符号12/12）は閾値上の1 cellとゼロ値の2 cellで外れた。 |
| class内のrecency順序は1 bitに情報を足していない（bitだけで足りる） | 否定。両class内を一様乱数で順序付けると、recency順序に12/12で一貫して負ける（−0.1〜−2.9点、1%×1で最大）。乱数は学習順序にも8/12で負け、会話0.25%×1でだけ勝つ。 | [事前登録](class-order-mix-plan.md)、[結果](matched-horizon-checks-findings.md)。乱数orderは1 seedに1 streamで、seed区間は標本抽出と乱数の両方の変動を含む。 |
| 合うhorizonの読み（1 bit、class、class内順序）はall16の到着条件の産物である | 部分的に肯定。h*をall16から持ち越したleaf16では、classの読み（R ≥ 0.9）は同じ10/12で保つが、1 bit＋recencyがlabelに届くcellは9/12（0.25%×1で0.20–0.21、会話2%×1で0.11不足。h* < 600秒の全cellで0.11–0.20悪化）、class内順序は逆転してrankerの順序がrecencyに11/12で一貫して勝ち、class内recencyは7/12でlabelの回復の0.9未満。admissionの差は0.05点以下（到着が候補になる判断は2.5–10%）。 | [事前登録](leaf-matched-horizon-plan.md)（実行前のaddendumで、公開行にdigestがないため再現検査を全公開counter列87列の一致に変更）、[結果](matched-horizon-checks-findings.md)。h*はall16の最良で、leaf16の最良horizonは探していない。leaf16は機構の対照であり提案ではない。 |
| 凍結rankerは、容量に合う境界での再利用classをrecencyより見分けられない | 容量による（保存ログ上）。合う境界での「再利用されるstateを退去させ非再利用候補を残す」判断は、0.25%×1（60秒）ではrecencyの0.64倍、1%の4 cell（150秒）では2.1–2.3倍、2%×1（300秒）では1.04倍と0.90倍。600秒fitのscoreの判断内分離は、h*のbitで600秒のbitより低いcellが8中6（登録予測）、0.25%×1では600秒のbitの分離が偶然並み（0.50）で60秒のbitが上回る。 | [事前登録](matched-horizon-diagnosis-plan.md)、[結果](matched-horizon-findings.md)。凍結rankerの自身のstoreのログで、exact classが保つstoreの候補集合ではない。自身のstore上の統計は、class-kept store上の順序の価値を予測しない（0.25%×1で逆向き）。 |
| 差がeviction側にあるのはall16の到着条件の産物である | 否定。leaf16でもeviction-locatedが12/12で、label evictionだけで差の97–100%を回復する。ただしleaf16では到着が候補になる判断が少数（0.25%×1で約半分、他は3–18%）で、「admissionは重要でない」とは区別できない。合うhorizonでのclassの読みもleaf16で保つ（上の行）。 | 同上、[三つの確認](matched-horizon-checks-findings.md)。 |
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
