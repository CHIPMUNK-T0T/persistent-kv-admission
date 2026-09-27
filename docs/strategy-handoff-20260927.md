# 次の研究戦略を検討するための引継ぎ — 2026-09-27

この文書はChatGPT等で次の戦略を検討するための入口である。研究結果の基準は
`5360dd87163a22d326006978cc1dbcac54a5e1ee`。今回追加したのは先行研究との位置づけと
未決の戦略候補であり、新しい実験結果・事前登録・policy設計ではない。
下記の候補は実施決定ではなく、実験や新policyの実装は開始していない。

## 1. 固定する研究目的

**有限容量のPersistent KV Cacheで、将来再利用価値の高いstateを選び、
限られた容量から得られるexact-prefix reuseを増やす。**

現在は新policyの一般化より、観測した結果に対する単純な説明・逆仮説の検証を優先する。
I/O throughput、SSD対recompute、embedding、semantic similarityを主題に変更しない。
実装・実行はSol、全体判断・結果レビューはAstraが担う方針を継続する。

## 2. 現在までに何を測ったか

正確な数値・分母は[研究の現在地](research-status.md)と各findingsを優先する。
Mooncakeの実trace 2本は同一deployment family由来で約59分、合成traceは別扱い。
ユーザー・session・prompt本文はなく、cross-session reuseや意味由来のreuseとは呼べない。
報酬は回避prefill tokensであり、GPU時間・TTFT・消費電力ではない。

| 論点 | 現在の証拠 | 言えないこと |
|---|---|---|
| 選択余地 | greedy future-aware comparatorと汎用policyの間に差がある | true optimumとの差、因果的に回収可能な量ではない |
| history prediction | global populationで高い予測精度があるが、decision populationやreplay効用とは一致しない | history一般が無力、semanticが必要とは言えない |
| target・model・population | 固定23特徴でtarget、モデル族、訓練母集団を変えても一貫してgapは閉じない | 全モデル・全目標を排除したわけではない |
| prefix dependency | heap LRU/LFU/2-hitではmissing-ancestorのhit損失がゼロ | sampled armや任意のtree-aware設計にも一般化できない |
| arrival protection | 弱いB armの2%×4では改善、小容量では悪化。direct_childとallは近い | ancestry固有の改善や汎用policyの優位ではない |
| on-policy学習 | 固定3更新のnext_useは小容量で悪化、2%×1・2%×4で両実traceとも改善 | 収束後の限界や独立workloadへの一般化ではない |
| per-block帰属 | loss会計がtoken単位で閉じる | そのdecisionを変えれば同量を回収できるとは言えない |
| counterfactual Q | 一つのvictimを変えると後続decisionと実現報酬が変わる | 安定した期待Qの優劣、trace全体の回収量ではない |

on-policy next_useの2%×4で、pi3−pi0は完全な評価窓ではconversation +0.538、
tool-agent +0.483 input-token points。一方、label-observableな短い窓では
+0.143 / +0.198である。順位指標は固定candidate population上の別の推定対象であり、
これらを同一の時間窓・同一の分母と扱わない。

Phase 0.97の「best learnedがbest generic heapを上回る」は、事後選択した
arm/targetの平均では5/12 real cells。以前の「2%×4のみ」は訂正済み。
小さな平均差を統計的優位と呼ばず、固定pi3の結果とも混同しない。

## 3. 最近の解釈を更新した重要点

単一continuationで見た事後最大Qとexact-label selectorとの差を、そのまま
「labelに欠けた回収可能な行動情報」と読んではいけない。

[fresh-stream実験](counterfactual-randomness-findings.md)は固定40状態・16新streamで、
8本でactionを選び、残り8本で評価し、方向を交換した。

| 選択範囲 | 選択側の平均利得 | held-out側の平均利得 |
|---|---:|---:|
| 全candidate | +15,003.0 tokens | −267.8 tokens |
| exact-labelの最小tie内 | +13,567.2 tokens | +90.3 tokens |

これは元actionとの差で、40状態等重みの記述平均である。期待Qが等しい証明でも、
「93%は乱数」という分散・regretの確定分解でもない。8-streamで選択したactionの
利得が別streamへ移らなかった、という範囲に留める。

[保持時間診断](counterfactual-residence-findings.md)では、同じ大きさで600秒内の
自己reuseがゼロの2候補を比較した。629/640ペアで後続のL2 eviction/rejection列の差が最初のhit差より先に
生じ、両候補が600秒前に消えた497ペアのうち491ペアでは、その後も報酬差が続いた。
一方、保持byte-secondsと実現ΔQの単純相関はほぼゼロで、平均差の符号は状態間で混在する。
これは経路依存の記述であり、複雑なtrajectory機構が必要という因果的証明ではない。

現在の停止点はこの診断の完了。sampling機構対照・label表現の追加検証・新policyは未実施。
既存結果を保存したまま次の戦略を検討する。

## 4. 先行研究との重複をどう見るか

以下は2026-09-27時点の一次資料の確認範囲。査読済み論文、公開原稿、公式実装を区別する。
全関連文献を網羅した新規性証明ではない。ユーザーが言及した「3日前の論文」は
タイトル未特定のため、その論文との重複判定は保留する。

| 先行研究 | 確認した内容 | こちらへの意味 |
|---|---|---|
| [LAH / S4-FIFO](https://arxiv.org/html/2608.27975v1)、[OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/xia) | object-level predictionとmiss削減のobjective mismatch、細粒度feedbackの不安定性を論じ、cache-level設定学習を提案 | 「prediction精度≠cache性能」という一般命題だけでは独自性が弱い |
| [PARROT, ICML 2020 §4.2](https://arxiv.org/pdf/2006.16239) | learned policyが訪れる状態を収集し、DAggerでcompounding errorsへ対応 | 自己生成母集団・closed-loop・on-policy再学習の発想そのものは既知 |
| [Decision-aligned eviction-value prediction](https://github.com/SoroushVahidi/Augmented-caching) | 著者公開資料にfinite-horizon downstream miss cost、tie-aware exact-target oracle、continuation、DAggerの対照がある。著者表記はKBS投稿中 | 「action valueまで測った」だけでも独自性を確保できない。全結果の独立再現や採択確認はしていない |
| [同研究のtie-aware監査](https://github.com/SoroushVahidi/Augmented-caching/blob/main/reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md) | exact targetの固定tie-breakによる不利が別tie-breakでは維持されず、target固有の劣位という読みを訂正 | exact-label比較はtargetとtie semanticsを分ける必要がある。こちらのcontinuation cross-fitと同一実験ではない |
| [KVLearn公式実装](https://github.com/FastLM/KVLearn) | reuse predictionとcost-aware retentionを組み合わせる | temporal/structural featuresでKVを選ぶという入口にも先行例がある。論文全実験との同条件比較は未完了 |
| [KVCache Cache in the Wild](https://arxiv.org/html/2506.02634v5) | 実LLM workloadのreuse・lifespanを分析し、workload-awareな保持を扱う | reuse偏り・時間特性のcharacterizationだけでは差別化しにくい |

反復学習やcounterfactualという方法の名前ではなく、既存研究からは得られない
具体的な説明・成立条件・設計判断を貢献にする必要がある。

## 5. サービング基盤との対応とズレ

有限容量のKV保持は実基盤の関心と整合する。ただし、このsimulatorのexclusive L2は
すべてのpersistent tierを代表しない。

- [SGLang HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design)はGPU・host・storageの階層と複数の書込み方針を持つ。[radix eviction](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)も葉の制約と複数方針を扱う。こちらのL1 victimのみを受けるモデルとの対応を明示する必要がある。
- [vLLM prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)と[LMCacheのcache管理](https://docs.lmcache.ai/kv_cache_management/index.html)は、保持・再利用を実装上の具体的な判断として提供する。単に「LLM KVへ学習cacheを適用する」だけでは差別化にならない。
- [Dynamo v1.5](https://docs.nvidia.com/dynamo/reference/releases/v1-5-0)のように基盤側の構成も更新されている。接続先の実装を固定せずに、特定tier構成を一般形として主張しない。

実基盤への実装やI/O実験を今すぐ追加するという提案ではない。まずモデルの適用範囲を
正しく限定し、その範囲内で選択の機構を説明する。

## 6. 残る新規性候補と、それぞれに足りない証拠

### 候補A: 容量によって判断の種類が変わり、学習の効果が反転する

問いは「L1から流入するKVを選ぶとき、admissionとresident evictionのどちらが
支配的になるかが、予測の有効性を左右するか」。小容量でarrival rejectionが多いこと、
on-policy学習とarrival protectionが容量によって逆の効果を持つことが動機である。

**現在は仮説。** 異なるarmの結果を一本の因果説明にまとめてはいけない。
同じscore・比較可能な条件で判断機構の寄与を分離し、その説明が未使用条件でも
反転方向を予測できることが必要。単に「容量依存だった」では貢献は弱い。
L1の選別が本当に原因なら、一般single-tierからの差も説明する。

### 候補B: 予測器の限界と、予測を適用する選択機構の限界を分離する

履歴が有用stateを区別できないのか、候補抽出・admission・後続evictionがその利得を
失わせるのかを区別する。既存のcandidate-search診断はあるが、最新のcontinuation
不安定性までその診断だけで説明したわけではない。

**現在は未決。** 同じ予測器で機構を変えた際にrankingとutilityの対応が回復するかが
重要で、単にsample widthを増やした図では足りない。乱数を消して分散がゼロになっても、
予測が実utilityを説明するようになった証拠にはならない。

### 候補C: retention評価方法としての貢献

母集団・時間窓・tie-break・continuation・事後選択を揃えて評価すると、
policy選択や研究上の結論がどう変わるかを示す。

**現状では補助貢献。** 自分たちの評価上の問題を修正しただけでは方法論論文として弱い。
外部の公開手法でも評価・設計判断が変わる証拠が必要であり、既存論文の誤りを
確認せず主張してはならない。

## 7. 戦略判断と停止条件

現時点でのAstraの優先案はAを中心候補、Bを原因の切り分け、Cを補助貢献とすること。
これは実施承認や事前登録ではない。新policyの一般化へはまだ進まない。

継続の価値があるのは、容量による効果の反転という具体的な未説明現象が残るため。
一方、次のような結果なら主張を縮小する。

- 機構を揃えると現象が消えるなら、history一般の限界ではなく当該機構の問題として扱う。
- 特定arm・Mooncakeの特定cellだけなら、一般原理ではなく限定的なfailure caseとする。
- 既存研究の説明だけで結果が説明でき、新しい設計判断を導けないなら、研究範囲を
  characterization / replicationとして評価し直す。

今の成果だけで「強い新規性が成立した」「historyでは解けない」「semanticが必要」
「期待action valueが本質的に不可予測」とは結論しない。
試した条件の多さと、原因を特定したことは別である。

## 8. 次のChatGPTへの相談文

> 研究目的は有限容量persistent KVのexact-prefix retentionです。
> まずこの引継ぎとresearch-status、randomness/residence findingsを読んでください。
> prediction≠utility、on-policy learning、counterfactual valueの発想には先行研究があり、
> 単一streamの事後Q最大による強い解釈はfresh-stream実験で弱まりました。
> 容量による学習効果の反転、L1 filtering、admissionとresident evictionの関係について、
> 既存研究を超える説明を作れるかを厳しく検討したいです。
> 新policy・新featureを直ちに設計せず、最も強い代替説明と、それを区別する最小限の
> 検証、成立時の論文貢献、不成立時の縮小方針を提案してください。
> 検証済み結果と仮説を分け、まだ期待Qや一般化が証明されたとは扱わないでください。

読む順序:

1. この文書と[research-status](research-status.md)
2. [on-policy findings](onpolicy-learning-findings.md)と[Phase 1介入](phase1-intervention-findings.md)
3. [fresh-stream findings](counterfactual-randomness-findings.md)
4. [residence findings](counterfactual-residence-findings.md)
5. 必要に応じて[実験計画の履歴](experiment-plan.md)と各phaseの詳細
