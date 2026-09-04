# Primer: machine learning for transaction monitoring

Companion to [primer-aml-domain.md](primer-aml-domain.md) (the domain) and the
[technical report](report.md). This primer explains the ML concepts the
workbench uses well enough to reason about them, with mini-derivations where
they are cheap. Every project number is reproduced from artifacts in
`data/reports/`; hand arithmetic is marked as such.

## 1. Class imbalance, and why it dominates everything

AML data is extremely imbalanced: fraud and laundering are rare events.
Consequences that drive every later choice:

- Overall accuracy is useless. With a 1% positive base rate, predicting
  "licit" for everything scores 99% while catching nothing.
- The positive class contributes most of what matters: each missed laundering
  account or forged-looking transaction is a real cost, and each false
  positive is an analyst investigation.
- Metrics must therefore be read **per class** and at **operating points**,
  not as single scalars. Hence precision, recall, PR-AUC, precision@k.

**Base rates in this project.** Elliptic's labeled transactions number 46,564,
of which 4,545 are illicit: $4545/46564 \approx 0.0976$, about 9.6%. The rate
varies across the 49 time steps by roughly an order of magnitude
(`docs/figures/base_rate_curve.png`). HI-Small's evaluation window holds 60
laundering accounts out of 515,080: $60/515080 \approx 0.000116$, about 1 in
8,600. Weeks of operations can pass between hits; that is what "imbalanced"
means operationally.

## 2. Confusion matrix, precision, recall

At a fixed threshold $t$, scores above $t$ are called positive:

| | actually illicit | actually licit |
|---|---|---|
| predicted illicit | TP | FP |
| predicted licit | FN | TN |

- **Precision** $= TP/(TP+FP)$: of what we flagged, how much was real.
- **Recall** $= TP/(TP+FN)$: of what was real, how much did we catch.
- Precision has an operational unit (alerts, investigations, cost per hit);
  recall is coverage of criminals. Tuning a monitor always prices this trade.

## 3. PR-AUC versus ROC-AUC

Both numbers summarize the whole threshold range. ROC-AUC is the probability
that a random positive scores above a random negative. It is threshold-free
and probability-faithful, but it is insensitive to the base rate: a fixed
false-positive rate $f$ admits $f \times (N-P)$ false positives, and when $P
\ll N$ the precision implied by $f$ can collapse without ROC-AUC noticing.

PR-AUC is the area under the precision-recall curve, which is constructed
directly in the units that matter here: precision is the axis of operational
cost. Because both axes depend on $P$, PR-AUC is not comparable between
datasets with different base rates, and small calibrations move it more than
ROC-AUC. Rule of thumb for a monitor: ROC-AUC says "ranking plausible",
precision@k and PR-AUC say "worth its queue".

**Why the project reports both.** ROC-AUC exists to keep comparability with
published benchmarks, which is why the README table carries it; PR-AUC was
predeclared as the challenger metric, in writing, before any model was
trained. Declaring the metric first prevents tuning the evaluation to the
result (a garden of forking paths; see Section 9).

**Why micro-F1 is suppressed.** At base rates this low, micro-averaged F1 at
a 0.5 threshold is dominated by the licit class: the licit majority
contributes most of the arithmetic, and a model can show a respectable
micro-F1 while quietly missing nearly all illicit transactions.
The report reads F1 per time step, next to precision and recall, at a fixed
threshold. It is not a headline.

## 4. Calibration

A probability estimate is **calibrated** when predicted probability equals
empirical frequency: among alerts scored 0.9, about 90% should be real. Two
monitors with identical ROC-AUC can differ enormously in calibration, and
calibration is what makes threshold selection and cost arithmetic meaningful:
cost per true positive at a threshold assumes the score is a rate rather than a bare
rank. Isotonic or Platt calibration can fix a miscalibrated model, but the
deeper point is principled: a fuzzy score is a ranking; a calibrated one is a
price. GraphSAGE's soft logits and a threshold-free AUC metric both dodge the
question; precision@k forces it.

## 5. The model family

### 5.1 Logistic regression

$\log \dfrac{p}{1-p} = \beta_0 + \beta^\top x$, fitted by maximum likelihood.
Linear in the features; it cannot express interactions or thresholds in
feature space without engineered help. Its PR-AUC 0.292 against random
forest's 0.798 (`data/reports/baselines_metrics.json`) is exactly what this
limitation predicts: monitoring signal lives in nonlinear combinations
(amount ratios, degree concentration, time-concentrated bursts), not single
directions.

### 5.2 Random forest (RF)

An ensemble of decision trees, each grown on a bootstrap sample and a random
feature subset; predictions are averaged. Two mechanisms matter: averaging
reduces variance, and feature subsampling decorrelates trees. It handles
nonlinearities and interactions natively. Weakness under imbalance: with
class priors untouched, trees grow deep and pay more attention to the
dominant class; the class-weight and threshold choices matter.

### 5.3 Gradient-boosted trees, and what LightGBM does

Boosting grows trees sequentially; each new tree fits the current residuals
(errors of the ensemble so far). GBDT performs gradient descent in function
space: each step moves the ensemble toward the negative gradient of the loss,
and the result typically gives state-of-the-art performance on tabular data.
**LightGBM** accelerates the standard algorithm:

- **histogram-based splits**: binning feature values into a few hundred bins
  makes split finding O(bins) instead of O(n) per feature pass.
- **leaf-wise growth**: grows the leaf with the largest gradient sum rather
  than level-by-level, reaching the same tree mass with fewer, deeper nodes.
- **GOSS** (gradient-based one-side sampling) optionally keeps
  large-gradient observations, those still being learned, and samples the
  rest.

Representation power + residual-fitting depth is why the challenger reached
PR-AUC 0.907 (`data/reports/challenger_metrics.json`), drawn from 3 seeds
(0.9072, 0.9100, 0.9041). No single hyperparameter produced it; the model's
expressiveness reaches combinations of features that shallow baselines miss.

## 6. Temporal validation and leakage

### 6.1 Why random splits are wrong here

Transactions and accounts interact through time: today's strategy is built
from yesterday's flows, and yesterday's outcome affects today. A random
row-split mixes future into training, producing scores that a deployable
system could not realize. This failure mode is called **look-ahead leakage**
(or data leakage through time) and it crudely inflates metrics.

**Strict-inductive** is the discipline against it: at an as-of time $t$, every
feature, every training row, and every graph edge used must come from
(strictly before) $t$. In the project, the one-sentence form appears in the
artifacts: "no test-period adjacency during training or scoring".

### 6.2 Walk-forward

**Walk-forward validation** (expanding-window temporal CV) refits the model at
each test step using only data before it, then scores that step, so the
evaluation traces are the same object a production retrain would execute.
Elliptic's protocol: train on labeled steps 1-34 for challenger selection,
then walk forward through the test steps 35-49, refitting with an expanding
window at each step. The validation artifact
(`data/reports/validation_metrics.json`) holds per-step precision, recall, F1
at threshold 0.5.

### 6.3 The leakage discipline as code and gates

The pipeline does not rely on reviewers noticing leakage. Documented
mechanisms:

- **Temporal splits only**, encoded in one helper
  (`split_temporal` in `src/aml_workbench/gnn.py` and sibling modules).
- **As-of feature construction** in triage: features for each alert's window
  end strictly before `window=` cutoffs.
- **Seam tests** assert exit-code behavior of the commands; the look-ahead
  proof is red-then-green (see P16's lookahead test as prior art).
- **Gates C1-C5** fail-closed: a gate violation exits non-zero before any
  output, so an invalid run cannot emit plausible-looking artifacts.

A subtlety specific to graphs: **transductive** GNN protocols often let
test-period edges into message passing. That inflates GNN metrics by a large
margin (protocol audits such as Maganti 2026 in the artifacts; see also
Weber et al. 2019). The GraphSAGE baseline here refuses it wholly, so its
honest loss to the GBM (Section 7) is the expected finding, not a tuning
failure.

### 6.4 Seeds and why ≥3

One random seed is one draw; model variance across seeds is part of the
result. The ≥3-seed rule standardizes against seed luck: the challenger
carries 3 seeds (42, 7, 2026) with bagging_fraction 0.8 making the seed
operative ("per-seed metrics are now independent draws", the artifact notes).
Walk-forward refits hold one fixed seed because the seed rule applies at
challenger selection, and tripling 15 refits adds cost with no decision
riding on it. That reasoning, written down in the artifact, is the right way
to justify a deviation.

## 7. Why the graph baseline lost, and what to conclude

A **graph neural network (GNN)** operates on transaction structure directly.
**Message passing**: each node's embedding is updated from a weighted
aggregation of its neighbors' embeddings, layer by layer; a k-layer GNN lets
a node compute with its k-hop neighborhood. **GraphSAGE** samples and
aggregates the neighborhood, rather than holding all nodes' embeddings in a
transductive memory, which makes inductive scoring on new nodes possible.

Track A result: strict-inductive GraphSAGE achieves mean PR-AUC 0.6009 vs
LightGBM's 0.9071 under identical protocol and edges
(`data/reports/gnn_comparison.json`). Concluding "GNNs do not work for AML"
from this is wrong for at least three reasons, and the report says so:

1. One architecture, one size, no tuning: the baseline exists for the
   comparison's integrity, not as a limit.
2. The engineered features the GBM consumes (degree, concentration, flags)
   already encode the first-order graph signal; a GNN's edge comes from
   letting raw structure propagate.
3. Class imbalance interacts badly with message passing: aggregating
   neighbors with an overwhelming licit-majority pattern smooths away
   minority signal without deliberate counter-strategy.

The honest conclusion, recorded before the race: GBM-on-engineered-features
is the selected model for this protocol and this data; GNNs remain an active
research direction for AML.

## 8. Feature attribution with SHAP

**SHAP** (SHapley Additive exPlanations; Lundberg and Lee, 2017) attributes a
prediction to features through Shapley values from cooperative game theory: the
fair share of the prediction owed to each feature, averaged over all
subsets of features. Additivity means
$\sum_i \phi_i = f(x) - \mathbb{E}[f]$: attributions sum exactly to the gap
between this prediction and the base rate, which is why SHAP numbers audit
cleanly rather than merely suggesting importance. Caveats worth knowing:
values depend on the background distribution used, correlated features share
credit in ways that can mislead substitution readings, and the exact
Shapley computation is exponential; TreeSHAP (Lundberg et al. 2020) makes GBDT
attribution polynomial.

The project computes SHAP over a seeded 20,000-row test subsample, written to
`data/reports/shap_summary.md`. It also serves the interview-grade question
"what does the model actually look at" without re-training anything.

## 9. Drift and the population stability index

**Data drift** is a change in the input distribution between development and
deployment. A related but distinct failure is **concept drift**, a change in
the relationship between inputs and labels. A monitor measured only at launch
can degrade quietly as either occurs, so an operational system measures drift
continuously.

The **PSI** (population stability index), the industry-standard drift metric
for scorecards, compares binned distributions reference $R$ (training) and
current $C$ (test):

$$\text{PSI} = \sum_i (R_i - C_i)\,\ln\frac{R_i}{C_i}$$

where $R_i, C_i$ are the share of the reference and current populations in
bin $i$. It is the Jeffreys divergence between the two binned distributions,
that is, the sum of the two directional KL divergences, so it penalizes
displacement in either direction. A common convention treats PSI below 0.10
as stable, 0.10-0.25 as watch, and above 0.25 as breach.

**Hand-worked example.** One continuous feature binned into two buckets.
Reference shares $R = (0.8, 0.2)$; current shares $C = (0.5, 0.5)$.

- bin 1: $(0.8 - 0.5)\ln(0.8/0.5) = 0.3 \times 0.4700 = 0.1410$
- bin 2: $(0.2 - 0.5)\ln(0.2/0.5) = (-0.3) \times (-0.9163) = 0.2749$
- PSI $= 0.1410 + 0.2749 = 0.4159 > 0.25$: breached.

The real artifact (`data/reports/drift_metrics.json`) holds this arithmetic
for 172 features: 42 breach the 0.25 threshold, 98 are in the 0.10-0.25 watch
band. The most drifted are `f137`/`f136`/`f139` at PSI near 11.6, an order of
magnitude above threshold, and the excluded categorical `louvain_community`
is monitored by category-share stability instead, because quantile-binned
PSI is meaningless on a categorical column. Measured breach that exceeds
belief by this much reflects the synthetic generator's known property of
shifting features across time; the honest reading is that the temporal
protocol is doing its job and the drift flag needs interpretation, not
silencing.

## 10. Reading list and where to go next

- **Weber et al. (2019), arXiv:1908.02573** — the Elliptic paper; states the
  benchmark, the dataset, and the protocol caveat this project follows.
- **Lundberg & Lee (2017), "A Unified Approach to Interpreting Model
  Predictions", NeurIPS** — SHAP foundations; the Shapley section is the part
  to read.
- **Lundberg et al. (2020), Nature Machine Intelligence, "From local
  explanations to global understanding with explainable AI for trees"** —
  TreeSHAP for GBDT; polynomial-time exact attribution.
- **Ke et al. (2017), "LightGBM: A Highly Efficient Gradient Boosting
  Decision Tree", NeurIPS** — histogram binning, GOSS, leaf-wise growth; the
  three mechanisms in Section 5.3 come from here.
- **Maganti et al. (2026), GNN protocol audit** — the paper cited in
  `data/reports/gnn_metrics.json` for transductive-protocol inflation; read
  next to the Track A comparison.
- **Breiman (2001), "Random Forests", Machine Learning** — the averaging and
  decorrelation argument in Section 5.2, in the original.
- **Chapter 7, "The Elements of Statistical Learning"** (Hastie, Tibshirani,
  Friedman) — the standard reference depth for boosting; free PDF exists.
- **scikit-learn's model evaluation docs** + `lightgbm` parameter docs —
  the engineering layer; read parameter names against the sweep config in
  `data/reports/tuning_params.json`.

Ordering that pays: Weber 2019 (anchoring) → Ke 2017 (challenger's
mechanism) → Lundberg & Lee (attribution) → ESL chapter 7 (depth).
