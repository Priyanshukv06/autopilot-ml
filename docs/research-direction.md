# Research Direction

Status: **Planned.** This document defines what will be measured and how. No results are reported here because the autonomous layer does not exist yet. Every number in a future version of this document must come from the benchmark harness.

## 1. Central question

> Can an autonomous experimentation layer improve the efficiency and effectiveness of conventional AutoML by learning from previous experiments, diagnosing model weaknesses, selecting appropriate improvement strategies, and allocating computational resources adaptively?

## 2. Research questions

| ID | Question |
|---|---|
| RQ1 | Does diagnosis-driven strategy selection reach a given target score with less compute than running the AutoML engine alone for the same total budget? |
| RQ2 | Does experiment memory (what worked on similar datasets) reduce the number of experiments needed on new datasets? |
| RQ3 | Does adaptive budget allocation (utility = improvement per unit compute) improve final performance under a fixed budget compared with uniform allocation? |
| RQ4 | How robust are these effects across dataset types (size, imbalance, missingness, cardinality)? |

## 3. Hypotheses

- **H1**: Full AutoPilot-ML achieves a lower median time-to-target than FLAML with the same total budget.
- **H2**: Removing experiment memory increases the number of experiments to reach the target (memory helps), measured on datasets not used to build the memory.
- **H3**: Removing strategy selection (random or fixed strategy order) lowers the anytime performance curve.
- **H4**: Final predictive performance of the full system is not worse than FLAML's at equal budget (non-inferiority), while using fewer experiments.

Each hypothesis will be tested with paired comparisons over datasets and seeds (see §7). A hypothesis that is not supported will be reported as such.

## 4. Strategy selection: planned approaches

1. **Rule-based** (Phase 2): diagnosis → candidate strategies by a documented rule table and fixed priority. This is the reference point.
2. **Multi-armed bandits** (Phase 3): each strategy is an arm; reward = Δ(primary metric) / compute seconds; UCB1 and Thompson sampling, optionally contextual on dataset meta-features.
3. **Meta-learning from memory** (Phase 3): k-nearest datasets in meta-feature space, prioritise strategies with the best observed reward on them.
4. **Bayesian optimisation over strategy parameters** (optional): e.g. Optuna over resampling ratio or feature-selection threshold, only if the simpler approaches leave a measurable gap.
5. **Local LLM reasoning** (optional, Future Work): explanation only, evaluated separately; it is not part of the core claims.

## 5. Resource-aware objective

For experiment *i* with primary score *sᵢ* and cost *cᵢ* (CPU-seconds or wall-clock seconds):

```text
Utility_i = (s_i − s_best_before_i) / c_i        (sign-adjusted for loss metrics)
```

The resource manager decides (a) how to split the total budget between experiments and (b) when to stop (e.g. when the expected utility of every remaining strategy falls below a threshold).

## 6. Datasets

- Development: the four bundled samples (Titanic, Breast Cancer Wisconsin, Palmer Penguins, California Housing).
- Evaluation: a suite of public tabular datasets from OpenML (for example tasks from the OpenML-CC18 suite and the AutoML Benchmark), selected to cover small/large, balanced/imbalanced, low/high missingness, numeric/categorical-heavy datasets. Exact task ids will be fixed and committed before any evaluation run, and datasets used to build the experiment memory will be disjoint from the evaluation datasets.
- Constraint: every dataset must run on a laptop CPU within the chosen budgets.

## 7. Protocol

- Budgets: several fixed total budgets (for example 2, 5 and 10 minutes) per dataset.
- Seeds: at least 5 seeds per (dataset, system, budget).
- Splits: identical outer train/test splits across systems for a given seed.
- Logged per run: final test score, anytime curve (best score vs elapsed time), number of experiments, CPU-seconds, wall-clock time, strategies chosen.
- Statistics: Wilcoxon signed-rank tests for pairwise comparisons over datasets, Friedman test with a post-hoc Nemenyi test for multiple systems (Demšar, 2006); effect sizes and confidence intervals reported alongside p-values.

## 8. Metrics

| Category | Metric |
|---|---|
| Predictive | Primary metric on the outer test split (F1 / ROC-AUC / R² / RMSE depending on task) |
| Efficiency | Time-to-target (first time the best score reaches a fraction of the final best of a reference system), area under the anytime curve |
| Experimentation | Number of experiments to target, number of strategies tried |
| Cost | CPU-seconds, wall-clock seconds, utility per experiment |
| Convergence | Score improvement per additional experiment, stopping point |
| Robustness | Average rank across datasets, variance across seeds, worst-case gap to the best system |

## 9. Baselines and ablations

| # | System | Purpose |
|---|---|---|
| 1 | Manual baseline (fixed sensible pipeline, e.g. default LightGBM / logistic regression) | Floor without search |
| 2 | FLAML alone, full budget | The engine AutoPilot-ML builds on |
| 3 | AutoGluon (CPU, matched budget) | Strong external AutoML reference |
| 4 | AutoPilot-ML without experiment memory | Tests H2 |
| 5 | AutoPilot-ML without strategy selection (random or fixed order) | Tests H3 |
| 6 | Full AutoPilot-ML | Tests H1, H4 |

## 10. Threats to validity

- **Time-based search noise**: FLAML and AutoGluon are budget-driven; mitigated by multiple seeds and paired comparisons.
- **Hardware variance**: all systems run on the same machine, sequentially, with fixed `n_jobs`.
- **Memory leakage**: memory built only from development datasets; evaluation datasets are never seen before evaluation.
- **Metric choice**: results reported for the primary metric chosen before running, plus secondary metrics.
- **Overfitting to the benchmark**: thresholds and rules frozen before the final evaluation runs.
