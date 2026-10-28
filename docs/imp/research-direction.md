# Research direction

Status (28 October 2026): this page sets out what we will measure and how. The systems, the benchmark harness, its statistics and analysis, and the frozen memory are built (Phase 3, tasks 20 to 26), and the benchmark has been running since 28 October 2026, so there are no results here yet. Any numbers added later must come from the benchmark runs.

## The main question

> Can an autonomous experimentation layer improve the efficiency and effectiveness of conventional AutoML by learning from previous experiments, diagnosing model weaknesses, selecting appropriate improvement strategies, and allocating computational resources adaptively?

Put simply: on the same machine and with the same time limit, can our loop reach a good model faster, or reach a better one, than FLAML using the whole budget by itself?

## Research questions

- **RQ1:** Does choosing strategies from diagnoses reach a target score with less compute than running the AutoML engine alone for the same total budget?
- **RQ2:** Does the experiment memory (what worked on similar datasets) reduce the number of experiments needed on new datasets?
- **RQ3:** Does adaptive budget allocation (gain per unit of compute) give a better final score under a fixed budget than splitting the budget evenly?
- **RQ4:** How consistent are these effects across different kinds of datasets (size, imbalance, missingness, cardinality)?

## Hypotheses

- **H1:** The full system needs less time (median) than FLAML to reach the target score, with the same total budget.
- **H2:** Without the memory, more experiments are needed to reach the target, measured on datasets that were not used to build the memory.
- **H3:** Without strategy selection (random or fixed order), the best-score-over-time curve is lower.
- **H4:** At the same budget, the final score of the full system is not worse than FLAML's, while using fewer experiments.

Each hypothesis is tested with paired comparisons over datasets and seeds. If a hypothesis is not supported, we will say so.

## How strategies will be chosen

1. **Rules (Phase 2):** diagnosis to candidate strategies through a documented rule table, in a fixed priority order. Everything else is compared against this.
2. **Multi-armed bandits (Phase 3):** each strategy is an arm, and the reward is the change in the primary metric divided by the compute seconds. We will try UCB1 and Thompson sampling, possibly using the dataset meta-features as context.
3. **Memory (Phase 3):** find the nearest past datasets by meta-features and try first the strategies that gave the best reward there.
4. **Bayesian optimisation of strategy settings (optional):** for example Optuna over the SMOTE ratio or a feature-selection threshold, only if the simpler methods leave a measurable gap.
5. **Local LLM (optional, later):** only to explain decisions. It is evaluated separately and is not part of the main claims.

## The resource-aware objective

For experiment i, with primary score s_i and cost c_i (CPU or wall-clock seconds):

```text
utility_i = (s_i − best score before experiment i) / c_i      (sign flipped for loss metrics)
```

The resource manager decides how to split the total budget between experiments, and when to stop. For example, it stops when no remaining strategy is expected to have a utility above a threshold.

## Datasets

- Development: the four sample datasets in `data/sample` (Titanic, Breast Cancer Wisconsin, Palmer Penguins, California Housing) and six OpenML datasets (wilt, tic-tac-toe, vehicle, balance-scale, kin8nm, student_performance_por).
- Evaluation: six OpenML datasets (pc4, cylinder-bands, segment, MiceProtein, energy_efficiency, solar_flare). Both lists were fixed and committed in `research/datasets.json` on 27 October 2026, before any evaluation run: from the laptop-sized datasets of the OpenML-CC18 and OpenML-CTR23 suites, one evaluation and one development dataset were drawn with a fixed seed from each of six groups that cover the three tasks, imbalance, missing values and categorical columns. Datasets used to build the memory are not used for evaluation.
- Every dataset has to run on a laptop CPU within the chosen budgets.

## Protocol

- Budgets: total budgets of 2 and 5 minutes per run. A third budget of 10 minutes was planned; it was dropped on 27 October 2026 to fit the laptop (it would add about 20 hours to a grid of about 22). The ablations run at 5 minutes only.
- Seeds: 5 (1 to 5) for every combination of dataset, system and budget.
- Splits: for a given seed, every system gets the same outer train/test split and the same columns.
- Logged for every run (one row per cell, `results/benchmark/benchmark_cells.csv`): the test score and the other test metrics, wall-clock seconds, the number of experiments, the CPU cores, the strategies kept and the system's settings; each Session's record keeps every step. CPU seconds are not measured: all runs are on one machine, one at a time, so wall-clock time is the cost measure.
- Statistics: Wilcoxon signed-rank tests for pairs of systems over datasets, and the Friedman test with a Nemenyi post-hoc test for several systems at once (Demšar, 2006). Effect sizes and confidence intervals are reported, not only p-values.
- With two budgets, the "best-score-over-time curve" of H3 is the test score at 2 and at 5 minutes (fixed on 27 October 2026, before any evaluation run).

## What we measure

- **Predictive quality:** the primary metric on the outer test split (F1, ROC-AUC, R² or RMSE, depending on the task).
- **Efficiency:** time to reach a target score (the smallest budget at which a system reaches 99% of the best final score of any system on that dataset and seed), and the score at each budget.
- **Experimentation:** number of experiments and strategies needed to reach the target.
- **Cost:** wall-clock seconds, and utility per experiment (CPU seconds are not measured; see the protocol).
- **Convergence:** how much each extra experiment adds, and where the system stops.
- **Consistency:** average rank across datasets, spread across seeds, and the worst-case gap to the best system.

## Baselines and ablations

1. Manual baseline: one fixed, sensible pipeline (for example default LightGBM or logistic regression). This is the floor, with no search.
2. FLAML alone with the full budget: the engine we build on.
3. AutoGluon (CPU only, same budget): a strong outside reference.
4. AutoPilot-ML without experiment memory: tests H2.
5. AutoPilot-ML without strategy selection (random or fixed order): tests H3.
6. The full AutoPilot-ML: tests H1 and H4.

As built (tasks 22 and 23): 1 `manual` (LightGBM with its library defaults), 2 `flaml`, 3 `autogluon` (AutoGluon-Tabular 1.6.3, preset `medium_quality`, in its own environment), 4 `autopilot_rules` (without the memory the learned choice keeps the rule order, so this is the same Session), 5 `random_choice`, 6 `autopilot_full` (Thompson sampling with the frozen memory). Two more ablations remove one part each from the full system: `no_critic` (random valid changes, no Diagnoses) and `no_resource_manager` (equal shares of the time).

## Studies of the components

Before the parts are trusted inside the loop, each one is measured on its own. Details are in the spec (`.kiro/specs/autopilot-ml/`, Requirements 6, 10 and 25):

- **Critic study:** precision and recall of each diagnosis on synthetic datasets with planted problems, plus clean datasets for false alarms. For overfitting and unreliable validation, the ground truth comes from a large fresh sample (20,000 rows) drawn from the same generator. Thresholds are tuned on one set of seeds and reported on another.
- **Judge study:** the false-keep rate and the power of the Judge, on pairs of setups whose true difference is measured on the fresh sample, for several values of alpha and of the number of fold repeats. Defaults are chosen on one set of seeds and reported on another.
- **Judge agreement study (Phase 3):** how often the cheap Judge (refits on the Session's 15 folds) gives the same verdict as a strict comparison that re-runs full FLAML searches on several inner splits, with Cohen's kappa and the cost of each method.

## Things that could make the results misleading

- **Noise from time-limited search:** FLAML and AutoGluon stop on time, not on trial count. We use several seeds and paired comparisons.
- **Hardware differences:** every system runs on the same machine, one at a time, with a fixed `n_jobs`.
- **Memory leaking into evaluation:** the memory is built only from development datasets, and evaluation datasets are never seen beforehand.
- **Choice of metric:** the primary metric is fixed before the runs, and the other metrics are reported as well.
- **Tuning the system to the benchmark:** thresholds and rules are frozen before the final evaluation.
