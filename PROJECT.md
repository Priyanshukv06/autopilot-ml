# AutoPilot-ML: Autonomous Machine Learning Experimentation & Self-Optimization Framework

Master project document · B.Tech Computer Science major project

| Item | Status |
|---|---|
| Phase 1 baseline (profiling, task detection, FLAML, evaluation, MLflow, dashboard) | **Implemented** |
| Autonomous layer (planner, critic, strategy engine, memory, resource manager) | **Planned** |
| Benchmark results | **None yet**: no performance claims are made in this document |

---

## Abstract

Automated machine learning (AutoML) systems search over models and hyperparameters, but the surrounding experimentation loop — diagnosing why a model is weak, choosing what to change next, and deciding where to spend the remaining compute — is still largely manual. AutoPilot-ML proposes an autonomous experimentation layer on top of an existing AutoML engine (FLAML). Given a tabular dataset and a compute budget, the system will profile the data, run an initial AutoML experiment, diagnose weaknesses (overfitting, underfitting, class imbalance, feature problems, wasted compute), select improvement strategies, run follow-up experiments, and store the outcome of every strategy together with dataset characteristics so that later runs on similar datasets can prioritise strategies that worked before. The first milestone, implemented and described here, is a fully local baseline: CSV ingestion, dataset profiling, task detection, FLAML-based search with a held-out evaluation, basic explainability, MLflow tracking and an experiment history. The research contribution will be evaluated against FLAML, AutoGluon and ablations of the proposed system on predictive performance, time-to-target, number of experiments, compute cost, convergence and robustness.

## 1. Introduction

Tabular data dominates applied machine learning in industry, healthcare, finance and public administration. For such data, gradient-boosted trees and well-tuned linear models remain strong, and AutoML tools such as auto-sklearn, TPOT, AutoGluon and FLAML can produce competitive models with little manual effort.

What these tools automate is the *inner* loop: selecting an algorithm and its hyperparameters for a fixed dataset representation and a fixed objective. A practitioner's *outer* loop is broader. After a first model they ask: Is it overfitting? Is accuracy hiding a minority class that is never predicted? Is an identifier column leaking? Would dropping a noisy high-cardinality column help? Was the last ten minutes of search worth it? They then change the data, the features, the objective or the search space and run again. That loop is where most experimentation time goes, and it is mostly undocumented and unlearned: the knowledge gained on one dataset is rarely reused on the next.

AutoPilot-ML treats the AutoML engine as an optimisation component and focuses on automating and learning the outer loop.

## 2. Motivation

1. **Compute is the constraint.** Students and small teams typically have a laptop CPU and no budget for paid inference APIs or cloud GPUs. Spending compute where it helps most matters more than adding more search.
2. **Diagnoses follow patterns.** A large train/validation gap, a 92 % majority class, a column unique per row, or a feature with 70 % missing values are detectable from data and metrics that an AutoML run already produces.
3. **Strategies are reusable.** "Class weighting helped on a severely imbalanced dataset with mostly numeric features" is knowledge that should transfer to the next similar dataset.
4. **Transparency.** A system that states its diagnosis and the reason for each next experiment is easier to trust, teach with and evaluate than an opaque search.
5. **No hype dependency.** The decision layer must be measurable and work offline. Large language models are not required for any core function; at most they are a later, optional explanation aid.

## 3. Problem statement

Given:

- a tabular dataset *D* with a target column *y* (selected by the user in the current version),
- a task *T* ∈ {binary classification, multiclass classification, regression},
- a primary metric *m*,
- a computational budget *B* (wall-clock time, number of experiments, CPU cores),

find a pipeline *p* (data treatment + features + model + hyperparameters) that maximises the expected test performance *m(p)*, **and** reach a good pipeline with as little of *B* as possible.

The baseline addresses the first part with a single AutoML run. The research system addresses both parts with a sequence of experiments *e₁, e₂, …, eₖ*, where each *eᵢ₊₁* is chosen from diagnoses of *e₁…eᵢ* and from a memory of experiments on other datasets.

## 4. Existing approaches

| Family | Representative work | What it automates |
|---|---|---|
| Manual / expert-driven | — | Everything, by hand |
| Grid and random search | Bergstra & Bengio (2012) | Hyperparameters, no model of the response surface |
| Bayesian optimisation | SMAC (Hutter et al., 2011); TPE (Bergstra et al., 2011); Spearmint (Snoek et al., 2012) | Sample-efficient hyperparameter search |
| Bandit / multi-fidelity | Successive Halving (Jamieson & Talwalkar, 2016); Hyperband (Li et al., 2018); BOHB (Falkner et al., 2018) | Early stopping of poor configurations |
| Combined algorithm selection + HPO (CASH) | Auto-WEKA (Thornton et al., 2013); auto-sklearn (Feurer et al., 2015, 2022) | Model choice + hyperparameters, meta-learned warm starts, ensembles |
| Evolutionary pipeline search | TPOT (Olson et al., 2016) | Pipeline structure via genetic programming |
| Stacking / ensembling | AutoGluon-Tabular (Erickson et al., 2020) | Multi-layer stack ensembles with strong defaults |
| Cost-frugal search | FLAML (Wang et al., 2021); CFO (Wu et al., 2021); BlendSearch (Wang et al., 2021) | Low-cost search that accounts for trial cost |
| Meta-learning | Vanschoren (2018); Feurer et al. (2015a) | Transfer of configurations between datasets |
| LLM-assisted data science | CAAFE (Hollmann et al., 2023); MLAgentBench (Huang et al., 2024) | Feature engineering suggestions, agentic experimentation |

## 5. Limitations of conventional AutoML

1. **No explicit diagnosis.** Search reports a best score, not *why* the model is weak or what kind of change could help.
2. **Fixed problem formulation.** Data treatment (resampling, column removal, encoding of high-cardinality features) is mostly outside the search space or fixed by defaults.
3. **Accuracy-centric defaults.** Imbalanced targets can produce high accuracy and useless minority-class recall unless the user picks a better metric.
4. **Limited cross-dataset learning.** Apart from meta-learned warm starts (auto-sklearn), most tools start every dataset from scratch and do not record which interventions helped.
5. **Static budget use.** The budget is spent on one search; there is no decision about whether more search, a different strategy, or stopping is the best use of the remaining compute.
6. **Opaque process.** Users see a leaderboard; the sequence of decisions is not exposed as an explainable experiment log.

## 6. Proposed system

AutoPilot-ML wraps an AutoML engine in an experiment loop:

1. **Dataset intelligence**: profile the data and detect the task (implemented).
2. **Experiment planning**: choose the next experiment configuration from the profile, memory and remaining budget (planned).
3. **AutoML core**: run the engine (FLAML today, AutoGluon as a comparison engine later) (implemented for FLAML).
4. **Evaluation**: metrics, errors and resource usage on a held-out split (implemented).
5. **Critic**: structured diagnosis of the experiment (planned).
6. **Strategy engine**: candidate improvement strategies ranked by expected utility (planned).
7. **Experiment memory**: what was tried, on what kind of data, at what cost, with what effect (planned; the baseline already writes the raw records).
8. **Resource manager**: allocate time, experiments and CPU across the loop and decide when to stop (planned).

## 7. Architecture

```text
                         USER
                           │
                           ▼
                    Dataset Upload                       [implemented]
                           │
                           ▼
                 ┌──────────────────┐
                 │ Dataset Profiler  │                    [implemented]
                 └─────────┬────────┘
                           ▼
                 ┌──────────────────┐
                 │ Task Detection   │                     [implemented]
                 └─────────┬────────┘
                           ▼
                ┌─────────────────────┐
                │ Experiment Planner  │                   [planned]
                └──────────┬──────────┘
                           ▼
                    ┌────────────┐
                    │ AutoML Core│                        [implemented: FLAML]
                    └─────┬──────┘
                          ▼
                     Evaluation                           [implemented]
              ┌───────────┼───────────┐
              ▼           ▼           ▼
           Metrics      Errors     Resources              [implemented: recorded]
              └───────────┼───────────┘
                          ▼
                       Critic                             [planned]
                          ▼
                 Strategy Engine                          [planned]
              ┌───────────┼───────────┐
              ▼           ▼           ▼
          Features      Model        Data                 [planned]
              └───────────┼───────────┘
                          ▼
                    Next Experiment                       [planned]
                          ▼
                  Experiment Memory                       [planned; raw records exist]
                          └──────────────► Next Cycle
```

Software layers (current code): Streamlit UI (`app/main.py`, `app/components/`) → services (`app/services/`) → core (`app/core/`) → external libraries (FLAML, scikit-learn, MLflow) → local storage (`experiments/`, `models/`, `mlruns/`). Services do not depend on the UI. See `docs/architecture.md` for class-level detail and the sequence diagram.

## 8. Detailed modules

### 8.1 Dataset profiler — implemented (`app/services/profiler.py`)

- **Input**: `pandas.DataFrame`, optional target and task.
- **Output**: `DatasetProfile` with rows, columns, dtypes, numerical/categorical/datetime features, missing values and ratios, duplicate rows, unique counts, constant / identifier-like / high-cardinality / high-missingness columns, class distribution, `ClassBalance` (majority and minority share, imbalance ratio, severity: none < 1.5 ≤ mild < 3 ≤ moderate < 9 ≤ severe), regression target summary, human-readable warnings and `meta_features()`.
- **Design**: no Streamlit dependency, JSON-serialisable, logged as an MLflow artifact.

### 8.2 Task detector — implemented (`app/services/task_detector.py`)

- Non-numeric or boolean targets: 2 classes → binary, otherwise multiclass.
- Numeric targets: 2 distinct values → binary; integer-valued with ≤ 20 distinct values → multiclass (medium/low confidence, with a suggestion to override if the values are ordinal or counts); otherwise regression.
- `validate()` rejects impossible overrides (regression on text, binary with ≠ 2 classes, multiclass with < 3 classes) and warns about missing targets and classes with fewer than two samples.
- `target_candidates()` describes every column's suitability as a target. `suggest_target()` is a transparent placeholder (generic keyword match, else last column); automatic target detection is future work.

### 8.3 AutoML runner and FLAML engine — implemented (`app/services/automl_runner.py`)

1. Validate configuration (target exists, excluded columns exist, budget 5–3600 s, test size 0.05–0.5, at least 20 labelled rows).
2. Drop rows with a missing target; separate *X* and *y*; remove excluded columns.
3. Detect or validate the task; validate the primary metric for the task.
4. Stratified train/test split (non-stratified with a warning if a class has < 2 samples).
5. `FLAMLEngine.fit`: FLAML `AutoML.fit` with the mapped metric (`f1` → `macro_f1`, `roc_auc` → `roc_auc_ovr` for multiclass), time budget, seed, `eval_method="auto"`, `log_type="all"`, `retrain_full=True`, `early_stop=True` and FLAML's own MLflow logging disabled.
6. Capture best estimator, best configuration, best validation score (converted from FLAML's loss), training time, trial history (from FLAML's log) and per-estimator leaderboard.
7. Evaluate on test and train splits; compute a trivial baseline; permutation importance.
8. Save `models/EXP-###/model.pkl` (model + feature columns + target + task + classes) and artifacts; log to MLflow; write `summary.json`.

`AutoMLEngine` is a small protocol; any backend returning an `EngineResult` can replace FLAML (tested with a scikit-learn logistic regression engine).

### 8.4 Evaluator — implemented (`app/services/evaluator.py`)

- Classification: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, log loss; macro averaging for multiclass; positive class = last sorted label for binary.
- Regression: MAE, MSE, RMSE, R².
- Selectable primary metrics: binary {F1, ROC-AUC, accuracy, log loss}; multiclass {macro F1, accuracy, macro ROC-AUC (one-vs-rest), log loss}; regression {R², RMSE, MAE, MSE}.
- `baseline_metrics`: majority-class or training-mean predictor on the same test split.
- `metric_warnings`: warns that accuracy is misleading when imbalance is moderate or severe.
- `feature_importance`: permutation importance with the primary metric on the test split (≤ 2000 rows, 5 repeats).

### 8.5 Experiment tracker — implemented (`app/services/experiment_tracker.py`)

- Atomic allocation of experiment ids (`EXP-001`, `EXP-002`, …).
- MLflow runs through `MlflowClient` against a local SQLite store, with parameters (dataset, target, task, metric, budget, seed, engine version, best hyperparameters, dataset meta-features), metrics (test/train/baseline metrics, validation score, timings, number of trials) and artifacts.
- JSON index (`experiments/EXP-###/summary.json`) read by the history view.

### 8.6 Streamlit dashboard — implemented (`app/main.py`, `app/components/`)

Seven sections: Dataset, Target, Experiment Configuration, Run AutoML (background thread with progress), Results (cards, test metrics with baseline deltas, train vs test, leaderboard, search progress), Explainability (permutation and native importance, confusion matrix, ROC curve, residual plots, best configuration), Experiment History (table, score chart, MLflow command).

## 9. Current implementation

| Aspect | State |
|---|---|
| Environment | Conda env `automl`, Python 3.11, dependencies pinned in `requirements.txt` |
| Data | CSV only; four bundled public sample datasets |
| Tasks | Binary, multiclass, regression |
| Engine | FLAML 2.7.0 with its default learners (LightGBM, XGBoost, random forest, extra trees, linear models, …) |
| Evaluation | Held-out test split + FLAML internal validation + trivial baseline |
| Explainability | Permutation importance, native importance, confusion matrix, ROC, residuals |
| Tracking | Local MLflow (SQLite) + JSON index |
| Tests | pytest suite covering profiler, task detector, runner (with real short FLAML runs), evaluator, failure handling and a custom engine |
| Not implemented | Planner, critic, strategy engine, memory, resource manager, SHAP, automatic target detection, feature engineering, resampling |

## 10. Proposed modules (Planned)

| Module | Interface sketch | Evaluation methodology |
|---|---|---|
| `experiment_planner.py` | `plan(profile, memory, budget_state) -> ExperimentConfig` | Number of experiments and time to reach target vs a fixed plan |
| `critic.py` | `diagnose(result, profile) -> list[Diagnosis]` with type, severity, evidence | Precision/recall of diagnoses on synthetic datasets with injected problems (known ground truth) |
| `strategy_engine.py` | `propose(diagnoses, memory) -> list[RankedStrategy]` | Regret and time-to-target vs random and fixed-order strategy selection |
| `experiment_memory.py` | `record(entry)`, `similar(meta_features, k)`, `strategy_stats(...)` | Ablation with/without memory on held-out datasets |
| `resource_manager.py` | `allocate(total_budget, history) -> per-experiment budget`, `should_stop(...)` | Final score and utility under fixed budgets vs uniform allocation |

## 11. Autonomous loop (Planned)

```text
profile  = DatasetProfiler.profile(df, target)
budget   = ResourceManager(total_time, max_experiments, cpu_cores)
config   = Planner.initial(profile, memory, budget)
best     = None

while budget.remaining() and not budget.should_stop(history):
    result     = AutoMLRunner.run_experiment(df, config)          # existing baseline
    history.append(result)
    best       = better_of(best, result)
    diagnoses  = Critic.diagnose(result, profile)
    candidates = StrategyEngine.propose(diagnoses, memory, profile)
    strategy   = StrategyEngine.select(candidates, budget)         # rules → bandit → meta-learned
    memory.record(profile.meta_features(), diagnoses, strategy, result, cost=result.total_time)
    config     = Planner.apply(strategy, config, budget.next_allocation())

return best, history
```

## 12. Experiment memory (Planned)

Each record will contain:

```text
dataset meta-features (from DatasetProfile.meta_features)
+ diagnoses before the experiment
+ strategy applied
+ engine, model family, hyperparameters
+ primary metric before / after (Δ score)
+ training cost (wall-clock, CPU-seconds)
```

Example of the kind of record the memory will hold (illustrative structure, not a measured result):

```text
high class imbalance + class_weight + LightGBM → Δ F1 = <measured value>, cost = <seconds>
```

Retrieval: standardised meta-feature vectors, k-nearest neighbours, strategy statistics aggregated over the neighbours (mean Δ score, mean utility, number of observations). Storage starts as the existing JSON index (and/or SQLite) and only moves to a vector store (FAISS/Chroma) if the number of records makes it necessary.

## 13. Strategy engine (Planned)

Diagnosis → candidate strategies (initial rule table):

| Diagnosis | Candidate strategies |
|---|---|
| Overfitting (large train–validation/test gap) | Regularisation-biased search space, fewer/shallower trees, feature selection, more data via smaller test split |
| Underfitting (low train and test score) | More budget, more expressive model families, feature interactions |
| Class imbalance (moderate/severe) | Class weights, resampling (SMOTE), metric change to F1/ROC-AUC, threshold tuning |
| High missingness | Missing-indicator features, column removal above a threshold |
| High cardinality | Frequency/target encoding, column removal |
| Identifier / constant columns | Column removal |
| Redundant features | Correlation-based removal |
| Excessive training time / low improvement per second | Restrict estimator list, reduce budget, stop |

Ranking evolves in three steps: fixed priority (Phase 2) → bandit with reward = Δscore / cost (Phase 3) → priors from experiment memory (Phase 3). Each step is compared with the previous one on the same benchmark.

## 14. Resource-aware optimisation (Planned)

User budget: maximum time (e.g. 10 minutes), maximum experiments (e.g. 15), CPU cores (e.g. 2). Objective: maximise performance improvement while minimising compute.

```text
Utility = Performance improvement / Computational cost
```

The resource manager will (a) assign per-experiment time budgets, (b) set `n_jobs`, (c) stop when the best expected utility of the remaining strategies drops below a threshold or the budget is exhausted. The baseline already measures training time, total time and number of trials for every experiment.

## 15. Research questions

- **RQ1**: Does diagnosis-driven strategy selection reach a target score with less compute than the AutoML engine alone?
- **RQ2**: Does experiment memory reduce the number of experiments needed on new datasets?
- **RQ3**: Does adaptive budget allocation improve final performance under a fixed budget compared with uniform allocation?
- **RQ4**: How robust are the effects across dataset characteristics?

## 16. Hypotheses

- **H1**: The full system has a lower median time-to-target than FLAML at equal total budget.
- **H2**: Removing memory increases the number of experiments to target on unseen datasets.
- **H3**: Removing strategy selection lowers the anytime performance curve.
- **H4**: Final performance of the full system is non-inferior to FLAML at equal budget.

## 17. Datasets

- **Current (development/demo)**: Titanic (binary, mixed types, missing values, identifier columns), Breast Cancer Wisconsin Diagnostic (binary, numeric), Palmer Penguins (multiclass, mixed types), California Housing (regression, 20 640 rows).
- **Planned evaluation suite**: public OpenML tabular tasks (e.g. from OpenML-CC18 and the AutoML Benchmark) chosen to vary size, imbalance, missingness and categorical share, small enough for a laptop CPU. Memory-building and evaluation datasets will be disjoint; the exact task list will be committed before evaluation.

## 18. Evaluation metrics

- **Predictive**: primary metric on the outer test split; secondary task metrics.
- **Efficiency**: time-to-target, area under the anytime (best-so-far) curve.
- **Experimentation**: number of experiments and strategies to reach the target.
- **Cost**: CPU-seconds and wall-clock seconds; utility per experiment.
- **Convergence**: improvement per additional experiment; stopping point.
- **Robustness**: average rank over datasets, variance over seeds, worst-case gap.

## 19. Baselines

1. Manual baseline (fixed reasonable pipeline)
2. FLAML
3. AutoGluon
4. AutoPilot-ML without experiment memory
5. AutoPilot-ML without strategy selection
6. Full AutoPilot-ML

## 20. Ablation study

| Variant | Removed component | Question answered |
|---|---|---|
| A0 | none (full system) | Reference |
| A1 | experiment memory | Value of cross-dataset learning (RQ2) |
| A2 | strategy engine (random / fixed order) | Value of diagnosis-driven selection (RQ1) |
| A3 | resource manager (uniform budget split) | Value of adaptive allocation (RQ3) |
| A4 | critic (strategies chosen without diagnoses) | Value of explicit diagnosis |

Protocol: same datasets, budgets, seeds and outer splits for every variant; paired statistical tests (Wilcoxon signed-rank; Friedman + Nemenyi for multiple variants) following Demšar (2006).

## 21. Expected contribution

1. An open, fully local experimentation layer that turns single AutoML runs into an explained, budget-aware sequence of experiments.
2. A structured critic with testable diagnoses for tabular ML experiments.
3. A strategy-selection method that uses past experiment outcomes (memory) and compute cost, evaluated against FLAML and AutoGluon.
4. A reproducible benchmark harness and ablation results. Whether the hypotheses hold is an open question; negative results will be reported.

## 22. Risks

| Risk | Mitigation |
|---|---|
| Improvements over FLAML are small or not significant | Report effect sizes honestly; focus claims on efficiency (time/experiments) where the loop has more leverage |
| Time-based search noise hides effects | Multiple seeds, paired comparisons, fixed hardware and `n_jobs` |
| Memory too small to help | Seed memory from a separate development suite of public datasets |
| Strategy rules overfit to sample datasets | Freeze rules before evaluation; evaluate on unseen datasets |
| Scope creep (LLMs, deep learning, other modalities) | Tabular only; LLM strictly optional and outside the core claims |
| Laptop compute limits | Small/medium datasets, bounded budgets, sequential runs |

## 23. Limitations

- Tabular, single-table, CSV input; no text, images or time series.
- User-selected target; no automatic leakage detection yet.
- Results depend on FLAML's learners and search; the system does not change FLAML internals.
- The baseline performs no feature engineering or resampling.
- Wall-clock budgets make runs only approximately reproducible.

## 24. Timeline

Relative weeks from project start (adjust to the academic calendar):

| Weeks | Milestone |
|---|---|
| 1–2 | Phase 1 baseline (done): profiling, task detection, FLAML, evaluation, MLflow, dashboard |
| 3–4 | Critic with synthetic-problem tests |
| 5–6 | Strategy catalogue and rule-based strategy engine |
| 7–8 | Planner loop, experiment memory v1, resource manager v1 |
| 9–10 | Bandit / memory-based strategy selection; AutoGluon engine |
| 11–12 | Benchmark harness, dataset suite, full runs |
| 13–14 | Ablations, statistical analysis, report and presentation |

## 25. Future scope

- Automatic target detection, leakage checks, data-validation rules.
- SHAP explanations; optional local LLM explanations of diagnoses (Ollama).
- Additional engines (AutoGluon as baseline, then optionally as a selectable engine).
- FastAPI service, Docker image, PostgreSQL-backed tracking.
- Extension to time-series and text features once the tabular loop is validated.

## 26. References

AutoML systems and hyperparameter optimisation

1. Wang, C., Wu, Q., Weimer, M., & Zhu, E. (2021). FLAML: A Fast and Lightweight AutoML Library. *Proceedings of Machine Learning and Systems (MLSys)*.
2. Wu, Q., Wang, C., & Huang, S. (2021). Frugal Optimization for Cost-related Hyperparameters. *AAAI Conference on Artificial Intelligence*.
3. Wang, C., Wu, Q., Huang, S., & Saied, A. (2021). Economical Hyperparameter Optimization with Blended Search Strategy. *International Conference on Learning Representations (ICLR)*.
4. Feurer, M., Klein, A., Eggensperger, K., Springenberg, J., Blum, M., & Hutter, F. (2015). Efficient and Robust Automated Machine Learning. *Advances in Neural Information Processing Systems (NeurIPS)*.
5. Feurer, M., Eggensperger, K., Falkner, S., Lindauer, M., & Hutter, F. (2022). Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning. *Journal of Machine Learning Research*, 23(261), 1–61.
6. Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., & Smola, A. (2020). AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data. *arXiv:2003.06505*.
7. Olson, R. S., Bartley, N., Urbanowicz, R. J., & Moore, J. H. (2016). Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science. *Genetic and Evolutionary Computation Conference (GECCO)*.
8. Thornton, C., Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2013). Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms. *ACM SIGKDD*.
9. Hutter, F., Kotthoff, L., & Vanschoren, J. (Eds.). (2019). *Automated Machine Learning: Methods, Systems, Challenges*. Springer.
10. Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2011). Sequential Model-Based Optimization for General Algorithm Configuration. *Learning and Intelligent Optimization (LION 5)*.
11. Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for Hyper-Parameter Optimization. *NeurIPS*.
12. Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research*, 13, 281–305.
13. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. *NeurIPS*.
14. Jamieson, K., & Talwalkar, A. (2016). Non-stochastic Best Arm Identification and Hyperparameter Optimization. *AISTATS*.
15. Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. *Journal of Machine Learning Research*, 18(185), 1–52.
16. Falkner, S., Klein, A., & Hutter, F. (2018). BOHB: Robust and Efficient Hyperparameter Optimization at Scale. *International Conference on Machine Learning (ICML)*.
17. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *ACM SIGKDD*.

Meta-learning, bandits and algorithm selection

18. Vanschoren, J. (2018). Meta-Learning: A Survey. *arXiv:1810.03548*.
19. Feurer, M., Springenberg, J. T., & Hutter, F. (2015a). Initializing Bayesian Hyperparameter Optimization via Meta-Learning. *AAAI*.
20. Rice, J. R. (1976). The Algorithm Selection Problem. *Advances in Computers*, 15, 65–118.
21. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem. *Machine Learning*, 47, 235–256.
22. Thompson, W. R. (1933). On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples. *Biometrika*, 25(3/4), 285–294.
23. Wolpert, D. H., & Macready, W. G. (1997). No Free Lunch Theorems for Optimization. *IEEE Transactions on Evolutionary Computation*, 1(1), 67–82.

Learning algorithms, data problems and explainability

24. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
25. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *ACM SIGKDD*.
26. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
27. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
28. Fisher, A., Rudin, C., & Dominici, F. (2019). All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously. *Journal of Machine Learning Research*, 20(177), 1–81.
29. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
30. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
31. He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.
32. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4).
33. Cawley, G. C., & Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. *Journal of Machine Learning Research*, 11, 2079–2107.

Evaluation methodology, benchmarks and tooling

34. Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*, 7, 1–30.
35. Gijsbers, P., Bueno, M. L. P., Coors, S., LeDell, E., Poirier, S., Thomas, J., Bischl, B., & Vanschoren, J. (2024). AMLB: an AutoML Benchmark. *Journal of Machine Learning Research*, 25(101), 1–65.
36. Vanschoren, J., van Rijn, J. N., Bischl, B., & Torgo, L. (2013). OpenML: Networked Science in Machine Learning. *SIGKDD Explorations*, 15(2), 49–60.
37. Zaharia, M., et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow. *IEEE Data Engineering Bulletin*, 41(4), 39–45.
38. Schwartz, R., Dodge, J., Smith, N. A., & Etzioni, O. (2020). Green AI. *Communications of the ACM*, 63(12), 54–63.
39. Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. *NeurIPS*.

LLM-assisted data science (related work only; not used by the system)

40. Hollmann, N., Müller, S., & Hutter, F. (2023). Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering. *NeurIPS*.
41. Huang, Q., Vora, J., Liang, P., & Leskovec, J. (2024). MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation. *ICML*.

Datasets

42. Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear Feature Extraction for Breast Tumor Diagnosis. *IS&T/SPIE International Symposium on Electronic Imaging: Science and Technology*, vol. 1905, 861–870.
43. Pace, R. K., & Barry, R. (1997). Sparse Spatial Autoregressions. *Statistics & Probability Letters*, 33(3), 291–297.
44. Gorman, K. B., Williams, T. D., & Fraser, W. R. (2014). Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus *Pygoscelis*). *PLoS ONE*, 9(3), e90081.
45. Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data. R package. doi:10.5281/zenodo.3960218.
46. Kaggle. Titanic – Machine Learning from Disaster (competition dataset).

Bibliographic details should be checked against the publishers' pages before final submission.
