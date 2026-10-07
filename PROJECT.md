# AutoPilot-ML: Autonomous Machine Learning Experimentation & Self-Optimization Framework

Main project document, B.Tech Computer Science major project.

Current status: the Phase 1 baseline (profiling, task detection, FLAML search, evaluation, MLflow tracking, dashboard) is implemented. The autonomous layer (planner, critic, strategy engine, experiment memory, resource manager) is planned and not built yet. No benchmark results exist yet, so this document makes no performance claims.

## Abstract

AutoML tools search for a good model and hyperparameters, but the loop around that search is still done by people: working out why a model is weak, deciding what to change, and judging whether more compute is worth it. AutoPilot-ML proposes an experimentation layer on top of an existing AutoML engine, FLAML. Given a tabular dataset and a compute budget, the system will run a first experiment and diagnose its weaknesses (overfitting, underfitting, class imbalance, problem columns, wasted compute). It will then pick an improvement strategy, run the next experiment, and record what each strategy did on what kind of dataset, so that later runs on similar data can try the most promising strategies first. The first milestone, described here, is a fully local baseline: CSV input, dataset profiling, task detection, FLAML search with an independent test evaluation, basic explainability, MLflow tracking and an experiment history. The full system will be compared with FLAML, AutoGluon and reduced versions of itself on test score, time to reach a target score, number of experiments, compute cost and consistency across datasets.

## Acknowledgement

This project grew out of working with FLAML (Wang, Wu, Weimer and Zhu, MLSys 2021), the open-source AutoML library from Microsoft Research. FLAML showed that strong models can be found quickly on ordinary laptops when the search pays attention to the cost of each trial. While using it, it became clear that the questions after each run were still left to the user, and AutoPilot-ML is our attempt to answer them. FLAML is used unchanged as a dependency (MIT licence). Everything this project adds sits around FLAML, not inside it.

## 1. Introduction

Most applied machine learning still works on tables: customer records, medical measurements, sensor logs, financial data. For this kind of data, gradient-boosted trees and well-tuned linear models are hard to beat, and AutoML libraries such as auto-sklearn, TPOT, AutoGluon and FLAML can produce competitive models with very little effort.

What these libraries automate is the inner loop: for a fixed dataset and a fixed objective, which algorithm and which hyperparameters work best? The loop a data scientist actually runs is bigger. After the first model they ask questions like these. Is it overfitting? Is accuracy hiding a minority class that is never predicted? Is an ID column leaking information? Would removing a messy column help? Was the last ten minutes of search worth it? Then they change something and run again. This outer loop takes most of the time, and the experience gained in it is rarely written down or reused.

AutoPilot-ML treats the AutoML engine as one component and focuses on automating, and later learning, the outer loop.

## 2. Motivation

1. Compute is the real limit. Students and small teams usually have a laptop CPU and no money for cloud GPUs or paid APIs, so the question is where to spend compute, not how to add more.
2. The problems found after a first run follow patterns. A large train/test gap, a 92% majority class, a column with a different value in every row, or a feature that is 70% empty can all be detected from data and metrics we already have.
3. Fixes can be reused. "Class weighting helped on a strongly imbalanced, mostly numeric dataset" is useful knowledge for the next dataset of that kind.
4. A system that says what it diagnosed and why it chose the next experiment is easier to trust, to teach with and to evaluate than a black-box search.
5. The decision layer should not depend on large language models or paid services. Every core decision must be measurable and work offline.

## 3. Problem statement

Given a tabular dataset D with a target column y (chosen by the user for now), a task T (binary classification, multiclass classification or regression), a primary metric m and a compute budget B (time, number of experiments, CPU cores), the goal is:

1. to find a pipeline p (data treatment, features, model, hyperparameters) with the best possible test performance m(p), and
2. to get there using as little of B as possible.

The baseline handles the first goal with a single AutoML run. The research system handles both, using a sequence of experiments e1, e2, ..., ek, where each next experiment is chosen from the diagnoses of the earlier ones and from a memory of experiments on other datasets.

## 4. Existing approaches

- **Manual work.** An expert does everything by hand, guided by experience.
- **Grid and random search** (Bergstra and Bengio, 2012). Simple hyperparameter search with no model of which settings are promising.
- **Bayesian optimisation**, for example SMAC (Hutter et al., 2011), TPE (Bergstra et al., 2011) and Spearmint (Snoek et al., 2012). It models the score as a function of the hyperparameters and samples where improvement is likely.
- **Bandit and multi-fidelity methods**: Successive Halving (Jamieson and Talwalkar, 2016), Hyperband (Li et al., 2018), BOHB (Falkner et al., 2018). They stop poor configurations early to save compute.
- **Combined model selection and tuning**: Auto-WEKA (Thornton et al., 2013) and auto-sklearn (Feurer et al., 2015, 2022). Auto-sklearn also uses meta-learning to choose starting configurations from similar datasets, and builds ensembles.
- **Evolutionary pipeline search**: TPOT (Olson et al., 2016) evolves whole pipelines with genetic programming.
- **Stacked ensembles**: AutoGluon-Tabular (Erickson et al., 2020) relies on strong defaults and multi-layer stacking rather than long searches.
- **Cost-aware search**: FLAML (Wang et al., 2021) with CFO (Wu et al., 2021) and BlendSearch (Wang et al., 2021) accounts for the cost of each trial, which makes it fast on small hardware.
- **LLM-assisted data science**: for example CAAFE for feature engineering (Hollmann et al., 2023) and MLAgentBench for language agents running ML experiments (Huang et al., 2024). These are related work only. Our system does not use LLMs.

## 5. Limitations of conventional AutoML

1. No diagnosis. The search reports a best score, not why the model is weak or what kind of change might help.
2. The problem is fixed in advance. Data treatment (resampling, removing columns, encoding high-cardinality features) is mostly outside the search space or decided by defaults.
3. Metric pitfalls. On imbalanced targets, accuracy can look high while the minority class is never predicted, unless the user picks a better metric.
4. Little learning across datasets. Apart from meta-learned starting points (auto-sklearn), most tools start every dataset from scratch and keep no record of which interventions helped.
5. The budget is spent in one go. There is no decision about whether more search, a different strategy, or stopping would be the better use of the remaining compute.
6. The process is hard to follow. Users see a leaderboard, not the reasoning behind a sequence of experiments.

## 6. Proposed system

AutoPilot-ML puts an experiment loop around the AutoML engine:

1. Dataset intelligence: profile the data and detect the task (implemented).
2. Experiment planning: choose the next experiment configuration from the profile, the memory and the remaining budget (planned).
3. AutoML core: run the engine. FLAML today, AutoGluon later as a comparison (FLAML implemented).
4. Evaluation: metrics, errors and resource usage on a held-out split (implemented).
5. Critic: structured diagnosis of each experiment (planned).
6. Strategy engine: candidate improvements ranked by expected gain per unit of cost (planned).
7. Experiment memory: what was tried, on what kind of data, at what cost, with what effect (planned; the baseline already writes the raw records).
8. Resource manager: share time, experiments and CPU across the loop and decide when to stop (planned).

## 7. Architecture

```text
                         USER
                           │
                           ▼
                    Dataset Upload                (implemented)
                           │
                           ▼
                   Dataset Profiler               (implemented)
                           │
                           ▼
                    Task Detection                (implemented)
                           │
                           ▼
                  Experiment Planner              (planned)
                           │
                           ▼
                 AutoML Core (FLAML)              (implemented)
                           │
                           ▼
                      Evaluation                  (implemented)
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           Metrics       Errors     Resources      (recorded today)
              └────────────┼────────────┘
                           ▼
                        Critic                    (planned)
                           ▼
                    Strategy Engine               (planned)
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Features       Model         Data       (planned)
              └────────────┼────────────┘
                           ▼
                    Next Experiment               (planned)
                           ▼
                   Experiment Memory              (planned; raw records exist)
                           └──────────► next cycle
```

In the code there are three layers: the Streamlit UI (`app/main.py`, `app/components/`), the services with all the logic (`app/services/`), and settings and logging (`app/core/`). Under them are the external libraries (FLAML, scikit-learn, MLflow) and local storage (`experiments/`, `models/`, `mlruns/`). The services never import the UI. Class-level detail and the sequence of calls are in `docs/architecture.md`.

## 8. Detailed modules

### 8.1 Dataset profiler (implemented, `app/services/profiler.py`)

Takes a pandas DataFrame and, optionally, the target and task, and returns a `DatasetProfile`. The profile contains rows, columns, types, numerical/categorical/datetime feature lists, missing values and ratios, duplicate rows and unique counts. It flags constant, identifier-like, high-cardinality and mostly-missing columns. For the target it gives the class distribution and a `ClassBalance` (majority and minority share, imbalance ratio, severity), or summary statistics for regression. It also writes readable warnings and a small set of numeric meta-features. Severity uses the ratio of majority to minority class: below 1.5 is none, 1.5 to 3 mild, 3 to 9 moderate, 9 and above severe. The profiler has no Streamlit dependency, and its output is saved with every experiment.

### 8.2 Task detector (implemented, `app/services/task_detector.py`)

- A text or boolean target with 2 classes is binary; with more classes it is multiclass.
- A numeric target with 2 distinct values is binary. An integer target with at most 20 distinct values is treated as multiclass, with medium or low confidence and a note that it may really be a count or an ordinal value. Anything else is regression.
- `validate()` rejects overrides that cannot work (regression on text, binary with a number of classes other than 2, multiclass with fewer than 3) and warns about missing targets and classes with fewer than two rows.
- `target_candidates()` describes how suitable every column is as a target. `suggest_target()` is a deliberately simple placeholder (a generic name such as "target" or "label", otherwise the last column). Proper automatic target detection is future work.

### 8.3 AutoML runner and FLAML engine (implemented, `app/services/automl_runner.py`)

1. Check the configuration: the target and excluded columns exist, the budget is 5 to 3600 seconds, the test size is 0.05 to 0.5, and there are at least 20 labelled rows.
2. Drop rows with a missing target, split off X and y, and remove the excluded columns.
3. Detect or validate the task, and check that the metric fits it.
4. Make a stratified train/test split. If a class has fewer than 2 rows, the split is not stratified and a warning is shown.
5. Run FLAML on the training part with the mapped metric (F1 becomes `macro_f1` and ROC-AUC becomes `roc_auc_ovr` for multiclass), the time budget and the seed. FLAML chooses cross-validation or holdout itself, logs every trial, retrains the best configuration on all training rows, and may stop early if the search has converged. FLAML's own MLflow logging is switched off so that tracking happens in one place.
6. Keep the best estimator, its configuration, the best validation score (converted back from FLAML's loss), the training time, the trial history and a leaderboard per model family.
7. Evaluate on the test and training parts, compute the trivial baseline and the permutation importance, and note any categorical columns whose test values never appeared in training.
8. Save the model, write the result files, log the MLflow run and write `summary.json`.

FLAML sits behind a small `AutoMLEngine` interface. Any engine that returns an `EngineResult` can replace it. The tests check this with a plain scikit-learn logistic regression engine.

### 8.4 Evaluator (implemented, `app/services/evaluator.py`)

- Classification: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC and log loss. Multiclass uses macro averages. For binary tasks the positive class is the last label in sorted order, which is the same convention FLAML uses internally.
- Regression: MAE, MSE, RMSE and R².
- Metrics you can optimise: binary {F1, ROC-AUC, accuracy, log loss}, multiclass {macro F1, accuracy, macro ROC-AUC, log loss}, regression {R², RMSE, MAE, MSE}.
- A trivial baseline (majority class or training mean) is scored on the same test rows.
- A warning appears when accuracy is chosen on a target with moderate or severe imbalance.
- Permutation importance uses the chosen metric on the test rows (at most 2000 rows, 5 repeats).

### 8.5 Experiment tracker (implemented, `app/services/experiment_tracker.py`)

Gives each experiment an ID (EXP-001, EXP-002, ...), creating its folder atomically so two runs can never get the same ID. It logs an MLflow run to a local SQLite database with the parameters (dataset, target, task, metric, budget, seed, engine version, best hyperparameters, dataset meta-features), the metrics (test, train and baseline metrics, validation score, timings, number of trials) and the files. It also writes `experiments/EXP-###/summary.json`, which the history table reads. Recent MLflow versions default to a database backend, and SQLite needs no server, so we chose it.

### 8.6 Dashboard (implemented, `app/main.py`, `app/components/`)

There are seven sections: Dataset, Target, Experiment Configuration, Run AutoML, Results, Explainability and Experiment History. The experiment runs in a background thread so the page stays responsive. The progress bar is estimated from elapsed time against the budget, because FLAML does not report progress during the search.

## 9. Current implementation

- Environment: conda environment `automl` with Python 3.11. Versions are pinned in `requirements.txt`.
- Data: CSV only, with four public sample datasets included.
- Tasks: binary, multiclass, regression.
- Engine: FLAML 2.7.0 with its default learners (LightGBM, XGBoost, random forest, extra trees, linear models and others).
- Evaluation: independent test split, FLAML's internal validation score and a trivial baseline.
- Explainability: permutation importance, the model's own importance, confusion matrix, ROC curve, residual plots.
- Tracking: local MLflow (SQLite) plus JSON records.
- Tests: 46 pytest tests for the profiler, the task detector, the runner (with short real FLAML runs), the evaluator, failure handling and a custom engine.
- Not implemented: planner, critic, strategy engine, memory, resource manager, SHAP, automatic target detection, feature engineering, resampling.

## 10. Proposed modules (planned)

Every planned module comes with a way to evaluate it:

- `experiment_planner.py`: `plan(profile, memory, budget_state) -> ExperimentConfig`. Evaluated by the number of experiments and the time needed to reach a target score, compared with a fixed plan.
- `critic.py`: `diagnose(result, profile) -> list[Diagnosis]`, each with a type, a severity and the evidence. Evaluated by precision and recall on synthetic datasets where we inject known problems, so the correct answer is known.
- `strategy_engine.py`: `propose(diagnoses, memory) -> ranked strategies`. Evaluated by time to target and final score against random and fixed-order selection.
- `experiment_memory.py`: `record(entry)`, `similar(meta_features, k)`, `strategy_stats(...)`. Evaluated by an ablation with and without memory on datasets the memory has never seen.
- `resource_manager.py`: `allocate(total_budget, history)` and `should_stop(...)`. Evaluated by final score and gain per unit of cost under fixed budgets, against an even split of the budget.

## 11. Autonomous loop (planned)

```text
profile  = DatasetProfiler.profile(df, target)
budget   = ResourceManager(total_time, max_experiments, cpu_cores)
config   = Planner.initial(profile, memory, budget)
best     = None

while budget.remaining() and not budget.should_stop(history):
    result     = AutoMLRunner.run_experiment(df, config)        # exists today
    history.append(result)
    best       = better_of(best, result)
    diagnoses  = Critic.diagnose(result, profile)
    candidates = StrategyEngine.propose(diagnoses, memory, profile)
    strategy   = StrategyEngine.select(candidates, budget)       # rules, then bandit, then memory-based
    memory.record(profile.meta_features(), diagnoses, strategy, result, cost=result.total_time)
    config     = Planner.apply(strategy, config, budget.next_allocation())

return best, history
```

## 12. Experiment memory (planned)

Each record will hold the dataset meta-features (from `DatasetProfile.meta_features()`), the diagnoses before the experiment, the strategy applied, the engine, the model family and hyperparameters, the primary score before and after, and the cost in wall-clock and CPU seconds.

For example, a record might read "severe class imbalance + class_weight + LightGBM gave a change in F1 of x for y seconds", where x and y are measured values.

To use the memory on a new dataset, the meta-features are standardised and the k nearest past datasets are found. For each strategy we then look at the average change in score, the average gain per second and how many times it was tried on those neighbours. The memory starts as the existing JSON records (or SQLite). A vector store such as FAISS or Chroma would only be added if the number of records made it necessary.

## 13. Strategy engine (planned)

A first rule table, from diagnosis to candidate strategies:

- Overfitting (large train vs validation/test gap): a search space biased towards regularisation, fewer or shallower trees, feature selection.
- Underfitting (low train and test scores): more budget, more expressive model families, feature interactions.
- Class imbalance (moderate or severe): class weights, SMOTE, switching the metric to F1 or ROC-AUC, threshold tuning.
- Heavy missingness: missing-value indicator features, or dropping columns above a threshold.
- High cardinality: frequency or target encoding, or dropping the column.
- Identifier or constant columns: drop them.
- Redundant features: remove one of each highly correlated pair.
- Too much time for too little gain: restrict the model list, reduce the budget, or stop.

The ranking will be developed in three steps: a fixed priority order first, then a multi-armed bandit whose reward is the gain in score divided by cost, then priors taken from the experiment memory. Each step is compared with the previous one on the same benchmark.

## 14. Resource-aware optimisation (planned)

The user gives a maximum time (say 10 minutes), a maximum number of experiments (say 15) and a number of CPU cores (say 2). The aim is to get the largest improvement for the least compute:

```text
Utility = performance improvement / computational cost
```

The resource manager will give each experiment its share of the time, set the number of cores FLAML may use, and stop when no remaining strategy is expected to be worth its cost, or when the budget runs out. The baseline already measures training time, total time and number of trials for every experiment, so this information is available.

## 15. Research questions

- RQ1: Does choosing strategies from diagnoses reach a target score with less compute than the AutoML engine on its own?
- RQ2: Does the experiment memory reduce the number of experiments needed on new datasets?
- RQ3: Does adaptive budget allocation give a better final score under a fixed budget than splitting the budget evenly?
- RQ4: How consistent are these effects across datasets with different characteristics?

## 16. Hypotheses

- H1: The full system reaches the target score in less time (median) than FLAML with the same total budget.
- H2: Without the memory, more experiments are needed to reach the target on unseen datasets.
- H3: Without strategy selection, the best-score-over-time curve is lower.
- H4: The final score of the full system is not worse than FLAML's at the same budget.

## 17. Datasets

- Development and demo: Titanic (binary, mixed types, missing values, ID columns), Breast Cancer Wisconsin Diagnostic (binary, numeric), Palmer Penguins (multiclass, mixed types) and California Housing (regression, 20,640 rows).
- Evaluation (planned): public OpenML tabular tasks, for example from the OpenML-CC18 suite and the AutoML Benchmark. They will be chosen to vary in size, imbalance, missingness and share of categorical columns, while staying small enough for a laptop CPU. Datasets used to build the memory and datasets used for evaluation will not overlap. The exact list will be fixed and committed before the evaluation runs.

## 18. Evaluation metrics

- Predictive quality: the primary metric on the outer test split, plus the other task metrics.
- Efficiency: time to reach a target score, and the area under the best-score-over-time curve.
- Experimentation: number of experiments and strategies needed to reach the target.
- Cost: CPU seconds, wall-clock seconds, and gain per experiment.
- Convergence: how much each extra experiment adds, and where the system stops.
- Consistency: average rank across datasets, spread across seeds, and the worst-case gap to the best system.

## 19. Baselines

1. Manual baseline (one fixed, sensible pipeline)
2. FLAML
3. AutoGluon
4. AutoPilot-ML without experiment memory
5. AutoPilot-ML without strategy selection
6. Full AutoPilot-ML

## 20. Ablation study

Each variant removes one part of the full system:

- A0, the full system: the reference.
- A1, no experiment memory: is learning across datasets worth it? (RQ2)
- A2, no strategy engine (random or fixed order): do diagnosis-driven choices help? (RQ1)
- A3, no resource manager (even budget split): does adaptive allocation help? (RQ3)
- A4, no critic (strategies chosen without diagnoses): does explicit diagnosis help?

All variants use the same datasets, budgets, seeds and outer splits. They will be compared with paired tests (Wilcoxon signed-rank, and Friedman with a Nemenyi post-hoc test for several variants at once), following Demšar (2006).

## 21. Expected contribution

1. An open, fully local experimentation layer that turns single AutoML runs into an explained, budget-aware sequence of experiments.
2. A critic with diagnoses for tabular experiments that can be tested.
3. A way to choose strategies using past results (memory) and compute cost, compared against FLAML and AutoGluon.
4. A reproducible benchmark setup with ablation results. Whether the hypotheses hold is an open question, and negative results will be reported as well.

## 22. Risks

- The gains over FLAML may turn out small or not significant. We will report effect sizes honestly and focus on efficiency (time and number of experiments), where the outer loop has the most room to help.
- Time-limited searches are noisy, which can hide real effects. We will use several seeds, paired comparisons and a fixed machine with a fixed number of cores.
- The memory may be too small to help. We will seed it from a separate set of public development datasets.
- The strategy rules may fit our sample datasets too closely. We will freeze them before evaluation and evaluate on unseen datasets.
- Scope creep (LLMs, deep learning, other data types). We will stay with tabular data, and any LLM use stays optional and outside the main claims.
- A laptop can only do so much. We will use small and medium datasets, bounded budgets and sequential runs.

## 23. Limitations

- Only tabular, single-table CSV input. No text, images or time series.
- The user picks the target. There is no automatic leakage check yet.
- Results depend on FLAML's learners and search, and we do not change FLAML itself.
- The baseline does no feature engineering or resampling.
- Because budgets are measured in wall-clock time, runs are only approximately reproducible.

## 24. Timeline

In weeks from the start of the project (to be adjusted to the semester):

- Weeks 1-2: Phase 1 baseline (done)
- Weeks 3-4: critic, tested on synthetic datasets with known problems
- Weeks 5-6: strategy catalogue and rule-based strategy engine
- Weeks 7-8: planner loop, experiment memory and resource manager
- Weeks 9-10: bandit and memory-based strategy selection; AutoGluon engine
- Weeks 11-12: benchmark setup, dataset suite, full runs
- Weeks 13-14: ablations, statistical analysis, report and presentation

## 25. Future scope

- Automatic target detection, leakage checks and data validation rules
- SHAP explanations, and an optional local LLM (through Ollama) to explain diagnoses in plain language
- More engines: AutoGluon as a baseline first, possibly also as a selectable engine
- FastAPI service, Docker image, PostgreSQL-backed tracking
- Time-series and text features, once the tabular loop is shown to work

## 26. References

1. Wang, C., Wu, Q., Weimer, M., and Zhu, E. (2021). FLAML: A Fast and Lightweight AutoML Library. Proceedings of Machine Learning and Systems (MLSys).
2. Wu, Q., Wang, C., and Huang, S. (2021). Frugal Optimization for Cost-related Hyperparameters. AAAI.
3. Wang, C., Wu, Q., Huang, S., and Saied, A. (2021). Economical Hyperparameter Optimization with Blended Search Strategy. ICLR.
4. Feurer, M., Klein, A., Eggensperger, K., Springenberg, J., Blum, M., and Hutter, F. (2015). Efficient and Robust Automated Machine Learning. NeurIPS.
5. Feurer, M., Eggensperger, K., Falkner, S., Lindauer, M., and Hutter, F. (2022). Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning. Journal of Machine Learning Research, 23(261).
6. Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., and Smola, A. (2020). AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data. arXiv:2003.06505.
7. Olson, R. S., Bartley, N., Urbanowicz, R. J., and Moore, J. H. (2016). Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science. GECCO.
8. Thornton, C., Hutter, F., Hoos, H. H., and Leyton-Brown, K. (2013). Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms. KDD.
9. Hutter, F., Kotthoff, L., and Vanschoren, J. (Eds.) (2019). Automated Machine Learning: Methods, Systems, Challenges. Springer.
10. Hutter, F., Hoos, H. H., and Leyton-Brown, K. (2011). Sequential Model-Based Optimization for General Algorithm Configuration. LION 5.
11. Bergstra, J., Bardenet, R., Bengio, Y., and Kégl, B. (2011). Algorithms for Hyper-Parameter Optimization. NeurIPS.
12. Bergstra, J., and Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research, 13, 281-305.
13. Snoek, J., Larochelle, H., and Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NeurIPS.
14. Jamieson, K., and Talwalkar, A. (2016). Non-stochastic Best Arm Identification and Hyperparameter Optimization. AISTATS.
15. Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., and Talwalkar, A. (2018). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. Journal of Machine Learning Research, 18(185).
16. Falkner, S., Klein, A., and Hutter, F. (2018). BOHB: Robust and Efficient Hyperparameter Optimization at Scale. ICML.
17. Vanschoren, J. (2018). Meta-Learning: A Survey. arXiv:1810.03548.
18. Auer, P., Cesa-Bianchi, N., and Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning, 47, 235-256.
19. Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.
20. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
21. Lundberg, S. M., and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
22. Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. Journal of Machine Learning Research, 7, 1-30.
23. Gijsbers, P., Bueno, M. L. P., Coors, S., LeDell, E., Poirier, S., Thomas, J., Bischl, B., and Vanschoren, J. (2024). AMLB: an AutoML Benchmark. Journal of Machine Learning Research, 25(101).
24. Vanschoren, J., van Rijn, J. N., Bischl, B., and Torgo, L. (2013). OpenML: Networked Science in Machine Learning. SIGKDD Explorations, 15(2).
25. Zaharia, M., et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow. IEEE Data Engineering Bulletin, 41(4).
26. Hollmann, N., Müller, S., and Hutter, F. (2023). Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering. NeurIPS.
27. Huang, Q., Vora, J., Liang, P., and Leskovec, J. (2024). MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation. ICML.

Datasets: Kaggle, Titanic: Machine Learning from Disaster; Street, Wolberg and Mangasarian (1993), Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository; Horst, Hill and Gorman (2020), palmerpenguins, doi:10.5281/zenodo.3960218; Pace and Barry (1997), Sparse Spatial Autoregressions, Statistics & Probability Letters, 33(3).
