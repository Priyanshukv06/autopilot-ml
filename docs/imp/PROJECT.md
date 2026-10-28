# AutoPilot-ML: Autonomous Machine Learning Experimentation & Self-Optimization Framework

Main project document, B.Tech Computer Science major project.

Current status (28 October 2026): the Phase 1 baseline (profiling, task detection, FLAML search, evaluation, MLflow tracking, dashboard) and the Phase 2 rule-based autonomous layer (Session loop, planner, assessor, critic, strategy engine, judge, resource manager, experiment memory, and the autonomous mode of the dashboard) are implemented and tested. Phase 3 is in progress: the development and evaluation datasets are fixed (task 20), and the learned strategy choice from the memory (task 21), the ablation switches (task 22), the comparison systems (task 23: a manual LightGBM baseline, FLAML alone and AutoGluon) a resumable benchmark harness (task 24), its statistics (task 25), its analysis and frozen experiment memory (task 26) and the judge agreement study (task 27) are built and tested (8.8). The benchmark has been running since 28 October 2026 and the judge agreement study runs after it, so there are no results from them yet. The only measurements so far are the component studies and the Phase 2 comparison with FLAML alone on the four sample datasets (`report-notes.md`, chapter 9); this document makes no other performance claims.

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
2. Experiment planning: build the first Setup and each candidate Setup from the profile, the chosen change and the remaining budget (implemented in Phase 2; choosing the change from the memory implemented in Phase 3, task 21).
3. AutoML core: run the engine. FLAML is the engine (implemented); AutoGluon runs only as a comparison system in the benchmark, in its own environment (Phase 3, task 23).
4. Evaluation: metrics, errors and resource usage on a held-out split (implemented), and, inside a Session, Assessments on fixed folds of the training part (implemented in Phase 2).
5. Critic: structured diagnosis of each experiment (implemented in Phase 2: 12 kinds, thresholds tuned by a study).
6. Strategy engine: candidate improvements ranked by a rule table (implemented in Phase 2: 9 Strategies, ranked by severity and cost), or by expected gain per unit of cost learned from the memory (implemented in Phase 3, task 21: Thompson sampling and UCB1; its effect is not measured yet).
7. Judge: keeps a change only if the fold differences support it (implemented in Phase 2).
8. Experiment memory: what was tried, on what kind of data, at what cost, with what effect (implemented in Phase 2; read by the learned strategy choice since task 21 when switched on).
9. Resource manager: share time, experiments and CPU across the loop and decide when to stop (implemented in Phase 2).

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
                  Experiment Planner              (implemented, Phase 2)
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
                        Critic                    (implemented, Phase 2)
                           ▼
                    Strategy Engine               (implemented, Phase 2: rules)
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Features       Model         Data       (implemented, Phase 2: 9 Strategies)
              └────────────┼────────────┘
                           ▼
                Next Experiment + Judge           (implemented, Phase 2)
                           ▼
                   Experiment Memory              (implemented, Phase 2: written; used in Phase 3)
                           └──────────► next cycle
```

In the code there are three layers: the Streamlit UI (`app/main.py`, `app/components/`), the services with all the logic (`app/services/`), and settings and logging (`app/core/`). Under them are the external libraries (FLAML, scikit-learn, MLflow) and local storage (`experiments/`, `models/`, `mlruns/`). The services never import the UI. Class-level detail and the sequence of calls are in [architecture.md](architecture.md).

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

There are seven sections: Dataset, Target, Experiment Configuration, Run AutoML, Results, Explainability and Experiment History. The experiment runs in a background thread so the page stays responsive. The progress bar is estimated from elapsed time against the budget, because FLAML does not report progress during the search. Since Phase 2 the sidebar has a mode switch; the autonomous mode is described in 8.7.

### 8.7 Autonomous layer (implemented in Phase 2)

A Session runs the loop on one dataset within a total time, a maximum number of Experiments and a number of CPU cores. It makes the training and test parts once (the same split as Phase 1) and 3 x 5 fixed folds of the training part; no decision ever reads the test part. The parts, all in `app/services/`:

- Planner (`experiment_planner.py`): the first Setup (the user's settings with identifier-like and constant columns left out) and each candidate (a Strategy applied to a copy of the Accepted_Setup, warm-started from its best configurations).
- Assessor (`assessor.py`): refits a Setup's best configuration on all 15 folds, with any data changes learned inside each fold, and records the fold scores.
- Critic (`critic.py`, thresholds in `critic_thresholds.toml`): 12 kinds of Diagnoses with severity and evidence; thresholds tuned by the critic study.
- Strategy catalogue and engine (`strategies.py`, `strategy_engine.py`, data changes in `transforms.py`): 9 Strategies and a rule table that ranks the Proposals by severity, cost and table order; each change is tried once per Session.
- Judge (`judge.py`): corrected resampled t-test on the paired fold differences; keep, keep_simpler (simpler and not worse by more than 0.005) or discard; alpha 0.20, from the judge study.
- Resource manager (`resource_manager.py`): 35% of the time for the first Experiment, later budgets from the measured time left, stop rules (Stop pressed, error, time, Experiments, no Proposals, three discards in a row).
- Experiment memory (`experiment_memory.py`): one record per Verdict in `experiments/memory/memory.jsonl`; nearest past datasets and per-Strategy statistics, read by the learned selectors since Phase 3 (task 21) when the memory is switched on.
- Records: `experiments/SES-###/session.json` after every step, step files per Experiment, and an MLflow parent run per Session.
- Dashboard autonomous mode (`app/components/autopilot_viewer.py`) and command line (`scripts/run_autopilot.py`).

The plain-language explanation is in [how-it-works.md](how-it-works.md), section 9, and the class-level view in [architecture.md](architecture.md).

### 8.8 Phase 3 so far (tasks 20 to 24, implemented)

- Datasets (`research/datasets.py`, `research/datasets.json`): 10 development datasets (the four samples and six OpenML datasets) and 6 evaluation datasets, fixed by a written rule before any evaluation run (section 17).
- Learned strategy choice (`strategy_engine.py`): the selectors `rules`, `random`, `thompson` and `ucb1`. An arm is a Strategy with the kind of Diagnosis it answers; its statistics come from the memory records of the 5 most similar other datasets, weighted by similarity; arms with fewer than 3 records keep their rule-order place.
- Ablation switches (`SessionSettings`): `use_memory`, the `random` selector, `use_critic=False` (random valid changes, no Diagnoses) and `use_resource_manager=False` (equal shares of the time; only the time and Experiment limits stop the Session). Each is recorded in `session.json`.
- Comparison systems (`research/systems.py`): `manual` (LightGBM with its library defaults, no search), `flaml` (one search with the whole budget), `autogluon` (AutoGluon-Tabular 1.6.3 in its own conda environment, called as a separate process) and the AutoPilot variants. Every system gets the same training and test rows and columns and is scored by the same Evaluator.
- Benchmark harness (`research/benchmark.py`, `scripts/run_benchmark.py`): 390 cells (5 main systems at 2 and 5 minutes, 3 ablations at 5 minutes, 6 evaluation datasets, seeds 1 to 5), one CSV row per finished cell, restart where it stopped, checks that the frozen memory comes from development datasets only, and a smoke setting.
- Statistics (`research/stat_tests.py`): the Friedman test, the Nemenyi critical difference, Wilcoxon tests with Holm's correction, rank-biserial effect sizes, bootstrap intervals and time-to-target (Demšar, 2006), checked against published reference values.
- Frozen memory (`research/memory_build.py`): 30 Sessions on the development datasets wrote 97 records, frozen on 27 October 2026 (`report-notes.md`, section 9.15); the benchmark reads it and never writes it.
- Analysis (`research/benchmark_analysis.py`): critical-difference diagrams, score-against-budget charts, per-dataset tables and time-to-target, with the comparisons fixed before any benchmark cell ran.
- Judge agreement study (`research/judge_agreement.py`): 57 decisions of the Judge from the memory-build Sessions, each made again by new FLAML searches on 5 inner splits; agreement, Cohen's kappa and the cost of each method.
- Running: the benchmark (since 28 October 2026, about a day), then the judge agreement study.

## 9. Current implementation

- Environment: conda environment `automl` with Python 3.11. Versions are pinned in `requirements.txt`.
- Data: CSV only, with four public sample datasets included.
- Tasks: binary, multiclass, regression.
- Engine: FLAML 2.7.0 with its default learners (LightGBM, XGBoost, random forest, extra trees, linear models and others).
- Evaluation: independent test split, FLAML's internal validation score and a trivial baseline.
- Explainability: permutation importance, the model's own importance, confusion matrix, ROC curve, residual plots.
- Tracking: local MLflow (SQLite) plus JSON records.
- Autonomous layer (Phase 2): see 8.7.
- Phase 3 so far: see 8.8.
- Tests: 47 pytest tests at the end of Phase 1; 468 after Phase 2 and 533 on 27 October 2026, including short real FLAML runs, whole Sessions with a fake clock, the dashboard through Streamlit's AppTest, smoke tests of the studies, the benchmark harness with stand-in systems, and the statistics against reference values (`report-notes.md`, section 8.6).
- Not finished: the benchmark and judge agreement runs and their results (Phase 3, tasks 26 and 27).
- Not implemented: SHAP, automatic target detection, feature engineering beyond the 9 Strategies, resampling.

## 10. Proposed modules (written for the Phase 1 review)

Status since Phase 2: the modules below are built as rule-based versions (8.7). The Critic was evaluated as planned, by precision and recall on synthetic datasets with planted problems, and the Judge by its false-keep rate and power (`report-notes.md`, sections 9.12 and 9.13). The memory, the learned strategy choice and the evaluation against other systems are Phase 3. The detailed design is the Kiro spec in `.kiro/specs/autopilot-ml/` (requirements, design and ordered tasks); it added two components to the list below: the Assessor, which refits a setup's best configuration on 15 fixed folds of the training data, and the Judge, which keeps a change only when the paired fold differences support it (corrected resampled t-test). The summary below is kept as written for the Phase 1 review.

Every planned module comes with a way to evaluate it:

- `experiment_planner.py`: `plan(profile, memory, budget_state) -> ExperimentConfig`. Evaluated by the number of experiments and the time needed to reach a target score, compared with a fixed plan.
- `critic.py`: `diagnose(result, profile) -> list[Diagnosis]`, each with a type, a severity and the evidence. Evaluated by precision and recall on synthetic datasets where we inject known problems, so the correct answer is known.
- `strategy_engine.py`: `propose(diagnoses, memory) -> ranked strategies`. Evaluated by time to target and final score against random and fixed-order selection.
- `experiment_memory.py`: `record(entry)`, `similar(meta_features, k)`, `strategy_stats(...)`. Evaluated by an ablation with and without memory on datasets the memory has never seen.
- `resource_manager.py`: `allocate(total_budget, history)` and `should_stop(...)`. Evaluated by final score and gain per unit of cost under fixed budgets, against an even split of the budget.

## 11. Autonomous loop (sketch from the Phase 1 review)

Built in Phase 2 in a refined form: every Setup is assessed on the same folds, and the Judge decides whether a change replaces the Accepted_Setup; the memory is written, and since Phase 3 (task 21) the learned selectors can use it to choose the change when it is switched on. The current loop is in [how-it-works.md](how-it-works.md), section 9.

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

## 12. Experiment memory (version 1 built in Phase 2; read by the learned strategy choice since Phase 3, task 21)

Version 1 writes one JSON line per Verdict, finds the nearest past datasets and summarises each Strategy on them; the learned selectors use these summaries. The plan below is from the Phase 1 review. Each record will hold the dataset meta-features (from `DatasetProfile.meta_features()`), the diagnoses before the experiment, the strategy applied, the engine, the model family and hyperparameters, the primary score before and after, and the cost in wall-clock and CPU seconds.

For example, a record might read "severe class imbalance + class_weight + LightGBM gave a change in F1 of x for y seconds", where x and y are measured values.

To use the memory on a new dataset, the meta-features are standardised and the k nearest past datasets are found. For each strategy we then look at the average change in score, the average gain per second and how many times it was tried on those neighbours. The memory starts as the existing JSON records (or SQLite). A vector store such as FAISS or Chroma would only be added if the number of records made it necessary.

## 13. Strategy engine (rule version built in Phase 2; learned ranking built in Phase 3, task 21)

The rule table as planned for the Phase 1 review is below. The built version (Phase 2) implements the rows for overfitting (smaller trees), underfitting and a still-improving search (more search time), class imbalance (class weights), heavy missingness (indicators or dropping), high cardinality (frequency encoding, grouping rare values, or dropping), identifier and constant columns (drop), redundant features (drop one of the pair), weak model families (fewer families) and an unreliable validation score (a smoother search metric). SMOTE, target encoding, threshold tuning and feature interactions are not built. A first rule table, from diagnosis to candidate strategies:

- Overfitting (large train vs validation/test gap): a search space biased towards regularisation, fewer or shallower trees, feature selection.
- Underfitting (low train and test scores): more budget, more expressive model families, feature interactions.
- Class imbalance (moderate or severe): class weights, SMOTE, switching the metric to F1 or ROC-AUC, threshold tuning.
- Heavy missingness: missing-value indicator features, or dropping columns above a threshold.
- High cardinality: frequency or target encoding, or dropping the column.
- Identifier or constant columns: drop them.
- Redundant features: remove one of each highly correlated pair.
- Too much time for too little gain: restrict the model list, reduce the budget, or stop.

The ranking will be developed in three steps: a fixed priority order first, then a multi-armed bandit whose reward is the gain in score divided by cost, then priors taken from the experiment memory. Each step is compared with the previous one on the same benchmark.

## 14. Resource-aware optimisation (built in Phase 2 as the Resource_Manager)

Built: the time, Experiment and CPU limits, budgets from measured time, and the stop rules (8.7); each step's utility (validation gain per second) is recorded. Not built: stopping because no remaining change is expected to be worth its cost, which needs the memory (Phase 3). The plan from the Phase 1 review:

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
- Fixed on 27 October 2026, before any evaluation run, in `research/datasets.json` (task 20). From the laptop-sized datasets (500 to 10,000 rows, at most 100 columns) of two curated OpenML suites, OpenML-CC18 (classification) and OpenML-CTR23 (regression), one evaluation and one development dataset were drawn with a fixed seed from each of six groups (binary imbalanced, binary with missing values or categorical columns, multiclass clean, multiclass messy, regression clean, regression messy).
- Development (10): the four samples, plus wilt, tic-tac-toe, vehicle, balance-scale, kin8nm and student_performance_por. They build the experiment memory.
- Evaluation (6): pc4, cylinder-bands, segment, MiceProtein, energy_efficiency and solar_flare. They are used only by the Phase 3 benchmark, and never to build the memory or tune anything.

## 18. Evaluation metrics

- Predictive quality: the primary metric on the outer test split, plus the other task metrics.
- Efficiency: time to reach a target score (the smallest budget at which a system reaches 99% of the best final score of any system on that dataset and seed), and the score at each budget.
- Experimentation: number of experiments and strategies needed to reach the target.
- Cost: wall-clock seconds, and gain per experiment. CPU seconds are not measured: all runs are on one machine, one at a time.
- Convergence: how much each extra experiment adds, and where the system stops.
- Consistency: average rank across datasets, spread across seeds, and the worst-case gap to the best system.

## 19. Baselines

1. Manual baseline (one fixed, sensible pipeline)
2. FLAML
3. AutoGluon
4. AutoPilot-ML without experiment memory
5. AutoPilot-ML without strategy selection
6. Full AutoPilot-ML

As built (task 23): 1 is `manual` (LightGBM with its library defaults), 2 `flaml`, 3 `autogluon`, 4 `autopilot_rules` (without the memory the learned choice keeps the rule order, so the Session without memory is the rule-order Session), 5 `random_choice`, and 6 `autopilot_full` (Thompson sampling with the frozen memory).

## 20. Ablation study

Each variant removes one part of the full system:

- A0, the full system: the reference.
- A1, no experiment memory: is learning across datasets worth it? (RQ2)
- A2, no strategy engine (random or fixed order): do diagnosis-driven choices help? (RQ1)
- A3, no resource manager (even budget split): does adaptive allocation help? (RQ3)
- A4, no critic (strategies chosen without diagnoses): does explicit diagnosis help?

All variants use the same datasets, budgets, seeds and outer splits. They will be compared with paired tests (Wilcoxon signed-rank, and Friedman with a Nemenyi post-hoc test for several variants at once), following Demšar (2006).

As built (task 22): A1 is `autopilot_rules` (the same Session, see section 19), A2 `random_choice`, A3 `no_resource_manager` and A4 `no_critic`; A3 and A4 keep Thompson sampling and the memory, so each removes one part from A0. The ablations run at the 5-minute budget only, to save time.

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
