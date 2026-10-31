# Architecture

This page describes the code as it is now: the Phase 1 baseline (one experiment), the Phase 2 autonomous layer (a Session of experiments) and the Phase 3 parts built so far (learned strategy choice, ablation switches, comparison systems, the benchmark harness with its statistics and analysis, the frozen memory build and the judge agreement study). The benchmark was stopped after 33 of its 390 cells, and the judge agreement study ran on 20 pairs; the results are in the report notes (sections 9.16 to 9.18). For a plain-language walkthrough of the same system and of how FLAML works inside it, see [how-it-works.md](how-it-works.md); the full design and the code shapes are in the spec's `design.md`.
 
## Layers

```text
UI (Streamlit)        app/main.py, app/components/      collects input, shows results
Services              app/services/                     all the logic
Core                  app/core/                         settings (.env) and logging
External libraries    FLAML, scikit-learn, MLflow,      used as normal dependencies,
                      pandas, Plotly                    never modified
Local storage         experiments/, models/, mlruns/    no cloud, no server
Research code         research/, scripts/, results/     studies and measurements; the app never imports it
```

We stick to three rules:

- Services never import Streamlit. Everything in `app/services/` can be used from a script, a notebook or a test.
- FLAML is only used inside `FLAMLEngine` in `automl_runner.py`. The rest of the code works with our own result types (`EngineResult`, `ExperimentResult`), so another engine can be added without touching the UI.
- MLflow is only used inside `ExperimentTracker`.

## Main classes

- `Settings` / `get_settings()` (`core/config.py`): paths, MLflow location and default values. Reads `AUTOPILOT_*` environment variables or a `.env` file.
- `DatasetProfiler` → `DatasetProfile` (`services/profiler.py`): the dataset profile, quality flags, class balance and meta-features.
- `analyze_class_balance` → `ClassBalance` (`services/profiler.py`): majority and minority share, imbalance ratio and severity.
- `TaskDetector` → `TaskDetectionResult` (`services/task_detector.py`): task detection, checking overrides, describing target candidates, and the default target suggestion.
- `TaskType` (`services/task_detector.py`): `binary`, `multiclass` or `regression`.
- `Evaluator` → `EvaluationResult` (`services/evaluator.py`): metrics, confusion matrix, trivial baseline and permutation importance.
- `METRICS`, `PRIMARY_METRICS` (`services/evaluator.py`): every metric we know about, and which ones can be optimised for each task.
- `ExperimentConfig` (`services/automl_runner.py`): everything needed to repeat an experiment. The planned planner will produce these.
- `AutoMLEngine`, `FLAMLEngine` → `EngineResult` (`services/automl_runner.py`): the engine interface and its FLAML implementation.
- `AutoMLRunner` → `ExperimentResult` (`services/automl_runner.py`): runs one experiment from start to finish.
- `ExperimentTracker` (`services/experiment_tracker.py`): MLflow runs, JSON records, experiment and Session IDs, the history tables, and atomic writes of `session.json` and the step files.

Autonomous layer (Phase 2):

- `AutopilotSession` → `SessionResult` (`services/autopilot.py`): runs a Session; `SessionSettings` (base Setup, `SessionBudget`, 5 folds x 3 repeats, `JudgeSettings`, thresholds file, selector) and `SessionStep` (one per Experiment). `run(df, settings, progress, stop_requested, on_update)`.
- `make_outer_split` → `OuterSplit`, `make_session_folds` → `SessionFolds` (`services/splits.py`): the Phase 1 split and the fixed folds, each with a fingerprint.
- `Planner` (`services/experiment_planner.py`): `first_setup`, `candidate`, `safe_exclusions`.
- `Assessor` → `Assessment` of `FoldScore`s (`services/assessor.py`): refits on the folds, with a cache per Setup and configuration.
- `summarize_search` → `SearchSummary` (`services/search_summary.py`): search time, time to the best trial, share after it, gain in the last quarter.
- `Critic` → `Diagnosis` (`services/critic.py`), with `CriticThresholds` read from `critic_thresholds.toml` (`tomllib`).
- `StrategySpec`, `Strategy` and `CATALOGUE` (`services/strategies.py`); `propose` → `Proposal`, `RulesSelector` (`services/strategy_engine.py`).
- `DataStepSpec`, `DataPipeline`, `PipelineModel` (`services/transforms.py`): the data changes and the model wrapper that applies them to raw rows.
- `Judge` → `JudgeDecision`, `JudgeSettings`, `corrected_resampled_t` (`services/judge.py`).
- `ResourceManager`, `SessionBudget`, `StepCost`, `StopReason` (`services/resource_manager.py`).
- `ExperimentMemory`, `MemoryRecord`, `StrategyStats`, `dataset_fingerprint` (`services/experiment_memory.py`).
- `FLAMLEngine.fit_config` refits one fixed configuration, and `fit` accepts `starting_points`, `search_space_limits`, `sample_weight` and `max_iter` (all FLAML's public options; FLAML is not changed).

## What happens during one experiment

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant Runner as AutoMLRunner
    participant Det as TaskDetector
    participant Eng as FLAMLEngine
    participant Eval as Evaluator
    participant Trk as ExperimentTracker

    User->>UI: upload CSV or choose a sample
    UI->>UI: DatasetProfiler.profile(df)
    User->>UI: target, task, budget, metric, seed
    UI->>Runner: run_experiment(df, config) in a background thread
    Runner->>Det: detect or validate the task
    Runner->>Runner: drop missing targets, remove excluded columns, stratified split
    Runner->>Trk: allocate_experiment_id() gives EXP-###
    Runner->>Eng: fit(training part, task, metric, budget, seed)
    Eng-->>Runner: model, best config, validation score, trials, leaderboard
    Runner->>Eval: test and train metrics, baseline, permutation importance
    Runner->>Runner: profile with target, save model, write result files
    Runner->>Trk: MLflow run (params, metrics, files)
    Runner->>Trk: summary.json
    Runner-->>UI: ExperimentResult
    UI-->>User: results, explanations, history
```

A few things worth knowing:

- The test split is made before FLAML sees any data, and it is used once, at the end. FLAML does its own validation on the training part (cross-validation for small data, holdout for larger data, chosen by FLAML). The app shows both numbers and labels them separately.
- The UI runs the experiment in a background thread and keeps the thread in the Streamlit session. If the page reruns during an experiment, the progress bar simply picks up again. FLAML gives no progress updates during its search, so the bar is estimated from elapsed time against the budget.
- A failed experiment writes `experiments/EXP-###/error.json` and does not appear in the history as completed.
- FLAML prints a warning on every prediction when a column has categories it never saw during training. During permutation importance that happens dozens of times, so we silence that one message and add a single note per column to the experiment instead.

## Data passed around

`ExperimentConfig` (input): `target_column`, `task_type` (None means detect it), `primary_metric` (None means the default for the task), `time_budget`, `random_seed`, `test_size`, `excluded_columns`, `estimator_list` (None means the engine decides), `dataset_name`, `n_jobs`. Phase 2 added the Setup fields `data_steps`, `class_weight`, `search_metric`, `search_space_limits` and `starting_points` (and `max_iter`, for tests); with their defaults the runner behaves exactly as in Phase 1. A Setup's identity is its `setup_key`, a hash of the fields that change what is learned; the time budget, the trial limit, the starting points, the dataset name and the CPU cores are left out.

Phase 2 records: `ExperimentResult` gained `schema_version` 2, `search_metric`, `test_evaluated`, `best_config_per_estimator`, `search_summary`, `session_id`, `parent_experiment_id`, `proposal`, `setup_key`, `setup` and `profile_rows`; Phase 1 records still load. A Session step holds the Proposal, the Diagnoses it answered, the brief Assessment (mean and spread of the primary metric over the folds), the Judge's decision and the `StepCost` (budget, search, Assessment and other seconds, overrun, utility).

`ExperimentResult` (output, saved as `summary.json`): the ID and time; the dataset, target, task and metric; the budget, seed and test size; the engine and its version; the train/test sizes and feature columns; the best estimator and its configuration; the validation, test and train scores; the full test, train and baseline metrics; the timings and number of trials; the leaderboard; the feature importances; the dataset meta-features; the warnings; the model path, the result folder and the MLflow run ID.

Kept in memory only, for the charts: the trial history, the test predictions and probabilities, and the full dataset profile.

Score conventions: every score is in the metric's normal units. FLAML minimises a loss, so `FLAMLEngine` converts it back (score = 1 − loss for F1, ROC-AUC, accuracy and R²; score = loss for MAE, MSE, RMSE and log loss). In binary problems the positive class is the last label in sorted order, the same as FLAML's label encoder. Multiclass precision, recall, F1 and ROC-AUC are macro averages.

## Files on disk

```text
experiments/EXP-001/
    summary.json                   the record the history table reads
    profile.json                   dataset profile including the target
    evaluation.json                test metrics, confusion matrix, per-class report, train and baseline metrics
    leaderboard.csv                best validation score per model family
    search_history.csv             every FLAML trial
    feature_importance.csv         permutation importance
    native_feature_importance.csv  the model's own importance (if it has one)
    predictions.csv                test predictions (with probabilities or residuals)
    flaml_search.log               FLAML's raw log
models/EXP-001/model.pkl           model + feature columns + target + task + classes
mlruns/mlflow.db                   MLflow database (SQLite)
mlruns/artifacts/                  MLflow copies of the model and the files above
```

These three folders are generated and are not committed to git.

## What happens during one Session

```mermaid
sequenceDiagram
    participant S as AutopilotSession
    participant RM as ResourceManager
    participant P as Planner
    participant R as AutoMLRunner
    participant A as Assessor
    participant C as Critic
    participant E as Strategy engine
    participant J as Judge
    participant M as ExperimentMemory
    S->>S: outer split (Phase 1 rule), 3 x 5 folds, profile of the training part
    S->>RM: first_budget()
    S->>P: first_setup(base, profile)
    S->>R: run_experiment(setup, split, evaluate_test=False)
    S->>A: assess(training part, setup, result)
    S->>C: diagnose(profile, setup, assessment, search summary)
    loop until a stop rule applies
        S->>E: propose(diagnoses, accepted setup, profile, tried)
        S->>RM: check_stop(proposals left), next_budget(proposal)
        S->>P: candidate(accepted setup, result, proposal, budget)
        S->>R: run_experiment(candidate, split, evaluate_test=False)
        S->>A: assess(training part, candidate, result)
        S->>J: compare(accepted assessment, candidate assessment, simpler)
        S->>M: add(record of the Verdict)
        S->>C: diagnose(...) if the candidate was kept
        S->>S: rewrite session.json, call on_update
    end
    S->>S: Final_Evaluation: first and final model on the test part
```

- Every Experiment of a Session runs with `evaluate_test=False`, `compute_importance=False` and `profile_rows="train"`; the Session raises an error if a result says the test part was used.
- A failed candidate (search, Assessment or Judge) is recorded as a discard and the loop goes on; a failed first Experiment ends the Session with status "failed" and stop reason "error".
- Settings errors (bad selector, folds, thresholds file, data) are raised before any file is written.

## Files of a Session

```text
experiments/SES-002/session.json        settings, code version, split and fold fingerprints, every step
                                        (proposal, Diagnoses, brief Assessment, decision, cost),
                                        stop reason and Final_Evaluation; rewritten after every step
experiments/SES-002/train_profile.json  the training-part profile behind the Diagnoses
experiments/EXP-0xx/                    each Experiment's usual files (no test files), plus
    assessment.json                     every fold score
    diagnoses.json                      the Diagnoses of a kept Setup
    decision.json                       the Judge's numbers and reason
experiments/memory/memory.jsonl         one record per Verdict, append-only
experiments/studies/                    saved inputs of the critic study (for replay)
experiments/benchmark/, models/benchmark/   the benchmark's Sessions, Experiments and models
                                        (benchmark-smoke/ for the smoke setting), MLflow off
experiments/benchmark-memory/           the 30 Sessions that built the frozen memory, and its log
experiments/studies/judge_agreement/    the judge agreement study's searches
results/studies/, results/phase2/       summary CSV files (in git)
results/benchmark/benchmark_cells.csv   one row per finished benchmark cell (in git)
results/benchmark/excluded_cells.csv    rows that must not be scored, with the reason (in git)
results/benchmark/frozen_memory.jsonl   the frozen memory the benchmark reads (in git), with
                                        frozen_memory_summary.csv and memory_sessions.csv
results/studies/judge_agreement_*.csv   the judge agreement study's rows and summaries (in git)
```

## Phase 3: built so far and planned

- Built (task 20): `research/datasets.py` and `research/datasets.json`, the fixed development and evaluation datasets.
- Built (task 21): `RandomSelector`, `ThompsonSelector` and `UCB1Selector` next to `RulesSelector` (`services/strategy_engine.py`), with `SelectionContext(memory, meta_features, task_type, fingerprint, rng)`; they read `ExperimentMemory.strategy_stats` for the nearest past datasets when `SessionSettings.use_memory` is on, and each step records its `selection`.
- Built (task 22): ablation switches in `SessionSettings`: `selector="random"`, `use_critic=False` (no Diagnoses; `random_proposals` picks random valid changes) and `use_resource_manager=False` (`EqualShareManager` in `services/resource_manager.py`: equal shares, no reserve, only the time and Experiment limits stop), next to `use_memory`; each is recorded in `session.json` and has a command-line flag.
- Built (task 23): `research/systems.py`. `Problem` is one benchmark cell and `prepare(problem, settings)` gives every system the same `Prepared` split and columns; `ManualSystem` (LightGBM defaults through `fit_config`), `FlamlSystem` (`AutoMLRunner` with the whole budget), `AutoGluonSystem` (runs `research/autogluon_worker.py` in the conda environment `autogluon` as a separate process and scores its predictions with the shared `Evaluator`) and `AutopilotSystem` (a Session with the settings of `AUTOPILOT_VARIANTS` and a read-only memory) each return a `SystemResult`, and a failure is a result, not an exception.
- Built (task 24): `research/benchmark.py`. `grid` and `smoke_grid` make the `Cell`s (system, dataset, budget, seed); `run` appends one row per finished cell to `results/benchmark/benchmark_cells.csv`, skips cells that have a row, and checks the frozen memory's hash; `preflight` checks the datasets, the AutoGluon Python and the memory's sources before a run; `read_cells`, `latest` and `timing` read the rows back. `scripts/run_benchmark.py run | status [--smoke]` is the command line.
- Built (task 25): `research/stat_tests.py`, the benchmark statistics: `friedman` (with Iman and Davenport's F), `nemenyi_cd` and `cliques` (the bars of a critical-difference diagram), `compare_pairs` (Wilcoxon with Holm's correction, rank-biserial correlation, bootstrap interval of the median difference) and `time_to_target`. It is not called `statistics.py`: the AutoGluon worker runs as a script from `research/`, and a `statistics.py` there would hide Python's own module from it.
- Built (task 26): `research/memory_build.py` with `scripts/build_memory.py run | status | freeze` builds the frozen memory (30 Sessions on the development datasets in a random order; only finished Sessions' records are copied to `results/benchmark/frozen_memory.jsonl`, which is never rewritten). `research/benchmark_analysis.py` with `scripts/run_benchmark.py analyse` turns the rows into the tests, per-dataset tables, time-to-target and charts; it compares systems only on the (dataset, seed) blocks where all of them completed.
- Built (task 27): `research/judge_agreement.py` with `scripts/run_study.py agreement run | analyse`: the judged steps of the memory-build Sessions, each compared again with new FLAML searches on 5 inner splits of the Session's training part and the Judge's own rules; agreement, Cohen's kappa and the cost ratio.
- Changed on 31 October 2026, after reading the results: `Judge.compare` treats a mean difference within `REPEAT_TOLERANCE` as 0; `Assessor.refit_key(columns, config, estimator, best_config)` replaces the cache key, so identical refits reuse their fold scores (`Assessment.reused_from`); `SelectionContext.min_datasets` (2) and `StrategyStats.datasets` keep an arm seen on one dataset in rule order, and `MemoryRecord.counts_as_kept` leaves out kept steps that changed no fold score; the benchmark and the judge agreement study wait while the laptop is on battery and exclude a cell or pair that ends on battery (`results/benchmark/excluded_cells.csv`, status `excluded`).

## Why it is built this way

- **Test split outside FLAML:** FLAML's validation score is the number it optimised, so it is a bit optimistic. A separate test set gives an honest final score.
- **Engine interface around FLAML:** FLAML can be replaced, and AutoGluon is needed later as a comparison.
- **`MlflowClient` instead of MLflow's global "active run":** run IDs are explicit, there is no hidden global state, and it works from a background thread.
- **SQLite for MLflow:** recent MLflow versions default to a database backend, and SQLite needs no server.
- **JSON records next to MLflow:** small, readable and independent of the engine, which is what the experiment memory needs.
- **Permutation importance first:** it works on the original columns for every model FLAML can return. SHAP can be added when it is actually needed.
- **Rule-based task detection with an override:** easy to explain and to test. Unclear cases, such as integer codes, are reported with lower confidence.
- **Assessments on fixed folds instead of FLAML's validation score:** FLAML's score is optimistic and comes from its own folds, so two searches cannot be compared fairly with it; refits on the same 15 folds can, fold by fold.
- **Search once, judge cheaply:** each Setup gets one search; the Judge compares refits, not repeated searches, so judging fits inside the loop's budget.
- **Decisions as pure functions of saved records:** the Critic, the Strategy engine and the Judge only read their inputs, which are saved, so every Diagnosis and Verdict can be replayed and tested.
- **Thresholds in a versioned file:** the Critic's numbers can be tuned by a study and frozen under a label without changing code.
- **The test part read once, at the end:** decisions use only the training part, so the final scores are not chosen by the test rows they are measured on.
