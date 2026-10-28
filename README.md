# AutoPilot-ML

**Autonomous Machine Learning Experimentation & Self-Optimization Framework**

B.Tech Computer Science major project.

The long-term aim is a system that takes a tabular dataset and a compute budget and runs the whole experimentation loop by itself: train a model, work out what is wrong with it, choose what to try next, run that, and keep the best result. It should also remember what worked on earlier datasets.

We are building it in stages, and this README keeps them apart:

- **Phase 1, the baseline (built):** CSV → dataset profiling → task detection → FLAML AutoML → evaluation → MLflow tracking → Streamlit dashboard.
- **Phase 2, the autonomous loop (built):** a Session that searches, assesses, diagnoses, changes and judges Setups by itself within a time budget, and records every decision in an experiment memory. It runs from the dashboard or the command line.
- **Phase 3 (in progress):** built so far: the fixed development and evaluation datasets, choosing changes from what the memory learned on similar datasets (Thompson sampling, UCB1), switches that turn single parts of the loop off (for the ablations), the comparison systems (a manual LightGBM baseline, FLAML alone and AutoGluon), a resumable benchmark harness, its statistics and analysis, the frozen experiment memory the benchmark reads, and the judge agreement study. The benchmark has been running since 28 October 2026 and the judge agreement study runs after it, so there are no benchmark results yet.

The detailed project document is [docs/imp/PROJECT.md](docs/imp/PROJECT.md). To understand how the current system and FLAML work, step by step and with diagrams, read [docs/imp/how-it-works.md](docs/imp/how-it-works.md). All maintained documents are in [docs/imp](docs/imp), including the running notes for the final report. The Phase 1 review material is archived in [docs/unimp](docs/unimp).

## Problem

AutoML tools are good at one step: choosing a model and its hyperparameters for a dataset. The work around that step is still manual. Someone has to notice that the model overfits, that accuracy is misleading because one class dominates, or that a column is just an ID. Then they have to decide what to change, run it again and compare. Whatever is learned on one dataset is usually lost by the next one.

## Motivation

- Most of us have a laptop CPU, no GPU and no budget for paid APIs. Compute has to be spent carefully.
- Many of the problems you find after a first run follow familiar patterns (a big train/test gap, a 90% majority class, a column with a unique value per row), and they can be detected from numbers the run already produces.
- If the fixes that worked are recorded together with a description of the dataset, the next similar dataset should need fewer experiments. Testing that idea is the research part of this project.

## Acknowledgement: FLAML

This project started from working with [FLAML](https://github.com/microsoft/FLAML), Microsoft Research's fast and lightweight AutoML library (Wang et al., MLSys 2021, MIT licence). FLAML finds good models quickly on ordinary hardware because its search methods (CFO and BlendSearch) take the cost of each trial into account. That is what made a fully local project like this realistic.

AutoPilot-ML uses FLAML unchanged as its search engine. The baseline adds the parts around it: data profiling, task and metric handling, an independent test evaluation, explanation views, experiment tracking and a dashboard. The autonomous layer sits on top of that. We do not modify FLAML, and we do not claim to outperform it. Comparing against FLAML is part of the evaluation plan.

## What it can do

Single experiment (the Phase 1 baseline):

- **Load data:** upload a CSV in the dashboard, or pick one of the bundled sample datasets.
- **Profile the dataset:** rows, columns, types, missing values, duplicates, unique counts, plus flags for identifier-like, constant, high-cardinality and mostly-empty columns, and the class balance of the target.
- **Detect the task:** you pick the target column and the app detects binary classification, multiclass classification or regression, and says why. You can override it, and overrides that cannot work (for example regression on a text column) are rejected with an explanation.
- **Train with FLAML:** a stratified 80/20 split, then a FLAML search within your time budget, with your choice of metric and random seed. The best model, its hyperparameters, the validation score, the training time, every trial and a leaderboard per model family are all kept.
- **Evaluate properly:** all metrics are computed on the 20% test set that FLAML never saw. Classification gets accuracy, balanced accuracy, precision, recall, F1, ROC-AUC and log loss. Regression gets MAE, MSE, RMSE and R². Train scores are shown next to test scores, and both are compared with a trivial baseline (always predict the majority class, or the mean). The app warns you when accuracy would be misleading because of class imbalance.
- **Explain the model:** permutation importance on the original columns, the model's own feature importance where it has one, a confusion matrix and per-class report, a ROC curve for binary problems, and predicted-vs-actual and residual plots for regression.
- **Track everything:** each experiment becomes an MLflow run in a local SQLite file (no server needed) with its parameters, metrics and files. A small JSON record is also written to `experiments/`, and the experiment history table in the app reads from those records.
- **Run fully offline:** no OpenAI, Anthropic, Gemini or other paid API, no cloud service and no GPU.

Autonomous run (Phase 2):

- **Run a Session:** give a dataset, a target, a total time, a maximum number of Experiments and the CPU cores. The Session keeps 20% of the rows aside as the test part and never uses them for a decision.
- **Assess fairly:** every Setup's best configuration is refitted on the same 15 folds of the training part (3 repeats of 5-fold cross-validation), so two Setups are compared fold by fold.
- **Diagnose:** a rule-based Critic gives Diagnoses of 12 kinds (for example overfitting, underfitting, class imbalance, high-cardinality, identifier, constant and mostly-empty columns, redundant columns, an unreliable validation score, wasted search time), each with its evidence numbers. Its thresholds were tuned on synthetic data with planted problems.
- **Change one thing at a time:** 9 Strategies answer the Diagnoses (drop columns, frequency-encode or group rare categories, add missing-value indicators, balanced class weights, smaller trees, fewer learner families, more search time, a smoother search metric). Data changes are learned inside each fold, so no validation row shapes them.
- **Judge:** a change is kept only if the paired fold differences support it (corrected resampled t-test), or if it is simpler and not worse.
- **Spend the budget:** the first Experiment gets 35% of the time; later ones get budgets from the measured time left; the Session stops for time, Experiments, no more changes to try, three discards in a row, or Stop.
- **Report:** the first and the final model are scored once on the test part, side by side. Every step, Diagnosis and Verdict is saved (`experiments/SES-###/session.json`) and can be replayed.
- **Remember:** every Verdict becomes a record in `experiments/memory/memory.jsonl`, with a description of the dataset, the change, its effect and its cost.
- **Learn which change to try (Phase 3):** with the memory switched on, the `thompson` (Thompson sampling) and `ucb1` selectors put first the changes that were kept, with a good gain per second, on the most similar past datasets; changes with fewer than 3 past records keep their rule-order place.

## Architecture

```text
CSV upload / sample
      │
      ▼
DatasetProfiler ──► TaskDetector + target chosen by the user
      │
      ▼
AutoMLRunner ──► stratified train/test split
      │
      ├──► FLAMLEngine (FLAML search on the training part)
      ├──► Evaluator (test + train metrics, trivial baseline, permutation importance)
      ├──► model saved to models/EXP-###/model.pkl
      └──► ExperimentTracker ──► MLflow (mlruns/mlflow.db) + experiments/EXP-###/summary.json
                                        │
                                        ▼
                              experiment history in the dashboard
```

An autonomous run wraps this in a loop (all in `app/services/`):

```text
Session (autopilot.py) ─► outer split + 3 x 5 folds of the training part (splits.py)
      │
      ▼
Planner: first Setup ─► AutoMLRunner (search only, no test rows) ─► Assessor (refit on the 15 folds)
      ▲                                                                  │
      │                                                                  ▼
Planner: candidate ◄── Resource_Manager ◄── Strategy_Engine ◄────────── Critic
      │
      ▼
search ─► Assessor ─► Judge: keep / keep_simpler / discard ─► Experiment_Memory ─► next step
      ...
Final_Evaluation: first and final model, once, on the test part
```

The code has three layers. `app/main.py` and `app/components/` are the Streamlit UI, which only collects input and shows results. `app/services/` holds all the logic, and none of it imports Streamlit, so it can be used from a script or a notebook. `app/core/` holds settings and logging. FLAML is only touched inside `FLAMLEngine`, and MLflow only inside `ExperimentTracker`.

More detail is in [docs/imp/architecture.md](docs/imp/architecture.md).

## Installation

You need [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and about 1.1 GB of free disk space for the environment. No GPU is needed. We developed and tested this on Windows 11 with Python 3.11. The libraries are cross-platform, but macOS and Linux have not been tried yet.

```bash
conda create -n automl python=3.11 -y
conda activate automl
cd autopilot-ml
pip install -r requirements.txt
```

If you want different defaults (time budget, seed, MLflow location), copy `.env.example` to `.env` and edit it. No API keys are needed.

## Usage

Run these from the `autopilot-ml` folder with the `automl` environment active.

Dashboard (then open http://localhost:8501):

```bash
streamlit run app/main.py --server.address localhost --browser.gatherUsageStats false
```

The app has no login, so `--server.address localhost` keeps it reachable from your own machine only. `--browser.gatherUsageStats false` turns off Streamlit's usage statistics. Choose **Single experiment** or **Autonomous run** in the sidebar.

An autonomous Session from the command line (prints every step; records go to `experiments/`, `models/` and `mlruns/`, or to a temporary folder with `--project-root`):

```bash
python scripts/run_autopilot.py --dataset data/sample/titanic.csv --target Survived --seconds 300 --max-experiments 8 --cores -1 --seed 42
```

The component studies and the Phase 2 measurement (each restarts where it stopped; summary CSV files go to `results/`):

```bash
python scripts/run_study.py critic run      # then: critic analyse [--freeze LABEL]
python scripts/run_study.py judge run       # then: judge analyse
python scripts/run_phase2.py run            # then: run_phase2.py analyse
python scripts/prepare_datasets.py download # the 12 OpenML datasets of research/datasets.json into data/raw/
```

The Phase 3 benchmark (records go to `experiments/benchmark/` and `models/benchmark/`, one row per finished cell to `results/benchmark/`; a restart skips finished cells):

```bash
python scripts/build_memory.py run                       # 30 Sessions on the development datasets (about 2 h)
python scripts/build_memory.py freeze                    # -> results/benchmark/frozen_memory.jsonl
python scripts/run_benchmark.py run --smoke --memory results/benchmark/frozen_memory.jsonl
python scripts/run_benchmark.py run --memory results/benchmark/frozen_memory.jsonl   # the 390 cells
python scripts/run_benchmark.py status                   # cells done, failures, time per system, time left
python scripts/run_benchmark.py analyse                  # tests, tables and charts in results/benchmark/
python scripts/run_study.py agreement run                # the judge agreement study; then: agreement analyse
```

AutoGluon needs pandas 2.3 and this project pins pandas 3.0, so the benchmark runs AutoGluon from its own conda environment (decision D4). Create it once (versions in `research/requirements-autogluon.txt`):

```bash
conda create -n autogluon python=3.11 -y
conda run -n autogluon python -m pip install -r research/requirements-autogluon.txt
```

MLflow UI, to browse every run (then open http://127.0.0.1:5000):

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000 --workers 1
```

On Windows, `--workers 1` is needed. With several workers we got a socket error (WinError 10022). The database file is created by the first experiment.

Tests (`-m "not slow"` skips the study smoke tests and the dashboard tests):

```bash
python -m pytest -q
```

From Python, without the UI:

```python
import pandas as pd
from app.services.automl_runner import AutoMLRunner

df = pd.read_csv("data/sample/titanic.csv")
result = AutoMLRunner().run(
    dataframe=df,
    target_column="Survived",
    task="binary",                  # or None to detect it
    time_budget=60,
    primary_metric="f1",
    excluded_columns=["PassengerId", "Name"],
    dataset_name="titanic",
)
print(result.experiment_id, result.best_model_name, result.test_score)
```

Comparing two setups fairly. One run on one split can mislead, so this script runs setup A and setup B on the same few random splits and prints the per-seed differences. It uses a temporary folder and does not touch the dashboard history.

```bash
python scripts/compare_seeds.py --dataset data/sample/titanic.csv --target Survived --exclude-a PassengerId,Name --exclude-b PassengerId,Name,Ticket,Cabin --seeds 1 2 3 4 5 --budget 30
```

Charts of one experiment's FLAML search: every trial over time, the search time each learner family got, and how the winning family's settings grew. The script only reads the experiment's saved files, and writes PNG files to `docs/imp/images`.

```bash
python scripts/plot_search_history.py --experiment EXP-001
```

## Screenshots

_To be added: dataset profile, results, explainability, experiment history and the MLflow run page._

## Example workflow (Titanic)

1. Start the dashboard and choose **Sample dataset → Titanic passenger survival**.
2. The profile shows 891 rows and 12 columns, with missing values in `Age`, `Cabin` and `Embarked`. The Data quality tab flags `PassengerId` and `Name` as identifiers, and `Ticket` and `Cabin` as high-cardinality.
3. Choose `Survived` as the target. The app detects binary classification and pre-selects `PassengerId` and `Name` for exclusion.
4. Set a time budget (30 to 60 seconds is enough), keep F1 as the metric, and run.
5. Look at the test scores against the trivial baseline, the train vs test gap, the leaderboard and the feature importance.
6. Exclude `Ticket` and `Cabin` as well, run again, and compare both runs in the experiment history and in MLflow.

In the single-experiment mode you make the decision of step 6 by hand. In the autonomous mode the Session makes it: in the first Session on Titanic (SES-001, 300 s) the Critic found overfitting and the two high-cardinality columns, the Session dropped `Cabin` (kept as simpler) and then `Ticket` (kept as better, p = 0.010), and the final model's test F1 was 0.724 against 0.684 for its first model.

## Technology stack

- pandas, NumPy: data handling
- scikit-learn: splitting, metrics, permutation importance
- FLAML (with LightGBM, XGBoost and scikit-learn learners): the AutoML search
- MLflow: experiment tracking, with a local SQLite backend
- Streamlit, Plotly: dashboard and charts
- python-dotenv: configuration
- pytest: tests

Exact versions are pinned in `requirements.txt`. Optuna and SHAP are part of the plan but are not used yet, so they are not installed.

## Current limitations

- CSV and single-table tabular data only. No text, images, time series or clustering.
- You choose the target column yourself. Automatic target detection is not implemented.
- The autonomous loop changes one thing at a time and tries each change at most once per Session. By default it follows a fixed rule order; the learned order needs a memory with records of similar datasets, and whether it helps is what the Phase 3 benchmark will measure.
- Its data changes are the 9 Strategies above. There is no other feature engineering and no resampling (SMOTE is an optional extra, not built).
- Every Setup is refitted 15 times for its Assessment, which takes a share of the budget, more so on bigger data.
- The Judge compares refits of configurations that FLAML chose on the same training part, so both scores are a little optimistic; pairing removes much of this, not all.
- In the single-experiment mode, ID-like and constant columns are suggested for exclusion and high-cardinality columns are only flagged.
- FLAML's search is limited by time, so two runs with the same seed can differ slightly.
- Permutation importance can understate features that are strongly correlated with each other. SHAP is not integrated yet.
- Models are saved with pickle, so only load model files this app created.
- MLflow stores absolute artifact paths. If you move the project folder, file links in old MLflow runs break (the JSON records are not affected).
- It is a single-user local app with no login. Keep it on localhost.

## Autonomous layer modules (Phase 2, built)

All in `app/services/`:

- `autopilot.py`: the Session loop, its records, the Final_Evaluation and (Phase 3) the ablation switches.
- `splits.py`: the outer split (the same as Phase 1) and the fixed folds of the training part.
- `experiment_planner.py`: the first Setup and each candidate Setup, warm-started from the best configurations found so far.
- `assessor.py`: refits a Setup's best configuration on every fold and records the fold scores.
- `critic.py` with `critic_thresholds.toml`: the Diagnoses and their thresholds (version `v1-critic-study`).
- `strategies.py` and `strategy_engine.py`: the 9 Strategies, the rule table that ranks them and (Phase 3) the learned selectors.
- `transforms.py`: the data changes, learned on training rows only and saved with the model.
- `judge.py`: the corrected resampled t-test and the keep, keep_simpler and discard rules.
- `resource_manager.py`: the budgets of each step and the stop rules (and, for an ablation, equal shares).
- `experiment_memory.py`: the memory records, the nearest past datasets and per-Strategy statistics.
- `search_summary.py`: how a search spent its time, for the budget Diagnoses.

The studies that tuned the Critic and the Judge, the Phase 2 measurement, the dataset registry, the comparison systems, the benchmark harness, its statistics and analysis, the frozen memory build and the judge agreement study are in `research/`.

## Research direction

The question we want to answer:

> Can an autonomous experimentation layer improve the efficiency and effectiveness of conventional AutoML by learning from previous experiments, diagnosing model weaknesses, selecting appropriate improvement strategies, and allocating computational resources adaptively?

The benchmark compares a manual LightGBM baseline, FLAML alone, AutoGluon, AutoPilot-ML with the rule order (which is also AutoPilot-ML without the experiment memory) and the full AutoPilot-ML (Thompson sampling with the memory), plus three ablations (random order, no Critic, no Resource_Manager), on 6 evaluation datasets with total budgets of 2 and 5 minutes and 5 seeds. They will be compared on test score, time to reach a target score, number of experiments, compute cost and consistency across datasets. The benchmark has been running since 28 October 2026; there are no results yet. The plan is in [docs/imp/research-direction.md](docs/imp/research-direction.md).

## Roadmap

1. Baseline: profiling, task detection, FLAML, evaluation, MLflow, dashboard, history (done)
2. Rule-based loop: assessor, critic, strategy catalogue and rule engine, judge, resource manager, planner, records and experiment memory, dashboard autonomous mode, critic and judge studies, measurement against FLAML alone (done)
3. Learned strategy selection (bandits, memory-based), benchmark datasets, ablation switches, comparison systems (manual, FLAML alone, AutoGluon), the benchmark harness, statistics and analysis, the frozen memory and the judge agreement study (built); the benchmark and judge agreement runs (in progress)
4. Optional extras: SMOTE, more data strategies, SHAP, local LLM explanations, an API, packaging

The summary is in [docs/imp/development-roadmap.md](docs/imp/development-roadmap.md). The detailed plan is a Kiro spec (requirements, design and ordered tasks) in the workspace folder `.kiro/specs/autopilot-ml/`, one level above this repository.

## Future work

- Automatic target detection and data leakage checks
- More feature engineering and data-level fixes: target encoding, resampling, outlier handling
- SHAP explanations for the whole model and for single predictions
- An optional local LLM (for example through Ollama) to explain diagnoses in plain language. The decisions themselves stay rule-based or learned, and measurable.
- FastAPI service, Docker image, PostgreSQL-backed tracking for multi-user use

## Project structure

```text
autopilot-ml/
├── app/
│   ├── main.py             Streamlit entry point (mode switch)
│   ├── components/         UI sections, including autopilot_viewer.py (autonomous mode)
│   ├── core/               settings and logging
│   └── services/           Phase 1 services and the autonomous layer (see above)
├── research/               synthetic data, the critic and judge studies, the Phase 2 measurement,
│                           the dataset registry, the comparison systems, the benchmark harness,
│                           statistics and analysis, the memory build, the judge agreement study
├── results/                summary CSV files of the studies, measurements and benchmark (in git)
├── data/sample/            small public demo datasets
├── scripts/                run_autopilot.py: one Session from the command line
│                           run_study.py: the critic, judge and judge agreement studies
│                           run_phase2.py: Sessions against FLAML alone
│                           prepare_datasets.py: the OpenML datasets of research/datasets.json
│                           build_memory.py: the frozen memory of the benchmark
│                           run_benchmark.py: the Phase 3 benchmark and its analysis
│                           compare_seeds.py: compare two setups over several splits
│                           plot_search_history.py: charts of one experiment's search
├── experiments/            JSON records of experiments and Sessions, and the memory (generated, not in git)
├── models/                 saved models (generated, not in git)
├── mlruns/                 MLflow database and files (generated, not in git)
├── tests/                  pytest tests
└── docs/
    ├── imp/                maintained: project document, how-it-works guide, architecture,
    │                       roadmap, research plan, notes for the final report, charts
    └── unimp/              archived Phase 1 review material
```

## Sample datasets

- `titanic.csv`: binary classification, target `Survived`. Kaggle "Titanic: Machine Learning from Disaster" training set, from [datasciencedojo/datasets](https://github.com/datasciencedojo/datasets).
- `breast_cancer.csv`: binary classification, target `diagnosis`. Breast Cancer Wisconsin (Diagnostic) from the UCI repository, exported from scikit-learn.
- `penguins.csv`: multiclass classification, target `species`. Palmer Penguins (Horst, Hill and Gorman), from [mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data).
- `california_housing.csv`: regression, target `MedHouseVal`. California Housing (Pace and Barry, 1997), exported from scikit-learn.
