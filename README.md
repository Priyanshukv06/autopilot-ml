# AutoPilot-ML

**Autonomous Machine Learning Experimentation & Self-Optimization Framework**

B.Tech Computer Science major project.

The long-term aim is a system that takes a tabular dataset and a compute budget and runs the whole experimentation loop by itself: train a model, work out what is wrong with it, choose what to try next, run that, and keep the best result. It should also remember what worked on earlier datasets.

We are building it in stages, and this README keeps them apart:

- **Phase 1, the baseline (built):** CSV → dataset profiling → task detection → FLAML AutoML → evaluation → MLflow tracking → Streamlit dashboard.
- **Phase 2, the autonomous loop (built):** a Session that searches, assesses, diagnoses, changes and judges Setups by itself within a time budget, and records every decision in an experiment memory. It runs from the dashboard or the command line.
- **Phase 3, learning and evaluation (built):** fixed development and evaluation datasets, choosing changes from what the memory learned on similar datasets (Thompson sampling, UCB1), switches that turn single parts of the loop off (for the ablations), comparison systems (a manual LightGBM baseline, FLAML alone and AutoGluon), a resumable benchmark with its statistics, a frozen experiment memory, and a study of how far the cheap Judge agrees with searching again. The benchmark was stopped early (23 usable runs on 2 datasets with 1 seed), so its results are observations, not tested claims; see Results below.
- **Optional extensions (built, 1 to 12 November 2026):** SMOTE, cross-fitted target encoding and word features of text columns as new Strategies, group-aware splits, SHAP explanations of a Session's final model, target suggestion and a single-column leakage warning, plain-language Session summaries by a local language model (labelled, never used for decisions), a local HTTP API and a Docker image. The benchmark ran before them, so none of its results include them. A second, full benchmark run with the final code is planned and has not been run yet.

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
- **Choose the target and detect the task:** the app suggests up to three likely target columns, each with its reasons (a target-like name, the last column, only 0/1 or true/false values, no empty cells), and the first is the default choice. You confirm it or pick another column. The app then detects binary classification, multiclass classification or regression, and says why. You can override it, and overrides that cannot work (for example regression on a text column) are rejected with an explanation.
- **Warn about possible leakage:** each column is scored alone as a predictor of the target (5-fold cross-validation). If one scores at least 0.99, the target section warns that it may give the answer away, for example because it is filled in after the outcome. Nothing is changed automatically; you decide whether to exclude it.
- **Train with FLAML:** a stratified 80/20 split, then a FLAML search within your time budget, with your choice of metric and random seed. The best model, its hyperparameters, the validation score, the training time, every trial and a leaderboard per model family are all kept.
- **Evaluate properly:** all metrics are computed on the 20% test set that FLAML never saw. Classification gets accuracy, balanced accuracy, precision, recall, F1, ROC-AUC and log loss. Regression gets MAE, MSE, RMSE and R². Train scores are shown next to test scores, and both are compared with a trivial baseline (always predict the majority class, or the mean). The app warns you when accuracy would be misleading because of class imbalance.
- **Explain the model:** permutation importance on the original columns, the model's own feature importance where it has one, a confusion matrix and per-class report, a ROC curve for binary problems, and predicted-vs-actual and residual plots for regression.
- **Track everything:** each experiment becomes an MLflow run in a local SQLite file (no server needed) with its parameters, metrics and files. A small JSON record is also written to `experiments/`, and the experiment history table in the app reads from those records.
- **Run fully offline:** no OpenAI, Anthropic, Gemini or other paid API, no cloud service and no GPU.

Autonomous run (Phase 2):

- **Run a Session:** give a dataset, a target, a total time, a maximum number of Experiments and the CPU cores. The Session keeps 20% of the rows aside as the test part and never uses them for a decision.
- **Assess fairly:** every Setup's best configuration is refitted on the same 15 folds of the training part (3 repeats of 5-fold cross-validation), so two Setups are compared fold by fold.
- **Diagnose:** a rule-based Critic gives Diagnoses of 14 kinds (for example overfitting, underfitting, class imbalance, high-cardinality, identifier, constant and mostly-empty columns, redundant columns, an unreliable validation score, wasted search time, a text column whose frequent words the model does not use, and possible leakage), each with its evidence numbers. Its thresholds were tuned on synthetic data with planted problems; the two newest kinds (unused words and possible leakage) came later and were not part of that study. Possible leakage is only a warning: no Strategy answers it, because only you know whether a column is known before the outcome.
- **Change one thing at a time:** 12 Strategies answer the Diagnoses (drop columns, frequency-encode, target-encode or group rare categories, replace a text column by its frequent words, add missing-value indicators, balanced class weights or SMOTE, smaller trees, fewer learner families, more search time, a smoother search metric). Data changes are learned inside each fold, so no validation row shapes them. Target encoding is also cross-fitted on the training rows, so no row's encoding uses its own label, and SMOTE adds synthetic rows to training rows only.
- **Keep groups together (optional):** name a grouping column (for example several rows per patient or per store) and whole groups go to the test part, each group stays inside one fold, and the column is not used as a feature.
- **Judge:** a change is kept only if the paired fold differences support it (corrected resampled t-test), or if it is simpler and not worse.
- **Spend the budget:** the first Experiment gets 35% of the time; later ones get budgets from the measured time left; the Session stops for time, Experiments, no more changes to try, three discards in a row, or Stop.
- **Report:** the first and the final model are scored once on the test part, side by side. Every step, Diagnosis and Verdict is saved (`experiments/SES-###/session.json`) and can be replayed.
- **Remember:** every Verdict becomes a record in `experiments/memory/memory.jsonl`, with a description of the dataset, the change, its effect and its cost.
- **Learn which change to try (Phase 3):** with the memory switched on, the `thompson` (Thompson sampling) and `ucb1` selectors put first the changes that were kept, with a good gain per second, on the most similar past datasets; changes with fewer than 3 past records keep their rule-order place.
- **Explain the final model:** on request, SHAP values of the Session's final model on at most 200 rows of the training part (the test part is not used): the columns it relies on most, and the contributions to one chosen row.
- **Summarise in plain words (optional):** on request, a local language model (`qwen2.5:7b-instruct` through Ollama, on the same machine) writes a short summary from the Session record. The text is labelled as generated, numbers that are not in the record are listed next to it, and no decision reads it.

As a service:

- **Local HTTP API:** start, list, read and stop Sessions over HTTP (FastAPI). It listens on 127.0.0.1 by default and runs one Session at a time.
- **Docker image:** the dashboard and the API in one image, with the results kept in Docker volumes.

## Architecture

```text
CSV upload / sample
      │
      ▼
DatasetProfiler ──► TaskDetector + target suggested, confirmed by the user
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

The code has three layers. `app/main.py` and `app/components/` are the Streamlit UI, which only collects input and shows results. `app/services/` holds all the logic, and none of it imports Streamlit, so it can be used from a script or a notebook. `app/core/` holds settings and logging. `app/api.py` is the local HTTP API, which runs the same Session code as the dashboard. FLAML is only touched inside `FLAMLEngine`, and MLflow only inside `ExperimentTracker`.

More detail is in [docs/imp/architecture.md](docs/imp/architecture.md).

## Installation

You need [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and about 1.1 GB of free disk space for the environment. No GPU is needed. We developed and tested this on Windows 11 with Python 3.11. The libraries are cross-platform, but macOS has not been tried yet, and Linux only inside the Docker image (a short Session on 12 November 2026; the tests were not run there).

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

Add `--group-column <column>` to keep the rows with the same value of that column together, in the test part and in every fold (the column is then not used as a feature).

After a Session: SHAP values of its final model on the training part (give the CSV file the Session ran on; writes `experiments/SES-###/shap_summary.json`), and a plain-language summary by a local language model (needs [Ollama](https://ollama.com) running on this machine with the model pulled; writes `experiments/SES-###/summary_generated.json`):

```bash
python scripts/explain_session.py SES-001 --dataset data/sample/titanic.csv
ollama pull qwen2.5:7b-instruct                          # once, for the summaries
python scripts/summarise_session.py SES-001
```

The component studies and the Phase 2 measurement (each restarts where it stopped; summary CSV files go to `results/`):

```bash
python scripts/run_study.py critic run      # then: critic analyse [--freeze LABEL]
python scripts/run_study.py judge run       # then: judge analyse
python scripts/run_phase2.py run            # then: run_phase2.py analyse
python scripts/prepare_datasets.py download # the 12 OpenML datasets of research/datasets.json into data/raw/
```

The single-column leakage check and the target suggestion on every registered dataset, on the training part of the seed-42 split (summary in `results/leakage_check.csv`; on 12 November 2026 no column of the 16 datasets reached 0.99):

```bash
python scripts/check_single_columns.py
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

A second, full run with the final code is planned and has not been run yet. Adding `--label v2` to `build_memory.py` and `run_benchmark.py` makes such a run write to its own folders (`experiments/benchmark-v2*/`, `models/benchmark-v2*/` and `results/benchmark-v2/`), so it can never touch the first run's rows or its frozen memory.

AutoGluon needs pandas 2.3 and this project pins pandas 3.0, so the benchmark runs AutoGluon from its own conda environment (decision D4). Create it once (versions in `research/requirements-autogluon.txt`):

```bash
conda create -n autogluon python=3.11 -y
conda run -n autogluon python -m pip install -r research/requirements-autogluon.txt
```

The local HTTP API (then open http://127.0.0.1:8000/docs for its endpoints). It has no login, so it listens on 127.0.0.1 by default, and a dataset must be a CSV file inside the project's `data/` folder:

```bash
python scripts/run_api.py
```

The Docker image runs the dashboard (port 8501) and the API (port 8000), and keeps `experiments/`, `models/` and `mlruns/` in volumes. Publish the ports on 127.0.0.1 only, because the API has no login. The local summaries are not available inside the image: they need Ollama, which the image does not contain, and the app only talks to a model on its own machine. Records made in a container have no commit hash, because the image holds no git history.

```bash
docker build -t autopilot-ml:0.1.0 .
docker run --rm -p 127.0.0.1:8501:8501 -p 127.0.0.1:8000:8000 \
    -v autopilot-experiments:/app/experiments -v autopilot-models:/app/models -v autopilot-mlruns:/app/mlruns \
    autopilot-ml:0.1.0
```

MLflow UI, to browse every run (then open http://127.0.0.1:5000):

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000 --workers 1
```

On Windows, `--workers 1` is needed. With several workers we got a socket error (WinError 10022). The database file is created by the first experiment.

Tests (`-m "not slow"` skips the tests marked slow: the study smoke tests, the dashboard tests and a few tests with real FLAML searches):

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
- imbalanced-learn: SMOTE, for the optional `smote` Strategy
- shap: SHAP explanations of a Session's final model
- FastAPI and uvicorn: the local HTTP API
- python-dotenv: configuration
- pytest: tests
- Ollama (optional, installed separately, not a Python package): runs the local language model for the plain-language summaries
- Docker (optional): the image with the dashboard and the API

Exact versions are pinned in `requirements.txt`. Optuna is not used, so it is not installed.

## Current limitations

- CSV and single-table tabular data only. No text, images, time series or clustering.
- The app suggests likely target columns with their reasons, but you confirm the choice. The suggestion uses simple signals (the name, the position, the values); its points are design choices, not measured. On the 16 registered datasets the registered target came first on 15 (on penguins it came third).
- The leakage check scores one column at a time and only warns. A column that gives the answer away together with other columns is not found, and only you can tell whether a flagged column is known before the outcome.
- The autonomous loop changes one thing at a time and tries each change at most once per Session. By default it follows a fixed rule order; the learned order needs a memory with records of similar datasets (at least 3 records from at least 2 datasets for a change), and the stopped benchmark could not show whether it helps.
- The Judge decides on refits of the configurations the searches found. A study that searched again for 20 of its decisions agreed with it on 8 (Cohen's kappa 0.05), so one verdict is evidence, not proof.
- Its changes are the 12 Strategies above, including SMOTE and target encoding. There is no other feature engineering (for example no threshold tuning, feature interactions or outlier handling). The newer Strategies were only checked with short runs; the benchmark ran before them.
- Every Setup is refitted 15 times for its Assessment, which takes a share of the budget, more so on bigger data.
- The Judge compares refits of configurations that FLAML chose on the same training part, so both scores are a little optimistic; pairing removes much of this, not all.
- In the single-experiment mode, ID-like and constant columns are suggested for exclusion and high-cardinality columns are only flagged.
- FLAML's search is limited by time, so two runs with the same seed can differ slightly.
- Permutation importance can understate features that are strongly correlated with each other. SHAP is only available for a Session's final model, not in the single-experiment mode. For learners other than tree models it uses a slower estimate on fewer rows (50 rows against 25 background rows).
- The plain-language summaries come from a language model. In the checks of 12 November 2026 one summary misread a verdict and another added judgements of its own, which is why the text is labelled and never used for decisions. They need Ollama and the model on the same machine.
- Models are saved with pickle, so only load model files this app created.
- MLflow stores absolute artifact paths. If you move the project folder, file links in old MLflow runs break (the JSON records are not affected).
- It is a single-user local app with no login. Keep it on localhost. The same holds for the HTTP API (on 127.0.0.1 by default, one Session at a time) and for the ports of the Docker image.
- A Session that is killed together with its process (for example when the API server is stopped while it runs) cannot mark itself as stopped, so its record stays at "running".

## Autonomous layer modules (Phase 2, built)

All in `app/services/`:

- `autopilot.py`: the Session loop, its records, the Final_Evaluation and (Phase 3) the ablation switches.
- `splits.py`: the outer split (the same as Phase 1) and the fixed folds of the training part; with a grouping column, both keep each group on one side.
- `experiment_planner.py`: the first Setup and each candidate Setup, warm-started from the best configurations found so far.
- `assessor.py`: refits a Setup's best configuration on every fold and records the fold scores.
- `critic.py` with `critic_thresholds.toml`: the Diagnoses and their thresholds (version `v1-critic-study`).
- `strategies.py` and `strategy_engine.py`: the 12 Strategies, the rule table that ranks them and (Phase 3) the learned selectors.
- `transforms.py`: the data changes, learned on training rows only and saved with the model (target encoding is cross-fitted on the training rows).
- `judge.py`: the corrected resampled t-test and the keep, keep_simpler and discard rules.
- `resource_manager.py`: the budgets of each step and the stop rules (and, for an ablation, equal shares).
- `experiment_memory.py`: the memory records, the nearest past datasets and per-Strategy statistics.
- `search_summary.py`: how a search spent its time, for the budget Diagnoses.

Optional extensions (tasks 29 to 35, built 1 to 12 November 2026):

- `resampling.py`: SMOTE for the `smote` Strategy, applied by FLAML and the Assessor to training rows only.
- `explainer.py`: SHAP explanations of a Session's final model on rows of its training part.
- `narrator.py`: the plain-language summaries by a local language model; no decision module reads them.
- `profiler.py` and `task_detector.py` gained the frequent words of text columns, the single-column leakage scores and the target suggestion.
- `app/api.py` (outside `app/services/`): the local HTTP API; `Dockerfile` and `scripts/docker_entry.sh`: the Docker image.

The studies that tuned the Critic and the Judge, the Phase 2 measurement, the dataset registry, the comparison systems, the benchmark harness, its statistics and analysis, the frozen memory build and the judge agreement study are in `research/`.

## Research direction

The question we want to answer:

> Can an autonomous experimentation layer improve the efficiency and effectiveness of conventional AutoML by learning from previous experiments, diagnosing model weaknesses, selecting appropriate improvement strategies, and allocating computational resources adaptively?

The benchmark compares a manual LightGBM baseline, FLAML alone, AutoGluon, AutoPilot-ML with the rule order (which is also AutoPilot-ML without the experiment memory) and the full AutoPilot-ML (Thompson sampling with the memory), plus three ablations (random order, no Critic, no Resource_Manager), on 6 evaluation datasets with total budgets of 2 and 5 minutes and 5 seeds. They are compared on test score, time to reach a target score, number of experiments, compute cost and consistency across datasets. The plan is in [docs/imp/research-direction.md](docs/imp/research-direction.md).

## Results

All measured on one laptop (CPU only, 16 logical cores); the details, settings and raw files are in [docs/imp/report-notes.md](docs/imp/report-notes.md), chapter 9.

- **Critic study** (270 synthetic datasets with planted problems): the data-level Diagnoses raised no false alarm; overfitting was found with precision 0.88 and recall 0.91; unreliable validation is weak (precision 0.27).
- **Judge study** (84 pairs of Setups with a known true difference): on the reporting seeds the Judge kept 0 of 18 harmful changes and 10 of 12 helpful ones.
- **Loop against FLAML alone** (4 sample datasets x 3 seeds, 300 s): about the same final test scores on average (4 better, 4 equal, 4 worse), using about two thirds of the time.
- **Benchmark** (stopped early: 23 usable runs, pc4 and cylinder-bands, seed 1): the full system was never below FLAML alone (pc4 F1 0.600 and 0.613 against 0.590 at 120 and 300 s; cylinder-bands equal at 120 s, 0.872 against 0.870 at 300 s). With 2 datasets no test can confirm this.
- **Judge agreement** (20 real decisions, each checked by new searches on 5 inner splits): 8 of 20 verdicts agreed (kappa 0.05), and the strict check cost 2.4 times as much.
- **From the development Sessions:** only changes kept as significantly better raised the test score (+0.016 on average, 8 of 11 Sessions better). Three flaws found in the records were fixed: rounding noise counted once as a gain, unchanged models counted as successes in the memory, and changes seen on only one dataset steered the learned order.

## Use of AI tools

This project was developed in Kiro, an AI-assisted code editor built on VS Code. Kiro's AI agent wrote most of the code, the tests and the documents, following the spec in `.kiro/specs/autopilot-ml/` (requirements, design and ordered tasks), and it also ran the experiments and analysed their results. Priyanshu Verma set the goals and constraints, made the design and run decisions recorded in the spec's `design.md` and in the report notes, and reviewed the work. The application itself uses no paid or cloud AI service: every decision in AutoPilot-ML comes from its own rules and statistics. The only language model it can call is the optional local one (through Ollama, on the same machine) that writes a labelled summary of a finished Session when you ask for it; no decision reads that text.

## Roadmap

1. Baseline: profiling, task detection, FLAML, evaluation, MLflow, dashboard, history (done)
2. Rule-based loop: assessor, critic, strategy catalogue and rule engine, judge, resource manager, planner, records and experiment memory, dashboard autonomous mode, critic and judge studies, measurement against FLAML alone (done)
3. Learned strategy selection (bandits, memory-based), benchmark datasets, ablation switches, comparison systems (manual, FLAML alone, AutoGluon), the benchmark harness, statistics and analysis, the frozen memory and the judge agreement study (done; the benchmark grid was stopped early)
4. Optional extras: SMOTE, target encoding and word features, group-aware splits, SHAP, target suggestion and a leakage check, local language-model summaries, a local HTTP API and a Docker image (done, 1 to 12 November 2026; 586 tests passing)
5. A second, full benchmark run with the final code (`--label v2`; planned, not run yet)

The summary is in [docs/imp/development-roadmap.md](docs/imp/development-roadmap.md). The detailed plan is a Kiro spec (requirements, design and ordered tasks) in the workspace folder `.kiro/specs/autopilot-ml/`, one level above this repository.

## Future work

- The second, full benchmark run with the final code, so that the hypotheses can be tested
- More feature engineering and data-level fixes, for example decision-threshold tuning and outlier handling
- PostgreSQL-backed tracking for multi-user use

Built since the first version of this list (November 2026): target suggestion and a single-column leakage check, target encoding, SMOTE, SHAP explanations of a Session's final model, the optional local language-model summaries (the decisions stay rule-based or learned, and measurable), a FastAPI service and a Docker image.

## Project structure

```text
autopilot-ml/
├── app/
│   ├── main.py             Streamlit entry point (mode switch)
│   ├── api.py              local HTTP API (FastAPI)
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
│                           explain_session.py: SHAP values of a Session's final model
│                           summarise_session.py: a plain-language summary by a local model
│                           check_single_columns.py: the leakage check on every registered dataset
│                           run_api.py: the local HTTP API
│                           docker_entry.sh: starts the API and the dashboard in the image
├── experiments/            JSON records of experiments and Sessions, and the memory (generated, not in git)
├── models/                 saved models (generated, not in git)
├── mlruns/                 MLflow database and files (generated, not in git)
├── tests/                  pytest tests
├── Dockerfile              the image with the dashboard and the API
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
