# AutoPilot-ML

**Autonomous Machine Learning Experimentation & Self-Optimization Framework**

B.Tech Computer Science major project.

The long-term aim is a system that takes a tabular dataset and a compute budget and runs the whole experimentation loop by itself: train a model, work out what is wrong with it, choose what to try next, run that, and keep the best result. It should also remember what worked on earlier datasets.

We are building it in two stages, and this README keeps them apart:

- **Current baseline (working now):** CSV → dataset profiling → task detection → FLAML AutoML → evaluation → MLflow tracking → Streamlit dashboard.
- **Proposed research system (planned, not built yet):** experiment planner, critic, strategy engine, experiment memory and resource-aware budgeting on top of the baseline.

The detailed project document is [docs/imp/PROJECT.md](docs/imp/PROJECT.md). To understand how the current system and FLAML work, step by step and with diagrams, read [docs/imp/how-it-works.md](docs/imp/how-it-works.md). All maintained documents are in [docs/imp](docs/imp), including the running notes for the final report. The Phase 1 review material is archived in [docs/unimp](docs/unimp).

## Problem

AutoML tools are good at one step: choosing a model and its hyperparameters for a dataset. The work around that step is still manual. Someone has to notice that the model overfits, that accuracy is misleading because one class dominates, or that a column is just an ID. Then they have to decide what to change, run it again and compare. Whatever is learned on one dataset is usually lost by the next one.

## Motivation

- Most of us have a laptop CPU, no GPU and no budget for paid APIs. Compute has to be spent carefully.
- Many of the problems you find after a first run follow familiar patterns (a big train/test gap, a 90% majority class, a column with a unique value per row), and they can be detected from numbers the run already produces.
- If the fixes that worked are recorded together with a description of the dataset, the next similar dataset should need fewer experiments. Testing that idea is the research part of this project.

## Acknowledgement: FLAML

This project started from working with [FLAML](https://github.com/microsoft/FLAML), Microsoft Research's fast and lightweight AutoML library (Wang et al., MLSys 2021, MIT licence). FLAML finds good models quickly on ordinary hardware because its search methods (CFO and BlendSearch) take the cost of each trial into account. That is what made a fully local project like this realistic.

AutoPilot-ML uses FLAML unchanged as its search engine. The baseline adds the parts around it: data profiling, task and metric handling, an independent test evaluation, explanation views, experiment tracking and a dashboard. The planned research layer sits on top of that. We do not modify FLAML, and we do not claim to outperform it. Comparing against FLAML is part of the evaluation plan.

## What the current baseline can do

- **Load data:** upload a CSV in the dashboard, or pick one of the bundled sample datasets.
- **Profile the dataset:** rows, columns, types, missing values, duplicates, unique counts, plus flags for identifier-like, constant, high-cardinality and mostly-empty columns, and the class balance of the target.
- **Detect the task:** you pick the target column and the app detects binary classification, multiclass classification or regression, and says why. You can override it, and overrides that cannot work (for example regression on a text column) are rejected with an explanation.
- **Train with FLAML:** a stratified 80/20 split, then a FLAML search within your time budget, with your choice of metric and random seed. The best model, its hyperparameters, the validation score, the training time, every trial and a leaderboard per model family are all kept.
- **Evaluate properly:** all metrics are computed on the 20% test set that FLAML never saw. Classification gets accuracy, balanced accuracy, precision, recall, F1, ROC-AUC and log loss. Regression gets MAE, MSE, RMSE and R². Train scores are shown next to test scores, and both are compared with a trivial baseline (always predict the majority class, or the mean). The app warns you when accuracy would be misleading because of class imbalance.
- **Explain the model:** permutation importance on the original columns, the model's own feature importance where it has one, a confusion matrix and per-class report, a ROC curve for binary problems, and predicted-vs-actual and residual plots for regression.
- **Track everything:** each experiment becomes an MLflow run in a local SQLite file (no server needed) with its parameters, metrics and files. A small JSON record is also written to `experiments/`, and the experiment history table in the app reads from those records.
- **Run fully offline:** no OpenAI, Anthropic, Gemini or other paid API, no cloud service and no GPU.

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

The app has no login, so `--server.address localhost` keeps it reachable from your own machine only. `--browser.gatherUsageStats false` turns off Streamlit's usage statistics.

MLflow UI, to browse every run (then open http://127.0.0.1:5000):

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000 --workers 1
```

On Windows, `--workers 1` is needed. With several workers we got a socket error (WinError 10022). The database file is created by the first experiment.

Tests:

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

Step 6 is the kind of decision the planned critic and strategy engine are meant to make automatically. In the baseline you make it by hand.

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
- One click runs one experiment. There is no automatic diagnosis or follow-up experiment yet.
- No feature engineering, resampling (such as SMOTE) or class weighting beyond what FLAML's learners do internally.
- ID-like and constant columns are suggested for exclusion. High-cardinality columns are only flagged.
- FLAML's search is limited by time, so two runs with the same seed can differ slightly.
- Permutation importance can understate features that are strongly correlated with each other. SHAP is not integrated yet.
- Models are saved with pickle, so only load model files this app created.
- MLflow stores absolute artifact paths. If you move the project folder, file links in old MLflow runs break (the JSON records are not affected).
- It is a single-user local app with no login. Keep it on localhost.

## Proposed autonomous architecture (planned)

```text
Dataset ─► Profiler + task detection ─► Experiment planner* ─► FLAML ─► Evaluation
                                              ▲                            │
                                              │                            ▼
                                       Strategy engine* ◄──────────── Critic*
                                              ▲                            │
                                              │                            ▼
                                     Resource manager*           Experiment memory*
```

\* not built yet. Each one will be a module in `app/services/`:

- `experiment_planner.py`: picks the next experiment configuration from the profile, the memory and the remaining budget.
- `critic.py`: produces structured diagnoses such as overfitting, underfitting, class imbalance, problem columns and wasted compute.
- `strategy_engine.py`: maps diagnoses to candidate fixes (class weights, SMOTE, feature selection, regularisation, a different model family) and ranks them.
- `experiment_memory.py`: stores dataset meta-features, the strategy, the model, the result and the cost, and finds what worked on similar datasets.
- `resource_manager.py`: enforces the time, experiment and CPU budgets and decides when to stop.

## Research direction

The question we want to answer:

> Can an autonomous experimentation layer improve the efficiency and effectiveness of conventional AutoML by learning from previous experiments, diagnosing model weaknesses, selecting appropriate improvement strategies, and allocating computational resources adaptively?

We plan to compare six systems: a manual baseline, FLAML, AutoGluon, AutoPilot-ML without experiment memory, AutoPilot-ML without strategy selection, and the full AutoPilot-ML. They will be compared on test score, time to reach a target score, number of experiments, compute cost, convergence and consistency across datasets. The plan is in [docs/imp/research-direction.md](docs/imp/research-direction.md).

## Roadmap

1. Baseline: profiling, task detection, FLAML, evaluation, MLflow, dashboard, history (done)
2. Rule-based critic, strategy catalogue, planner loop, experiment memory, budget enforcement (next)
3. Learned strategy selection (bandits, memory-based), benchmark datasets, AutoGluon baseline, ablation study
4. Optional extras: local LLM explanations, an API, packaging

Steps and exit criteria are in [docs/imp/development-roadmap.md](docs/imp/development-roadmap.md).

## Future work

- Automatic target detection and data leakage checks
- Feature engineering and data-level fixes: encoding high-cardinality columns, resampling, outlier handling
- SHAP explanations for the whole model and for single predictions
- Explicit CPU limits and stopping rules in the resource manager
- An optional local LLM (for example through Ollama) to explain diagnoses in plain language. The decisions themselves stay rule-based or learned, and measurable.
- FastAPI service, Docker image, PostgreSQL-backed tracking for multi-user use

## Project structure

```text
autopilot-ml/
├── app/
│   ├── main.py             Streamlit entry point
│   ├── components/         UI sections
│   ├── core/               settings and logging
│   └── services/           profiler, task detector, AutoML runner, evaluator, tracker
├── data/sample/            small public demo datasets
├── scripts/                compare_seeds.py: compare two setups over several splits
│                           plot_search_history.py: charts of one experiment's search
├── experiments/            JSON experiment records (generated, not in git)
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
