# Implementation (tools and technology used)

## Hardware and environment

We did all the development and every measurement in this report on one laptop with an AMD Ryzen 9 6900HS processor (8 cores and 16 threads), 15.2 GB of memory and Windows 11, without a graphics processor. Once the datasets are downloaded, no step needs a network connection. The project runs in a conda environment named `automl` with Python 3.11. AutoGluon, which we use only as a comparison system, has an environment of its own because it needs an older version of pandas, and the benchmark runs it as a separate process.

## Tools and technology used

Table @tab:tools lists the software the project uses and the versions we ran, and the Python libraries are pinned to these versions in the project's requirements files. Everything except Docker Desktop is open source, Docker Desktop is free for personal and educational use, and all of it runs offline on a laptop CPU. Each tool is the work of its own authors and is used under its own licence. We chose FLAML as the search engine because its cost-aware search gives good models on small budgets [@wang2021flaml], and we chose the other libraries because they are widely used, well documented and well tested.

::table tab:tools | Tools and libraries used, with their versions | widths=3.2,4.4,2.0,6.2 | align=cccc
| Area | Tool | Version | Used for |
|---|---|---|---|
| Language | Python | 3.11.16 | All the code |
| Environment | Miniconda (conda) | 26.5.3 | The environments `automl` and `autogluon` |
| Data | pandas | 3.0.6 | Loading and handling tables |
| Numerics | NumPy | 2.4.6 | Arrays and numeric work |
| Statistics | SciPy | 1.17.1 | Student t distribution and statistical tests |
| Machine learning | scikit-learn [@pedregosa2011sklearn] | 1.9.1 | Splits, metrics, permutation importance |
| AutoML engine | FLAML [@wang2021flaml; @flamlrepo] | 2.7.0 | Model and hyperparameter search, used unchanged |
| Learners | LightGBM [@ke2017lightgbm], XGBoost [@chen2016xgboost] | 4.7.0, 2.1.4 | Gradient-boosted trees searched by FLAML |
| Resampling | imbalanced-learn | 0.14.2 | SMOTE, SMOTE-NC and SMOTE-N |
| Explanations | SHAP [@lundberg2017shap] | 0.51.0 | SHAP values of the final model |
| Tracking | MLflow | 3.17.0 | Experiment runs in a local SQLite file |
| Dashboard | Streamlit, Plotly | 1.65.0, 7.1.0 | Web interface and its interactive charts |
| Static charts | Matplotlib | 3.11.2 | Charts in the documents and this report |
| Web API | FastAPI, Uvicorn | 0.142.2, 0.54.0 | The local HTTP service |
| Local language model | Ollama with Qwen2.5 7B Instruct | 0.32.1 | Labelled plain-language summaries only |
| Container | Docker Desktop | 29.6.1 | The image with the API and the dashboard |
| Comparison system | AutoGluon-Tabular [@erickson2020autogluon] | 1.6.3 | The AutoGluon baseline of the benchmark |
| Datasets | OpenML [@vanschoren2013openml] | API v1 | Downloading the 12 benchmark datasets once |
| Testing | pytest | 9.1.1 | The 586 automated tests |
| Version control | Git | 2.55.0 | History, one change per commit |
::end

## Structure of the code

The code is organised into four folders. The `app` folder holds everything a user runs: the services of the loop, the wrapper around the search engine, profiling, evaluation and tracking, the Streamlit pages and the web API. The `research` folder holds the synthetic data, the studies, the comparison systems, the benchmark harness and its statistics. It imports `app`, but `app` never imports it. The `scripts` folder holds the command-line programs, and the `tests` folder holds the automated tests. Altogether the code has 20,237 lines in 97 Python files.

Table @tab:modules lists the main modules of the autonomous layer with their size and their job. The work is spread over small, separate services: the largest is the experiment runner around FLAML, and the session loop coordinates all the others in 680 lines. Each service has typed inputs and outputs, raises clear errors and never imports the user interface, so exactly the same code runs from the dashboard, the scripts, the API and the tests.

::table tab:modules | Main modules of the autonomous layer | widths=6.0,1.4,8.4 | align=jcj
| Module | Lines | What it does |
|---|---|---|
| services/autopilot.py | 680 | The session loop, its settings, step records and the final evaluation |
| services/automl_runner.py | 1,368 | One experiment: checks, split, FLAML search with warm starts and search limits, evaluation, records |
| services/assessor.py | 254 | Refits of a configuration on the 15 session folds, with reuse of identical refits |
| services/critic.py | 504 | The 14 rules that turn evidence into diagnoses, with thresholds read from a file |
| services/strategies.py | 430 | The 12 strategies and the changes they make to a setup |
| services/strategy_engine.py | 415 | The rule table, the proposals and the four selectors |
| services/judge.py | 161 | The corrected resampled t-test and the three verdicts |
| services/resource_manager.py | 257 | Budgets, the reserve, the stop rules and the cost ledger |
| services/experiment_memory.py | 218 | The append-only memory, similar datasets and arm statistics |
| services/transforms.py | 400 | The data steps, fitted on training rows only, and the saved pipeline model |
| services/splits.py | 249 | The outer split and the session folds, also by groups |
| services/explainer.py, narrator.py | 197, 203 | SHAP explanations and labelled local summaries |
| api.py | 180 | The local web API |
::end

## How one session runs in the code

A session is run by the `run` method of the `AutopilotSession` class in `services/autopilot.py`. The dashboard, the command-line script and the web API all call this one method with the dataset and the session settings, along with a function that reports progress and a function that tells whether the user has pressed Stop. The method follows Algorithm @alg:session in five parts.

1. It checks the settings, detects the task and its metric, makes the outer split and the 15 session folds, and profiles the training part, which includes checking each column on its own for possible leakage.
2. It allocates a session ID, creates the Resource Manager, writes the first version of `session.json` and opens the parent run in MLflow.
3. It asks the Planner for the first setup, searches it with the first budget, assesses it on the folds and has the Critic diagnose it.
4. In the loop, the Strategy Engine proposes changes for the diagnoses of the accepted setup, the selector orders them, and the Resource Manager checks the stop rules and sets the budget. The Planner, the search, the Assessor and the Judge then try the first proposal. Each verdict goes to the Experiment Memory and each step to `session.json`.
5. After the loop, the first and the final model are scored once on the test part, and the record, the MLflow run and the model files are completed.

## Reliability and repeatability

A number of rules in the code keep a session safe and its results repeatable.

- The experiments inside a session run without the test part, and a guard raises an error for any experiment that used it, so such a result can never be accepted.
- All settings are checked before any work starts, so a wrong setting never leaves a half-made record behind.
- When a candidate step fails, the step is recorded with its error, the candidate is discarded and the session carries on with its accepted setup. A failure to write the memory never stops a session either.
- Every random choice is seeded: the outer split, the session folds, the selectors and FLAML's searches. A search limited by time can still differ slightly between two runs, which is why the benchmark uses 5 seeds.
- The frozen memory of the benchmark is opened read-only, and its SHA-256 hash is stored with every run, so we can show that every run of the full system read the same memory.

## Records and data formats

Every session writes a folder `experiments/SES-###/` containing `session.json`. This file holds the settings, the split and fold fingerprints, the profile of the training part and, for every step, the diagnoses, the proposal and how it was ranked, the search budget and time, the assessment summary, the Judge's numbers and verdict, and the cost of the step. Each experiment of the session has a folder of its own with its summary, its assessment and its diagnoses or decision. Its model is saved in `models/EXP-###/` together with its fitted data steps, so that a reloaded model predicts from raw rows. The same runs appear in MLflow as a parent run with one child run per experiment. The Experiment Memory is one JSON line per verdict in `experiments/memory/memory.jsonl`, a file that is only ever appended to. Records are first written to a temporary file and then renamed, so a crash never leaves a half-written record.

## Configuration

Project-wide settings, such as the logging level, the MLflow address, the default budget, seed and test share, and the number of CPU cores, are read from environment variables that start with `AUTOPILOT_` and are documented in `.env.example`. The settings of one session form a `SessionSettings` object. It holds the target, metric, seed and test share of the base setup, the budget (time, largest number of experiments and CPU cores), the fold layout of 5 folds and 3 repeats, the Judge's alpha and margin, the file of Critic thresholds, the selector (`rules`, `random`, `thompson` or `ucb1`), whether the selector reads the memory, and an optional grouping column. The whole object is written into `session.json`, so every session record states exactly how the session was run.

## User interfaces

The Streamlit dashboard has two modes. The single-experiment mode has seven sections: the dataset (an upload or one of four samples, with a profile and data-quality warnings), the target (with the detected task and ranked suggestions), the settings, the run, the results (test scores next to the trivial baseline, training against test scores, the leaderboard and the search progress), the explanations (permutation importance, confusion matrix, ROC curve or residuals) and the history of all runs. The autonomous mode reuses the first two sections and then shows the session settings, Start and Stop buttons, a live timeline of the steps with each diagnosis, change, mean difference, p-values and verdict, a chart of the validation score per step, the first and the final model side by side on the test part, the memory records of the dataset, the SHAP explanation of the final model, and an optional generated summary that is marked as not used for any decision.

The same work can also be started from command-line scripts, for one experiment, one session, the studies or the benchmark, and from a local web API built with FastAPI and Uvicorn. Through the API one can list the session records, fetch a single record, start a session in the background and stop it, and the API documents its own endpoints on a page where they can also be tried out. By default it listens only on the local machine, accepts only CSV files inside the project's `data` folder, checks every field of a request and runs one session at a time. The Docker image contains the code, the samples and the pinned libraries, and it runs the API on port 8000 and the dashboard on port 8501, with the result folders kept in volumes.

## Key parts of the code

Figure @fig:judgecode shows the core of the Judge, taken from `app/services/judge.py`. The fold differences are signed so that a positive value favours the candidate. If all the differences are equal, the verdict is decided on the mean alone; otherwise the corrected t statistic of Equation (@eq:ttest) gives the two p-values that decide the verdict [@nadeau2003inference].

::code fig:judgecode | The decision of the Judge (app/services/judge.py, shortened and re-wrapped)
sign = 1.0 if METRICS[metric].greater_is_better else -1.0
raw = sign * (candidate.values(metric, "validation") - accepted.values(metric, "validation"))
d = raw[np.isfinite(raw)]
mean, std = float(d.mean()), float(d.std(ddof=1))
if std <= REPEAT_TOLERANCE:  # all equal: decide on the mean alone
    gain = mean if abs(mean) > REPEAT_TOLERANCE else 0.0
    keep = gain > s.min_gain
    keep_simpler = candidate_is_simpler and gain >= -s.non_inferiority_margin
else:
    t, df = corrected_resampled_t(d, self.test_train_ratio)
    p_better = float(stats.t.sf(t, df))
    p_worse = float(stats.t.cdf(t, df))
    keep = mean > s.min_gain and p_better <= s.alpha
    keep_simpler = (candidate_is_simpler and mean >= -s.non_inferiority_margin
            and p_worse > s.alpha)
verdict = "keep" if keep else "keep_simpler" if keep_simpler else "discard"
::end

## Testing

The project has 586 automated tests in 38 test files, run with pytest. They cover the baseline (task detection, profiling and real short FLAML searches), every Critic rule at and around its thresholds, every strategy, the Judge against a worked example, budgets and stop rules, whole sessions, the memory and the selectors, the optional parts, the benchmark harness, and the real dashboard pages, including a whole autonomous session and the Stop button. Most tests use small stand-in engines, scripted assessments and a fake clock, so that a whole session can be checked in about a second, and the statistics functions are checked against published reference values [@demsar2006]. All 586 tests pass, in about 332 seconds.
