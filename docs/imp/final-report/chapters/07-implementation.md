# Implementation (tools and technology used)

## Hardware and environment

All development and every measurement in this report ran on one laptop with an AMD Ryzen 9 6900HS processor (8 cores and 16 threads), 15.2 GB of memory and Windows 11, without a graphics processor. No step needs a network connection after the datasets are downloaded. The project runs in a conda environment named `automl` with Python 3.11; AutoGluon, used only as a comparison system, has its own environment because it needs an older version of pandas, and the benchmark runs it as a separate process.

## Tools and technology used

Table @tab:tools lists the software the project uses, with the versions pinned in its requirements file. Everything is open source and runs offline on a laptop CPU. FLAML was chosen as the search engine because its cost-aware search gives good models on small budgets [@wang2021flaml]; the other libraries were chosen because they are widely used, documented and tested.

::table tab:tools | Tools and libraries used, with their versions | widths=3.2,4.4,2.0,6.2 | align=cccj
| Area | Tool | Version | Used for |
|---|---|---|---|
| Language | Python | 3.11.16 | All the code |
| Environment | Miniconda (conda) | 26.5.3 | The environments `automl` and `autogluon` |
| Data | pandas [@mckinney2010pandas] | 3.0.6 | Loading and handling tables |
| Numerics | NumPy [@harris2020numpy] | 2.4.6 | Arrays and numeric work |
| Statistics | SciPy [@virtanen2020scipy] | 1.17.1 | Student t, Wilcoxon and Friedman tests |
| Machine learning | scikit-learn [@pedregosa2011sklearn] | 1.9.1 | Splits, metrics, permutation importance |
| AutoML engine | FLAML [@wang2021flaml; @flamlrepo] | 2.7.0 | Model and hyperparameter search, used unchanged |
| Learners | LightGBM [@ke2017lightgbm], XGBoost [@chen2016xgboost] | 4.7.0, 2.1.4 | Gradient-boosted trees searched by FLAML |
| Resampling | imbalanced-learn [@lemaitre2017imblearn] | 0.14.2 | SMOTE, SMOTE-NC and SMOTE-N |
| Explanations | SHAP [@lundberg2017shap] | 0.51.0 | SHAP values of the final model |
| Tracking | MLflow [@zaharia2018mlflow] | 3.17.0 | Experiment runs in a local SQLite file |
| Dashboard | Streamlit [@streamlit], Plotly [@plotly] | 1.65.0, 7.1.0 | Web interface and its interactive charts |
| Static charts | Matplotlib [@hunter2007matplotlib] | 3.11.2 | Charts in the documents and this report |
| Web API | FastAPI [@fastapi], Uvicorn [@uvicorn] | 0.142.2, 0.54.0 | The local HTTP service |
| Local language model | Ollama [@ollama] with Qwen2.5 7B Instruct [@qwen2024] | 0.32.1 | Labelled plain-language summaries only |
| Container | Docker Desktop [@merkel2014docker] | 29.6.1 | The image with the API and the dashboard |
| Comparison system | AutoGluon-Tabular [@erickson2020autogluon] | 1.6.3 | The AutoGluon baseline of the benchmark |
| Datasets | OpenML [@vanschoren2013openml] | API v1 | Downloading the 12 benchmark datasets once |
| Testing | pytest [@pytest] | 9.1.1 | The 586 automated tests |
| Version control | Git [@git] | 2.55.0 | History, one change per commit |
| Development environment | Kiro [@kiro] | | AI-assisted editor used to write the specification, code and documents |
| Report | python-docx [@pythondocx] | 1.2.0 | Building this report in the department's format |
::end

## Structure of the code

The code is split into four folders. The `app` folder holds everything a user runs: the services of the loop, the wrapper around the search engine, profiling, evaluation and tracking, the Streamlit pages and the web API. The `research` folder holds the synthetic data, the studies, the comparison systems, the benchmark harness and its statistics; it imports `app` but is never imported by it. The `scripts` folder holds the command-line programs, and the `tests` folder holds the automated tests. In total the code has 20,237 lines in 97 Python files.

Table @tab:modules lists the main modules of the autonomous layer. Each service has typed inputs and outputs, raises clear errors and never imports the user interface, so the same code runs from the dashboard, the scripts, the API and the tests.

::table tab:modules | Main modules of the autonomous layer | widths=6.0,1.4,8.4 | align=jcj
| Module | Lines | What it does |
|---|---|---|
| services/autopilot.py | 680 | The Session loop, its settings, step records and the Final_Evaluation |
| services/automl_runner.py | 1,368 | One Experiment: checks, split, FLAML search with warm starts and search limits, evaluation, records |
| services/assessor.py | 254 | Refits of a configuration on the 15 Session_Folds, with reuse of identical refits |
| services/critic.py | 504 | The 14 rules that turn evidence into Diagnoses, with thresholds read from a file |
| services/strategies.py | 430 | The 12 Strategies and the changes they make to a Setup |
| services/strategy_engine.py | 415 | The rule table, the Proposals and the four selectors |
| services/judge.py | 161 | The corrected resampled t-test and the three Verdicts |
| services/resource_manager.py | 257 | Budgets, the reserve, the stop rules and the cost ledger |
| services/experiment_memory.py | 218 | The append-only memory, similar datasets and arm statistics |
| services/transforms.py | 400 | The Data_Steps, fitted on training rows only, and the saved pipeline model |
| services/splits.py | 249 | The outer split and the Session_Folds, also by groups |
| services/explainer.py, narrator.py | 197, 203 | SHAP explanations and labelled local summaries |
| api.py | 180 | The local web API |
::end

## How one Session runs in the code

A Session is run by the `run` method of the `AutopilotSession` class in `services/autopilot.py`. The dashboard, the command-line script and the web API all call this one method with the dataset and the Session settings, together with a function that reports progress and a function that tells whether the user has pressed Stop. The method follows Algorithm @alg:session in five parts.

1. It checks the settings, detects the task and its metric, makes the outer split and the 15 Session_Folds, and profiles the Training_Part, including the check of each column alone for possible leakage.
2. It allocates a Session ID, creates the Resource_Manager, writes the first version of `session.json` and opens the parent run in MLflow.
3. It asks the Planner for the first Setup, searches it with the first budget, assesses it on the folds and diagnoses it with the Critic.
4. In the loop, the Strategy_Engine makes the Proposals for the Diagnoses of the Accepted_Setup and the chosen selector orders them, the Resource_Manager checks the stop rules and gives the next budget, the Planner applies the first Proposal to a copy of the Accepted_Setup, and the search, the Assessor and the Judge decide on the candidate. Each Verdict is appended to the Experiment_Memory and each step to `session.json`.
5. After the loop, the first and the final model are scored once on the Test_Part, and the record, the MLflow run and the model files are completed.

## Reliability and repeatability

Several rules in the code keep a Session safe and its results repeatable.

- The Experiments inside a Session are run without the Test_Part, and a guard raises an error for any Experiment that used it, so such a result can never be accepted.
- All settings are checked before any work starts, so a wrong setting never leaves a half-made record.
- When a candidate step fails, the step is recorded with its error, the candidate is discarded and the Session goes on with its Accepted_Setup, and a failure to write the memory never stops a Session.
- Every random choice is seeded: the outer split, the Session_Folds, the selectors and FLAML's searches. A search limited by time can still differ slightly between two runs, which is why the benchmark uses 5 seeds.
- The frozen memory of the benchmark is opened read only, and its SHA-256 hash is stored with every run, so every run of the full system can be shown to have read the same memory.

## Records and data formats

Every Session writes a folder `experiments/SES-###/` with `session.json`: the settings, the split and fold fingerprints, the profile of the Training_Part, and for every step the Diagnoses, the Proposal and how it was ranked, the search budget and time, the Assessment summary, the Judge's numbers and Verdict, and the step's cost. Each Experiment of the Session has its own folder with its summary, its Assessment and its Diagnoses or decision, and its model is saved in `models/EXP-###/` together with its fitted Data_Steps, so a reloaded model predicts from raw rows. The same runs appear in MLflow as a parent run with one child run per Experiment [@zaharia2018mlflow]. The Experiment_Memory is one JSON line per Verdict in `experiments/memory/memory.jsonl`, only ever appended to. Records are written to a temporary file and renamed, so a crash never leaves a half-written record.

## Configuration

Project-wide settings are read from environment variables that start with `AUTOPILOT_`, documented in the example file `.env.example`: the logging level, the MLflow tracking address and experiment name, the default time budget, seed and test share, the number of CPU cores for FLAML and the largest CSV file the dashboard accepts. The settings of one Session form a `SessionSettings` object: the target, metric, seed and test share of the base Setup, the budget (time, largest number of Experiments and CPU cores), the fold layout of 5 folds and 3 repeats, the Judge's alpha and margin, the file of Critic thresholds, the selector (`rules`, `random`, `thompson` or `ucb1`), whether the selector reads the memory, and an optional grouping column. The whole object is written into `session.json`, so every Session record states exactly how it was run.

## User interfaces

The Streamlit dashboard has two modes [@streamlit]. The single-experiment mode has seven sections: the dataset (upload or one of four samples, with a profile and data-quality warnings), the target (with the detected task and ranked suggestions), the settings, the run, the results (test scores next to the trivial baseline, training against test scores, the leaderboard and the search progress), the explanations (permutation importance, confusion matrix, ROC curve or residuals) and the history of all runs. The autonomous mode reuses the first two sections and then shows the Session settings, a Start and a Stop button, a live timeline of the steps with each Diagnosis, change, mean difference, p-values and Verdict, a chart of the validation score per step, the first and the final model side by side on the Test_Part, the memory records of the dataset, the SHAP explanation of the final model and an optional generated summary, marked as not used for any decision.

The same work can be started from command-line scripts, for one Experiment, one Session, the studies or the benchmark, and from a local web API built with FastAPI and Uvicorn [@fastapi; @uvicorn]. The API can list the Session records, return one record, start a Session in the background and stop it, and it documents its own endpoints on a page that can also try them. It listens only on the local machine by default, accepts only CSV files inside the project's `data` folder, checks every field of a request and runs one Session at a time. The Docker image holds the code, the samples and the pinned libraries and runs the API on port 8000 and the dashboard on port 8501, with the result folders kept in volumes [@merkel2014docker].

## Key parts of the code

Figure @fig:judgecode shows the centre of the Judge, from `app/services/judge.py`. The fold differences are signed so that a positive value favours the candidate, equal differences are decided on the mean alone, and otherwise the corrected t statistic of Equation (@eq:ttest) gives the two p-values that decide the Verdict [@nadeau2003inference].

::code fig:judgecode | The decision of the Judge (app/services/judge.py, shortened and re-wrapped)
sign = 1.0 if METRICS[metric].greater_is_better else -1.0
raw = sign * (candidate.values(metric, "validation")
              - accepted.values(metric, "validation"))
d = raw[np.isfinite(raw)]
mean, std = float(d.mean()), float(d.std(ddof=1))
if std <= REPEAT_TOLERANCE:  # all equal: decide on the mean alone
    gain = mean if abs(mean) > REPEAT_TOLERANCE else 0.0
    keep = gain > s.min_gain
    keep_simpler = (candidate_is_simpler
                    and gain >= -s.non_inferiority_margin)
else:
    t, df = corrected_resampled_t(d, self.test_train_ratio)
    p_better = float(stats.t.sf(t, df))
    p_worse = float(stats.t.cdf(t, df))
    keep = mean > s.min_gain and p_better <= s.alpha
    keep_simpler = (candidate_is_simpler
                    and mean >= -s.non_inferiority_margin
                    and p_worse > s.alpha)
verdict = ("keep" if keep
           else "keep_simpler" if keep_simpler else "discard")
::end

## Testing

The project has 586 automated tests in 38 test files, run with pytest [@pytest]. They cover the baseline (task detection, profiling and real short FLAML searches), every Critic rule at and around its thresholds, every Strategy, the Judge against a worked example, budgets and stop rules, whole Sessions, the memory and the selectors, the optional parts, the benchmark harness and the real dashboard pages, including a whole autonomous Session and the Stop button. Most tests use small stand-in engines, scripted Assessments and a fake clock, so that a whole Session can be checked in about a second, and the statistics functions are checked against published reference values [@demsar2006; @virtanen2020scipy]. All 586 tests pass, in about 332 seconds.
