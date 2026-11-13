# Implementation (tools and technology used)

## Hardware and environment

All development and every measurement in this report ran on one laptop, described in Table @tab:machine. No GPU was used, and no step needs a network connection after the datasets are downloaded. The project runs in a conda environment named `automl` with Python 3.11; AutoGluon, used only as a comparison system, has its own environment because it needs an older version of pandas.

::table tab:machine | The machine used for all runs | widths=4,10 | align=jj
| Item | Value |
|---|---|
| Laptop | ASUS ROG Zephyrus G15 (GA503RW) |
| Processor | AMD Ryzen 9 6900HS, 8 cores and 16 threads |
| Memory | 15.2 GB, as reported by Windows |
| Graphics processor | Not used |
| Operating system | Windows 11 Home Single Language, version 10.0.26200 |
::end

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

The code is split into four folders, shown in Table @tab:structure. The `app` folder holds everything a user runs; `research` holds the studies, the benchmark and its statistics, which import `app` but are never imported by it. At the end of the project, on 13 November 2026 and without the tools that build this report, the code had 20,237 lines in 97 Python files, counted line by line from the files tracked by Git.

::table tab:structure | Folders of the code and their size | widths=2.8,8.6,1.4,1.6 | align=jjcc
| Folder | Content | Files | Lines |
|---|---|---|---|
| app/ | Services (the loop, the search engine wrapper, profiling, evaluation, tracking), the Streamlit pages and the web API | 33 | 8,939 |
| research/ | Synthetic data, the Critic and Judge studies, the dataset registry, the comparison systems, the benchmark harness, its statistics and analysis | 13 | 3,294 |
| scripts/ | Command-line programs: one Experiment, one Session, the studies, the benchmark, the memory build, explanations and summaries | 12 | 1,183 |
| tests/ | The automated tests, with small stand-in engines for fast checks | 39 | 6,821 |
::end

Table @tab:modules lists the main modules of the autonomous layer. Each service has typed inputs and outputs, raises clear errors, and never imports the user interface, so the same code runs from the dashboard, the scripts, the API and the tests.

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

## Records and data formats

Every Session writes a folder `experiments/SES-###/` with `session.json`: the settings, the split and fold fingerprints, the profile of the Training_Part, and for every step the Diagnoses, the Proposal and how it was ranked, the search budget and time, the Assessment summary, the Judge's numbers and Verdict, and the step's cost. Each Experiment of the Session has its own folder with its summary, its Assessment and its Diagnoses or decision, and its model is saved in `models/EXP-###/` together with its fitted Data_Steps, so a reloaded model predicts from raw rows. The same runs appear in MLflow as a parent run with one child run per Experiment [@zaharia2018mlflow]. The Experiment_Memory is one JSON line per Verdict in `experiments/memory/memory.jsonl`, only ever appended to. Records are written to a temporary file and renamed, so a crash never leaves a half-written record.

## User interfaces

The Streamlit dashboard has two modes [@streamlit]. The single-experiment mode has seven sections: the dataset (upload or one of four samples, with a profile and data-quality warnings), the target (with the detected task and ranked suggestions), the settings, the run, the results (test scores next to the trivial baseline, training against test scores, the leaderboard and the search progress), the explanations (permutation importance, confusion matrix, ROC curve or residuals) and the history of all runs. The autonomous mode reuses the first two sections and then shows the Session settings, a Start and a Stop button, a live timeline of the steps with each Diagnosis, change, mean difference, p-values and Verdict, a chart of the validation score per step, the first and the final model side by side on the Test_Part, the memory records of the dataset, the SHAP explanation of the final model and an optional generated summary, marked as not used for any decision.

The same work can be started from the command line, as Figure @fig:commands shows (an indented line there continues the command above it), and from the local web API of Table @tab:api, built with FastAPI and Uvicorn [@fastapi; @uvicorn]. The API listens only on the local machine by default, accepts only CSV files inside the project's `data` folder, checks every field of a request and runs one Session at a time. The Docker image holds the code, the samples and the pinned libraries and runs the API on port 8000 and the dashboard on port 8501, with the result folders kept in volumes [@merkel2014docker].

::code fig:commands | Commands to run the tests, the dashboard, one Session, the benchmark and the container
conda activate automl
python -m pytest -q
streamlit run app/main.py --server.address localhost
python scripts/run_autopilot.py --dataset data/sample/titanic.csv
    --target Survived --seconds 300 --max-experiments 8 --seed 42
python scripts/run_benchmark.py run --label v2
python scripts/run_benchmark.py analyse --label v2
docker run --rm -p 127.0.0.1:8501:8501 -p 127.0.0.1:8000:8000
    autopilot-ml:0.1.0
::end

::table tab:api | Endpoints of the local web API | widths=4.6,9.8 | align=jj
| Request | What it does |
|---|---|
| GET /health | Answers when the service is running |
| GET /sessions | Lists the Session records |
| GET /sessions/{id} | Returns one whole `session.json` |
| POST /sessions | Starts one Session in a background thread and returns its ID |
| POST /sessions/{id}/stop | Asks the running Session to stop after its current step |
| GET /docs | FastAPI's own page that documents and tries the endpoints |
::end

## Key parts of the code

Figure @fig:judgecode shows the centre of the Judge, from `app/services/judge.py`: the fold differences are signed so that positive favours the candidate, equal differences are decided on the mean alone, and otherwise the corrected t statistic of Equation (@eq:ttest) gives the two p-values [@nadeau2003inference]. Figure @fig:thompsoncode shows the Thompson sampling score of an arm from `app/services/strategy_engine.py`, which is Equation (@eq:thompson) in code [@thompson1933].

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

::code fig:thompsoncode | The Thompson sampling score of one arm (app/services/strategy_engine.py, re-wrapped)
def _score(self, arm, stats, context):
    keep_probability = float(context.rng.beta(
        1.0 + arm.weighted_keeps, 1.0 + arm.weighted_discards))
    gain = (arm.mean_gain_when_kept
            if arm.mean_gain_when_kept is not None else 0.0)
    return keep_probability * gain / max(arm.mean_cost_seconds,
                                         MIN_COST_SECONDS)
::end

## Testing

The project has 586 automated tests in 38 test files, summarised in Table @tab:tests [@pytest]. Most tests use small stand-in engines, scripted Assessments and a fake clock, so that a whole Session can be checked in about a second; others run short real FLAML searches. The statistics functions are checked against published reference values, such as Demšar's critical-difference tables and SciPy's documented Wilcoxon example [@demsar2006; @virtanen2020scipy]. The last full run, on 13 November 2026, passed all 586 tests in 332.4 seconds.

::table tab:tests | The automated tests by area | widths=3.2,1.2,1.4,8.6 | align=jccj
| Area | Files | Tests | Examples of what they check |
|---|---|---|---|
| Baseline | 8 | 116 | Task detection, profiling, real short FLAML runs, the FLAML features the loop relies on, splits, records and MLflow runs |
| Autonomous loop | 11 | 321 | Every Critic rule at and around its thresholds, every Strategy and rule, the Judge against a worked example, budgets and stop rules with a fake clock, whole Sessions, the memory and the selectors |
| Optional parts | 7 | 45 | SMOTE only on training rows, cross-fitted target encoding, grouped folds, SHAP values that add up, planted leaks, local summaries, the API |
| Research tools | 11 | 99 | Synthetic data, the studies, the benchmark harness and its restarts, the statistics against reference values |
| Dashboard | 1 | 5 | The real pages in Streamlit's test tool, including a whole autonomous Session and the Stop button |
::end
