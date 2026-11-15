# Notes for the final project report

This file collects the material for the final report while the project is being built, so nothing has to be dug out of old chats or commits later. It is a set of notes, not the report itself.

Chapters 1 to 10 follow the report format. Each chapter has notes and a short "still to add" list. The appendices hold the references, the work log, the figure checklist, open items and where the raw data is.

Rules for keeping it up to date:

- After each piece of work (a module, a design decision, an algorithm, a problem solved, an experiment), add the facts to the right chapter and one line to the work log in Appendix B.
- Only real, measured numbers, each with its settings (dataset, time budget, seed, date) and where the raw files are.
- Mark anything that is not built yet as planned.
- Longer text that is already written lives in `PROJECT.md` (same folder). This file points to it instead of copying it.

Last updated: 2026-10-26

---

## 0. Front matter

- Title submitted to the department: **AutoPilot-ML - An Autonomous, Self-Optimizing Machine Learning Experimentation Framework**. The repository and `PROJECT.md` use the subtitle "Autonomous Machine Learning Experimentation & Self-Optimization Framework". Use the submitted title everywhere in the final report.
- Members (B.Tech CSE): Priyanshu Verma, Sch. No. 23U02016; Salil Kushwah, Sch. No. 23U02003.
- Guide, department and institute: the guide is Dr. Gagan Vishwakarma, Assistant Professor, Department of CSE, IIIT Bhopal. The final report (`docs/imp/final-report/front.toml`) is the major project report and uses the Department of Computer Science and Engineering, IIIT Bhopal, the coordinator Dr. Rahul Chaurasia (the department's template spells it Chourasiya) and the Director Dr. Ashutosh Kumar Singh. The department's template has no field for a group number.
- Phase 1 review: October 2026, completed successfully (exact date to fill in).
- Abstract: write it last, from the final results. Starting points are the abstract in `PROJECT.md` and the project description posted for the department, copied below. When it was posted only the baseline existed; the critic, strategy engine, memory, resource manager and SHAP were still planned.

> AutoPilot-ML is an autonomous machine learning experimentation framework that goes beyond conventional AutoML. Instead of a single model search, it works like a data scientist in a loop. It profiles the dataset (data types, missing values, duplicates, class imbalance, high-cardinality and identifier columns), detects the problem type (binary, multiclass or regression) and trains models with FLAML as its AutoML engine. It then evaluates them on held-out data against baseline models and with explainability. A Critic module diagnoses model weaknesses such as overfitting, underfitting, class imbalance and misleading validation scores. A Strategy Engine then selects the next improvement (feature engineering, class weighting, SMOTE, regularisation or a change of model family) using rule-based selection, multi-armed bandits and meta-learning. An Experiment Memory stores dataset meta-features, strategies, results and compute cost, so the system learns from past experiments and tries first the strategies that worked on similar datasets. A Resource Manager spends the user's time and CPU budget on the experiments that give the most improvement per unit of compute. The system includes MLflow experiment tracking, explainability (permutation importance, SHAP) and an interactive Streamlit dashboard, and runs fully locally with no paid APIs. It will be evaluated against FLAML and AutoGluon with ablation studies. The baseline pipeline (profiling, task detection, FLAML training, evaluation, tracking and dashboard) is already implemented.

---

## 1. Introduction

Notes:

- Tabular data (records in rows and columns) is still the most common kind of data in applied machine learning. Gradient-boosted trees such as LightGBM and XGBoost are strong on it.
- AutoML libraries (auto-sklearn, TPOT, AutoGluon, FLAML) automate choosing a model and its hyperparameters, known as the combined algorithm selection and hyperparameter optimisation (CASH) problem (Thornton et al., 2013).
- They do not automate the loop around that search: reading the results, finding out why the model is weak, choosing what to change in the data, the features or the model family, running again, comparing fairly and deciding when to stop. What is learned on one dataset is rarely reused on the next.
- Motivating example, measured on Titanic (EXP-001, 60 s budget, seed 42): test F1 0.673 against train F1 0.934, so the model overfits. `Ticket` has 681 different values in 891 rows, and 63% of the test passengers have a ticket number the model never saw in training. A person would drop or re-encode the column, run again and compare. AutoPilot-ML aims to do this loop itself.
- Origin of the project: it started from working with FLAML (Microsoft Research; Wang et al., MLSys 2021; MIT licence). FLAML finds good models quickly on a laptop because its search takes the cost of each trial into account. The questions after every run were still left to the user, and that gap became this project. FLAML is used unchanged as the search engine. We do not modify it and do not claim to beat it; comparing against it is part of the evaluation.
- Constraints that shape the design: a laptop CPU, no GPU, no paid APIs, no cloud services. Every decision must be measurable and work offline.
- Research question: "Can an autonomous experimentation layer improve the efficiency and effectiveness of conventional AutoML by learning from previous experiments, diagnosing model weaknesses, selecting appropriate improvement strategies, and allocating computational resources adaptively?"
- Text already written: `PROJECT.md` Abstract, Acknowledgement, sections 1 and 2.

Draft of the paragraph on how the report is organised (31 October 2026; adjust it to the final chapter titles): Chapter 2 reviews AutoML systems and the ways they learn across datasets. Chapter 3 states the problem, the objectives and the hypotheses. Chapter 4 describes the method: the phases, the datasets, the benchmark and the studies. Chapter 5 gives the algorithms, from one experiment to the autonomous loop, the learned choice of changes and the statistics. Chapter 6 shows the diagrams, chapter 7 the tools, and chapter 8 the implementation and its tests. Chapter 9 reports every measurement with its settings: the Phase 1 experiments, the component studies, the loop against FLAML alone, the frozen memory, the stopped benchmark and the judge agreement study. Chapter 10 draws the conclusions and lists the future work.

---

## 2. Literature review

Notes by topic. Numbers in brackets refer to the reference list in `PROJECT.md` section 26; other sources are in Appendix A.2.

- Hyperparameter search: grid and random search [12]; Bayesian optimisation with SMAC [10], TPE [11] and Gaussian processes [13]; early stopping of weak configurations with Successive Halving [14], Hyperband [15] and BOHB [16].
- Model selection together with tuning (CASH): Auto-WEKA [8]; auto-sklearn [4, 5], which also starts its search from configurations that worked on similar datasets (meta-learning) and builds ensembles.
- Pipeline search: TPOT evolves whole pipelines with genetic programming [7].
- Ensembles with strong defaults: AutoGluon-Tabular stacks many models instead of searching for long [6].
- Cost-aware search: FLAML [1] with CFO [2] and BlendSearch [3]. FLAML also has a "zero-shot" mode that picks data-dependent default configurations mined offline from many datasets (Kayali and Wang, 2022).
- Time-limited model selection using other datasets: OBOE fits a low-rank model to a matrix of model errors across many datasets and uses it to choose promising models for a new dataset within a time budget (Yang et al., 2019).
- Meta-learning in general: survey by Vanschoren [17]. OpenML as a source of datasets and results [24]. The AutoML Benchmark for fair comparisons [23].
- Bandits, for choosing between options under uncertainty: UCB1 [18] and Thompson sampling (Thompson, 1933).
- Data problems the critic will look for: class imbalance and SMOTE [19]; data leakage, for example through ID-like columns (Kaufman et al., 2012).
- Honest evaluation: picking the best of many configurations on the same validation data makes the winner's validation score optimistic (Cawley and Talbot, 2010). Comparing methods over many datasets needs paired tests such as Wilcoxon and Friedman [22].
- Explanations: permutation importance, introduced with random forests [20], in its model-agnostic form (Fisher et al., 2019); SHAP [21] (built in optional task 32, 12 November 2026: exact tree SHAP for tree models, the permutation estimate for other models).
- LLM-based data science: CAAFE for feature engineering [26] and MLAgentBench for agents that run ML experiments [27]. These depend on a large language model, and the papers used hosted models. Related work only: AutoPilot-ML does not use an LLM for decisions.
- Experiment tracking: MLflow [25].

Draft comparison. Check each cell against the papers before using it in the report.

| System | Model and hyperparameter search | Changes to data and features | Explains why a model is weak | Learns across datasets | Budget decisions across experiments |
|---|---|---|---|---|---|
| auto-sklearn | Yes, Bayesian optimisation | Partly: preprocessing steps inside the search space | No | Yes: warm start from data collected offline | Time limit only |
| TPOT | Yes, genetic programming | Yes: pipeline operators | No | No | Generations or time limit |
| AutoGluon-Tabular | Little: strong defaults and stacking | Built-in preprocessing | No | Fixed presets; newer versions also use portfolios built offline | Time limit and presets |
| FLAML | Yes, cost-aware | Built-in preprocessing | No | Zero-shot defaults mined offline | Cost-aware inside one search |
| OBOE | Model selection | No | No | Yes: error matrix over many datasets | Time-limited model selection |
| LLM agents (CAAFE, MLAgentBench) | Through the LLM | Yes, written by the LLM | Implicitly, inside the LLM | No | No |
| AutoPilot-ML (planned) | Through FLAML | Yes: strategy engine | Yes: critic | Yes: memory of changes with their effect and cost, from the user's own runs | Yes: resource manager |

Points to make in the chapter:

- Existing cross-dataset learning (auto-sklearn's warm start, FLAML's zero-shot defaults, OBOE) transfers good model and hyperparameter choices mined offline. AutoPilot-ML's memory stores changes to the experiment (drop or encode a column, class weights, resampling, another model family) with their measured effect and cost, collected from the user's own runs.
- None of the AutoML systems above explains why a model is weak or decides to change the data. LLM agents can, but they need a large language model and are hard to reproduce and measure.
- Phase 1 already showed why careful comparison matters (chapter 9). This connects to Cawley and Talbot (2010) and to Demšar [22].

Still to add: read the papers in Appendix A.3, check the comparison table, write one paragraph per group of works.

---

## 3. Problem definition and objectives

Problem (from `PROJECT.md` section 3): given a tabular dataset D with a target column y, a task T (binary classification, multiclass classification or regression), a primary metric m and a compute budget B (time, number of experiments, CPU cores):

1. find a pipeline p (data treatment, features, model, hyperparameters) with the best possible test performance m(p), and
2. reach it using as little of B as possible.

The baseline solves part 1 with one AutoML run. The research system works on both parts with a sequence of experiments e1, e2, ..., ek, where each next experiment is chosen from the diagnoses of the earlier ones and from a memory of experiments on other datasets.

| No. | Objective | Status |
|---|---|---|
| O1 | Baseline: profiling, task detection, FLAML training, held-out evaluation, explanations, experiment tracking, dashboard | Done (Phase 1) |
| O2 | Critic: automatic, structured diagnosis of each experiment, with evidence | Done (Phase 2: 12 kinds, measured by the critic study; the optional tasks added unused_words and the warning-only possible_leakage, not measured by a study) |
| O3 | Strategy engine: map diagnoses to candidate changes and rank them (rules, then bandits, then memory) | Rules done (Phase 2: 9 Strategies; 12 with the optional smote, target_encode and token_features); bandits on memory records built (Phase 3, task 21: Thompson sampling and UCB1); measured in the second, full benchmark run (13 November 2026, section 9.20): no difference beyond the intervals over 6 datasets (H2 and H3 not shown) |
| O4 | Planner and autonomous loop, with a fair comparison of every change over several splits | Done (Phase 2: Assessor on 15 folds and the Judge, measured by the judge study) |
| O5 | Experiment memory: reuse what worked on similar datasets | Recording done (Phase 2); reuse built (Phase 3, task 21: the learned selectors read the nearest datasets' records); the frozen memory built on 27 October 2026 (97 records, section 9.15); a second frozen memory built by the final code on 12 November 2026 (95 records); its effect, measured in the second benchmark run, is within the intervals (H2 not shown; section 9.20) |
| O6 | Resource manager: share the time and CPU budget and decide when to stop (utility = improvement / cost) | Done (Phase 2: budgets, stop rules and the utility of each step recorded); stopping on expected utility needs the memory (Phase 3) |
| O7 | Evaluation against a manual baseline, FLAML and AutoGluon, with ablations, on public datasets and several seeds | Built (tasks 20 to 26). The first run (started 28 October 2026; 390 cells: 6 OpenML evaluation datasets, budgets of 120 and 300 s, 5 seeds) was stopped after 33 cells, 23 usable (section 9.16). The second run, with the final code, completed all 390 cells on 12 and 13 November 2026 (section 9.20): H1 contradicted, H2 to H4 not shown |

- Scope: CSV, single-table tabular data; binary, multiclass and regression; laptop CPU; fully offline.
- Out of scope: text, images, time series, clustering; LLMs as decision makers; cloud or paid services.
- Limits in the current code: time budget 5 to 3600 s, at least 20 labelled rows, test share 0.05 to 0.5. The user chooses the target column.
- Research questions RQ1 to RQ4 and hypotheses H1 to H4: `PROJECT.md` sections 15 and 16, and `research-direction.md`.

Success criteria for the benchmark, written on 28 October 2026 at about 14:25. At that time 8 of the 390 cells had finished, all on pc4 with seed 1, and their scores had been printed in the run's log; the criteria below only translate the hypotheses (written in October 2026) into the analysis of section 4.10 and do not use those scores. "Difference" is the per-dataset difference of signed test scores (each dataset's mean over its seeds; positive means the first system is better), and "interval" is its 95% bootstrap interval of the median over the 6 evaluation datasets.

| Hypothesis | Comparison | Supported if | Contradicted if |
|---|---|---|---|
| H4: the full system is not worse than FLAML at the same budget | `autopilot_full` against `flaml`, at 120 s and at 300 s | the interval lies above -0.005 (the Judge's margin) at both budgets | the interval lies below -0.005 at either budget |
| H1: the full system reaches the target sooner than FLAML | time-to-target, `autopilot_full` against `flaml` | it reaches the target at an earlier budget in more (dataset, seed) pairs than the other way round, and its median time-to-target is not later | the other way round |
| H2: without the memory, the loop does worse | `autopilot_full` against `autopilot_rules`, at 300 s (also shown at 120 s) | the interval lies above 0 | the interval lies below 0 |
| H3: without strategy selection, the loop does worse | `autopilot_full` against `random_choice`, at 300 s | the interval lies above 0 | the interval lies below 0 |
| RQ3 and the Critic: adaptive budgets and diagnoses help | `autopilot_full` against `no_resource_manager` and against `no_critic`, at 300 s | the interval lies above 0 | the interval lies below 0 |

Outcome (31 October 2026): the benchmark stopped after 23 usable cells on 2 datasets with 1 seed (section 9.16), so none of H1 to H4 can be decided by these criteria: every interval over 2 datasets is wide and rests on one seed. The observations are reported as such in section 9.16 and chapter 10.

Outcome of the second, full run (13 November 2026; all 390 cells, code af8f14d, section 9.20): H1 contradicted (FLAML reached the target at an earlier budget in 10 of 30 (dataset, seed) pairs, the full system in 2; median time-to-target 120 s against 300 s). H4 not shown: not supported at either budget, and not contradicted, although at 120 s the full system was behind FLAML on all 6 datasets (median -0.0137, interval -0.0183 to -0.0022). H2, H3 and the ablations of RQ3 and the Critic not shown: every interval contains 0.

Otherwise the result is "not shown". The Wilcoxon p-values with Holm's correction are reported next to these, but with 6 datasets they cannot fall below 0.05 for any pair once two or more pairs are compared (section 5, Algorithm 21), so no claim rests on them alone. The Friedman test and the Nemenyi critical difference show whether any system stands apart from the others at a budget.

---

## 4. Methodology and work description

### 4.1 Approach

- Build and measure a reliable baseline first, then add one autonomous module at a time. Each module gets tests and an evaluation method before it is used in any claim.
- Use mature open-source libraries instead of rebuilding them (FLAML, scikit-learn, MLflow, Streamlit). FLAML is used unchanged.
- Rules set at the start of the project: do not over-engineer; no LLM just for the label "agentic"; no paid APIs; no cloud; business logic separate from the UI; reusable modules with type hints; clear error handling; tests for the important parts; an evaluation method for every research component; never invent results, and label anything not built as planned.
- One logical change per commit, with conventional commit messages.
- Spec-driven development (from Phase 2): the plan is a Kiro spec in `.kiro/specs/autopilot-ml/` (in the workspace folder, outside this repository): `requirements.md` (32 requirements, 160 numbered acceptance criteria written as WHEN/IF/THEN/SHALL statements), `design.md` (components, code shapes, rules, data formats, decisions D1 to D8) and `tasks.md` (35 tasks plus 21 test and run sub-tasks, each one commit, each linked to the criteria it satisfies, with a dependency graph and groups of tasks that can run in parallel). It was written on 26 October 2026, after the Phase 1 review, to fix the order of work and the interfaces before building. A script checked that every task's requirement links exist and that every criterion is covered.

### 4.2 Phases

| Phase | Content | Status |
|---|---|---|
| 1 | Baseline: profiling, task detection, FLAML, evaluation, MLflow, dashboard, history | Built 7 October 2026; reviewed in October 2026 |
| 2 | Rule-based critic, strategy catalogue and engine, fair comparison of changes, planner loop, memory, resource manager | Next |
| 3 | Learned strategy selection (bandits, memory), AutoGluon engine, benchmark, ablations | Planned |
| 4 | Optional local LLM explanations, SHAP, API, packaging | Later |

Steps and "done when" checks for each phase: `development-roadmap.md` (summary) and the spec's `tasks.md` (detail). Phase 2 is built in four milestones: M0 groundwork (tasks 1 to 4.1), M1 a thin loop that runs end to end with three diagnoses and three strategies (tasks 5 to 11.2), M2 the full version with the component studies (tasks 12 to 17), M3 the dashboard and the Phase 2 measurement (tasks 18 and 19).

### 4.3 Work done in Phase 1

In order (dates and commits in Appendix B):

1. Checked the machine and tools, created the conda environment `automl` with Python 3.11 and installed pinned libraries.
2. Created the project structure, settings and logging.
3. Built the services: task detector, dataset profiler, evaluator, experiment tracker, and the AutoML runner with the FLAML engine.
4. Added four small public datasets.
5. Built the Streamlit dashboard with seven sections, including the experiment history.
6. Wrote the tests, checked the dashboard headlessly on all four datasets and checked metric values against scikit-learn.
7. Wrote the documentation, ran the first experiments, wrote the seed comparison script, then prepared and gave the Phase 1 review.

### 4.4 How correctness was checked

- 47 pytest tests (section 8.6), including short real FLAML runs: about 29 s on 7 to 12 October, about 41 s in three runs on 22 October, with no code change in between.
- Streamlit's headless test tool drove the real dashboard on all four sample datasets without errors. The demo flow was rehearsed with 23 automated checks, all passed (7 October 2026).
- ROC-AUC, F1 and accuracy shown by the app were recomputed with scikit-learn from the saved predictions and matched exactly (Titanic: ROC-AUC 0.7868247694334651 both ways; the predicted class matched the higher probability on every row).
- MLflow runs, metrics and files were read back through the MLflow server's REST API.
- A saved model was reloaded and used to predict.

### 4.5 Experimental method so far

- Outer split: stratified 80/20 train/test split, made before FLAML sees any data. The test part is used once, at the end.
- FLAML searches on the training part and validates each trial with 5-fold cross-validation or a holdout split, using its own rule (section 5.2).
- Reported for every run: test metrics, train metrics, FLAML's validation score, a trivial baseline on the same test rows, training time and number of trials.
- Comparing two setups: both are run on the same splits (same seed) and compared split by split (`scripts/compare_seeds.py`), with a paired t-test and a Wilcoxon signed-rank test on the differences. For a single split, a paired bootstrap over the test rows gave a confidence interval for the difference (section 9.3).
- Since Phase 2: the loop decides on the training part only, over 15 fixed folds (Assessor and Judge, sections 5.4 and 9.13); the test part only reports the first and the final model at the end. Component studies use synthetic data with a fresh sample as the truth, with separate tuning and reporting seeds. The loop against FLAML alone: same total budget, same split and columns, 3 seeds per dataset, compared pair by pair (section 9.14).

### 4.6 Design decisions

| Decision | Reason |
|---|---|
| Test split made outside FLAML | FLAML's validation score is the number it optimised, so it cannot also be the independent final score |
| FLAML behind a small engine interface | FLAML stays replaceable; AutoGluon is needed later for comparison |
| Services never import Streamlit | The same code runs from the UI, scripts, notebooks and tests |
| `MlflowClient` with explicit run IDs instead of MLflow's global "active run" | No hidden global state; works from a background thread |
| MLflow on a local SQLite file | MLflow 3 defaults to a database backend; SQLite needs no server |
| A small JSON record per experiment, next to MLflow | Readable and independent of the engine; raw material for the experiment memory |
| Permutation importance first, SHAP later | Works on the original columns for every model FLAML returns |
| Rule-based task detection with a user override | Easy to explain and test; unclear cases such as integer codes get lower confidence |
| Experiment IDs reserved by creating the folder | Two runs started together can never get the same ID |
| Experiment runs in a background thread | Streamlit reruns the script on every click; the run and its progress survive reruns |
| Autonomous run: the page does not wait in a loop; a fragment redraws the live view every second (task 18) | A Session takes minutes, and Stop must stay clickable while it runs; the dashboard tests can then press Start and Stop between page runs |
| The live view is drawn only from the session record, which the Session hands to the page after every save (`on_update`) | One source of truth: the page shows exactly what `session.json` holds, without reading a file that the Session may be rewriting |
| Benchmark budgets of 120 and 300 s only; ablations at 300 s only (task 24, agreed with the user on 27 October 2026) | The 600 s budget would add about 20 hours to a grid of about 22 hours on one laptop; two budgets still give a score-against-budget comparison and time-to-target |
| Benchmark rows are only appended; a failed cell is run again as a new row and the analysis reads each cell's last row | Raw results are never edited, and an overnight run can be stopped and restarted at any point |

### 4.7 Problems met and how they were solved

| Problem | Solution |
|---|---|
| FLAML printed a warning on every prediction when a column had values not seen in training: 96 warning lines in one 10-second Titanic run, mostly during permutation importance | Silenced that message and added one note per column with the share of affected test rows (commit 2fddc57) |
| In longer runs the warnings came back, because scikit-learn's parallel helper calls `warnings.resetwarnings()` in worker threads, which wipes the process-wide filter list | Also drop these two kinds of warnings (unseen categories and convergence) where Python prints warnings, with a test (commit ca536c7) |
| `mlflow ui` crashed on Windows with several workers (`OSError: [WinError 10022]`) | Start it with `--workers 1` |
| MLflow 3 prints an "agent hint" banner on import, and its SQLite store prints a deprecation warning | `MLFLOW_DISABLE_AGENT_HINT=1`, and that warning is filtered |
| MLflow 3 creates its database in whatever folder it is started from | Fixed locations: `mlruns/mlflow.db` and `mlruns/artifacts` |
| Streamlit listens on all network interfaces by default and has no login | Start it with `--server.address localhost` |
| FLAML reports no progress during its search | The progress bar is estimated from elapsed time against the budget |
| Tests with a 3-second budget failed the 5-second minimum | Tests use 5 s |
| pandas 3.0 has a new default text type (`str`) | Checked that FLAML 2.7 handles it |
| Streamlit 1.65 deprecates `use_container_width` | `width="stretch"` |
| Saved models are pickle files, which can run code when loaded | Documented: only load files this app created |
| MLflow stores absolute file paths | Documented: moving the project folder breaks file links in old MLflow runs; the JSON records still work |
| Ranking tie in SES-001 (section 9.11): with overfitting high, "drop `Cabin`" and "drop `Ticket`" both answered a high Diagnosis, had the same cost and table position, and the tie was broken by the key in alphabetical order, so `Cabin` (only moderately high-cardinality) was tried before `Ticket` (high). It cost one extra step and did not change the final Setup | Not applied (decided on 26 October 2026). The proposed tie-break used the severities of all the Diagnoses a Proposal answers. With the full Critic (task 13), `Cabin` (77% empty) also gets a high missing-values Diagnosis, so "drop `Cabin`" answers two high Diagnoses too and that tie-break would keep the same order. The extra step was not wasted either: dropping `Cabin` was kept as the simpler Setup. The ranking rule of Requirement 8.2 stays as it is |
| Warm starts lost after "smaller trees" in SES-001 (EXP-006): the Accepted_Setup's best configurations of LightGBM, XGBoost and depth-limited XGBoost were outside the limited search space, so the engine left them out (with warnings, as designed) and those searches started from FLAML's defaults | Not applied (decided on 26 October 2026). A configuration moved into the limited space has never been evaluated, and in a changed space FLAML's own cheap starting point is a sound start, since its search grows from low-cost configurations. Whether moving would help needs its own measurement; listed as future work (chapter 10) |
| A dataset with repeated row labels would silently pair features with the wrong target values: the runner matches rows by label (`df.loc[y.index]`) and splits by position (found while reviewing task 3, 26 October 2026; CSV uploads always have a plain row index, so no run was affected) | The runner refuses such a dataset with a clear message before an experiment starts; any unique index works (test: a shuffled index gives the same test metrics as the plain one) |
| Equal paired differences could still give a tiny non-zero variance through rounding of their mean, and so a meaningless, huge t statistic (found by the Judge tests, task 8.1, 26 October 2026) | `corrected_resampled_t` checks whether the values themselves are all equal; the Judge treats a standard deviation of at most 1e-12 (the refit tolerance) as equal and then decides on the mean alone |
| FLAML never searches the upper value of an integer range, but its validity check accepts it (found in the source, task 3, 26 October 2026) | Limits are written inclusive ("at most 32") and the engine adds 1 for FLAML; starting values are checked against the range FLAML really searches |
| The judge study's fixed rule chose 5 fold repeats over 3 on one tuning pair whose verdict changed only from `keep_simpler` to `keep`, with the same change accepted either way; on the reporting seeds both gave the same rates, and 5 repeats make every Assessment 25 refits instead of 15 (section 9.13, 26 October 2026) | Decided on 26 October 2026: alpha 0.20 as the rule says, but 3 repeats instead of 5, because both accept the same changes on both seed sets and 3 take three fifths of the Assessment time. Reported as a departure from the rule fixed before the run (section 9.13) |
| The judge study did not create the search folders of its two Setups, so FLAML could not write its log (found by a trial run before the real run, task 17, 26 October 2026; the smoke test uses a stand-in engine that writes no log) | The study creates each search folder before the search |
| The final smoke run's log held about 37,800 lines of one scikit-learn warning ("`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`", 11.8 MB), printed during the FLAML-alone cell and three Session cells, while the logs of the earlier runs have none (section 8.8). No package other than scikit-learn uses that helper; scikit-learn shows the warning when a job finds the warning filters it was given empty, and the known reset of the filters in worker threads (the problem of 7 October above) can empty them while another thread hands out jobs. This is the likely cause, not a confirmed one: a test with 8 threads fitting random forests did not reproduce it, and a 20 s FLAML search showed no warning (1 November 2026) | Dropped like the other two kinds of search noise, by a filter and in `showwarning` (test extended). The project sets no scikit-learn configuration, so the jobs compute the same with or without the warning |
| The Judge kept a change that made no difference: in the memory build (SES-020, student_performance_por, smaller trees) the candidate's refits matched the accepted ones to the 16th digit (mean difference 7e-18, standard deviation 2e-16), and the rule for equal differences (`mean > min_gain`, with min_gain 0) read that rounding noise as a gain, so the verdict was `keep` (found on 31 October 2026 while checking the steps that changed nothing). Replaying all saved decisions: this was the only such `keep` (the other 32 no-difference decisions in the Session records were `keep_simpler` or `discard`, and none in the judge study's 756 rows was `keep`) | A mean within the refit tolerance (1e-12) now counts as no difference, so such a step is `keep_simpler` if simpler and `discard` otherwise (1 test). The frozen memory keeps the old record; section 9.17 and the memory's statistics treat it as a step without effect |
| The benchmark run was cut short by the laptop: 55 minutes after the start it switched to battery and went into Modern Standby again and again, although the script asked Windows not to sleep (`ES_SYSTEM_REQUIRED`); on laptops with Modern Standby the screen turning off starts the standby. 10 of the 33 finished cells had been measured partly in standby (found on 31 October 2026 from cells that took up to 40 times their budget, matched against the Windows System log) | The 10 cells are listed in `results/benchmark/excluded_cells.csv` and not scored; rows are never edited. The script now also asks to keep the screen on (`ES_DISPLAY_REQUIRED`), waits before each cell while the laptop is on battery, and lists a cell that ends on battery as excluded, so it runs again (2 tests) |
| A statistics module named `research/statistics.py`, as the design said, broke AutoGluon: the AutoGluon worker runs as a script from `research/`, so that folder comes first on its import path and the file hid Python's own `statistics` module (checked in the `autogluon` environment before committing, task 25, 27 October 2026: the import failed, and worked again after the rename) | The module is `research/stat_tests.py` |
| Wilcoxon gave a different p-value for differences that are equal on paper: 0.9 - 0.85 and 0.95 - 0.9 differ in the 17th digit, so they got different ranks (found by the statistics tests, task 25.1, 27 October 2026) | Differences are rounded to 12 decimals before ranking, so equal differences share a rank |

### 4.8 Phase 1 review

- Presented in October 2026 using the live dashboard and `docs/unimp/Project_Presentation.md`. Result: completed successfully.
- Correction made afterwards (12 October 2026): the presentation explained the validation/test disagreement on seed 42 with shared ticket numbers. Measurement showed this does not hold up (section 9.7). The archived files carry a note.
- Feedback from the guide: to fill in.

Phase 2 and Phase 3 work: sections 4.6 to 4.11, chapter 9 and the work log (Appendix B).

### 4.9 Datasets for Phase 3 (task 20, fixed on 27 October 2026)

Requirement 22: development datasets (to build the memory, free to look at) and evaluation datasets (only for the benchmark) must not overlap, must be chosen by written criteria, and must be fixed in a committed file before any evaluation run. How they were chosen (`research/datasets.py`; the rule was written after listing the suites' OpenML summary numbers, and before any dataset was downloaded or any model was run on one):

1. Pool: all 107 datasets of two curated OpenML suites, OpenML-CC18 (72 classification datasets, study 99) and OpenML-CTR23 (35 regression datasets, study 353). Kept: 500 to 10,000 rows and at most 100 feature columns (laptop-sized for budgets of 2 and 5 minutes); left out: the two versions of the Wisconsin breast cancer data, since the `breast_cancer` sample is one of them. 64 datasets remained. The pool with its OpenML qualities is in `results/datasets/openml_pool.csv`.
2. Six groups, two per task, so that each list covers imbalance, missing values and categorical columns: binary imbalanced (largest class at least 3 times the smallest; 11 candidates); binary with missing values or categorical columns (7); multiclass with neither and balanced (10); multiclass with missing values, categorical columns or imbalance (12); regression with neither (13); regression with missing values or categorical columns (6). Balanced, numeric-only binary datasets fall into no group.
3. In each group the candidates were sorted by OpenML ID and shuffled with one generator seeded with 2026; the first became an evaluation dataset, the second a development dataset.

| Group | Evaluation (OpenML ID) | Development (OpenML ID) |
|---|---|---|
| binary imbalanced | pc4 (1049): 1,458 rows, 37 numeric columns, ratio 7.2 | wilt (40983): 4,839 rows, 5 columns, ratio 17.5 |
| binary messy | cylinder-bands (6332): 540 rows, 37 columns (14 text), 5% empty cells | tic-tac-toe (50): 958 rows, 9 text columns |
| multiclass clean | segment (40984): 2,310 rows, 18 columns, 7 classes | vehicle (54): 846 rows, 18 columns, 4 classes |
| multiclass messy | MiceProtein (40966): 1,080 rows, 77 columns, 8 classes, 1.7% empty cells | balance-scale (11): 625 rows, 4 columns, 3 classes, ratio 5.9 |
| regression clean | energy_efficiency (44960): 768 rows, 8 columns | kin8nm (44980): 8,192 rows, 8 columns |
| regression messy | solar_flare (44966): 1,066 rows, 10 columns (3 text) | student_performance_por (44967): 649 rows, 30 columns (17 text) |

With the four samples there are 10 development and 6 evaluation datasets. Licences: "Public" on OpenML for the CC18 datasets, CC BY 4.0 for the CTR23 datasets (as recorded by OpenML, in `datasets.json`). Each dataset is downloaded once with scikit-learn's `fetch_openml` (cache in `data/raw/openml_cache`) and saved as `data/raw/<name>.csv`; OpenML's ignored columns are left out (for MiceProtein that removes `MouseID` and the three columns from which the class is made). Two regression targets have few integer values (student grades, solar flare counts), so the app's own detector would call them multiclass; the registry records OpenML's task, which the benchmark passes to every system, and the detected one. `python scripts/prepare_datasets.py download` recreates the files from the IDs and versions and checked all 12 against the registry (27 October 2026).

### 4.10 Benchmark harness and grid (task 24, fixed on 27 October 2026)

Requirement 23. `research/benchmark.py` runs the comparison systems of Algorithm 20 over a grid; `scripts/run_benchmark.py run | status [--smoke]` is the command line. A cell is one (system, dataset, budget, seed). In a cell the system gets the same training and test rows and columns as every other system for that dataset and seed, and all CPU cores.

The grid, agreed with the user on 27 October 2026 before any evaluation run:

| Part | Systems | Budgets | Datasets and seeds | Cells |
|---|---|---|---|---|
| Main | `manual`, `flaml`, `autogluon`, `autopilot_rules`, `autopilot_full` | 120 s and 300 s | the 6 evaluation datasets x seeds 1 to 5 | 300 |
| Ablations | `random_choice`, `no_critic`, `no_resource_manager` | 300 s only | the same | 90 |

The design planned a third budget of 600 s (Requirement 23.1 names 2, 5 and 10 minutes). It was dropped to fit the laptop: it would add about 20 hours (4 systems that use their whole budget x 600 s x 30 pairs of dataset and seed). By the same arithmetic the reduced grid needs about 22 hours plus the time above budget, which the smoke run measures; this is an estimate, not a measurement. The command line can still add 600 s with `--budgets`.

How the harness keeps the results safe and fair:

- One CSV row per finished cell, appended at once and forced to disk: the test score and all 11 test metrics, wall time, number of Experiments, model, stop reason, first model's test score and kept Strategies (Sessions), the run ID of the records, training and test rows, columns left out, the split's fingerprint, CPU cores asked for and present, the system's settings, the frozen memory's file, SHA-256 and number of records, the code version (commit, uncommitted changes, library versions), the error of a failed cell, and the start and end times.
- Restart: cells that have a row are skipped; an interrupted cell has none and runs again. A system that fails gives a row with status `failed`; `--retry-failed` runs those cells again and the analysis uses each cell's last row, so no row is ever edited.
- Cells run seed by seed (then dataset by dataset and budget by budget), so a run stopped early still has complete seeds for the statistics.
- The frozen memory: the run refuses to start if it is missing for a system that reads it, if any record comes from an evaluation dataset or from any dataset that is not a development dataset (by name or by the dataset fingerprint; Requirement 22.3), or if earlier rows were made with a memory of another hash; it stops if the file changes during the run. The memory is opened read-only.
- Only evaluation datasets can enter the grid's rows; trials on other datasets use the smoke setting or a temporary folder.
- The systems' records go to `experiments/benchmark/` and `models/benchmark/`, apart from the dashboard's history, with MLflow off; the rows go to `results/benchmark/benchmark_cells.csv`.
- While it runs, the script asks Windows not to sleep (`SetThreadExecutionState`; the request ends with the process), because a laptop that sleeps during a time-limited search would cut that search short.
- Smoke setting (Requirement 23.4): every system once on titanic (a development dataset, so looking at its result tunes nothing), seed 1, the smallest budget; its own folders (`experiments/benchmark-smoke/`, `results/benchmark/smoke/`). It took 12 min 50 s (below), more than the "few minutes" of the requirement, because it uses the real budget to measure the time above budget.

The frozen memory (task 26, `research/memory_build.py`, `scripts/build_memory.py run | status | freeze`), decided on 27 October 2026 before the build: one Session per development dataset and seed (1 to 3), 30 Sessions of 300 s, at most 8 Experiments, starting from an empty memory file (the 34 Phase 2 records are not used, so the frozen memory comes from one written procedure). The Sessions try the applicable changes in a random order without reading the memory. Reason: with the rule order the same changes come first on the same data, so most (Strategy, diagnosis kind) arms would stay below the 3 records a learned selector needs before it moves an arm; a random order spreads the records over every change that applies. One row per finished Session goes to `results/benchmark/memory_sessions.csv`, and `freeze` copies only the records of finished Sessions to `results/benchmark/frozen_memory.jsonl` (in git, never rewritten) with a summary per arm; its SHA-256 is then in every benchmark row.

The analysis (task 26, `research/benchmark_analysis.py`, `scripts/run_benchmark.py analyse`), fixed on 27 October 2026 before any benchmark cell ran:

- Each cell's last row is used; failed cells are listed apart and not scored. A comparison at one budget uses only the (dataset, seed) blocks where every compared system completed, so all systems are scored on the same rows; a dataset's score is the mean over its seeds, with error metrics negated so that higher is better.
- Three families of comparisons, each with the Friedman test, the Nemenyi critical difference (alpha 0.05) and Wilcoxon tests with Holm's correction over its own pairs (Algorithm 21): the 5 main systems at 120 s and again at 300 s (pairs: the full system against each of the other four, and the rule-order Session against FLAML alone), and the ablations at 300 s (the full system, the rule-order Session and the three ablations; pairs: the full system against each of the other four).
- Also reported: wins, ties and losses over the 30 blocks of each pair (as a description only, since seeds of one dataset are not independent), per-dataset tables (mean and standard deviation of the test score over seeds, mean wall time), time-to-target of the main systems, critical-difference diagrams and a score-against-budget chart per dataset.
- Time-to-target summary: per system, how many (dataset, seed) pairs reached the target by 120 s and by 300 s, and the median time-to-target with a run that never reached the target counted as later than every budget (so the median is "never" when fewer than half reached it); per pair of main systems, how often one reached the target at an earlier budget than the other. Changed on 28 October 2026 at about 14:20, after the first 6 cells had run and their scores had been printed in the log: until then the median left out the runs that never reached the target, so a system that rarely reached it could still show a short median, and the pair counts did not exist. The tests, families and pairs above were not changed.

A throwaway trial on 27 October 2026 in a temporary folder (penguins, a development dataset, 15 s budget, seed 1, the Phase 2 memory as a stand-in) ran all 8 systems, skipped the 3 finished cells after a stop, wrote every field and left the memory file unchanged (same SHA-256 before and after). Its scores are not results.

Smoke run (task 26, 27 October 2026, 19:22 to 19:35; code 41737c8 with no uncommitted changes; the frozen memory, SHA-256 4baeb012..., 97 records; all 16 cores): every system completed on titanic with seed 1 and the 120 s budget. Wall times: manual 4.2 s, FLAML 120.5 s, AutoGluon 48.1 s (it stopped well before its time limit), and the five Sessions 113.0 to 127.3 s, at most 7.3 s above the budget (`no_resource_manager`); all five Sessions stopped for time, after 2 to 7 Experiments. The full system made its 5 changes in rule order: titanic's arms (missing values, high cardinality) have no records from the other binary development datasets (the memory leaves out titanic's own records), and the one arm with a record had 1, below the 3 needed. From these times the 390 cells need about 18 to 22 hours (arithmetic, not measured; most of the range is how long AutoGluon runs). Titanic is a development dataset, so these scores are a check of the harness, not results. Raw rows: `results/benchmark/smoke/benchmark_cells.csv`; the Sessions and the log `run.log` in `experiments/benchmark-smoke/`.

Before the grid, a check of every system type on a synthetic binary dataset with a boolean target (400 rows, ratio 4, read back from CSV as True/False, as pc4's target is; no development dataset has a boolean target) in a temporary folder: manual, FLAML, AutoGluon and two Session variants completed with all 7 classification metrics. Only the evaluation datasets' column types and target types were looked at for this, not their values.

### 4.11 Judge agreement study design (task 27, fixed on 27 October 2026)

Requirement 25: does the cheap Judge (refits of FLAML's chosen configuration on the Session's 15 folds, decision D2) decide like a strict comparison that searches again? `research/judge_agreement.py`, `scripts/run_study.py agreement run | analyse`. Fixed before any strict comparison ran:

- Pairs: the judged steps of the 30 Sessions that built the frozen memory (development datasets only), so they are decisions the Judge really made; at most 2 per dataset, drawn with seed 2026 where a dataset has more, so that no dataset dominates. The first design allowed 8 per dataset (57 pairs, about 2.5 hours); it was lowered to 2 on 31 October 2026, before any strict comparison ran, because the user asked for shorter runs. That gives 20 pairs from 17 Sessions; the Judge had said keep 4, keep_simpler 8 and discard 8.
- Power: the study waits before each pair while the laptop is on battery, and a pair that ends on battery is marked `excluded` and runs again (as for the benchmark, section 9.16).
- For each pair: the accepted and the candidate Setup as the Session recorded them, on the Session's training part (rebuilt from its seed and checked against the Session's split fingerprint; the test part is never read). 5 random 80/20 splits of the training part (stratified for classification); for each split and Setup a new FLAML search on the inner training rows (at most 25 trials with a 60 s cap, as in the judge study, no warm start), scored on the inner validation rows. The 5 differences are judged with the Judge's own code and settings (alpha 0.20, margin 0.005, the same "simpler" flag), with the corrected resampled t-test for 20% validation rows (ratio 0.25).
- Reported: agreement and Cohen's kappa on the three verdicts and on accepted (keep or keep_simpler) against discarded, the 3 x 3 table, and the cost ratio (the strict comparison's 10 searches against the Judge's two Assessments, per pair and in total).

---

## 5. Proposed algorithms

### 5.1 Implemented in Phase 1

**Algorithm 1: one experiment** (`AutoMLRunner.run_experiment`)

```text
input: dataset D, target, task (optional), metric (optional), budget, seed, test share, excluded columns
1  check: target and excluded columns exist, 5 <= budget <= 3600, 0.05 <= test share <= 0.5
2  y = target values without missing ones; X = the same rows without the target and excluded columns
3  require at least 20 rows and at least one feature column
4  task = given task, or detect it (Algorithm 2); check that task and metric fit y
5  split X, y into train and test (stratified for classification; plain if a class has fewer than 2 rows)
6  model, trial history, leaderboard = FLAML search on train (section 5.2)
7  test metrics, train metrics, trivial baseline on the test rows (Algorithm 4)
8  permutation importance on the test rows (Algorithm 5); unseen-category notes (Algorithm 6)
9  save the model bundle, write result files, log an MLflow run, write summary.json
output: ExperimentResult
```

**Algorithm 2: task detection** (`TaskDetector.detect`), on the target values without missing ones:

1. Fewer than 2 distinct values: rejected.
2. Text or boolean target: 2 classes gives binary; more classes gives multiclass, with high confidence up to 20 classes and low confidence above.
3. Numeric target with exactly 2 distinct values: binary.
4. Integer values with at most 20 distinct values: multiclass, medium confidence if the distinct values are at most 5% of the rows, otherwise low; the user is told it may be a count or an ordinal value.
5. Other integer values: regression, medium confidence. Anything else: regression, high confidence.

User overrides are checked by `TaskDetector.validate`: regression needs a numeric target, binary needs exactly 2 classes, multiclass needs at least 3. Warnings for missing targets, more than 20 classes, or classes with fewer than 2 rows.

**Algorithm 3: column checks** (`DatasetProfiler`):

- Constant: at most 1 distinct value.
- Identifier-like: every value is different, nothing is missing, and the column is not decimal or a date; or the name looks like an ID (`id`, `..._id`, `...Id`, `...ID`) and more than 95% of the values are distinct. Identifier-like and constant columns are suggested for exclusion.
- High cardinality: a text column with more than 50 distinct values (flagged only).
- Mostly empty: at least 40% missing.
- Class balance: ratio of majority to minority class. Below 1.5 none, 1.5 to 3 mild, 3 to 9 moderate, 9 or more severe.
- Meta-features for the planned memory (`DatasetProfile.meta_features`): rows, columns, numerical and categorical feature counts, share of numerical features, overall and largest missing share, duplicate share, numbers of ID-like, high-cardinality and constant columns; for classification the number of classes, imbalance ratio and majority share; for regression the skew of the target.

**Algorithm 4: scores and metrics** (`Evaluator`, `FLAMLEngine`):

- Primary metric choices: binary F1, ROC-AUC, accuracy, log loss; multiclass macro F1, accuracy, macro ROC-AUC (one-vs-rest), log loss; regression R², RMSE, MAE, MSE. Defaults: F1 and R².
- Names passed to FLAML: multiclass F1 becomes `macro_f1` and ROC-AUC becomes `roc_auc_ovr`; the others keep their names.
- FLAML minimises a loss. For metrics where higher is better the score is 1 - loss; for error metrics the score is the loss itself.
- Binary positive class: the last label in sorted order (Titanic: 1, survived; breast cancer: "malignant").
- Trivial baseline on the same test rows: always the most common training class, or the training mean for regression.
- Accuracy warning: shown when accuracy is chosen and the imbalance is moderate or severe.

**Algorithm 5: permutation importance** (`Evaluator.feature_importance`, built on scikit-learn):

```text
use at most 2000 test rows
s0 = score of the model with the primary metric, signed so that higher is better
for each original column j:
    repeat 5 times: shuffle column j and score again -> s(j, r)
    importance(j) = mean over r of (s0 - s(j, r)); spread(j) = standard deviation
sort the columns by importance
```

**Algorithm 6: unseen categories** (`unseen_category_warnings`): for each non-numeric column, the share of test rows whose value never appears in the training part. One note per affected column.

### 5.2 FLAML's search (external, used unchanged)

- Learner choice: FLAML keeps a search state for each learner and picks the next one using an estimate of the cost for improvement (ECI) [1]. In our classification runs the learners were LightGBM, XGBoost, XGBoost with limited depth, random forest, extra trees, an SGD linear model and L1 logistic regression (CatBoost is not installed).
- Hyperparameters: CFO, a local search that starts from cheap configurations and moves to expensive ones only when that looks worthwhile [2]. FLAML's "auto" setting resolves to CFO in our runs because trials run one at a time without Ray or Spark (checked in the FLAML 2.7.0 source). BlendSearch [3] is used only for parallel trials and needs the optional Optuna package, which is not installed.
- Validation rule (FLAML 2.7.0 source): 5-fold cross-validation if rows x columns / 0.9 < 10,000,000 x budget / 3600 and rows < 100,000; otherwise a holdout split. Titanic used cross-validation (recorded at 10 s and 60 s); California housing used holdout at 10 s.
- After the search, the best configuration is retrained on all training rows (`retrain_full=True`). The search may stop early once it has converged (`early_stop=True`).

Task 1 of the spec (26 October 2026): automated checks in `tests/test_flaml_capabilities.py` (8 tests) of the FLAML features the autonomous layer relies on. All measured on the Titanic seed-42 training part (712 rows) unless stated, with `n_jobs=-1` on the laptop in chapter 7:

- `starting_points`: a search started from a given LightGBM configuration evaluated exactly that configuration as its first trial.
- `max_iter=1` with a starting configuration trains exactly that configuration on all given rows, with no search. This is how the Assessor refits a setup. Two refits gave identical predictions (largest difference 0 for XGBoost and LightGBM, 3.3e-16 for random forest), so the tolerance is set to 1e-12.
- Cost of one refit: XGBoost with 304 trees on 570 Titanic rows, median 0.24 s (largest 0.36 s, 5 runs); XGBoost with 300 trees and 64 leaves on 13,210 California housing rows, median 1.49 s (3 runs). So 15 refits cost about 4 s on Titanic and about 22 s on California housing for these configurations.
- Using FLAML's `DataTransformer` and the XGBoost learner class directly did not reproduce the predictions of the same configuration trained through `AutoML` (largest difference in predicted probability 0.245), so it is not used as a fallback.
- `sample_weight` (balanced class weights): accepted; the mean predicted survival probability on the test part rose from 0.371 to 0.442 (LightGBM, 37 trees). FLAML also weights its validation score: the same first configuration scored a validation loss of 0.2286 without weights and 0.2045 with them. The Judge therefore compares unweighted fold scores.
- `custom_hp`: with the LightGBM `num_leaves` domain capped at 32 (FLAML's own starting value of 4 kept), all 40 trials stayed within the cap.
- Read in the FLAML 2.7.0 source while building task 3: the search never tries the `upper` value of an integer range (`normalize` and `denormalize` in `tune/space.py` use `upper - 1` unless the range is quantized). FLAML's own spaces are written this way, for example `min_child_samples` uses `upper=2**7 + 1` for "at most 128". So our limit "at most 32" becomes `upper=33`. Task 3.1 confirms it with a test: the top of the scaled range maps to 32 for `lograndint(4, 33)`, to 8 for `randint(1, 9)` and to 10 for the quantized `qrandint(2, 10, 2)`. FLAML's own validity check of a starting value (`is_valid`) still accepts `upper` itself, so our engine checks against the range that is really searched.
- A starting configuration outside the search space (`num_leaves` = 1,000,000) was dropped and the search continued from FLAML's default, but FLAML's warning could not be captured, so our engine checks starting points itself and records the warning. FLAML makes this check only when `max_iter` is above 1, which is why a fixed refit (`max_iter=1`) trains the configuration as given.
- `resampler`: a stand-in resampler was called on the training rows of each fold (569 or 570 rows) and once on all 712 training rows, 11 times for 2 trials, never on validation rows. Combining it with `sample_weight` raises `ValueError`.

Details checked in the FLAML 2.7.0 source and in the EXP-001 and EXP-002 trial logs (22 October 2026). The plain-language version, with diagrams, is `how-it-works.md` section 6.

- One pass of the search loop runs exactly one trial: pick a learner, let its CFO search propose one configuration, train and score it (5 models with cross-validation), update that learner's records. EXP-001 made 290 passes in 60 s, and 59.7 s of that went into training and scoring.
- Starting cost guesses relative to LightGBM (`cost_relative2lgbm` in `automl/model.py`): LightGBM 1, SGD 1, XGBoost 1.6, extra trees 1.9, random forest 2, XGBoost depth-limited 64, L1 logistic regression 160. EXP-001 tried the learners for the first time in exactly this order, at trials 0, 3, 12, 14, 21, 258 and 289.
- Learner choice (`_select_estimator` in `automl/automl.py`, `estimated_cost4improvement` in `automl/state.py`): expected cost of the next improvement = the larger of the learner's own time spent on its last improvement and its own time since then. A learner behind the leader by a gap g, improving at speed v, gets at least 2g/v. An untried learner keeps its starting guess times the cost of the first trial, and is chosen when it is the cheapest. Otherwise the learner is drawn at random with probability proportional to 1 / expected cost (FLAML's own fixed seed 1). Learners whose best trial is slower than the time left are skipped.
- CFO details (`tune/searcher/flow2.py`, `blendsearch.py`): the first configuration uses small values for the cost-related settings (4 trees and 4 leaves for LightGBM, XGBoost and random forest) and defaults for the rest. From the best point, it steps in a random direction in a 0-to-1 scaled space (log scale for wide ranges) and tries the opposite step if that fails. The first step length is 0.1 x sqrt(d), where d is the number of settings. The step shrinks only after 2^min(d, 9) consecutive proposals without improvement (512 for XGBoost, 256 for LightGBM). When the step falls below a lower bound the search has converged, and CFO restarts from the starting point with random noise. Our `seed` drives these random directions.
- Search ranges on Titanic: up to 712 trees and 712 leaves (FLAML caps both at the number of training rows; the largest tried in EXP-001 was 711 trees).
- Early stop needs every learner's local search to have converged at least once and the search to have run more than 10 times as long as it took to find the current best. It did not trigger in EXP-001 or EXP-002, which both searched for the full 60 s. In EXP-001, XGBoost never had more than 34 failed tries in a row, far below the 512 needed before its step shrinks.
- Retraining after the search has no time limit (`time_budget = -1` in `automl/automl.py`), which explains why runs overrun the budget (California at 10 s: 16.9 s).
- Folds and holdout: `RepeatedStratifiedKFold(5)` for classification (`RepeatedKFold(5)` for regression) after a shuffle, and the holdout is 10% of the training part (`SPLIT_RATIO = 0.1`), all with FLAML's fixed seed 1, so they do not change with our seed. A trial's score is the mean of its fold scores. California switches from holdout to cross-validation at a 53 s budget (16,512 x 8 / 0.9 = 146,773 against 147,222).
- Data preparation (`DataTransformer` in `automl/data.py`, run on the EXP-001 split). Text columns with one value, or a different value in every filled row, are dropped: FLAML would drop `Name` by itself but would keep `PassengerId`, a number. Other text columns become categories, and empty text becomes `__NAN__`. Empty numbers get the training median: 137 missing ages became 28.5. Labels are encoded 0/1.
- How each learner sees categories: LightGBM uses them directly. XGBoost, random forest, extra trees and the linear models get alphabetical codes (female 0, male 1; tickets 0 to 570). Unseen test tickets become `__NAN__`, code 571, for 112 of the 179 test rows in EXP-001.
- Sampling (training on a growing sample) is used only with a holdout and more than 40,000 training rows, so it never applied to our datasets: every trial used the whole training part. Classes with fewer than 20 training rows would be copied (`auto_augment`); this also never applied.

### 5.3 Planned (draft designs; they will change as they are built)

Update, 26 October 2026: the current design is the spec's `design.md` (`.kiro/specs/autopilot-ml/`). The drafts P1 to P7 below are kept for the history of the design. What changed:

- New component, the Assessor: it refits a setup's best configuration (no search) on fixed Session_Folds of the training part, 3 repeats of 5-fold cross-validation (kept after the judge study, section 9.13), and records per-fold scores. The Critic and the Judge both read these scores.
- P6 is replaced by the Judge: paired differences on the same folds, the corrected resampled t-test of Nadeau and Bengio (2003) in the repeated form of Bouckaert and Frank (2004), and three verdicts: keep, keep_simpler (simpler and not worse) and discard. The expensive version in P6 (full searches on several inner splits) becomes the strict reference in the judge agreement study (Requirement 25). Decision D2, confirmed on 26 October 2026.
- P1: the thresholds move to a TOML file with a version label (`critic_thresholds.toml`). Two more kinds are added: budget_limited and wasted_budget. The thin loop starts with overfitting, high_cardinality and class_imbalance.
- P2: strategies are split into a thin set (drop_columns, class_weight_balanced, regularised_search_space) and a full set. Data changes (frequency encoding, rare-category grouping, missing indicators) are learned inside each fold.
- P3: the learned selectors treat each pair of strategy and diagnosis kind as an arm, with statistics from similar past datasets in the memory.
- P5: the first experiment gets 35% of the time by default (30% to 40% allowed), budgets are counted in measured wall-clock time, and the default patience is 3 discards in a row.
- P7: group-aware folds are now optional (Requirement 28).
- New rule: inside a Session, the test part is used once, at the end, for the first and the final model only (the Final_Evaluation), and decision components only see records with every test number removed.

**P1. Critic.** Decisions use training and validation data only, never the test set. Starting rules, to be tuned on synthetic datasets with planted problems and frozen before the final evaluation:

| Diagnosis | Evidence | Starting rule |
|---|---|---|
| Overfitting | train score minus validation score | gap above 0.10 moderate, above 0.20 high (scores from 0 to 1) |
| Underfitting | train and validation scores against the trivial baseline | both low; exact rule to be set |
| Class imbalance | class balance severity | moderate or severe |
| Missing values | missing share per column | at least 40% in a column |
| High cardinality | distinct values; unseen values between training and validation folds | more than 50 distinct text values, or many unseen values |
| Unreliable validation | spread of validation scores across folds or splits | spread larger than the change being tested |
| Wasted compute | improvement per second over recent trials or experiments | below a threshold |
| Possible leakage (later) | one column with very high importance; ID-like names | flag for the user |

Example from Phase 1 (EXP-001): train F1 0.934 against validation F1 0.802, a gap of 0.13, so "overfitting: moderate" under the starting rule.

Each diagnosis will be checked for precision and recall on synthetic datasets where the problem is planted on purpose.

**P2. Strategy rule table** (first version, also in `PROJECT.md` section 13):

| Diagnosis | Candidate strategies |
|---|---|
| Overfitting | drop or encode high-cardinality columns; search space biased towards regularisation; fewer or shallower trees; feature selection |
| Underfitting | more budget; more expressive model families; new features |
| Class imbalance | class weights; SMOTE; switch the metric to F1 or ROC-AUC; tune the decision threshold |
| Missing values | missing-value indicator columns; drop mostly empty columns |
| High cardinality | frequency or target encoding; drop the column |
| ID-like or constant columns | drop them |
| Redundant features | drop one column of each highly correlated pair |
| Unreliable validation | more splits or repeated cross-validation; group-aware folds where rows are related |
| Wasted compute | restrict the model list; smaller budget; stop |

**P3. Strategy selection.**

- Version 1: fixed priority from the rule table, cheapest strategy first.
- Version 2, bandit: each strategy is an arm. Reward = (validation score after - before) / compute seconds, sign flipped for error metrics and scaled to 0..1 over the strategies tried. UCB1 picks the arm with the highest mean reward + sqrt(2 ln N / n_a), where N is the total number of tries and n_a the tries of arm a [18]. Thompson sampling keeps a Gaussian belief about each arm's mean reward and picks the arm with the highest sampled value.
- Version 3: start each arm's belief from what the strategy achieved on similar datasets (P4).

**P4. Memory retrieval.** Each record stores meta-features, diagnoses, strategy, engine, model family, hyperparameters, score before and after (with its spread over splits) and cost. For a new dataset: standardise the meta-features over the stored records, find the k nearest past datasets by Euclidean distance (k = 5 to start), and rank strategies by distance-weighted mean reward. Fall back to the rule order when there are fewer than 3 observations.

**P5. Resource manager.** Utility = improvement in validation score / compute seconds. The first run gets a fixed share of the budget (for example 30%), and each later experiment gets a share of what is left, weighted by expected utility. Stop when the time or experiment limit is reached, when no remaining strategy has an expected utility above a threshold, or after two experiments in a row without a kept improvement. `n_jobs` follows the CPU limit.

**P6. Fair comparison of a change** (added after Phase 1):

```text
input: current setup A, candidate setup B, number of splits k (k >= 5)
for i = 1..k:
    split the training part into a search part and a decision part (80/20, seed i)
    run FLAML for A and for B on the search part, same budget
    d(i) = score of B minus score of A on the decision part
keep B if the mean of d is above 0 and B wins on most splits
(use the Wilcoxon signed-rank test once k >= 6)
the outer test set is never used for this decision
```

This costs 2k runs per decision, so the budget split has to allow for it. Cheaper variants (shorter comparison budgets, reusing A's runs) are to be explored.

**P7. Group-aware validation** (option): keep rows of the same group, for example the same `Ticket`, inside one fold, for data where new rows will come from new groups. On Titanic it is not needed to make validation and test agree (section 9.7); it is a choice about what kind of new data to expect.

### 5.4 Implemented in Phase 2

Added as each spec task is built. Full code shapes are in the spec's `design.md`.

**Algorithm 7: search-space limits and starting points** (`FLAMLEngine.custom_hp` and `check_starting_points`, task 3)

1. For each limited setting, take FLAML's own spec of that learner's search space for the training data size.
2. Work out the range FLAML really searches: `lower .. upper - 1` for a plain integer range, `lower .. upper` otherwise (section 5.2).
3. Narrow it with the user's inclusive bounds (never widen it); stop with an error if fewer than two values are left.
4. Write the new range back in FLAML's form (upper + 1 for a plain integer range), keeping the domain type, and move `init_value` and `low_cost_init_value` inside it.
5. For each starting configuration, drop FLAML's bookkeeping keys, then check every value against the limited space: right type and inside the searched range, or one of the choices. A configuration with any value outside is left out, with a warning naming the settings.

**Algorithm 8: synthetic dataset with planted problems** (`research/synthetic.py`, task 4), used to test the Critic and the Judge against known answers.

1. Make the rows of the dataset and of a fresh sample (20,000 rows by default) in one call to `make_classification` (8 informative features, class weights when imbalance is planted, no label flips) or `make_regression` (8 informative features plus noise with a third of the signal's standard deviation, so the best possible R2 is 0.9).
2. Split them into the dataset and the fresh sample, stratified for classification.
3. Plant each requested problem in both parts in a fixed order, each from its own random stream derived from the seed: change labels (label noise), add noisy copies of 2 columns with a chosen correlation, add random numeric columns, add an integer ID column with a chosen share of distinct values, add a column with a chosen share of one main value, add a category column with strength x rows levels drawn at random, and empty a chosen share of the cells of 2 columns. Counts are exact in each part.
4. Return the dataset, the fresh sample and the list of planted problems with their columns.

Because both parts come from one generator, a model's score on the fresh sample estimates its true score. Checked on 26 October 2026: a binary dataset with all eight problems (1,000 rows plus 20,000 fresh rows) is made in 0.1 s.

**Algorithm 9: Assessment of a Setup** (`app/services/assessor.py`, task 5)

1. Look up the refit key: the feature columns, the Data_Steps and class weights, the winning learner family and its configuration (everything the refits depend on). If the same refits were made before in the Session, reuse their fold scores without refitting, in a copy that names the Experiment they were made for (`reused_from`). Before 31 October 2026 the key was the whole Setup with the learner and configuration, so a Setup that only searched fewer learners and found the same model was refitted again for the same scores: 18 of the 97 candidate steps of the memory build, 149.7 s of its 6,203.6 s (2.4%).
2. Check that the given rows are the Training_Part the Session_Folds were made on (same number of rows).
3. For each of the 15 folds (3 repeats of 5 folds, the default; kept after the judge study in section 9.13): take the fold's training rows; compute balanced sample weights from them if the Setup uses class weights; train the winning configuration on them with `fit_config`; score it on the fold's training rows and validation rows with every task metric, record each class's validation recall, the share of validation rows with a category not seen in the fold's training rows (per column), the trivial baseline's validation scores and the fit time.
4. If a fold fails, record the error, mark the Assessment failed and stop.
5. Report the total time, and per metric the mean and standard deviation over the folds.

Throwaway check on 26 October 2026 (Titanic, setup A, seed-42 split, session folds seed 43, one 10 s FLAML search, `n_jobs=-1`; output printed only, not saved): FLAML chose XGBoost with a validation F1 of 0.774; the 15 refits took 4.6 s and gave validation F1 0.751 (standard deviation 0.051 over the folds) and training F1 0.854. The trivial baseline (always "did not survive") scores F1 0. In the folds, 72% of validation rows had a ticket not seen in the fold's training rows (section 9.7 found 71%) and 15.7% a cabin not seen. Asking again for the same configuration returned the cached Assessment.

**Algorithm 11: Critic, thin version** (`app/services/critic.py`, thresholds in `app/services/critic_thresholds.toml`, version `v0-start`, task 6). Inputs: the Training_Part profile, the Setup, its Assessment and the search summary. Rules run in a fixed order; columns are taken in name order.

1. Diagnostic metric m: the primary metric if it is bounded and higher is better (F1, accuracy, ROC-AUC, R2), otherwise F1 for classification or R2 for regression.
2. Overfitting: gap = mean over the folds of (training m minus validation m). Moderate from 0.10, high from 0.20.
3. Class imbalance (classification): ratio = largest class / smallest class in the Training_Part; recall = mean validation recall of the smallest class. Moderate if the ratio is at least 3 or the recall is below 0.5; high if the ratio is at least 9 or the recall is below 0.3. The recall test counts only from a ratio of 1.5, because with balanced classes a low recall means a weak model, not an imbalance (a design choice, to be tuned in task 16).
4. High cardinality (each used text column): moderate if it has more than 50 different values or at least 30% of validation rows have a value unseen in the fold's training rows; high from 60% unseen.
5. Each Diagnosis carries its evidence numbers, a plain-language message and the thresholds version. No rule firing gives an empty list.

Throwaway check on 26 October 2026 (Titanic, setup A, seed-42 split, session folds seed 43, one 20 s FLAML search, `n_jobs=-1`; output printed only, not saved): FLAML chose XGBoost; the Critic reported overfitting moderate (training F1 0.919, validation F1 0.725, gap 0.194 over 15 folds), high cardinality high for `Ticket` (571 values, 72% unseen) and moderate for `Cabin` (127 values, 16% unseen), and no class imbalance (ratio 1.6). This matches what the design expected from the Phase 1 numbers, except that the gap was larger than EXP-001's 0.13 (EXP-001 measured the gap between FLAML's validation score and the training score of one model, not over folds).

**Algorithm 12: Strategy_Engine with the rule table, thin version** (`app/services/strategy_engine.py` and `app/services/strategies.py`, task 7). Strategies: drop_columns (simpler, cheap), class_weight_balanced (same complexity, cheap, classification only) and regularised_search_space (simpler, cheap: at most 32 leaves for LightGBM and XGBoost, LightGBM leaves of at least 10 rows, XGBoost minimum child weight at least 1, depth at most 4 for depth-limited XGBoost, at most 64 leaves for random forest and extra trees; design choices, not measured values). Each returns a changed copy of the Setup.

1. For each Diagnosis, in the Critic's order, take its row of the rule table: high_cardinality gives "drop the column"; overfitting gives "smaller trees", then "drop each high-cardinality column"; class_imbalance gives "balanced class weights". One Proposal per column.
2. Leave out a change already tried in the Session, or one that does not apply to the Accepted_Setup (for example a column that is already dropped, or class weights for regression).
3. Merge a change asked for by several Diagnoses into one Proposal that answers all of them, with the highest severity and the earliest table position.
4. Rank by severity (high first), then cost class (cheap first), then table position, then the change's key (canonical JSON), so the order is always the same.

On the Titanic Diagnoses of Algorithm 11 this gives: drop `Ticket` (high; answers overfitting and high cardinality), drop `Cabin` (moderate), then smaller trees (moderate).

**Algorithm 12, completed: all nine Strategies and the full rule table** (task 14)

| Diagnosis | Strategies tried, in order |
|---|---|
| identifier_column, constant_column | drop the column |
| high_cardinality | drop the column; frequency encoding; group rare values |
| missing_values | drop the column (only if high); missing-value indicator (numeric columns) |
| redundant_features | drop the later column of the pair |
| overfitting | smaller trees; drop each high-cardinality column |
| underfitting, budget_limited | more search time |
| class_imbalance | balanced class weights; smoother search metric (log loss) |
| unreliable_validation | smoother search metric |
| weak_model_families | search only the other learner families |
| wasted_budget | none: the Resource_Manager shortens the next search |

New Strategies: frequency encoding (same complexity), grouping rare values (simpler), missing-value indicator (more complex), fewer learner families (simpler), more search time (same Setup, twice the time, expensive) and smoother search metric (same complexity, normal cost). Each changes a copy of the Setup and applies only when it makes sense for it (for example, a column can get only one of the two encodings, and dropping a column also removes its Data_Steps). A 90 s throwaway Session on Titanic (seed 42, temporary folder, 26 October 2026) ran the new data Strategies end to end through FLAML: it dropped `Ticket` (kept), frequency-encoded `Cabin` (discarded), grouped the rare values of `Cabin` (kept as simpler) and tried the smoother search metric (discarded), and stopped for time at 90.5 s.

**Algorithm 13: Judge** (`app/services/judge.py`, task 8; settings alpha 0.20 (0.10 until the judge study, section 9.13), minimum gain 0, non-inferiority margin 0.005, decision D2)

1. Check that both Assessments are complete, use the same folds (same fingerprint and number of folds) and the same primary metric; otherwise stop with an error, and the Session discards the candidate.
2. On each fold j, d_j = candidate score minus accepted score on the validation rows, with the sign flipped for error metrics (RMSE, MAE, MSE, log loss), so positive always means the candidate is better. Folds where either score is missing are left out (at least 2 must remain).
3. Count wins, losses and ties (|d_j| at most 1e-12, the refit tolerance of task 1).
4. Corrected resampled t: t = mean(d) / sqrt((1/J + r) x var(d)), with J folds, r = validation rows per training row in a fold (1/4 for 5 folds) and var with J - 1 in the denominator; p(better) = P(T > t) and p(worse) = P(T < t) for Student's t with J - 1 degrees of freedom.
5. Keep if mean(d) > minimum gain and p(better) <= alpha. Otherwise keep the simpler Setup if the candidate is simpler, mean(d) >= -margin and p(worse) > alpha. Otherwise discard.
6. If all differences are equal (standard deviation at most 1e-12), decide on the mean alone and record no p-values.

Why the correction: the 15 folds share most of their training rows, so their scores are not independent. With 15 folds the variance of the mean is taken as (1/15 + 1/4) times the fold variance instead of 1/15 of it. Worked example: d = (0.02, 0.01, 0.03, -0.01, 0.02) gives mean 0.014 and variance 0.00023, corrected t = 1.376 with 4 degrees of freedom and p(better) = 0.120, so the change is not kept at alpha 0.10 (it is kept at the default 0.20); an uncorrected paired t-test would give t = 2.064 and p = 0.054 and would keep it at 0.10. A side effect of the correction: the 1/4 term does not shrink with more folds, so going from 15 to 25 folds makes the standard error of the mean only about 4% smaller.

**Algorithm 14: Resource_Manager** (`app/services/resource_manager.py`, task 9). Budget: total seconds T, maximum Experiments, CPU cores, minimum 5 s per Experiment, first share 35%, patience 3. All time is measured wall-clock time.

1. First Experiment: max(5, floor(0.35 x T)) seconds of search; it must finish by T.
2. Reserve: 1.2 x (Assessment + other seconds) of the previous step, kept back for the next Assessment and the final steps.
3. Time left R = T - elapsed - reserve; Experiments left s. If R < 5, stop (time); if s = 0, stop (experiments).
4. Next search budget = max(5, floor(R / s)). For "more search time": at least min(2 x the Accepted_Setup's budget, R). If a wasted-budget Diagnosis is present: at most max(5, ceil(1.5 x the time the Accepted_Setup's search needed to find its best)).
5. A later step must finish by "now + its budget + the reserve". After each step, record the budget, search time, Assessment time, other time, overrun past the deadline and utility (validation gain per second).
6. Stop checks, in order: user request, error, time, experiments, no Proposals left, 3 discards in a row.

Since every deadline lies inside T, the steps end at most one step's overrun after T, and that overrun is recorded.

**Algorithm 11, completed: all twelve Diagnoses** (task 13). Rules, in this order (m is the diagnostic metric; the table gives the starting values `v0-start`; the critic study, section 9.12, changed the overfitting gap to 0.11, the missing share to 0.30 and the near-constant share to 0.98, frozen as `v1-critic-study`):

| Diagnosis | Evidence | Moderate | High |
|---|---|---|---|
| overfitting | gap = mean of (training m - validation m) over the folds | 0.10 or more | 0.20 or more |
| underfitting | skill = (validation m - baseline m) / (1 - baseline m), only if the gap is below 0.03 | below 0.50 | below 0.25 |
| budget_limited | gain of the search's best score in the last quarter of its time (bounded metrics only) | 0.005 or more | 0.02 or more |
| wasted_budget | share of the search time after its best configuration | 0.50 or more | 0.75 or more |
| class_imbalance | largest / smallest class; validation recall of the smallest class | ratio 3 or more, or recall below 0.5 | ratio 9 or more, or recall below 0.3 |
| missing_values | share of empty cells of a used column | 0.40 or more | 0.70 or more |
| high_cardinality | different values of a used text column; share of validation rows with a value unseen in the fold's training rows | more than 50 values, or 0.30 or more unseen | 0.60 or more unseen |
| identifier_column | the profiler's identifier flag | | always high |
| constant_column | one value, or the same value in 99% or more of rows | low | |
| redundant_features | absolute correlation of two used numeric columns | 0.95 or more | 0.99 or more |
| unreliable_validation | standard deviation of validation m over the folds | 0.05 or more | 0.08 or more |
| weak_model_families | a learner family 0.10 or more behind the best family that used 5% or more of the search time | low | |

The profiler now also reports, on the rows it is given, each column's most common value share, near-constant columns and highly correlated numeric pairs. On the saved SES-001 records the full Critic adds "missing values high" for `Cabin` (77% empty) to the first Setup's Diagnoses, and finds "budget limited" (the search's best F1 still rose by 0.006 in its last quarter) and "weak model families" for the final Setup.

**Algorithm 18: learned strategy choice** (`app/services/strategy_engine.py`, task 21; Requirement 19). Four selectors re-rank the Proposals of the rule table:

- `rules`: the written order (severity, cost, table position, key), as in Phase 2.
- `random`: the Proposals in a random order drawn with the Session's generator (seeded with the Session seed).
- `thompson` and `ucb1` learn from the Experiment_Memory. An arm is a pair of Strategy and diagnosis kind; a Proposal's arm uses the kind of the most severe Diagnosis it answers. The arm statistics come from the records of the k = 5 nearest past datasets of the same task (Algorithm 17), never from the current dataset, each record weighted by 1 / (1 + distance): weighted keeps K and discards D, the mean Judge difference of the kept records G, the mean cost in seconds C, and the number of records n.
  - `thompson` (Thompson, 1933): score = p x G / max(C, 1 s), with p drawn from Beta(1 + K, 1 + D): a sampled expected validation gain per second.
  - `ucb1` (Auer, Cesa-Bianchi and Fischer, 2002): score = K / (K + D) + sqrt(2 ln N / n), with N the records of all arms of the same diagnosis kind; equal scores go to the cheaper arm.
  - Too little experience: Proposals whose arm has fewer than 3 records, or records from only one dataset (since 31 October 2026), keep their place in the rule order; only the places of the others are re-filled, best score first. With no memory (`use_memory` off) every place keeps the rule order.
  - What counts as kept (since 31 October 2026): a kept record whose fold differences were all zero (the change gave the same model) counts as not kept, so a Strategy that often changes nothing no longer looks reliable. The records themselves are unchanged.
  - Effect on the frozen memory, computed from its records: 6 of the 20 arms can move (13 before); 34 of the 97 records count as kept (53 before), because 21 changed no fold score; restrict_model_families on multiclass data goes from 13 kept of 13 to 1 of 13. The class-weight arm on binary data (3 records, all from wilt) no longer counts, so on a new binary dataset class weights keep their rule-order place. These changes were made after the benchmark was stopped and have not been measured on new data.
- The memory is used only when the Session's `use_memory` switch is on. Each step records how its Proposal was placed (arm, records, score, rule rank and rank). A read-only memory, as the benchmark will use, is read but never written.

Built on 27 October 2026 (task 21). Tests: section 8.6.

**Algorithm 19: ablation switches** (`app/services/autopilot.py`, `strategy_engine.py`, `resource_manager.py`, task 22; Requirement 20). Each switch turns one part off, so the benchmark can measure what that part adds; all are recorded in `session.json`.

| Switch | What the Session does instead |
|---|---|
| `use_memory` off (default) | The selectors see no memory, so `thompson` and `ucb1` keep the rule order |
| `selector="random"` | The Proposals of the rule table in a random order (no strategy choice) |
| `use_critic` off | No Diagnoses. For each of the 9 Strategies one change with random valid parameters (a random used column to drop, a random text column to encode or group, a random numeric column with empty cells for an indicator, all tried learner families but one), kept if it applies and was not tried, in a random order. Since the optional tasks, for each of the 12 (also a random text column for target encoding and a random text column with frequent words for word features) |
| `use_resource_manager` off | Every search gets total time / maximum Experiments (at least 5 s); nothing is kept back for Assessments, no shorter search after a wasted budget and no longer one for more search time; only the time and Experiment limits stop the Session (and Stop, an error or no change left; no "three discards in a row" rule) |

The random choices use the Session's generator, seeded with the Session seed, so an ablation run can be repeated. Built on 27 October 2026 (task 22). Tests: section 8.6.

**Algorithm 20: comparison systems** (`research/systems.py`, `research/autogluon_worker.py`, task 23; Requirement 21). For one dataset and seed, every system gets the same training and test rows (the Phase 1 split) and the same columns (identifier-like and constant columns of the training part left out, as in a Session's first Setup), the same time budget and CPU cores, and its test predictions are scored by the same Evaluator, so all systems report the same metrics on the same rows.

| System | What it does |
|---|---|
| `manual` | LightGBM with its library defaults (100 trees, 31 leaves, learning rate 0.1), trained once with FLAML's data preparation; no search |
| `flaml` | One FLAML search with the whole budget |
| `autogluon` | AutoGluon-Tabular 1.6.3, preset `medium_quality`, CPU only, in its own conda environment as a separate process; it writes its predictions, which our Evaluator scores |
| `autopilot_rules` | A Session with the rule order (Phase 2) |
| `autopilot_full` | A Session with Thompson sampling on the frozen memory |
| `random_choice`, `no_critic`, `no_resource_manager` | The ablations of Algorithm 19 (the last two with Thompson sampling and the memory) |

Without memory the learned choice keeps the rule order, so the "no memory" ablation is the same Session as `autopilot_rules` and is not run twice. AutoGluon could not be installed next to our libraries: a pip dry run on 27 October 2026 showed it would replace pandas 3.0.6 by 2.3.3, so it got the separate environment allowed by decision D4. A throwaway run in a temporary folder (30 to 40 s budgets, tic-tac-toe, kin8nm and balance-scale) ran every system end to end; those scores are not results.

**Algorithm 21: benchmark statistics** (`research/stat_tests.py`, task 25; Requirement 24), following Demšar [22]. Per budget, a system's score on a dataset is the mean over its 5 seeds, so each dataset is one observation; error metrics are negated first so that higher is always better.

- Ranks: within each dataset, rank 1 is the best system; tied systems share the mean rank. R_j is system j's average rank over the N datasets.
- Friedman test (Friedman, 1937) over k systems: chi2_F = 12N / (k(k+1)) x (sum of R_j^2 - k(k+1)^2 / 4), from `scipy.stats.friedmanchisquare` (which also corrects for ties), and Iman and Davenport's (1980) F_F = (N - 1) chi2_F / (N(k - 1) - chi2_F) with k - 1 and (k - 1)(N - 1) degrees of freedom, which Demšar recommends as less conservative.
- Nemenyi critical difference (Nemenyi, 1963): CD = q_alpha x sqrt(k(k+1) / (6N)), with q_alpha the studentized range quantile for k groups and infinite degrees of freedom divided by sqrt(2). Two systems differ when their average ranks differ by at least CD; the bars of a critical-difference diagram join the runs of systems that do not.
- Pairs: the Wilcoxon signed-rank test (Wilcoxon, 1945) on the per-dataset differences (two-sided; zero differences dropped; exact p-values for small N), Holm's (1979) step-down correction over the pairs compared together, the matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-) as the effect size (Kerby, 2014), and a 95% percentile bootstrap interval of the median difference (10,000 resamples, fixed seed; Efron and Tibshirani, 1993). Differences equal up to floating-point noise are rounded to 12 decimals so that they share a rank.
- Time-to-target (Requirement 24.3): for each dataset and seed, the target is 99% of the best score any system reached at any budget (for a negative best, 1% of its size below it); a system's time-to-target is the smallest budget at which it reached the target, or none.

What the tests can show with 6 evaluation datasets (worked out from the formulas, not measured): the smallest two-sided Wilcoxon p-value is 2 / 2^6 = 0.031, reached only when one system is better on all 6 datasets, so after Holm's correction no pair can reach p < 0.05 once two or more pairs are compared; the Nemenyi CD at alpha 0.05 is 2.49 ranks for the 5 main systems and 4.29 for all 8. The effect sizes and intervals therefore carry most of the information. With 12 datasets these would be 0.0005, 1.76 and 3.03.

Built on 27 October 2026 (task 25). Tests (section 8.6) check every function against SciPy's documented Wilcoxon example, Demšar's Table 5 and worked example, and values worked out by hand.

**Algorithm 17: Experiment_Memory, version 1** (`app/services/experiment_memory.py`, task 15). After every Verdict the Session appends one line to `experiments/memory/memory.jsonl`: the dataset's fingerprint (a hash of its column names and types, number of rows and target) and the Meta_Features of its training part, the task and metric, the Diagnoses the Proposal answered (kind and severity), the Strategy and its parameters, the Verdict, the Judge's mean and standard deviation of the differences, the number of folds, the step's cost in seconds, its Utility, the engine version and the date. The file is only appended to, and a damaged line is skipped with a warning.

Finding similar datasets: average each stored dataset's Meta_Features over its records, standardise every feature over the stored datasets (dropping features that do not vary), compare only the features both datasets have, and rank by Euclidean distance (same task type only, optionally leaving one dataset out). For the k nearest datasets (5 by default), each (Strategy, diagnosis kind) gets its number of observations, its keep rate with each observation weighted by 1 / (1 + distance), its mean gain when kept and its mean cost. In Phase 2 the memory is written and shown but does not influence any decision; Phase 3's learned strategy choice will read it. SES-001 ran before the memory existed, so the memory starts with the next Session.

**Algorithm 16: Data_Steps** (`app/services/transforms.py`, task 12). A Data_Step is learned from the rows it is fitted on and then applied unchanged to other rows; no step reads the target.

| Step | Learned from the training rows | Applied to other rows |
|---|---|---|
| drop_columns | nothing | the columns are removed |
| frequency_encode | the share of training rows with each value | the value becomes that share; an unseen value becomes 0; empty cells stay empty |
| group_rare_categories | the values seen in at least 1% of the training rows (`min_share`) | other values become `__other__`; empty cells stay empty |
| missing_indicator | nothing | a new 0/1 column `<column>__missing` for each listed numeric column |

Where they are fitted: in an Experiment, on the Training_Part before the search, so FLAML searches the changed columns; in an Assessment, again on each fold's training rows. The model is saved together with its fitted steps (`PipelineModel`), so a reloaded model predicts from raw rows, and permutation importance is measured on the original columns. A throwaway check on 26 October 2026 (Titanic, frequency encoding of `Ticket`, rare `Cabin` values grouped, a missing-age indicator, 8 s search, temporary folder) ran the search, the evaluation, saving and reloading, MLflow logging and a 15-fold Assessment without errors.

**Algorithm 15: autonomous Session, thin loop** (`app/services/autopilot.py` and `app/services/experiment_planner.py`, task 11)

1. Check the settings (selector, folds, thresholds file); any error stops here and nothing is written.
2. Drop rows without a target; make the outer split (training and test parts, as in Phase 1) and 3 x 5 Session_Folds of the training part (seed + 1); fingerprint them; profile the training part only.
3. Reserve `SES-###`, write `session.json` (status running) and the training-part profile, and open the MLflow parent run.
4. First Setup: the user's settings plus the identifier-like and constant columns of the training part left out. Search it with the first budget, on training rows only (no test score, no importance), assess it on the Session_Folds and diagnose it. If this fails, the Session stops with status failed.
5. Loop: if the user asked to stop, stop after the current step. Ask the Strategy_Engine for Proposals for the Accepted_Setup (changes not tried yet); ask the Resource_Manager whether to stop (user, error, time, Experiments, no Proposals, patience) and for the next budget. Take the first Proposal: apply its Strategy to a copy of the Accepted_Setup, start the search from the Accepted_Setup's best configuration of each learner family, search, assess and judge it against the Accepted_Setup. If it is kept (keep or keep_simpler) it becomes the Accepted_Setup and is diagnosed; otherwise the Accepted_Setup stays. A failure is recorded and counts as a discard. Record the step's cost and rewrite `session.json`.
6. Final_Evaluation, the only use of the test part: score the first and the final model on it once each, with permutation importance for the final model; close the MLflow run; write `session.json` (status completed).

Trial runs on 26 October 2026 in a temporary folder, not kept (Titanic, seed 42): a 60 s Session (at most 5 Experiments, MLflow off) and a 40 s Session (at most 4, MLflow on) both ran end to end and stopped for time within their budgets (52.3 s and 39.8 s). In both, the first Setup left out `PassengerId` and `Name`, and the first candidate dropped `Ticket`, as the rule table says. The 40 s Session's MLflow parent run had both Experiments as child runs. The measured results of the first real Session are in chapter 9 (task 11.2).

**Algorithm 10: search summary** (`app/services/search_summary.py`, task 5): from the trial history, the search time (end of the last trial), the time of the best trial (the first one if tied), the share of the search time after it, the gain of the best score in the last quarter of the time (signed so that positive is better), the number of trials, and the best score and total trial time of each learner family. On the saved EXP-001 and EXP-002 histories it gives exactly the numbers of section 9.10 (best at 42.9 s and 23.2 s; 17.1 s and 36.8 s after the best; XGBoost 34.6 s in EXP-001).

### 5.5 Optional extensions (tasks 29 to 35, built from 1 November 2026)

**Algorithm 22: SMOTE through FLAML's resampler** (`app/services/resampling.py`, Strategy `smote`, task 29; Requirement 26). SMOTE (Chawla et al. [19]) adds synthetic rows of the smaller classes: each new row lies between a row of the class and one of its k nearest neighbours of the same class.

1. A Setup may name a resampler (`resampler="smote"`). FLAML applies a fresh copy to the training rows of each of its search folds and to the rows of the final model, never to validation rows (checked in task 1); the Assessor gives it to every refit, so only a Session fold's training rows are resampled and its validation rows are scored as they are. FLAML does not allow a resampler together with sample weights, so the Strategy does not apply when class weights are set, and class weights no longer apply once a resampler is set.
2. FLAML prepares the rows before the resampler sees them: text columns become category columns and empty numeric cells are filled (checked on Titanic, 1 November 2026). Plain SMOTE refuses category columns, so `FoldSafeSmote` uses SMOTE for numeric columns only, SMOTE-NC for mixed columns and SMOTE-N for text columns only.
3. Every smaller class is oversampled to the size of the largest. The number of neighbours (5) is lowered to the smallest of those classes minus 1; with a class of 1 row nothing is resampled. The original rows come back first and unchanged, followed by the new ones, in the original column types.
4. In the rule table, SMOTE is the third answer to class imbalance, after class weights and the smoother search metric. Like every change, it is kept only if the Judge finds it better or simpler and not worse.

Checked on 1 November 2026 (throwaway, temporary folder): on FLAML's prepared Titanic rows (891 rows, 549 and 342 per class) it gave 1,098 rows, 549 per class, with the original rows unchanged, the column types kept and only seen text values in the new rows; FLAML fits with it for LightGBM, XGBoost, random forest, extra trees and L1 logistic regression. On wilt (a development dataset, class ratio 17.5), one 20 s search with seed 1 through the runner, with the saved model reloaded and a 15-fold Assessment, ran end to end with and without SMOTE (test F1 0.852 with it and 0.780 without, but mean validation F1 over the 15 folds 0.846 and 0.854; one search each, so a check that the parts work together, not a result).

**Algorithm 23: cross-fitted target encoding** (Data_Step `target_encode`, Strategy `target_encode`, task 30; Requirement 27.1). A text column is replaced by the mean target of its value, which a tree can split on even when the column has hundreds of values.

1. Turn the target into numbers: 1 for the positive class (binary; the last sorted label, as in FLAML), one 0/1 column per class (multiclass), or the value itself (regression).
2. For each value v of the column, on the rows the step learns from: encoding(v) = (sum of the targets of the rows with v + m x mean of all rows) / (number of rows with v + m), with m = 10. A rare value gets almost the overall mean; a common value gets almost its own mean. A value not seen gets the overall mean; an empty cell stays empty.
3. Cross-fitting on the training rows themselves: split them into 5 parts (shuffled, seed 0) and encode each part with the encodings learned from the other 4 parts only. So no row's encoding uses its own label, and the model cannot learn the encoding of its own label (target leakage). The encodings for other rows (validation and test rows, new data) are learned from all training rows.
4. The runner trains on the cross-fitted rows of the Training_Part; the Assessor does the same within each Session_Fold's training rows, with that fold's labels only, and scores the validation rows through the encodings of all of that fold's training rows.
5. In the rule table it is the fourth answer to high_cardinality, after dropping, frequency encoding and grouping.

The 10 and the 5 parts are design choices made before any run, not tuned. scikit-learn 1.9.1's `TargetEncoder` was not used: on Titanic's `Cabin` its `transform` raised `TypeError` for rows whose cells were all empty, and it treats empty cells as a value of their own, while the other Data_Steps keep them empty (checked 12 November 2026).

**Algorithm 24: word features** (Data_Step `token_features`, Strategy `token_features`, Diagnosis `unused_words`, task 30; Requirement 27.2).

1. The words of a cell are the runs of letters and the runs of digits of its lower-case text ("Braund, Mr. Owen Harris" gives braund, mr, owen, harris; "C85" gives c, 85).
2. On the training rows, keep the words found in at least 2% of the rows and in at least two different values, but not in every row; at most the 20 most frequent. A word that only ever appears as one whole value (male, female) is not kept, because the value itself already says it.
3. Replace the column by one 0/1 column per kept word. The original values are not used.
4. The profiler lists the frequent words of each text column of the Training_Part. The Critic's unused_words Diagnosis names a text column with frequent words that no encoding changes: low if the model uses the column (it sees each value only as a whole), moderate if the first Setup left it out by itself as identifier-like (the model sees none of it). Columns the user excluded are never brought back.
5. The answer is token_features(column). For a column the first Setup left out, it takes the column out of the exclusions and adds the step, so only the words enter the model.

On Titanic's Training_Part (seed 42 split) the words of `Name` begin with mr, miss, mrs, william, john, master; `Cabin` gives the deck letters c, b, d, e; `Ticket` gives prefixes such as pc, soton and ston. `Sex` and `Embarked` give none. There is no Titanic-specific code: the same rule runs on every text column.

Checked on 12 November 2026 (throwaway, temporary folder, code of task 30 before its commit, Titanic, seed 1, 4 cores). One 15 s search through the runner: without the new steps (PassengerId and Name excluded) test F1 0.750 (LightGBM, 75 trials); with token_features on `Name` and target_encode on `Cabin` and `Ticket` (PassengerId excluded) test F1 0.758 (XGBoost, 70 trials), and permutation importance put `Name` second after `Sex`. One search each, so this only shows that the parts work together. A 150 s Session (at most 6 Experiments) gave the first Setup the Diagnoses unused_words moderate for `Name` and low for `Cabin` and `Ticket`; its fifth step tried token_features on `Name` and the Judge discarded it (mean difference +0.0014 F1 over the 15 folds, not significant). The Session's test F1 went from 0.750 (first model) to 0.754 (final model, after keeping drop `Ticket` and grouping the rare values of `Cabin`).

**Algorithm 25: group-aware splits** (`app/services/splits.py`, `SessionSettings.group_column`, task 31; Requirement 28). For data where new rows come from new groups (several rows per patient, store or family), a model scored on rows of groups it has already seen looks better than it will be on new groups.

1. The user names a grouping column. It is never a feature: the Session adds it to the user's exclusions.
2. Outer split: whole groups go to the test part (`GroupShuffleSplit`, seeded; the test share is then a share of groups, and the test part is not stratified).
3. Session_Folds: in each of the 3 repeats, the groups of the Training_Part are shuffled with that repeat's seed and dealt into 5 folds, keeping the class balance where possible (`StratifiedGroupKFold`; `GroupKFold` for regression). A group is never split between the training and validation rows of a fold.
4. The choice is recorded in `session.json` (`settings.group_column`, `splits.grouped`, and notes with the group counts). Without a grouping column nothing changes, fingerprints included.

The outer split was made group-aware too, although Requirement 28 only names the folds, because with groups on both sides the final test score would still be optimistic for new groups. Checked on 12 November 2026 (throwaway, temporary folder): a 60 s Session on Titanic grouped by `Ticket` (families share tickets), seed 1, 4 cores, through `scripts/run_autopilot.py --group-column Ticket`: 137 of the 681 ticket groups (178 of 891 rows) formed the test part, the 544 groups of the Training_Part each stayed inside one fold, and `Ticket` was left out of the features; first and final test F1 0.783 and 0.767 (one short Session, not a result).

**Algorithm 26: SHAP explanations of the final model** (`app/services/explainer.py`, task 32; Requirement 29). SHAP (Lundberg and Lee [21]) gives each input column a share of one prediction; the shares add up to the prediction minus the average prediction.

1. Rebuild the Session's Training_Part from the dataset and the recorded settings (seed, test share, exclusions, group column), and refuse the data if the rebuilt split's fingerprint is not the recorded one. The Test_Part is never used.
2. Take a seeded sample of at most 200 Training_Part rows, and pass them along the final model's own path: its Data_Steps, then FLAML's preparation of the rows, down to the fitted learner. So the columns explained are the learner's inputs, for example `Name__tok_mr` after word features.
3. Tree learners (LightGBM, XGBoost, random forest, extra trees): shap's exact tree algorithm, in log-odds for the boosted models, in probabilities for the forests and in the predicted value for regression.
4. Other learners: the permutation estimate on the predicted probabilities (or value), against 25 background rows of the Training_Part, for the first 50 sampled rows (it calls the model thousands of times per row).
5. Show the mean |SHAP value| of the top 15 columns (per class for multiclass), and the contributions to one chosen row.

Measured on 12 November 2026 (throwaway, temporary folder, one FLAML search of 4 trials per learner, seed 1, 4 cores, 200 explained rows): tree SHAP took at most 1.9 s (the first call, which includes loading shap) and otherwise under 0.1 s for each tree learner on Titanic, penguins and a 1,500-row sample of California housing; the permutation estimate with 50 background rows took 28.5 s for L1 logistic regression and 169 s for k-nearest neighbours on Titanic, which is why it now uses 50 rows and 25 background rows. For the grouped Titanic Session of Algorithm 25 (final model LightGBM), `scripts/explain_session.py` ranked Sex (1.52 log-odds), Fare (0.80), Pclass (0.79) and Age (0.72) first. The contributions add up to the prediction (checked by `tests/test_explainer.py` for a random forest and for logistic regression).

**Algorithm 27: single-column leakage check** (`profiler.single_column_scores`, Diagnosis `possible_leakage`, task 33; Requirement 30.2). Target leakage (Kaufman et al., 2012) is a column that gives the answer away, for example one filled in after the outcome; a model built on it looks excellent and fails on new data.

1. On the Session's Training_Part (at most 5,000 rows, a seeded sample), score each feature column alone in 5-fold cross-validation: a numeric column with a depth-6 decision tree (at least 5 rows per leaf), and for regression also with an isotonic (monotone) fit, the better counting; any other column by the most common target of each value (the mean for regression), with unseen values getting the most common target of the training rows.
2. The score is balanced accuracy for classification (a constant guess gets 1 / classes, whatever the class balance) or R2 for regression.
3. A used column scoring at least 0.99 gets the Diagnosis possible_leakage (high). It is only a warning: no rule of the Strategy_Engine answers it, because only the user knows whether the column is known before the outcome. The dashboard also warns in the target section, before any run.

The 0.99 was set before any run and not tuned; it is an optional threshold with a default, so the frozen thresholds file stayed as it was. The isotonic fit was added after a test: with the tree alone, a near-copy of a continuous target scored R2 0.987 on 300 rows.

**Algorithm 28: target suggestion** (`TaskDetector.suggest_targets`, task 33; Requirement 30.1). Among the columns that can be a target (not constant, not identifier-like, not text with very many values), points for a generic target name (target, label, class, y, outcome, response: 3) or such a word inside the name (2), being the last column (1), holding only 0 and 1 or true and false (1), and having no empty cells (0.5); ties go to the later column. The best three are shown with their reasons, and the first is the default choice. Design choices, not measured.

Measured on 12 November 2026 with `scripts/check_single_columns.py` (code of task 33 before its commit; Training_Part of the seed-42 split, test share 0.2, every registered dataset; summary in `results/leakage_check.csv`): no column of the 16 datasets reached 0.99, so no false alarm; the highest scores were 0.905 (energy_efficiency, `relative_compactness`, R2) and 0.900 (breast_cancer, `mean concave points`, balanced accuracy); the check took 0.06 to 1.62 s per dataset (MiceProtein, 77 columns, the slowest). The registered target was the first suggestion on 15 of the 16 datasets; on penguins it came third, after `sex` (the last column) and `island`.

**Algorithm 29: plain-language summary by a local language model** (`app/services/narrator.py`, task 34; Requirement 31). The model explains; it never decides.

1. Take facts from the Session record only: the data and target, the budget, the first Setup's Diagnoses, each step's change and outcome in words with its numbers, the changes kept, the stop reason, and the first and final test scores. No data rows.
2. Send them with fixed instructions (120 to 180 words, plain English, only these facts, no new numbers) to a model on this machine through Ollama (`qwen2.5:7b-instruct`, temperature 0, seed 0). Any address other than this machine is refused.
3. List the numbers in the text that are not in the facts (after rounding; a percentage may stand for a share).
4. Save the text with its label ("Generated text ... not checked by a person and no decision used it"), the facts and the list in `summary_generated.json`, apart from the record. No module of the loop reads it (checked by a test).

Checked on 12 November 2026 (throwaway, temporary folder, Ollama 0.32.1 on this laptop): for the grouped Titanic Session of Algorithm 25 the first summary took 13.5 s (including loading the model) and said that dropping `Cabin` had not helped, although the Judge had kept it as simpler and not worse: the model misread the verdict code. After the facts gave each outcome in words ("kept, because it was simpler and not worse"), a second summary took 6.4 s, described the kept change correctly, gave the two test scores (0.7826 and 0.7667) and had no number outside the facts. It still added judgements of its own ("still within acceptable limits"), which is why the text is labelled and never used.

**Local HTTP API and Docker image** (`app/api.py`, `Dockerfile`, task 35; Requirement 32). Not an algorithm; the service around the same Session code.

- Endpoints: `GET /health`, `GET /sessions`, `GET /sessions/{id}` (the whole `session.json`), `POST /sessions` (starts one Session in a background thread and returns its ID; one at a time) and `POST /sessions/{id}/stop`. FastAPI also serves its own documentation at `/docs`.
- Safety for a service without a login: it listens on 127.0.0.1 by default; a dataset must be a CSV file inside the project's `data/` folder (checked first, before anything runs); session IDs must look like `SES-001`, so no path can be passed as an ID; the request fields have ranges (for example 10 s to 24 h of budget).
- The Docker image (Python 3.11 slim) holds the code, the bundled samples and the pinned packages, but no results; it runs the API on port 8000 and the dashboard on 8501, keeps `experiments/`, `models/` and `mlruns/` in volumes, and is meant to be run with the ports published on 127.0.0.1 only. The local summaries are not available inside it (they need Ollama on the host).

Checked on 12 November 2026 (throwaway, temporary folder, real FLAML, uvicorn on 127.0.0.1): a 45 s Titanic Session started through `POST /sessions` (seed 1, 4 cores) ran 2 Experiments in 38.0 s and its record came back through `GET /sessions/SES-002`; while it ran, a dataset outside `data/` was refused with 400. A first attempt was cut off when the test server was stopped too early, and that Session's record stayed at "running": a Session killed with its process cannot mark itself as stopped.

The image was built on the same day with Docker Desktop 29.6.1 (started for the purpose): 2.12 GB, about 5 minutes to install the packages (299.7 s for that step). A check container with the ports published on 127.0.0.1 answered `/health` within 5 s, the dashboard answered HTTP 200 and its own health check "ok", Docker's health check reported "healthy" and the processes ran as the non-root user. A 40 s Titanic Session started through the container's API (seed 1, 2 cores) ran 2 Experiments in 35.0 s, with first and final test F1 0.75. Its record had no commit hash, because the image holds no git history; records made in the container can be traced to the image tag only. The check container and its volumes were then removed; the image was kept.

---

## 6. Proposed flowchart, DFD and block diagrams

To put these in the report, paste each block into mermaid.live and export PNG or SVG, or use a Mermaid preview extension in the editor. The call sequence of one experiment is in `architecture.md`.

More diagrams are in `how-it-works.md`: the layers (what we wrote and what we use), the code map, the dashboard's background worker (sequence diagram), the data split, inside one FLAML call, FLAML's search loop, CFO's step rule, and where the planned parts connect. All diagrams in the maintained documents parse with Mermaid 11.15.0, the version in Kiro's Markdown preview (checked on 22 October 2026, and again on 27 October 2026 after the Phase 2 diagrams were added: all 18 blocks in `how-it-works.md`, `architecture.md` and this file).

### 6.1 Block diagram: current baseline

```mermaid
flowchart LR
    U["User"] --> UI["Streamlit dashboard"]
    UI --> PR["Dataset profiler"]
    UI --> TD["Task detector"]
    UI --> RU["AutoML runner"]
    RU --> SP["Stratified train/test split"]
    SP --> FE["FLAML engine"]
    FE --> EV["Evaluator and explanations"]
    EV --> TR["Experiment tracker"]
    RU --> MS[("Model files")]
    TR --> ML[("MLflow store")]
    TR --> JR[("Experiment records")]
    JR --> UI
    EV --> UI
```

### 6.2 Block diagram: proposed system

```mermaid
flowchart TD
    IN["Dataset, target and budget"] --> DI["Dataset intelligence: profiler and task detector"]
    DI --> PL["Experiment planner (planned)"]
    PL --> AM["AutoML core: FLAML"]
    AM --> EV["Evaluation: metrics, errors, cost"]
    EV --> CR["Critic (planned)"]
    CR --> SE["Strategy engine (planned)"]
    SE --> PL
    EV --> ME[("Experiment memory (planned)")]
    ME --> SE
    RM["Resource manager (planned)"] --> PL
    RM --> SE
    EV --> OUT["Best pipeline and experiment log"]
```

### 6.3 DFD level 0 (context)

```mermaid
flowchart LR
    U["User"] -->|CSV file, target, budget, metric, seed| S(("AutoPilot-ML"))
    S -->|profile, results, charts, history, trained model| U
```

### 6.4 DFD level 1 (current baseline)

```mermaid
flowchart LR
    U["User"]
    P1(("1.0 Load and profile data"))
    P2(("2.0 Detect task and check settings"))
    P3(("3.0 Split data"))
    P4(("4.0 Search models with FLAML"))
    P5(("5.0 Evaluate and explain"))
    P6(("6.0 Save and track"))
    P7(("7.0 Show results and history"))
    D1[("D1 Model store")]
    D2[("D2 Experiment records")]
    D3[("D3 MLflow store")]

    U -->|CSV file| P1
    P1 -->|dataset profile| U
    U -->|target, task, budget, metric, seed, excluded columns| P2
    P1 -->|data table| P2
    P2 -->|checked configuration| P3
    P3 -->|training part| P4
    P3 -->|test part| P5
    P4 -->|best model, trial history, leaderboard| P5
    P5 -->|metrics, importance, predictions| P6
    P6 -->|model bundle| D1
    P6 -->|summary and result files| D2
    P6 -->|parameters, metrics, files| D3
    D2 -->|past experiments| P7
    P5 -->|results| P7
    P7 -->|tables and charts| U
```

### 6.5 Flowchart: one experiment (current)

```mermaid
flowchart TD
    A(["Start"]) --> B["Load CSV"]
    B --> C["Profile the dataset"]
    C --> D["User selects the target column"]
    D --> E["Detect the task; user may override"]
    E --> F{"Settings valid?"}
    F -->|No| G["Show the error"] --> D
    F -->|Yes| H["Drop rows without a target and the excluded columns"]
    H --> I["Stratified 80/20 split"]
    I --> J["FLAML search on the training part within the time budget"]
    J --> K{"At least one trial finished?"}
    K -->|No| L["Ask for a larger time budget"] --> Z(["End"])
    K -->|Yes| M["Retrain the best configuration on all training rows"]
    M --> N["Evaluate on test and train parts, compare with the trivial baseline"]
    N --> O["Permutation importance on the test part"]
    O --> P["Save the model, write result files, log the MLflow run"]
    P --> Q["Show results and update the history"]
    Q --> Z
```

### 6.6 Flowchart: autonomous loop (built in Phase 2)

```mermaid
flowchart TD
    A(["Start: dataset, target, total time, maximum Experiments, CPU cores"]) --> B["Training part 80% and test part 20%, 3 x 5 folds of the training part, profile of the training part"]
    B --> C["Planner: first Setup without ID-like and constant columns"]
    C --> D["FLAML search on the training part, 35% of the time"]
    D --> E["Assessor: refit the best configuration on the 15 folds"]
    E --> F["Critic: Diagnoses with evidence"]
    F --> G["Strategy engine: Proposals ranked by severity, cost and table order"]
    G --> H{"Resource manager: stop?"}
    H -->|Stop pressed, error, time, Experiments, no Proposal or 3 discards in a row| Z["Final evaluation: first and final model on the test part"] --> DONE(["End"])
    H -->|go on| I["Planner: candidate = Accepted Setup + the top Proposal, warm-started, with the next budget"]
    I --> J["FLAML search, then Assessor on the same 15 folds"]
    J --> K{"Judge: corrected resampled t-test on the fold differences"}
    K -->|keep or keep_simpler| L["The candidate becomes the Accepted Setup"]
    K -->|discard| M["The Accepted Setup stays"]
    L --> N["Memory: one record of the Verdict, its effect and its cost"]
    M --> N
    N --> F
```

The draft of this flowchart (Phase 1 review) used the memory to plan the first Setup and ranked changes by expected gain per second; both need the memory and are Phase 3. Still to add: render the diagrams as images.

---

## 7. Tools and technology used

Machine used for all runs so far:

| Item | Value |
|---|---|
| Laptop | ASUS ROG Zephyrus G15 (GA503RW) |
| CPU | AMD Ryzen 9 6900HS, 8 cores, 16 threads |
| RAM | 15.2 GB, as reported by Windows |
| GPU | Not used |
| Operating system | Windows 11 Home Single Language, version 10.0.26200 |

Software (versions pinned in `requirements.txt`):

| Area | Tool | Version | Used for |
|---|---|---|---|
| Language | Python | 3.11.16 | Everything |
| Environment | Miniconda (conda) | 26.5.3 | Separate environment `automl` (about 1.1 GB on disk) |
| Data | pandas | 3.0.6 | Loading and handling tables |
| Numerics | NumPy | 2.4.6 | Arrays and numeric work |
| Machine learning | scikit-learn | 1.9.1 | Splitting, metrics, permutation importance |
| Scientific | SciPy | 1.17.1 | Dependency; statistical tests later |
| AutoML | FLAML | 2.7.0 | Model and hyperparameter search |
| Learners | LightGBM, XGBoost | 4.7.0, 2.1.4 | Gradient-boosted trees searched by FLAML |
| Tracking | MLflow | 3.17.0 | Experiment runs on a local SQLite file |
| Dashboard | Streamlit | 1.65.0 | Web interface |
| Charts | Plotly | 7.1.0 | Interactive charts |
| Static charts | matplotlib | 3.11.2 | PNG charts for the documents and the report (`scripts/plot_search_history.py`) |
| Configuration | python-dotenv | 1.2.4 | Reading `.env` settings |
| Testing | pytest | 9.1.1 | Automated tests |
| Version control | Git | 2.55.0 | History, one change per commit |
| Editor | Kiro (based on VS Code) | | Development |

Why these: all open source, all run offline on a laptop CPU, widely used and documented. FLAML was chosen as the engine because its cost-aware search gives good models on small budgets.

Added in Phase 3 (27 October 2026):

| Area | Tool | Version | Used for |
|---|---|---|---|
| Datasets | OpenML (through scikit-learn's `fetch_openml`) | API v1 | Downloading the 12 benchmark datasets once (task 20) |
| Comparison system | AutoGluon-Tabular, with LightGBM 4.7.0, CatBoost 1.2.10 and xgboost-cpu 3.2.0 | 1.6.3 | The AutoGluon baseline of the benchmark (task 23) |
| Environment | conda environment `autogluon` (Python 3.11.17, pandas 2.3.3; about 1.1 GB) | | AutoGluon needs pandas below 2.4, the project pins 3.0.6; exact versions in `research/requirements-autogluon.txt` |

Planned, not used yet: Optuna (searching strategy settings), SHAP (explanations), imbalanced-learn (SMOTE).

Sample datasets (all public, small):

| File | Size | Rows x columns | Task and target | Notes from the profiler |
|---|---|---|---|---|
| `titanic.csv` | 59 KB | 891 x 12 | Binary, `Survived` (549 died, 342 survived; majority 61.6%, mild imbalance) | 866 missing values (Age 177, Cabin 687, Embarked 2); ID-like `PassengerId`, `Name`; high-cardinality `Ticket` (681), `Cabin` (147); `Cabin` 77% missing |
| `breast_cancer.csv` | 122 KB | 569 x 31 | Binary, `diagnosis` (357 benign, 212 malignant; majority 62.7%, mild) | 30 numeric features, no missing values |
| `penguins.csv` | 13 KB | 344 x 7 | Multiclass, `species` (Adelie 152, Gentoo 124, Chinstrap 68; majority 44.2%, mild) | 4 numeric and 2 text features; 19 missing values |
| `california_housing.csv` | 1.9 MB | 20,640 x 9 | Regression, `MedHouseVal` (mean 2.07, median 1.80, range 0.15 to 5.00, skew 0.98) | 8 numeric features, no missing values; the target is capped at about 5.0 in the original data |

Sources: Titanic, Kaggle "Titanic: Machine Learning from Disaster" training set (via datasciencedojo/datasets on GitHub); Breast Cancer Wisconsin (Diagnostic), UCI, exported from scikit-learn; Palmer Penguins (Horst, Hill and Gorman), via mwaskom/seaborn-data; California Housing (Pace and Barry, 1997), exported from scikit-learn.

---

## 8. Implementation and coding

### 8.1 Code size (22 October 2026)

| File | Lines | Main parts |
|---|---|---|
| `app/services/automl_runner.py` | 817 | `ExperimentConfig`, `AutoMLEngine`, `FLAMLEngine`, `EngineResult`, `ExperimentResult`, `AutoMLRunner`, `unseen_category_warnings` |
| `app/services/profiler.py` | 328 | `DatasetProfiler`, `DatasetProfile`, `ColumnProfile`, `ClassBalance`, `analyze_class_balance` |
| `app/services/evaluator.py` | 318 | `Evaluator`, `EvaluationResult`, `METRICS`, `PRIMARY_METRICS`, `metric_warnings` |
| `app/services/task_detector.py` | 228 | `TaskDetector`, `TaskType`, `TaskDetectionResult`, `TargetCandidate` |
| `app/services/experiment_tracker.py` | 197 | `ExperimentTracker` |
| `app/core/config.py`, `logging_config.py` | 130, 41 | `Settings`, `build_settings`, `get_settings`, `setup_logging` |
| `app/components/results_viewer.py` | 293 | Sections 5 and 6 of the dashboard |
| `app/components/experiment_viewer.py` | 206 | Sections 3, 4 and 7 |
| `app/components/dataset_profiler.py` | 185 | Sections 1 (profile) and 2 |
| `app/components/dataset_uploader.py` | 124 | Section 1 (input), CSV reading with fallbacks |
| `app/main.py` | 91 | Page layout and sidebar |
| `scripts/compare_seeds.py` | 89 | Paired comparison of two setups over several seeds |
| `scripts/plot_search_history.py` | 172 | Charts of one experiment's FLAML search from its saved trial history |
| `tests/` (3 files) | 469 | 47 tests |

Total: 3,711 lines in 22 Python files (2,981 in `app/`, 469 in `tests/`, 261 in `scripts/`), including comments and blank lines.

After Phase 2 (26 October 2026, code of commit 940225a), counted the same way:

| File | Lines | Main parts |
|---|---|---|
| `app/services/automl_runner.py` | 1,326 | Phase 1 runner, plus the Setup fields, warm starts, search-space limits, sample weights, `fit_config` and the options that keep the test part out of a Session |
| `app/services/autopilot.py` | 620 | `AutopilotSession`, `SessionSettings`, `SessionStep`, `SessionResult`, the records and the Final_Evaluation |
| `app/services/critic.py` (+ `critic_thresholds.toml`, 64) | 453 | `Critic`, `Diagnosis`, `CriticThresholds` |
| `app/services/profiler.py` | 376 | Phase 1 profiler, plus main-value shares, near-constant columns and correlated pairs |
| `app/services/strategies.py` | 356 | the 9 Strategies, `CATALOGUE`, `REGULARISED_LIMITS` |
| `app/services/experiment_tracker.py` | 347 | Phase 1 tracker, plus Session IDs, `session.json`, step files, MLflow parent and child runs |
| `app/services/assessor.py` | 233 | `Assessor`, `Assessment`, `FoldScore` |
| `app/services/transforms.py` | 208 | `DataStepSpec`, `DataPipeline`, `PipelineModel` |
| `app/services/resource_manager.py` | 208 | `ResourceManager`, `SessionBudget`, `StepCost`, `StopReason` |
| `app/services/strategy_engine.py` | 200 | the rule table, `propose`, `Proposal`, `RulesSelector` |
| `app/services/experiment_memory.py` | 200 | `ExperimentMemory`, `MemoryRecord`, `StrategyStats` |
| `app/services/splits.py` | 167 | `make_outer_split`, `make_session_folds` |
| `app/services/judge.py` | 158 | `Judge`, `JudgeSettings`, `JudgeDecision`, `corrected_resampled_t` |
| `app/services/search_summary.py`, `experiment_planner.py` | 72, 42 | `summarize_search`; `Planner` |
| `app/components/autopilot_viewer.py` | 523 | the autonomous mode of the dashboard |
| `app/main.py` | 134 | page layout, mode switch and sidebar |
| `research/` (5 files) | 1,154 | `synthetic.py` 287, `critic_study.py` 422, `judge_study.py` 212, `phase2.py` 229 |
| `scripts/` (5 files) | 613 | `run_autopilot.py` 96, `run_study.py` 166, `run_phase2.py` 90, plus the two Phase 1 scripts |
| `tests/` (22 files) | 4,461 | 468 tests |

Total: 13,400 lines in 62 Python files (7,172 in `app/`, 1,154 in `research/`, 613 in `scripts/`, 4,461 in `tests/`), plus the 64-line thresholds file.

After Phase 3 (31 October 2026, code of commit 812eec4), counted the same way: 17,610 lines in 82 Python files (7,511 in 29 files of `app/`, 3,278 in 13 of `research/`, 965 in 8 of `scripts/`, 5,856 in 32 of `tests/`, of which 31 test files). The Phase 3 files:

| File | Lines | Main parts |
|---|---|---|
| `research/benchmark.py` | 446 | `Cell`, `grid`, `smoke_grid`, `run`, `run_cell`, `preflight`, the exclusions and the power checks |
| `research/judge_agreement.py` | 353 | `StudyPair`, `study_pairs`, `run_pair`, `strict_decision`, `cohen_kappa`, `agreement` |
| `research/systems.py` | 342 | `Problem`, `prepare`, `ManualSystem`, `FlamlSystem`, `AutoGluonSystem`, `AutopilotSystem` |
| `research/benchmark_analysis.py` | 292 | `blocks`, `compare_family`, `targets`, `target_pairs`, the CD diagram and the score chart |
| `research/datasets.py` | 255 | the dataset registry and its selection rule |
| `research/stat_tests.py` | 247 | Friedman, Iman and Davenport, Nemenyi, Wilcoxon with Holm, rank-biserial, bootstrap, time-to-target |
| `research/memory_build.py` | 133 | the frozen memory's build, `freeze` and `summary` |
| `research/autogluon_worker.py` | 56 | the AutoGluon run in its own environment |
| `scripts/run_benchmark.py`, `build_memory.py`, `run_study.py` | 142, 84, 217 | the command lines of the benchmark, the memory build and the three studies |
| `app/services/strategy_engine.py` | 406 | the rule table and the four selectors (200 lines after Phase 2) |

After the optional tasks 29 to 35 (12 November 2026, code of commit 40c868f), counted the same way (`git ls-files "*.py"`, every line): 20,235 lines in 97 Python files (8,938 in 33 files of `app/`, 3,294 in 13 of `research/`, 1,183 in 12 of `scripts/`, 6,820 in 39 of `tests/`, of which 38 test files). The new modules: `app/services/narrator.py` 203 lines, `app/services/explainer.py` 197, `app/api.py` 180 and `app/services/resampling.py` 101; `transforms.py` grew to 400 lines, `profiler.py` to 470, `critic.py` to 504 and `strategies.py` to 430.

### 8.2 Main interfaces

```python
result = AutoMLRunner().run(
    dataframe=df, target_column="Survived", task=None,      # None means detect
    time_budget=60, primary_metric="f1", random_seed=42, test_size=0.2,
    excluded_columns=["PassengerId", "Name"], dataset_name="titanic",
    estimator_list=None, progress_callback=None,
)
```

- `AutoMLEngine.fit(X_train, y_train, *, task_type, metric, time_budget, seed, n_jobs, estimator_list, work_dir)` returns an `EngineResult`: model, engine name and version, best estimator, best configuration, best validation score, validation method, training time, trial history, leaderboard, model's own feature importance, class labels.
- `ExperimentResult` (saved as `summary.json`): ID and time; dataset, target, task, metric; budget, seed, test share; engine and version; train and test sizes; feature and excluded columns; best model and configuration; validation, test and train scores; full test, train and baseline metrics; training time, total time, number of trials; leaderboard; feature importance; dataset meta-features; warnings; model path, result folder and MLflow run ID.
- Progress stages reported to the UI: preparing (2%), searching (5% to 85%, estimated from time), evaluating (85%), explaining (90%), saving (95%), done (100%).

Phase 2 extensions (spec task 3, 26 October 2026). Every new option defaults to the Phase 1 behaviour, and the 70 existing tests pass unchanged except for the test engine, which takes the new arguments.

- `ExperimentConfig` gains `data_steps` (rejected until task 12), `class_weight` (None or "balanced"), `search_metric`, `search_space_limits` (plain JSON such as `{"lgbm": {"num_leaves": {"upper": 32}}}`, bounds inclusive), `starting_points` (learner family to configuration) and `max_iter`. `setup()` lists the settings that define a Setup and `setup_key()` is a 16-character SHA-256 hash of it; the time budget, `max_iter`, starting points, dataset name and `n_jobs` are left out.
- `AutoMLEngine.fit(...)` gains `starting_points`, `search_space_limits`, `sample_weight` and `max_iter`; `metric` is now the metric the search optimises. `FLAMLEngine` passes each one to FLAML only when it is given. `EngineResult` gains `best_config_per_estimator` and `warnings`.
- `FLAMLEngine.fit_config(X, y, *, task_type, estimator, config, seed, n_jobs, sample_weight=None)` trains one fixed configuration through FLAML (`max_iter=1`) and returns a model that predicts the original labels.
- `FLAMLEngine.custom_hp` turns the limits into FLAML's `custom_hp`: it copies FLAML's own spec of the setting, narrows the bounds and moves the starting values inside them. `check_starting_points` leaves out a starting configuration with a value outside the (limited) space and records a warning.
- `AutoMLRunner.run_experiment(..., split=None, evaluate_test=True, compute_importance=True, profile_rows="all", session_id=None, parent_experiment_id=None, proposal=None)`. With `evaluate_test=False` the test part is not used at all (no test metrics, baseline, predictions, importance or test-row notes). With `class_weight="balanced"` the runner passes scikit-learn's balanced sample weights to FLAML.
- `ExperimentResult` gains `schema_version` (2), `search_metric`, `test_evaluated`, `best_config_per_estimator`, `search_summary`, `session_id`, `parent_experiment_id`, `proposal`, `setup_key`, `setup` and `profile_rows`.
- `decision_view(result)` returns the record without the test score, test and baseline metrics, permutation importance and warnings, and without the meta-features unless the profile was made from the training part only. Decision components only ever see this view.
- Task 18, the dashboard: `AutopilotSession.run` gained `on_update`, called with a copy of the session record each time `session.json` is written (an error in it is logged and never stops the Session); `app/components/autopilot_viewer.py` has `render_session_settings`, `render_session_run` and `render_memory_records`, and `new_session(settings)` builds the Session the page runs.
- Task 11, the Session: `AutopilotSession(settings, engine, tracker, clock, enable_mlflow).run(df, SessionSettings(base, budget, n_splits=5, n_repeats=3, judge, thresholds_path, selector="rules"), progress, stop_requested)` returns a `SessionResult` (status, steps, first and final Experiment IDs, stop reason, Final_Evaluation, total seconds). `Planner.first_setup` and `Planner.candidate` build the Setups. Command line: `python scripts/run_autopilot.py --dataset data/sample/titanic.csv --target Survived --seconds 300 --max-experiments 8 --cores -1 --seed 42`.
- Task 10, records and tracking (`app/services/experiment_tracker.py`): `allocate_session_id()` reserves `SES-###` by creating its folder, as for experiments. `save_session`, `save_record` and `save_step_file` (`assessment.json`, `diagnoses.json`, `decision.json`) write JSON to a temporary file and then rename it over the old file, so a crash while writing never leaves a half-written record. `load_records()` reads Phase 1 records as schema version 1. `history_frame()` has a Session column, and `session_history_frame()` lists Sessions with their first and final validation and test scores. MLflow gets one parent run per Session (`start_session_run`, `finish_session_run`) and the runner logs each Experiment of a Session as its child run (tag `mlflow.parentRunId`).
- Task 5: `Assessor(engine, folds, task_type, primary_metric, n_jobs, seed).assess(X_train, y_train, config, result)` returns an `Assessment` (one `FoldScore` per fold; `values(metric, part)`, `summary()`, `to_dict()`). `summarize_search(history, metric)` returns a `SearchSummary`, which the runner now saves in every record (`search_summary`). `unseen_category_shares(X_train, X_other)` gives the unseen share per categorical column; the Phase 1 test-row note now uses it, with the same text.

### 8.3 Dashboard sections

1. Dataset: upload or sample; cards for rows, columns, missing values, duplicates, numerical and categorical features; tabs for columns, preview, data quality and target candidates.
2. Target: target column; detected problem with reason and confidence; override; class distribution or target histogram; columns to exclude.
3. Experiment configuration: time budget, primary metric, random seed, test split; metric warnings.
4. Run AutoML: button, progress bar, success or error message.
5. Results: best model, test score, validation score, training time; notes and warnings; test metrics with differences from the baseline; train vs test table; leaderboard; search progress chart.
6. Explainability: permutation importance with spread, the model's own importance, confusion matrix with per-class report, ROC curve (binary), predicted vs actual and residual plots (regression), best configuration.
7. Experiment history: table of all experiments, chart of test scores, command to open MLflow.

Since task 18 (26 October 2026) the sidebar has a mode switch. "Single experiment" keeps sections 1 to 7 above. "Autonomous run" reuses sections 1 and 2 and then shows:

3. Session settings: total time in minutes, maximum Experiments, CPU cores, primary metric, seed, test split; advanced settings: fold repeats (1, 3 or 5), Judge alpha (0.05, 0.10 or 0.20), the Strategy choice (`rules`; since task 21 also `random`, `thompson` and `ucb1`) and, since task 21, a switch to let the choice use the memory, with the code defaults preselected.
4. Autonomous run: Start; while the Session runs, a progress bar of the time used, the current activity, Stop, and the view below, redrawn every second; afterwards the stop reason, the time and Experiments used, and the first and final model side by side on the test part (test score and its change, learner, changes kept, all test metrics next to the trivial baseline) under the caption "Not used for decisions". The view: a timeline table (step, Experiment, Diagnoses answered, Strategy, change, mean difference and its standard deviation, p(better), p(worse), Verdict, validation score, seconds), a chart of the Accepted_Setup's validation score per step with the candidates marked by Verdict, and the Diagnosis cards with their evidence for every accepted Setup.
5. Experiment memory for this dataset: the memory records of the current dataset and target (Strategy, settings, Diagnoses answered, Verdict, mean difference and spread, folds, cost).

The sidebar lists the implemented modules and the planned ones. Since 26 October 2026 (after task 11.2) it listed the first version of the autonomous layer as built; since task 18 it lists the planner and session loop, the Critic (12 diagnoses, thresholds from the critic study), the Strategy_Engine (9 strategies), the Judge, the Resource_Manager and the Experiment_Memory (written and shown, not used for decisions yet) as built, and since its tests (task 18.1) also the autonomous mode in the dashboard. Since task 21.1 the learned strategy choice is listed as built and the memory without "not used for decisions yet"; planned: the benchmark against FLAML alone, AutoGluon and a manual baseline. Since 31 October 2026 the benchmark is listed as a built research tool, and since 12 November 2026 (optional tasks 29 to 35) the sidebar lists the Critic as 12 diagnoses with thresholds from the critic study plus unused words and a leakage warning, the Strategy engine with 12 strategies, the optional strategies (SMOTE, target encoding, word features), group-aware splits, SHAP, target suggestion with the leakage warning, the local summaries, and the API with the Docker image; the only planned item is the second, full benchmark run.

### 8.4 Files written by one experiment

```text
experiments/EXP-001/  summary.json, profile.json, evaluation.json, leaderboard.csv, search_history.csv,
                      feature_importance.csv, native_feature_importance.csv, predictions.csv, flaml_search.log
models/EXP-001/       model.pkl (model, feature columns, target, task, class labels)
mlruns/               mlflow.db and artifacts/ (copies of the model and the files above)
```

### 8.5 Commands

```bash
conda activate automl
cd C:\Users\priya\Desktop\project\AutoML\autopilot-ml
python -m pytest -q
streamlit run app/main.py --server.address localhost --browser.gatherUsageStats false
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000 --workers 1
python scripts/compare_seeds.py --dataset data/sample/titanic.csv --target Survived --exclude-a PassengerId,Name --exclude-b PassengerId,Name,Ticket,Cabin --seeds 1 2 3 4 5 --budget 30
python scripts/run_autopilot.py --dataset data/sample/titanic.csv --target Survived --seconds 300 --max-experiments 8 --cores -1 --seed 42
python scripts/run_study.py critic run      # and: critic analyse [--freeze LABEL]
python scripts/run_study.py judge run       # and: judge analyse
python scripts/run_phase2.py run            # and: run_phase2.py analyse
python scripts/prepare_datasets.py download # recreate data/raw/ from research/datasets.json
```

### 8.6 Tests (586, about 276 s on 12 November 2026)

Phase 1 ended with the first three files below (47 tests, about 29 to 41 s). Phase 2 has added the others so far. Full suite on 26 October 2026 after task 16: 457 passed in 139.7 s on the laptop in chapter 7; without the 4 study smoke tests marked slow (`python -m pytest -q -m "not slow"`, marker registered in `pytest.ini`), 453 passed in 123.6 s. After task 17: 458 passed in 150.4 s. After task 18.1: 465 passed in 175.8 s; without the 9 tests marked slow (4 study smoke tests and the 5 dashboard tests), 456 passed in 127.6 s. After task 19 (27 October 2026): 468 passed in 174.0 s. After task 21.1: 482 passed in 173.2 s. After task 22.1: 486 passed in 182.0 s. After task 23.1: 493 passed in 221.9 s. After task 24.1: 504 passed in 218.6 s. After task 25.1: 517 passed in 222.0 s. With the memory build (task 26): 521 passed in 217.1 s. With the benchmark analysis and the judge agreement study: 533 passed in 229.2 s. With the analysis fix of 28 October and the power checks of 31 October 2026: 536 passed in 234.6 s (a first run had one failure, a rounding tolerance in `test_judge_agreement.py`, fixed in e6de199). With the Judge fix, the Assessment reuse and the learned-choice changes (31 October 2026): 541 passed in 208.8 s. With SMOTE (optional task 29, 1 November 2026): 550 passed in 221.1 s. With target encoding and word features (optional task 30, 12 November 2026): 561 passed in 217.3 s. With group-aware splits (optional task 31, the same day): 568 passed in 221.0 s. With SHAP (optional task 32, the same day): 573 passed in 251.4 s, with 3 warnings from shap's own colour-map code (PendingDeprecationWarning). With target suggestion and the leakage check (optional task 33, the same day): 578 passed in 270.8 s; a first run showed 55 warnings from the new check (isotonic fits of weak or constant columns, and a pandas lookup that turned class 1 into 1.0, a real bug, fixed before the commit). With the local summaries (optional task 34, the same day): 583 passed in 265.6 s. With the API (optional task 35, the same day): 586 passed in 276.1 s. After the second benchmark run, with the sidebar change of 13 November 2026: 586 passed in 332.4 s. Run again on 15 November 2026 before the presentation, with no code change since: 586 passed in 367.5 s (PowerPoint was building the slides at the same time), with the same 3 warnings from shap's colour-map code. The counts below were taken from pytest's list of collected tests.

| File | Tests | What they cover |
|---|---|---|
| `tests/test_task_detector.py` | 16 | Detection for 7 kinds of target, confidence, missing values, unusable targets, rejected and allowed overrides, target candidates, default target |
| `tests/test_profiler.py` | 13 | Shape and types, missing values and duplicates, column flags, class distribution, the 4 imbalance levels, regression summary, JSON output and meta-features, bad input; since task 13.1 also the most-common-value shares, near-constant columns and correlated pairs (target left out), and profiles that round-trip through JSON, including old profiles without the new fields |
| `tests/test_automl_runner.py` | 21 | Real short FLAML runs (binary with MLflow and model reload, multiclass with ROC-AUC, regression), 7 invalid configurations, impossible override, too few rows, warning handling, unseen categories, a non-FLAML engine, the record left by a failed run, evaluator metrics against scikit-learn, baselines, permutation importance, metric checks; and (26 October 2026) datasets with repeated row labels refused while any unique index works |
| `tests/test_flaml_capabilities.py` | 9 | The FLAML features the autonomous layer relies on (section 5.2): starting points, fixed refits and their repeatability, sample weights, `custom_hp`, invalid starting points, the resampler, and that the upper value of an integer range is never searched |
| `tests/test_splits.py` | 15 | Outer split equal to Phase 1 (including EXP-001's saved test rows), session folds, stratification fallbacks, fingerprints |
| `tests/test_engine_extensions.py` | 29 | Without new options FLAML gets exactly the Phase 1 call; starting points, limits, weights and `max_iter` reach FLAML; `fit_config` predicts the original labels; limit conversion and starting-value checks; runner options; with `evaluate_test=False` no test row is fitted, predicted or profiled; `decision_view` has no test information; `setup_key` stability (pinned value); 10 invalid options rejected before an experiment ID is given out (including two invalid Data_Steps since task 12) |
| `tests/test_search_summary.py` | 5 | A small history worked by hand, error metrics, unusable histories, and the saved EXP-001 and EXP-002 histories (search time, time of the best trial, share after it, number of trials) |
| `tests/test_assessor.py` | 9 | Since 31 October 2026: a Setup that refits like an earlier one (here: fewer learners, same model) reuses its fold scores in a copy that names both Experiments, without a refit, while another configuration, class weights or other columns are assessed again. Before: One score per fold with every metric; means and standard deviations; only the fold's training rows of the Training_Part are refitted (the whole dataset is refused); class weights from each fold's training rows; a cached Setup is not refitted but a new configuration is; a failing fold fails the Assessment; unseen shares per fold against a hand-made answer; regression; two separate Assessments with real FLAML refits agree within 1e-12 |
| `tests/test_critic.py` | 112 | Hand-made profiles and Assessments, no FLAML. Task 13.1 added: every kind of the full Critic at high, at moderate and just below its threshold, for binary, multiclass and regression where it applies (underfitting, including the ROC-AUC baseline; budget_limited and weak_model_families, which skip unbounded search metrics; wasted_budget; missing_values; identifier_column; constant and near-constant columns; redundant_features; unreliable_validation); Data_Steps change which columns are diagnosed; all kinds at once come in table order. From task 6.1: each thin rule at high, at moderate, exactly at a threshold and just below it, with its evidence numbers; gaps and recall as means over folds; the smallest class in multiclass; no imbalance rule for regression; excluded and numeric columns skipped; fixed output order and repeatable output; the version label on every Diagnosis; an empty list when nothing applies; the choice of diagnostic metric; the default thresholds file holds the design values; missing, unreadable and invalid thresholds files are refused with a clear message |
| `tests/test_strategies.py` | 51 | Task 14.1 added: the exact change of each new Strategy (one Data_Step per column strategy, and the changed Setup passes the runner's check; restricted learner lists keep the Setup's order; more search time leaves the Setup and its key unchanged; the smoother metric sets log loss), their applicability (encodings only for text columns in use and only one encoding per column; indicators only for numeric columns with empty cells; no smoother metric for regression or when the search already uses log loss), conflicts (dropping a column removes its Data_Steps), malformed parameters, and that the original Setup is never changed. From task 7.1: the exact change of each Strategy to a Setup, with the original unchanged and nothing shared; drop_columns only for columns in use, with at least one left; class weights only for classification without weights; smaller trees keep tighter limits, never widen and apply once; the limits fit FLAML's search spaces for all three task types; canonical keys |
| `tests/test_strategy_engine.py` | 16 | Task 14.1 added: every new row of the rule table (identifier and constant columns dropped; missing values: drop only if high, indicator only for numeric columns; the later column of a redundant pair; one more-search-time Proposal for underfitting and budget_limited together; the smoother metric for unreliable validation, classification only; weak families left out in the order the search tried them; a treated column is not encoded again) and cost order within one severity. From task 7.1: the Titanic Proposals in rule order with merged answers; tried changes and changes that conflict with the Accepted_Setup left out; class weights for classification only; severity before table order; the full ranking key; empty lists; repeatable output |
| `tests/test_judge.py` | 27 | Since 31 October 2026: a mean difference within the refit tolerance is no gain (discard, or keep_simpler when simpler). Before: The defaults after the judge study (alpha 0.20 and 3 fold repeats; the worked example is kept at 0.20 but not at 0.10; the rule tests use alpha 0.10, as written); the corrected t and p-values against the worked example computed by hand; keep, keep_simpler and discard, including "within the margin but consistently worse"; the minimum gain; RMSE with the sign flipped; wins, losses and ties; equal differences decided on the mean without p-values; folds without scores left out; the decision record; different folds, metrics, failed Assessments and invalid settings refused |
| `tests/test_resource_manager.py` | 41 | With a fake clock: the first budget at its share and never below 5 s; later budgets from measured time and the reserve, not from the budgets handed out; the wasted-budget cap; more search time capped by the time left; the accepted budget follows kept steps only; each stop reason, and their order when all apply at once; patience counts discards in a row; every part of a step in the ledger; overruns against each step's deadline; 20 simulated Sessions with random overruns never end later than the budget plus the last step's overrun; invalid budgets refused |
| `tests/test_tracking.py` | 8 | Session IDs never clash (also with a hand-made folder in between and with 16 threads at once); `session.json` keeps its earlier steps and leaves no temporary file when writing is interrupted before or at the rename; a file held open for a moment is retried; step files; copies of the EXP-001 and EXP-002 records load as version 1 and appear in the history next to a Session record; the session history; an Experiment run in a Session is an MLflow child run of the Session's parent run, and one outside a Session is not |
| `tests/test_autopilot.py` | 14 | Whole Sessions with a recording engine, a scripted Assessor and a fake clock (about 1 s each; 3 repeats, to match the scripted 15 fold scores). Task 18.1 added: the update callback gets every saved record, the last one equal to `session.json`, each its own copy; a failing callback does not stop the Session. From task 11.1: no Test_Part row reaches the engine, the Assessor or the Critic's profile, and no Experiment record of a Session has a test number; every step's record and files; each stop reason (time, Experiments, patience, user request, no Proposals); a stop request still runs the Final_Evaluation; failed candidates (a failed search and a failed Assessment) are discarded and the Session goes on; a failed first Experiment stops the Session with an error record; bad settings write nothing; the Diagnoses and Verdicts replay exactly from the saved records |
| `tests/test_transforms.py` | 20 | Each Data_Step's exact output on a small frame, including unseen values and empty cells; steps in order; a step learns nothing when applied and each output row depends only on its own input row; the Assessor fits the steps on each fold's training rows only; specs round-trip through JSON and give the same setup key in either form; invalid specs and steps that do not fit the data are refused; a reloaded bundle predicts exactly as before saving from raw rows; importance on the original columns; without Data_Steps nothing changes |
| `tests/test_experiment_memory.py` | 15 | Since 31 October 2026: a kept step whose fold differences were all zero (also rounding noise of 7e-18) counts as not kept, while a mean of 0 with differing folds still counts; the arm statistics give the number of datasets and of records without effect. Before: Records survive a restart, in order, one JSON object per line, and adding only appends; damaged lines (not JSON, missing fields, a line cut off by a crash) are skipped with a warning; fields from a later schema are ignored; read-only memories and NaN values are refused; the snapshot hash; the dataset fingerprint; the nearest datasets in the right order with distances worked by hand (records averaged per dataset, features without spread and features one side lacks left out); other task types and an excluded dataset left out; strategy statistics on a small hand-made memory (weights, keep rate, gain when kept, cost); a Session adds one record per Verdict and none for a failed candidate; a pre-filled memory does not change any decision |
| `tests/test_critic_study.py` | 5 | The study's cases (every problem and strength, separate seeds, a different dataset each); with a fast stand-in engine, rows carry the truth and the evidence, the study's rule agrees with the real Critic for two sets of thresholds, tuning gives valid thresholds, and the written thresholds file loads (3 of them marked slow) |
| `tests/test_judge_study.py` | 4 | The study's pairs and seeds; one pair with a fast stand-in engine gives one row per setting (marked slow); the false-keep rate, the power and the choice of defaults on hand-made verdicts, including when no setting is safe enough |
| `tests/test_dashboard.py` | 5 | Streamlit's `AppTest` on `app/main.py` in a temporary project folder with the Titanic sample, real FLAML searches stopped after 3 trials, and no MLflow (all marked slow, about 40 s together): the sidebar marks the built modules and the default mode is unchanged; the single-experiment mode runs an experiment and shows its results; an autonomous run shows its steps, the first and final model and the memory records (one per Verdict); a Session whose searches fail is shown as an error and a new one can be started; Stop ends a running Session with stop reason "user" and the Final_Evaluation; no exceptions on the page |
| `tests/test_ablations.py` | 4 | With stand-in engines and a fake clock: without the Critic no Diagnoses are made and every change is a random valid one that answers nothing, is never repeated, differs from the rule order and repeats with the seed; without the Resource_Manager every search gets total time / maximum Experiments (the last one the time left), and all-discarded candidates no longer stop the Session after three (6 steps against 4); the equal-share manager on its own (no reserve, a wasted budget does not shorten a search, only time stops it); the random order changes the choices; every switch is recorded in `session.json` (the memory switch is shown in `test_selectors.py`) |
| `tests/test_selectors.py` | 12 | Since 31 October 2026: for `thompson` and `ucb1`, an arm with 6 records from one dataset keeps the rule order and has no score, and with `min_datasets` 1 the same records move it (the memories of the other tests now spread each arm over 2 datasets). Before: The rule order and the arm of a Proposal (its most severe Diagnosis); `random` shuffles exactly the applicable Proposals (class weights never appear for regression) and follows the seed; with a hand-made memory, a Strategy often kept on similar datasets moves up under `thompson` and `ucb1`, and the arm statistics are recorded; arms with fewer than 3 records keep the rule order, and only the places of known arms are re-filled; the current dataset's own records are left out; the same memory and seed give the same choices and scores; the UCB1 score worked by hand, with equal scores going to the cheaper arm; in a whole Session, the learned choice changes the first change only when the memory is switched on |
| `tests/test_datasets.py` | 4 | The dataset groups and the size rules; the draw is fixed by the seed, does not depend on the order of the pool, gives one dataset of each group to each list and never the same one twice; the description of a dataset uses the given task; the committed registry has 10 development and 6 evaluation datasets, no overlap, no sample in the evaluation list, all three tasks, and every required field |
| `tests/test_benchmark.py` | 13 | Since 31 October 2026: the run waits while the laptop is on battery (one message per wait), records the wait and the power state in each row, and lists a cell that ends on battery as excluded so it runs again; excluded rows are left out. Before: the default grid (390 cells: 300 main cells at 120 and 300 s and 90 ablation cells at 300 s, seed by seed) and the smoke setting (every system once on titanic, seed 1, the smallest budget, in its own folders). With stand-in systems on small synthetic datasets: every cell runs once and a restart runs none again; an interrupted cell has no row and runs again; a failed system and a failed split are rows, and running failed cells again adds new rows without editing old ones; every row has the memory's hash, a memory with another hash is refused, and a memory that changes during the run stops it without a row for that cell; memory records must come from development datasets (by name or fingerprint); the checks before a run; a cut-off last line and other columns are refused, and a commit such as `1e84091` stays text; the time-left estimate and the timing table; one real cell with the manual LightGBM system |
| `tests/test_resampling.py` | 9 | Optional task 29: SMOTE balances binary and multiclass targets with and without text columns, keeps the original rows first and unchanged, keeps FLAML's column types, adds only seen text values, repeats with the seed; SMOTE-N for text columns only; empty numeric cells stay empty in the original rows; fewer neighbours for a small class and no change for a one-row class or balanced data; cloneable; the Strategy and class weights exclude each other, apply once, only to classification, and the Setup key changes only when a resampler is set; class imbalance proposes SMOTE after the other two answers; the runner refuses an unknown resampler, one with class weights and one for regression; the Assessor resamples only each fold's training rows; with the real engine, FLAML resamples the refit's rows once and the search's training partitions, never a number of rows that includes validation rows (marked slow) |
| `tests/test_text_steps.py` | 11 | Optional task 30: flipping any one label never changes that row's own cross-fitted target encoding, and empty cells stay empty; other rows get the smoothed mean of all fitted rows (unseen values the overall mean); multiclass gives one column per class whose shares add up to 1, regression the smoothed mean; missing target, task type or text column are refused; the words of Titanic's names (mr, miss, mrs first; none for Sex and Embarked); word columns on other rows; the runner searches the cross-fitted rows and the reloaded bundle predicts from raw rows; the Assessor cross-fits within each fold's training rows; the profile's frequent words; token_features brings back the left-out `Name` (words only); the unused_words Diagnosis (moderate when left out, low when used, never for the user's own exclusions) and its rule. Also updated for task 30: the catalogue, rule-table and table-order tests, and the no-Critic test (searches capped at 50 s, so a random more_search_time no longer ends it) |
| `tests/test_group_folds.py` | 7 | Optional task 31: the outer split keeps every group on one side, with the test share counted in groups, fixed by the seed and with its own fingerprint; every Session_Fold keeps each group inside one fold, for classification (stratified) and regression, each row is a validation row once per repeat and the repeats differ; too few groups, empty cells, a wrong length and a single group are refused; a Session with a group column records it, never gives the column to the engine, makes the test part of whole groups and gives the Assessor grouped folds; a missing column or the target as the group column is refused before anything is written |
| `tests/test_explainer.py` | 5 | Optional task 32: tree SHAP values of a random forest add up to its predicted probabilities and rank the two signal columns above the noise column; logistic regression gets the permutation estimate, whose values also add up to the prediction; multiclass gives one output per class and regression one; the columns are the learner's inputs after the Data_Steps and FLAML's preparation (a real FLAML LightGBM with `Name__tok_mr`); a Session's final model is explained on Training_Part rows only, and other data (a different split fingerprint) or no final model is refused. `tests/test_dashboard.py` also clicks "Explain the final model" after a finished Session and picks one row |
| `tests/test_leakage.py` | 5 | Optional task 33: on Titanic the first suggested target is `Survived` with its reasons, identifier and name columns are never suggested, and a name containing "class" comes first; planted leaks (a numeric and a text copy of a binary target, a near-copy of a continuous target) score at least 0.99 while honest columns do not, and the profile computes the scores only when asked; the Critic warns about a used leak column (high, balanced accuracy, threshold 0.99 from the default) and not about an excluded one, no Proposal answers it, a file may set the threshold and an invalid one is refused; a Session warns about a planted leak on its Training_Part and never changes it. Also updated: the table-order and frozen-thresholds tests of `tests/test_critic.py` and `tests/test_critic_study.py` |
| `tests/test_narrator.py` | 5 | Optional task 34, with a stand-in for Ollama: the facts come from the Session record (dataset, target, metric, every step's verdict, the final test score) and the prompt holds no data; the request goes to the local `/api/generate` with temperature 0 and seed 0 and the default model; the text is labelled, a number that is not in the facts is listed and a rounded score is not; percentages match shares; only a server on this machine is used, and an empty answer or no server give clear errors; no decision module mentions the narrator or its file. `tests/test_dashboard.py` also writes a summary with a stand-in model and checks its label, its warning and its file |
| `tests/test_api.py` | 3 | Optional task 35, with FastAPI's test client and the stand-in engine and Assessor of `tests/test_autopilot.py`: a Session started through `POST /sessions` finishes, and `GET /sessions` and `GET /sessions/{id}` return its record as written on disk; files outside `data/`, a missing file, a budget below 10 s, an unknown selector, a wrong target and IDs that are not session IDs are refused, and nothing is written for them; a second Session is refused while one runs, and stop ends the running one with the stop reason "user" |
| `tests/test_benchmark_analysis.py` | 7 | On hand-made rows: only the last completed row of each cell is scored and error metrics are negated; blocks keep only (dataset, seed) pairs where every compared system completed; the three families with their systems, pairs, average ranks, critical difference, Wilcoxon and Holm values (4 datasets, the full system best everywhere: p = 2/16, Holm 5 x 2/16); time-to-target of the main systems only, with the pair counts; the median time-to-target counts a run that never reached the target as later than every budget; every table and chart is written; a smoke-sized analysis (1 dataset, no 300 s cells) does not fail |
| `tests/test_judge_agreement.py` | 7 | Since 31 October 2026: a pair that ends on battery is written as excluded, runs again and is left out of the agreement. Before: Cohen's kappa equal to scikit-learn's on three cases and to a hand-worked one (0.5), undefined with one label; the agreement summary (agreement, accepted-against-discarded agreement, kappa, the 3 x 3 table, cost ratios) on hand-made rows; the strict verdict uses the Judge's rules (keep, keep_simpler only when simpler, discard; an error metric with its sign flipped; 5 splits, ratio 0.25); pairs from a hand-made Session record (failed steps and the first step left out, the accepted Setup is the step's parent, the Judge's cost is both Assessments); the rebuilt training part, its fingerprint check and the 5 inner splits; one strict comparison end to end with 2-trial searches, run once, skipped on restart, and a wrong fingerprint recorded as a failed row (marked slow) |
| `tests/test_memory_build.py` | 4 | The build's 30 Sessions (10 development datasets x seeds 1 to 3 at 300 s) with the random order and a writable memory; with a stand-in Session: each Session runs once and a restart runs none again, each row shows the memory as its Session found it (0, 1, 2, 3 records), and an evaluation dataset is refused; the frozen copy keeps only the records of finished Sessions (not those of a failed or an interrupted one) and is never rewritten; the summary counts a record under every diagnosis kind it answered |
| `tests/test_stat_tests.py` | 13 | Against reference values: SciPy's documented Wilcoxon example (Darwin's plant data: statistic 24, exact p 0.041259765625, also found by counting all 2^15 sign patterns); Demšar's (2006) Table 5 critical values for 2 to 10 systems at alpha 0.05 and 0.10, and his worked example (chi2_F 9.28, F_F 3.69, CD 1.25 and 1.12); worked by hand: ranks with ties, a Friedman table (chi2 4.5, p = exp(-2.25)), Holm's adjusted p-values, rank-biserial correlations with tied sizes, the bootstrap interval of 5 values, time-to-target with a negated error metric and a run that never reached the target; equal differences share a rank; the smallest Wilcoxon p-value with 6 datasets (1/32); errors for missing scores |
| `tests/test_systems.py` | 7 | Every system gets the Phase 1 split and leaves out the first Setup's columns; the manual system is LightGBM with the library's defaults; FLAML alone searches the shared training rows; a failing system returns a failed result; the AutoGluon bridge with a stand-in worker on the same Python (binary metrics equal to our Evaluator's on the worker's predictions, the settings it is given, regression, a crash and a timeout); all systems report the same metrics on the same rows (marked slow) |
| `tests/test_phase2.py` | 3 | The Phase 2 measurement's runs (each Session before its FLAML run), the pairing of runs with the sign set so that positive favours the Session (also for an error metric), and the summary and Strategy counts on hand-made rows |
| `tests/test_synthetic.py` | 32 | Same seed gives the same data; problem order does not matter; each of the 8 problems measured at its requested strength in both the dataset and the fresh sample (imbalance ratio, share of changed labels, correlation of copies, distinct share of IDs, share of the main value, unseen share of categories, empty share); our profiler flags the planted ID, constant and high-cardinality columns; clean data has none of them; a model scores the same on the fresh sample as in cross-validation; 11 invalid requests rejected |

### 8.7 Code listings worth including in the report

- `FLAMLEngine.fit`: the call to FLAML and the metric mapping (`_METRICS`)
- The loss-to-score conversion inside `FLAMLEngine.fit`
- `Evaluator.compute_metrics` and `Evaluator.feature_importance`
- `TaskDetector.detect` and `analyze_class_balance`
- `unseen_category_warnings` and `_showwarning_without_search_noise`
- `ExperimentTracker.allocate_experiment_id` and `ExperimentTracker.log_run`

Phase 2 and 3 code worth including:

- `corrected_resampled_t` and `Judge.compare` (`app/services/judge.py`): the corrected t-test and the three verdicts
- `Assessor.assess` and `Assessor.refit_key` (`app/services/assessor.py`): refits on fixed folds, and when they can be reused
- `Critic.diagnose` and one rule, for example `_redundant_features` (`app/services/critic.py`)
- `propose` and `ThompsonSelector._score` (`app/services/strategy_engine.py`): the rule table and the learned score
- `ResourceManager.next_budget` and `should_stop` (`app/services/resource_manager.py`)
- `ExperimentMemory.strategy_stats` (`app/services/experiment_memory.py`): the arm statistics
- `friedman`, `nemenyi_cd` and `compare_pairs` (`research/stat_tests.py`)

### 8.8 Final check (task 28, 31 October 2026)

- Whole test suite: 541 passed in 214.6 s (code of commit d6cdf1c; only documents uncommitted).
- Smoke benchmark of the final code in a temporary folder, with the frozen memory (4baeb012...) and all 16 cores, from 23:11:56 to 23:24:37 on mains power (no wait for power, no cell ended on battery): all 8 systems completed on titanic with seed 1 and 120 s. The library versions in the rows are the same as in the smoke run of 27 October. 6 of the 8 test scores equal those of 27 October; FLAML alone scored 0.7692 (0.7500 before; 357 trials against 276, since a time-limited search depends on timing) and `no_resource_manager` 0.8000 (0.7500 before). The full system made 7 Experiments, kept the same changes as the rule order, and its last step (fewer model families, same model) reused the fold scores of an earlier step instead of refitting (Algorithm 9), so the reuse works in a real Session. Wall times 4.3 s (manual) to 122.1 s. Titanic is a development dataset: this is a check, not a result. Raw rows: `results/benchmark/smoke/final_check_cells.csv` (copied from the temporary folder `%TEMP%\final_smoke`, which also holds the Sessions and the log).
- Dashboard: the 5 dashboard tests ran with the suite (Streamlit's AppTest, including a whole autonomous Session through the page with 3-trial searches), and the app was started with `streamlit run app/main.py` and answered its health check (HTTP 200, "ok") and its page (HTTP 200). No Session was clicked through by hand in a browser.
- Found during the check: the smoke run's log held about 37,800 lines of one scikit-learn warning ("`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`", 11.8 MB), printed during the FLAML-alone cell and three Session cells, while the logs of the earlier runs (smoke, memory build, benchmark, judge agreement) have none. The cause was not found; the rows and scores look normal. Open item (Appendix D).

---

## 9. Result analysis

All numbers below are real measurements. FLAML's search is limited by time, so repeated runs can differ slightly.

### 9.1 Setup

Machine and software as in chapter 7. Outer split: stratified 80/20; seed 42 unless stated. Titanic runs exclude `PassengerId` and `Name` (setup A); setup B also excludes `Ticket` and `Cabin`.

### 9.2 All four datasets through the dashboard (10 s budget, seed 42, 7 October 2026)

| Dataset | Task | Train / test rows | Best model | Test score | FLAML validation | Trivial baseline on the same test rows | Other test metrics | Training time |
|---|---|---|---|---|---|---|---|---|
| Titanic | Binary | 712 / 179 | XGBoost | F1 0.6935 | 0.7868 (CV) | F1 0.0000, accuracy 0.6145 | accuracy 0.7877, ROC-AUC 0.7868 | 10.1 s |
| Breast cancer | Binary (positive: malignant) | 455 / 114 | LightGBM | F1 0.9512 | 0.9675 | F1 0.0000, accuracy 0.6316 | accuracy 0.9649, ROC-AUC 0.9904 | 10.1 s |
| Penguins | Multiclass | 275 / 69 | LightGBM | macro F1 0.9829 | 0.9881 | macro F1 0.2020, accuracy 0.4348 | accuracy 0.9855, macro ROC-AUC 0.9991 | 10.0 s |
| California housing | Regression | 16,512 / 4,128 | XGBoost | R² 0.8566 | 0.8648 (holdout) | R² -0.0002, RMSE 1.1449 | MAE 0.2791, RMSE 0.4335 | 16.9 s |

California's training time is above the 10 s budget because FLAML retrains the best configuration on all 16,512 training rows after the search. Baselines were recomputed on 12 October 2026 on the same seed-42 splits.

### 9.3 Titanic EXP-001 vs EXP-002 (60 s budget, seed 42, run on 7 October 2026)

| | EXP-001 (setup A) | EXP-002 (setup B) |
|---|---|---|
| Best model | XGBoost | XGBoost (depth-limited) |
| Test accuracy | 0.793 | 0.793 |
| Test balanced accuracy | 0.748 | 0.767 |
| Test precision | 0.864 | 0.776 |
| Test recall | 0.551 | 0.652 |
| Test F1 | 0.673 | 0.709 |
| Test ROC-AUC | 0.792 | 0.823 |
| Test log loss | 0.551 | 0.499 |
| Confusion matrix (TP, FP, FN, TN) | 38, 6, 31, 104 | 45, 13, 24, 97 |
| Train F1 / train ROC-AUC | 0.934 / 0.987 | 0.873 / 0.960 |
| Train F1 minus test F1 | 0.261 | 0.164 |
| Train F1 minus validation F1 | 0.132 | 0.090 |
| FLAML validation F1 (5-fold CV) | 0.802 | 0.782 |
| Trials in 60 s | 290 | 501 |
| Training time / total time | 60.1 s / 63.1 s | 60.0 s / 61.0 s |
| Top permutation importances | Sex 0.219, Pclass 0.098, Ticket 0.043, Fare 0.027, SibSp 0.021 | Sex 0.293, Pclass 0.179, Age 0.070, Fare 0.045, SibSp 0.019 |
| Notes | `Ticket`: 63% of test rows unseen in training; `Cabin`: 11% | none |

Leaderboards (best validation F1 per learner, trials in brackets):

- EXP-001: XGBoost 0.802 (114), XGBoost depth-limited 0.792 (21), LightGBM 0.785 (83), random forest 0.742 (36), extra trees 0.715 (7), SGD 0.531 (28), L1 logistic regression 0.454 (1).
- EXP-002: XGBoost depth-limited 0.782 (187), LightGBM 0.774 (155), XGBoost 0.771 (38), random forest 0.743 (42), extra trees 0.733 (21), SGD 0.647 (46), L1 logistic regression 0.465 (12).

How the two runs differ, row by row on the same 179 test passengers: both right 132, only EXP-001 right 10, only EXP-002 right 10, both wrong 27. Accuracy is therefore the same. EXP-002 finds 7 more survivors (TP 45 against 38) at the cost of 7 more false alarms (FP 13 against 6), which raises recall and F1 and lowers precision.

Paired bootstrap over the 179 test rows (12 October 2026; 5,000 resamples with replacement, NumPy seed 0, from the `predictions.csv` files of both runs):

- 95% intervals: EXP-001 F1 0.564 to 0.768; EXP-002 F1 0.612 to 0.791.
- F1 difference (EXP-002 minus EXP-001): +0.036, 95% interval -0.042 to +0.117. The difference was zero or negative in 19% of resamples.
- ROC-AUC difference: +0.031, 95% interval -0.014 to +0.080, zero or negative in 9% of resamples.
- Conclusion: on this single split the improvement is not statistically clear.

### 9.4 Same pair with a 30 s budget (seed 42, demo rehearsal, 7 October 2026)

| | Setup A | Setup B |
|---|---|---|
| Best model | XGBoost (depth-limited) | XGBoost (depth-limited) |
| Test F1 / accuracy / ROC-AUC | 0.695 / 0.799 / 0.807 | 0.709 / 0.793 / 0.823 |
| Train F1 / train ROC-AUC | 0.994 / 1.000 | 0.873 / 0.960 |
| FLAML validation F1 | 0.796 | 0.782 |
| Trials | 188 | 237 |
| Total time | 37.0 s | 31.8 s |

### 9.5 Effect of the time budget (seed 42; single split, so indicative only)

| Budget | A validation | A test F1 | A train F1 | A trials | B validation | B test F1 | B train F1 | B trials |
|---|---|---|---|---|---|---|---|---|
| 10 s | 0.785 | 0.694 | 0.910 | 77 | 0.774 | 0.698 | 0.851 | 80 |
| 30 s | 0.796 | 0.695 | 0.994 | 188 | 0.782 | 0.709 | 0.873 | 237 |
| 60 s | 0.802 | 0.673 | 0.934 | 290 | 0.782 | 0.709 | 0.873 | 501 |

Sources: 10 s rows are development runs (7 October 2026, temporary folders); 30 s from 9.4; 60 s from 9.3. The 10 s Titanic run in 9.2 is a separate run of setup A with the same seed: validation 0.787 and test F1 0.694, so repeated runs moved the validation score by about 0.002 here.

- Setup B reached its final model by 30 s: the 30 s and 60 s runs gave the same test result. More budget only added trials.
- Setup A's validation F1 rose with the budget while its test F1 did not. That is what selection bias would look like (the best of more trials looks better than it is; Cawley and Talbot, 2010), but one split cannot prove it.

### 9.6 Paired comparison over five splits (30 s budget, seeds 1 to 5, 7 October 2026)

Difference B minus A on each split:

| Seed | Test F1 | Test ROC-AUC | Test accuracy | Train-test F1 gap | Validation F1 | Trials |
|---|---|---|---|---|---|---|
| 1 | +0.020 | +0.004 | +0.006 | +0.050 | +0.006 | +44 |
| 2 | +0.030 | +0.030 | +0.017 | +0.090 | +0.002 | -12 |
| 3 | -0.003 | -0.007 | -0.006 | +0.016 | +0.006 | +44 |
| 4 | -0.006 | -0.016 | -0.006 | -0.066 | +0.004 | +15 |
| 5 | +0.020 | +0.021 | +0.006 | -0.036 | +0.006 | +86 |
| Mean | +0.012 | +0.006 | +0.003 | +0.011 | +0.005 | +35.4 |

Means per setup:

| Setup | Test F1 | Test ROC-AUC | Test accuracy | Train F1 | Validation F1 | Train-test gap | Trials |
|---|---|---|---|---|---|---|---|
| A | 0.772 | 0.870 | 0.836 | 0.852 | 0.766 | 0.080 | 186.4 |
| B | 0.784 | 0.877 | 0.839 | 0.875 | 0.771 | 0.091 | 221.8 |

- B was better on test F1 on 3 of 5 splits and ran more trials on 4 of 5.
- Validation F1 preferred B on all 5 splits.
- The train-test gap did not shrink on average, unlike on seed 42.
- Paired tests on the five differences (computed 12 October 2026 with SciPy, from the differences rounded to 3 decimals): test F1 paired t-test p = 0.16, Wilcoxon p = 0.25; test ROC-AUC p = 0.50 and 0.63. Not significant. With only five pairs, an exact two-sided Wilcoxon test cannot go below p = 0.0625, so the benchmark needs more splits or more datasets.

A shorter check (10 s, seeds 1 to 3, the version used in the demo, 7 October 2026): test F1 A 0.750 / 0.794 / 0.800, B 0.752 / 0.762 / 0.806; mean difference -0.008; B ran more trials on 3 of 3 (+15 to +16).

Repeatability: running seed 1 at 30 s a second time gave the same test F1 for both setups (A 0.750, B 0.770) with slightly different trial counts (A 213 against 210, B 245 against 254). On seed 2, setup A's test F1 was 0.794 at 10 s but about 0.73 at 30 s, so the budget can change the chosen model, and its test score, on the same split.

### 9.7 Do shared tickets explain the validation and test disagreement? (12 October 2026)

At the Phase 1 review we suggested that passengers sharing a ticket made cross-validation look better than the test set. Measured on the same splits:

- 681 different tickets for 891 passengers; 38.6% of passengers share a ticket with someone.
- Test rows whose ticket also appears in the training part: 37.4% on seed 42; 30% to 37% on the six splits (seeds 42 and 1 to 5), 33.1% on average.
- Validation rows whose ticket appears in the other folds (5-fold stratified cross-validation on the training part, 5 shuffles): 29.1% on seed 42; 29% to 32% on the six splits, 30.4% on average.

On seed 42, test rows shared a ticket with a training row more often than validation rows did with their training folds. Shared tickets would therefore help the test score at least as much as the validation score, so they cannot explain why validation preferred setup A on that split. Method, to repeat it for the report: the same outer splits as the runner (stratified 80/20, `random_state` = seed); for validation, `StratifiedKFold(5, shuffle=True)` on the training part with shuffle seeds 0 to 4; count the rows whose `Ticket` value appears in the other part. What the data does show:

- Split luck: on seed 42 setup A's test F1 (0.67 to 0.69 over three budgets) was lower than on every other split (0.73 to 0.81 on seeds 1 to 5). Setup B was also lowest on seed 42 (0.70 to 0.71, against 0.75 to 0.81). At the same time setup A's validation F1 on seed 42 was higher than on the other seeds (0.796 at 30 s, against 0.763 to 0.768 on seeds 1 to 5). That split put harder passengers into the test part.
- Across seeds 1 to 5 the mean validation F1 (A 0.766) was close to the mean test F1 (A 0.772), and validation picked the same winner as the average test result.
- Selection bias may add to this (section 9.5), but these runs are too few to measure it.

### 9.8 Running times

- 30 s budget: Titanic 32 to 37 s total, Penguins 32 s, California housing 40 s. The time above the budget goes into retraining the best model, evaluation, permutation importance and logging.
- 60 s budget on Titanic: 61 to 63 s total.
- One 30 s experiment started from Python, including imports: 38 s.
- Test suite: about 29 s (7 to 12 October), about 41 s (three runs on 22 October, no code change in between).

### 9.9 What these results show, and what they do not

- The baseline beats the trivial baselines by a wide margin on every dataset (for example, Titanic F1 0.69 against 0; California R² 0.86 against 0.00).
- Dropping the two high-cardinality columns on Titanic is a small, cheap change: +0.012 test F1 on average over five splits, better on 3 of 5. In most runs it lets FLAML run more trials in the same time (4 of 5 splits at 30 s, +35 on average; 3 of 3 at 10 s; 290 to 501 in 60 s on seed 42).
- The choice of split matters more than the change: for setup A the test F1 ranged from 0.67 to 0.81 over the six splits and budgets tried, and from 0.70 to 0.81 at a fixed 30 s budget. The 95% interval of the single-split difference (-0.04 to +0.12) includes zero, and the five-split paired tests are not significant.
- So changes have to be compared over several splits, on validation data, with their uncertainty recorded. This is the basis of P6 (section 5.3) and of the memory design.
- Not shown yet: anything about the autonomous layer, which does not exist yet.

### 9.10 How FLAML spent the budget (EXP-001 and EXP-002, from the saved trial logs, analysed 22 October 2026)

Source: `experiments/EXP-00x/search_history.csv` and `flaml_search.log`. Charts of EXP-001: `docs/imp/images/exp001_search_timeline.png`, `exp001_time_per_learner.png`, `exp001_best_learner_settings.png` (drawn with `scripts/plot_search_history.py`).

| | EXP-001 (setup A) | EXP-002 (setup B) |
|---|---|---|
| Trials in 60 s | 290 | 501 |
| Time inside trials | 59.7 s | 59.5 s |
| New overall bests | 12 | 9 |
| Last new best | 42.9 s (XGBoost, trial 240) | 23.2 s (XGBoost depth-limited, trial 188) |
| Time after the last new best | 17.1 s | 36.8 s |
| Trials with validation F1 0 | 6 (SGD 4, LightGBM 1, XGBoost 1) | 5 (SGD 3, LightGBM 1, XGBoost 1) |
| Share of trials below 0.70 | 19.3% | 19.6% |

EXP-001 search time per learner (trials, best validation F1): XGBoost 34.6 s (114, 0.802), random forest 9.6 s (36, 0.742), LightGBM 9.1 s (83, 0.785), XGBoost depth-limited 2.5 s (21, 0.792), SGD 2.2 s (28, 0.531), extra trees 1.7 s (7, 0.715), L1 logistic regression 0.1 s (1, 0.454). The winner got 58% of the time.

EXP-001 best validation F1 over time: 0.631 after the first trial (0.1 s), 0.769 after 8 trials (0.8 s), 0.785 at 9.6 s, 0.802 at 42.9 s. About 81% of the total rise came in the first 0.8 s and 90% by 9.6 s.

XGBoost in EXP-001 started at 4 trees with at most 4 leaves (validation F1 0.631) and beat its own best 10 times, ending at 304 trees and 11 leaves (0.802). It never had more than 34 failed tries in a row.

In EXP-002, XGBoost depth-limited got its first trial at 20.0 s (trial 159) and produced the run's best 3.2 s later.

What this shows: on small data FLAML finds most of its gain in the first seconds and then keeps searching until the budget runs out, because early stop cannot trigger (section 5.2). This is evidence for the planned resource manager. It is a single run per setup, so the exact shares are indicative.

### 9.11 First autonomous Session on Titanic (SES-001, thin loop, 26 October 2026)

Settings: `python scripts/run_autopilot.py --dataset data/sample/titanic.csv --target Survived --seconds 300 --max-experiments 8 --cores -1 --seed 42` (primary metric F1, detected; thresholds `v0-start`; Judge alpha 0.10, margin 0.005, minimum gain 0; rule selector). Code at commit 489c048 with no uncommitted changes, FLAML 2.7.0, laptop of chapter 7. Raw files: `experiments/SES-001/` (`session.json`, `train_profile.json`) and `experiments/EXP-003` to `EXP-006` (each with `summary.json`, `assessment.json`, `decision.json` or `diagnoses.json`), models in `models/EXP-003` to `EXP-006`, MLflow parent run `SES-001` with the four Experiments as child runs.

Splits: the same outer split as Phase 1 (seed 42, stratified, 712 training and 179 test rows); 3 x 5 stratified Session_Folds of the training part with seed 43. The first Setup left out `PassengerId` and `Name` (identifier-like on the training part), which is Phase 1's setup A.

| Step | Experiment | Change | Search budget / wall time | Learner FLAML chose | Refit validation F1 (mean, sd over 15 folds) | Judge: mean difference, wins/losses/ties, corrected t, p(better) | Verdict |
|---|---|---|---|---|---|---|---|
| 0 | EXP-003 | first Setup | 105 s / 106.1 s | XGBoost depth-limited (368 trials) | 0.7056 (0.0443) | | first |
| 1 | EXP-004 | drop `Cabin` | 25 s / 25.7 s | XGBoost (64 trials) | 0.7143 (0.0479) | +0.0087, 9/5/1, 0.584, 0.284 | keep_simpler |
| 2 | EXP-005 | drop `Ticket` | 23 s / 23.5 s | LightGBM (69 trials) | 0.7649 (0.0446) | +0.0506, 14/1/0, 2.616, 0.010 | keep |
| 3 | EXP-006 | smaller trees | 23 s / 23.5 s | LightGBM (95 trials) | 0.7628 (0.0398) | -0.0022, 7/7/1, -0.157, 0.561 | keep_simpler |

Diagnoses: for the first Setup, overfitting high (training F1 0.996 against validation F1 0.706 on the folds, gap 0.291), high cardinality high for `Ticket` (571 values; 72% of validation rows had a ticket not seen in the fold's training rows) and moderate for `Cabin` (127 values; 16% unseen); no class imbalance (ratio 1.6). After dropping `Cabin` the overfitting gap was 0.207, after dropping `Ticket` 0.127 (moderate), and for the final Setup 0.134 (moderate). With both text columns dropped and the smaller trees already tried, no Proposal was left, so the Session stopped (`no_proposals`) after 206.0 s of its 300 s budget.

Costs: the Assessments took 4.1 to 6.4 s each (15 refits); the overruns past each step's deadline were 0.95 s (step 1), 0.14 s (step 3) and 0 for the others; the Final_Evaluation took 1.8 s.

Final_Evaluation on the 179 test rows (used once, at the end):

| Model | F1 | Accuracy | Balanced accuracy | ROC-AUC | Log loss |
|---|---|---|---|---|---|
| First (EXP-003) | 0.6838 | 0.7933 | 0.7535 | 0.7808 | 0.6411 |
| Final (EXP-006) | 0.7244 | 0.8045 | 0.7788 | 0.8123 | 0.5261 |

The trivial baseline (always "did not survive") scores F1 0 on the same rows. Permutation importance of the final model (mean drop in F1 on the test rows): `Sex` 0.276, `Pclass` 0.119, `Age` 0.070, `Fare` 0.061, `SibSp` 0.023, `Embarked` 0.016, `Parch` -0.009.

What this shows, and what it does not:

- The loop ran end to end without help and every decision can be checked from its numbers. It reached, by itself, the change that the Phase 1 analysis pointed to (dropping `Ticket`; sections 9.3 and 9.7), and only that step passed the corrected t-test (p = 0.010). The other two kept steps were kept as simpler Setups that were not shown to be worse.
- On the test rows the final model scored F1 0.724 against 0.684 for the first model. For reference, on the same test rows Phase 1's EXP-001 (setup A, 60 s) scored 0.673 and EXP-002 (setup B, 60 s) 0.709. This is one Session on one split, so it is not evidence that the loop beats a plain search; that needs the Phase 3 benchmark.
- FLAML's own validation scores (0.804, 0.793, 0.775, 0.777 in the history) went down while the refit scores on the Session_Folds went up (0.706 to 0.763). FLAML's scores come from its own folds and from the best of hundreds of trials, so they are optimistic and not comparable across Setups; this is why the Judge uses the refits (decision D2).
- Two weaknesses showed up; both are in section 4.7.

### 9.12 Critic study: how often the Critic is right (task 16, 26 October 2026)

Setup: 270 synthetic binary datasets of 1,000 rows, each with a fresh sample of 20,000 rows from the same generator; one planted problem per dataset at three strengths (class imbalance 3, 5, 10; label noise 0.1, 0.2, 0.3; useless columns 5, 20, 50; identifier column 0.95, 0.99, 1.0 of values distinct; constant column 0.98, 0.995, 1.0 main-value share; high-cardinality column 0.2, 0.5, 2 categories per row; missing values 0.3, 0.5, 0.8 in two columns; duplicated columns with correlation 0.95, 0.98, 0.995) plus 3 clean datasets, for each of 10 seeds: 5 tuning seeds (0 to 4) and 5 reporting seeds (100 to 104). Per dataset: one FLAML search (F1, at most 25 trials), an Assessment on 3 x 5 folds and the Critic. Truth: the planted problem for data-level kinds; for overfitting, training F1 minus fresh F1 of at least 0.10; for underfitting, that gap below 0.03 and a fresh skill over the trivial baseline below 0.50; for unreliable validation, a fold mean at least 0.05 from the fresh F1. Code at commit 0ddc314 plus this task's study code (committed afterwards as 5ef7ca7), laptop of chapter 7, all cores; 43.3 min in total (9.6 s per dataset on average). FLAML chose LightGBM for 216 datasets, SGD 25, XGBoost 20, random forest 7, extra trees 2. Raw files: `results/studies/critic_study_cases.csv` (one row per dataset), `critic_study_grid.csv`, `critic_study_summary.csv`, and the saved Critic inputs in `experiments/studies/critic_study_inputs.jsonl`.

Results on the reporting seeds (135 datasets the tuning never saw), starting thresholds `v0-start` against the tuned ones:

| Diagnosis | True cases | Precision, start | Recall, start | Precision, tuned | Recall, tuned |
|---|---|---|---|---|---|
| class_imbalance | 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| constant_column | 15 | 1.000 | 0.667 | 1.000 | 1.000 |
| high_cardinality | 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| identifier_column | 15 | 1.000 | 0.667 | 1.000 | 0.667 |
| missing_values | 15 | 1.000 | 0.667 | 1.000 | 1.000 |
| overfitting | 32 | 0.711 (13 false alarms) | 1.000 | 0.879 (4 false alarms) | 0.906 (3 missed) |
| redundant_features | 15 | 1.000 | 0.933 | 1.000 | 0.933 |
| underfitting | 5 | 1.000 | 0.600 | 1.000 | 0.600 |
| unreliable_validation | 4 | 0.273 (8 false alarms) | 0.750 | 0.273 | 0.750 |

Tuning changed three thresholds: overfitting gap 0.10 to 0.11, missing share 0.40 to 0.30, and near-constant share 0.99 to 0.98. They are frozen as version `v1-critic-study` in `app/services/critic_thresholds.toml`. Replaying the real Critic with the tuned thresholds on all 270 saved inputs gave exactly the study's answers (0 disagreements).

What this shows, and what it does not:

- The data-level Diagnoses are reliable on these datasets: no false alarm for any of them. Two of the three tuned changes (missing share 0.30, near-constant share 0.98) only move the threshold down to the weakest planted strength, so they follow from the study's own choice of what counts as a problem rather than from new evidence; the evidence is that nothing else in the data comes near those levels.
- The identifier rule (from the Phase 1 profiler, not tuned) finds IDs that are all distinct or 99% distinct, but not 95% distinct: the name rule asks for more than 95% distinct values, and 95% is exactly the boundary.
- Overfitting is the hardest kind with real noise; the tuned gap of 0.11 cut the false alarms from 13 to 4 at the cost of 3 missed cases.
- Unreliable validation is weak: the fold spread did not predict when the cross-validated score was off by 0.05 (3 of 11 alarms were right), and the tuning seeds had no true case at all, so its threshold could not be tuned. Its Proposal (the smoother search metric) is cheap and the Judge still decides, so it stays, with this result reported.
- Underfitting and unreliable validation had only 5 and 4 true cases in the reporting seeds, so their rates are rough.
- These are synthetic binary datasets of one size; the Phase 3 benchmark on real datasets is the test that matters for the report's claims.

### 9.13 Judge study: false keeps and power (task 17, 26 October 2026)

Setup: 84 pairs of Setups on synthetic binary datasets of 1,000 rows, each with a fresh sample of 20,000 rows from the same generator. Setup A is the plain Setup; Setup B changes one thing. Seven pair kinds per seed: drop one informative column (`x0`); drop two (`x0`, `x1`); drop a random high-cardinality column (`code`, 2 categories per row); balanced class weights on imbalanced classes (ratio 5); balanced class weights on balanced classes; drop 20 random columns; smaller trees with 20% label noise. Six tuning seeds (0 to 5, 42 pairs) and six reporting seeds (100 to 105, 42 pairs). Each Setup: one FLAML search on all 1,000 rows (default learners, F1, at most 25 trials, 60 s cap, all cores) and an Assessment of its best configuration on 5 x 5 folds (seed + 1); the 1- and 3-repeat Judges use the first 5 and 15 of these folds, which are exactly the folds a Session with 1 or 3 repeats makes (checked: identical). True difference: B's F1 on the fresh sample minus A's, each from the model FLAML trained on all rows. Judge: margin 0.005, minimum gain 0, alpha 0.05, 0.10 or 0.20, and 1, 3 or 5 repeats. A pair is harmful if its true difference is below -0.005 (the margin) and helpful if it is above 0.01; a false keep is `keep` or `keep_simpler` on a harmful pair, and the power is the share of helpful pairs with `keep` (Requirement 10.2). Code at commit 5ef7ca7 plus one fix of this task (the study did not create its search folders; found by a trial run before the real one), laptop of chapter 7; 32.4 min in total (23.1 s per pair on average, 17.8 to 41.0 s), no failed pair. Raw files: `results/studies/judge_study_pairs.csv` (756 rows: one per pair and setting) and `judge_study_summary.csv`.

The rule for the defaults was fixed before the run: on the tuning seeds, the highest power among the settings with a false-keep rate of at most 0.10; ties to fewer repeats, then to the smaller alpha.

| Repeats | Alpha | False keeps, tuning (of 22) | Power, tuning (of 11) | False keeps, reporting (of 18) | Power, reporting (of 12) |
|---|---|---|---|---|---|
| 1 | 0.05 | 3 (0.136) | 4 (0.364) | 0 | 6 (0.500) |
| 1 | 0.10 | 3 (0.136) | 7 (0.636) | 0 | 7 (0.583) |
| 1 | 0.20 | 3 (0.136) | 9 (0.818) | 0 | 9 (0.750) |
| 3 | 0.05 | 2 (0.091) | 7 (0.636) | 0 | 7 (0.583) |
| 3 | 0.10 | 2 (0.091) | 8 (0.727) | 0 | 8 (0.667) |
| 3 | 0.20 | 2 (0.091) | 9 (0.818) | 0 | 10 (0.833) |
| 5 | 0.05 | 2 (0.091) | 6 (0.545) | 0 | 5 (0.417) |
| 5 | 0.10 | 2 (0.091) | 9 (0.818) | 0 | 8 (0.667) |
| 5 | 0.20 | 2 (0.091) | 10 (0.909) | 0 | 10 (0.833) |

Chosen by the rule: 5 repeats and alpha 0.20 (false-keep rate 0.091 and power 0.909 on the tuning seeds; 0 of 18 false keeps and power 0.833 on the reporting seeds). Adopted: alpha 0.20 (`JudgeSettings.alpha`, was 0.10). Not adopted: 5 repeats. The default stays 3 repeats (`SessionSettings.n_repeats`), a deliberate departure from the rule, decided on 26 October 2026 for the reasons in the third point below. For a short time (commit bad02ec) the default was 5; no recorded Session ran with it.

What this shows, and what it does not:

- Alpha 0.20 had the highest power at every number of repeats on both seed sets, and alpha never changed the false keeps: all 21 false keeps (over all settings) were `keep_simpler`, and a larger alpha makes `keep` easier but `keep_simpler` harder.
- One repeat had 3 false keeps of 22 on the tuning seeds, above the 0.10 limit. Three and five repeats had 2 each (not the same pairs).
- The choice of 5 over 3 repeats rests on one tuning pair (seed 3, drop 20 random columns), where 3 repeats said `keep_simpler` and 5 said `keep`. Both accept the change, because dropping columns is simpler. Counted as accepted (`keep` or `keep_simpler`), 3 and 5 repeats at alpha 0.20 accept the same helpful pairs on both seed sets (10 of 11 and 12 of 12), and on the reporting seeds their rates are identical. The corrected t-test explains why more repeats add little: the 1/4 term does not shrink, so 25 folds give a standard error only about 4% smaller than 15. The cost is certain: every Assessment refits 25 instead of 15 times. So the rule had a flaw: it counted only `keep` as power, although `keep_simpler` also accepts a change, and it used the cost only to break exact ties. With 3 repeats and alpha 0.20 on the reporting seeds: 0 of 18 false keeps and power 10 of 12 (0.833), the same as the rule's choice.
- The true differences come mostly from the searches, not from the changes: two searches on nearly the same data often ended with different learners (in 27 of 84 pairs) or settings. True differences ranged from -0.200 to +0.439 (median -0.002); the largest came from a pair where A's search ended with the SGD linear model (fresh F1 0.402) and B's with LightGBM (0.840). So dropping an informative column was helpful in 4 of 12 pairs, and 11 pairs had a true difference of exactly 0 (both searches found the same model). The study therefore measures what a Session faces (is the candidate's search result better than the accepted one?), but it does not separate the effect of a change from the luck of a search.
- FLAML's learner choice depends on measured trial times, so a rerun gives somewhat different pairs: in a trial run before the study, the same seed-0 pairs "drop 20 random columns" and "smaller trees" gave true differences of -0.0072 and -0.0013 against +0.0193 and +0.0075 in the study.
- The counts are small (22 harmful and 11 helpful pairs on the tuning seeds), so one pair moves a rate by 0.05 to 0.09, and all data is synthetic, binary and of one size.

### 9.14 Phase 2 measurement: the loop against FLAML alone (task 19, 27 October 2026)

Setup: the four sample datasets (Titanic, breast cancer, penguins, California housing), seeds 1, 2 and 3, and 300 s of total time per run (the middle budget of the planned Phase 3 grid). For each dataset and seed, two runs on the same training and test parts (test share 0.2, the Phase 1 split): a Session with the default settings (rules, 3 x 5 folds, Judge alpha 0.20, thresholds `v1-critic-study`, at most 8 Experiments, first share 35%, patience 3), and one FLAML search with the whole 300 s (the Phase 1 runner) that leaves out the same columns as the Session's first Setup (`PassengerId` and `Name` on Titanic, none elsewhere). Default primary metrics: F1 (binary), macro F1 (penguins) and R2. All cores, MLflow on. Code at commit 940225a; only documents were edited while it ran, so the rows say `git_dirty`. 24 runs on 27 October 2026 from 00:06 to 01:48 (1 h 42 min), laptop of chapter 7. Raw files: `results/phase2/phase2_runs.csv` (one row per run), `phase2_pairs.csv`, `phase2_summary.csv`, `phase2_strategies.csv`; the Sessions are SES-002 to SES-013 with their Experiments in `experiments/`, and the 34 memory records they wrote.

Test score of the final model, Session minus FLAML alone (positive: the Session did better):

| Dataset (test rows) | Seed 1 | Seed 2 | Seed 3 | Mean | Session time (s), mean | FLAML time (s), mean | Experiments per Session |
|---|---|---|---|---|---|---|---|
| Titanic (179) | 0.0000 | +0.0336 | +0.0288 | +0.0208 | 197 | 304 | 4, 4, 4 |
| Breast cancer (114) | +0.0129 | -0.0256 | -0.0347 | -0.0158 | 293 | 307 | 8, 8, 7 |
| Penguins (69) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 132 | 303 | 2, 2, 1 |
| California housing (4,128) | +0.0001 | -0.0039 | -0.0069 | -0.0035 | 201 | 312 | 2, 2, 2 |
| All 12 pairs | 4 better | 4 equal | 4 worse | +0.0004 | 206 | 306 | 3.8 |

Session test scores: Titanic F1 0.750, 0.794, 0.823 (FLAML 0.750, 0.760, 0.794); breast cancer 0.951, 0.927, 0.953 (0.938, 0.952, 0.988); penguins 0.983, 0.983, 1.000 (the same); California R2 0.852, 0.849, 0.852 (0.852, 0.853, 0.858).

Inside the Sessions:

- Stop reasons: no Proposal left 7 times, time 4 times, three discards in a row once (Titanic seed 1). So Sessions often ended early: on average after 206 s of the 300 s, while FLAML alone always used its whole budget.
- 34 changes were judged: `drop_columns` 25 times (19 on breast cancer, 6 on Titanic; kept 21 times, 19 of them as simpler), `restrict_model_families` 5 (kept 3, as simpler), `regularised_search_space` 3 (kept once, California seed 1, as better), `frequency_encode` once (Titanic, discarded). No candidate failed.
- Titanic: the Critic found `Cabin` (77% empty, high-cardinality) and `Ticket` (high-cardinality) in all three seeds. In seeds 2 and 3 the Session dropped both, and the final model beat FLAML alone by 0.034 and 0.029 F1. In seed 1 the Judge discarded all three changes (dropping `Cabin`: mean fold difference -0.054; dropping `Ticket`: -0.010; frequency-encoding `Ticket`: +0.005, not significant), and the Session stopped at 196 s after three discards in a row, with the same test F1 as FLAML alone.
- Search budgets: every first search got 105 s; the 34 candidate searches got 5 to 58 s (median 21 s).
- Breast cancer: the Critic found redundant columns (radius, perimeter and area are almost the same measurement), and the Session dropped five or six of them as simpler and not worse. Its validation score rose (by 0.007, 0.025 and 0.005), but on the 114 test rows the final model was behind FLAML alone in two seeds. A difference of 0.03 in F1 there is two or three test rows, and the first model of seed 2 (0.952) was better on the test part than the final one (0.927), against the validation evidence.
- California housing: each Assessment refits 15 models on 13,200 rows; the first one took 19 to 52 s, and Assessments used 19 to 41% of the Session's time. Only 2 Experiments fitted in 300 s, and the loop was slightly behind FLAML alone (mean R2 -0.003).
- Penguins is easy (test F1 0.98 to 1.00): both reached the same test score, and the Sessions stopped after 113 to 145 s with nothing left to try.
- First against final model of the same Session (the loop's own effect on the test part): mean +0.0065 (Titanic +0.024, breast cancer +0.000, California +0.002, penguins 0).

What this shows, and what it does not:

- At 300 s on these four small datasets, the rule-based loop was as good as FLAML alone on average (4 better, 4 equal, 4 worse; mean +0.0004), and it used two thirds of the time. It helped where the Critic found a real data problem that FLAML cannot fix by itself (Titanic's high-cardinality columns), and it cost a little where its changes were neutral (breast cancer) or where Assessments are expensive (California).
- "Simpler and not worse" accepted many small changes in a row on breast cancer. Each passed the Judge alone; whether their sum helps on new rows is not tested by the Judge. This is a known property of the keep_simpler rule to check in Phase 3.
- 3 seeds and test parts of 69 to 179 rows for three of the datasets give no statistical power: these are observations, not tested differences. The Phase 3 benchmark (more datasets, budgets of 2, 5 and 10 minutes, at least 5 seeds, the statistics of section 5) is the test for the report's claims.

### 9.15 The frozen benchmark memory (task 26, 27 October 2026)

Setup (section 4.10): 30 Sessions, one per development dataset (10) and seed (1, 2, 3), 300 s each, at most 8 Experiments, changes tried in a random order without reading the memory, all other Session settings at their defaults (3 x 5 folds, Judge alpha 0.20, thresholds `v1-critic-study`), all 16 cores, MLflow off, starting from an empty memory. Code at commit 94714d2 with no uncommitted changes; laptop of chapter 7; 27 October 2026 from 17:10 to 18:54 (1 h 44 min). No Session failed.

Results:

- The Sessions ran 127 Experiments (4.2 per Session on average) and took 113 to 314 s each (mean 206.9 s). Stop reasons: no change left to try 20, three discards in a row 4, time 4, the Experiment limit 2.
- They wrote 97 memory records, all kept by `freeze` (no Session was unfinished): 44 discard, 42 keep_simpler, 11 keep. Per dataset: breast_cancer 21, vehicle 18, balance-scale 15, titanic 13, wilt 13, kin8nm 7, california_housing 3, student_performance_por 3, penguins 2, tic-tac-toe 2. Per Strategy: drop_columns 36, restrict_model_families 24, regularised_search_space 13, smoother_search_metric 7, class_weight_balanced 6, frequency_encode 5, more_search_time 4, group_rare_categories 2. A record's cost (the step's wall time) was 26.3 s on average (median 27.7 s).
- Arms (Strategy and diagnosis kind, per task type, counted as the selectors count them): 20, of which 13 have the 3 or more records a learned selector needs before it moves an arm. The largest are drop_columns for redundant features on binary data (22 records, 14 kept), restrict_model_families for weak model families on multiclass data (13 records, all kept) and on binary data (8 records, 7 kept), and regularised_search_space for overfitting on regression data (8 records, 5 kept).
- Frozen memory: `results/benchmark/frozen_memory.jsonl` (97 records, 100,403 bytes), SHA-256 `4baeb012a2037904c0a4a5303d016c3c59084326562a33670061994e70361316`; summary per arm `frozen_memory_summary.csv`; one row per Session `memory_sessions.csv`. The Sessions' records are in `experiments/benchmark-memory/` (14.3 MB, with the growing memory and `build.log`) and their models in `models/benchmark-memory/` (97.3 MB).

What this shows, and what it does not:

- Records are uneven across datasets: the datasets with many redundant or weak model families got many records, while penguins and tic-tac-toe stopped after one or two changes because nothing else applied. A learned selector will mostly reorder drop_columns, restrict_model_families and regularised_search_space; for most other arms it keeps the rule order.
- The memory holds few datasets per task type (4 binary, 3 multiclass, 3 regression), so "the 5 nearest datasets" means all development datasets of the same task.

### 9.16 Benchmark, stopped early: 23 usable cells (task 26, 28 to 31 October 2026)

What happened: the 390 cells started on 28 October 2026 at 14:02 (code ec2be03, no uncommitted changes, frozen memory 4baeb012..., all 16 cores). 33 cells finished. At 14:57:27 the laptop switched to battery and went into Modern Standby again and again (Windows System log: Kernel-Power 105, then 506 and 507 "Idle Timeout" and "Austerity Battery Drain Budget Exceeded"); the run's process was suspended for hours at a time and ended with an unclean reboot on 31 October at 20:02 (Kernel-Power 41). Time-limited searches count wall time, so the cells that ran into this period got far less computing: for example `no_critic` on cylinder-bands took 11,868 s for a 300 s budget, and FLAML on segment ran 85 trials in 579 s against 188 trials in 120 s before. The 10 cells that finished after 14:57:27 are listed in `results/benchmark/excluded_cells.csv` with this reason and are not scored; the raw rows are kept unchanged. On 31 October 2026 the user decided not to run the rest of the grid ("no need for that big a test"); the analysis below uses the 23 usable cells.

Usable cells: seed 1 only; pc4 (binary, 1,458 rows, 292 test rows) with all 13 systems and budgets, and cylinder-bands (binary, 540 rows, 108 test rows) without the three ablations at 300 s. Test F1 (the first model's test F1 in brackets for the Sessions; E = Experiments, stop reason):

| System | pc4, 120 s | pc4, 300 s | cylinder-bands, 120 s | cylinder-bands, 300 s |
|---|---|---|---|---|
| manual (LightGBM defaults) | 0.6250 | 0.6250 | 0.8593 | 0.8593 |
| flaml | 0.5902 | 0.5902 | 0.8788 | 0.8702 |
| autogluon | 0.5246 (31 s) | 0.5246 (31 s) | 0.8438 (74 s) | 0.8438 (68 s) |
| autopilot_rules | 0.5424 (0.5902; 6 E, patience) | 0.5902 (0.5902; 5 E, patience) | 0.8788 (0.8788; 4 E, no changes left) | 0.8722 (0.8788; 4 E, no changes left) |
| autopilot_full | 0.6000 (0.5902; 7 E, time) | 0.6129 (0.5902; 5 E, patience) | 0.8788 (same steps as rules) | 0.8722 (same steps as rules) |
| random_choice | | 0.5938 (0.5538; 6 E) | | excluded |
| no_critic | | 0.6364 (0.5902; 6 E) | | excluded |
| no_resource_manager | | 0.6154 (0.5902; 7 E) | | excluded |

What these cells show (descriptive only; one seed and two datasets):

- On pc4 the full system ended above its first model and above FLAML alone at both budgets; the rule-order Session ended below its first model at 120 s, after keeping two column drops as "simpler". On cylinder-bands the full system made the same choices as the rule order (its arms there had fewer than 3 records), and both ended slightly below their first model at 300 s.
- The best pc4 score came from `no_critic` (0.6364), the only Session that tried class weights (by its random order); that was the only change on pc4 with a significant gain (`keep`, validation +0.0667). The rule order never reached class weights: pc4 has many high-severity "drop one of a redundant pair" Proposals ahead of the moderate class-imbalance one, and three discards in a row stopped the Session. The full system did not try them either: its 3 records of that arm on binary data, all from wilt, were discards, and an arm with no keeps has an expected gain of 0.
- The library defaults of LightGBM beat FLAML's search on pc4 (0.6250 against 0.5902), as they did on titanic in the smoke run (0.7941 against 0.7500); FLAML was better on cylinder-bands (0.8788 against 0.8593).
- AutoGluon (`medium_quality`) stopped after 31 to 74 s, far below its time limit, and had the lowest F1 on both datasets.
- The Sessions on cylinder-bands stopped because no change was left, after 113 s of 120 and 211 s of 300.
- No statistical test can separate systems here: with 2 datasets the smallest two-sided Wilcoxon p-value is 0.5. The analysis files (`results/benchmark/analysis_*.csv`, `figures/`) were written from these 23 cells on 31 October 2026 and are descriptive only.

### 9.17 What the development Sessions show about the loop (31 October 2026)

From the records of the 42 development-data Sessions at 300 s: the 30 that built the frozen memory (random order, section 9.15) and the 12 of the Phase 2 measurement (rule order, section 9.14). Test gain = final model's test score minus the first model's, signed so that higher is better. Raw data: `experiments/benchmark-memory/SES-###/` and `experiments/SES-002` to `SES-013`, with `results/benchmark/memory_sessions.csv` and `results/phase2/phase2_runs.csv`.

- Only significant gains carry over to the test part. The 11 Sessions whose final Setup contains at least one `keep` gained +0.0162 on average (8 better, 1 equal, 2 worse). The 18 whose final Setup contains only `keep_simpler` steps changed by -0.0031 on average (4 better, 8 equal, 6 worse): simpler and about as good, as the rule promises, but not better. The 13 that kept nothing return their first model (gain 0).
- Steps that change nothing: 18 of the 97 candidate steps of the memory build ended with exactly the same model as the Accepted Setup (same learner, configuration and data treatment), 14 of them `restrict_model_families` (of 24 such steps). They used 7.7% of all Session time (searches 299 s, Assessments 150 s); in Phase 2, 2 of 34 steps and 2.5%. In the memory these count as kept: the arm `restrict_model_families` / weak model families has 21 kept of 24, 16 of them with a difference of exactly 0, which is why the full system moved it from place 14 of the rule order to the front on pc4 at 120 s.
- Time: the memory-build Sessions used 206.8 s of their 300 s on average; 20 of 30 stopped because no change was left, after 182 s on average. Searches took 78% of Session time and Assessments 20%.
- Thin memory: of the 13 arms with the 3 records a learned selector needs, 7 have all their records from one dataset (for example class weights on binary data: 3 records, all from wilt).

### 9.18 Judge agreement study (task 27, 31 October 2026)

Setup (section 4.11): 20 pairs of Setups, 2 per development dataset, drawn with seed 2026 from the judged steps of the 30 memory-build Sessions; for each pair, new FLAML searches of both Setups on 5 random 80/20 splits of the Session's training part (at most 25 trials, 60 s cap, all 16 cores, no warm start), judged with the Judge's own code and settings. Run on 31 October 2026 from 22:14 to 22:47 on mains power, stopped once after pair 10 and resumed (the harness skipped the finished pairs). Pairs 1 to 10 ran with code fd97c00 and pairs 11 to 20 with fabac45, which adds the Judge fix of section 4.7; that fix changes none of the 20 strict verdicts, since every pair's 5 strict differences vary. No pair failed or was excluded. Raw files: `results/studies/judge_agreement_pairs.csv` (one row per pair, with the 5 differences and the searches' Experiment IDs), `judge_agreement_summary.csv`, `judge_agreement_confusion.csv`; the 200 searches in `experiments/studies/judge_agreement/` and their models in `models/studies/judge_agreement/` (not in git), and the log `run.log` there.

| Judge (rows) against strict (columns) | keep | keep_simpler | discard |
|---|---|---|---|
| keep | 0 | 1 | 3 |
| keep_simpler | 2 | 5 | 1 |
| discard | 1 | 4 | 3 |

- Agreement on the three verdicts: 8 of 20 (0.40, 95% bootstrap interval 0.20 to 0.60); Cohen's kappa 0.048 (interval -0.23 to 0.34). On accepted (keep or keep_simpler) against discarded: 11 of 20 (0.55, interval 0.35 to 0.75); kappa 0.043 (interval -0.38 to 0.47). Intervals: 10,000 resamples of the 20 pairs.
- With the fixed Judge, the one `keep` that came from rounding noise (SES-020, section 4.7) is `keep_simpler`, which the strict comparison also gave: agreement 9 of 20 (0.45), kappa 0.102; accepted against discarded is unchanged.
- Cost: the Judge's two Assessments took 519.6 s over the 20 pairs and the strict comparison's 10 searches per pair 1,237.1 s, 2.38 times as much in total (median per pair 4.38). The ratio is below 1 only on kin8nm (0.49), where the Session's Assessments were long (8,192 rows). The strict comparison was capped at 25 trials per search (median 25), fewer than the Sessions' searches of the same Setups (median 66 trials of 40 searches), so a strict comparison at the Sessions' search size would cost more.

What this shows, and what it does not:

- The cheap Judge and this strict comparison agree little more than chance would give. Of the Judge's 4 significant gains (`keep`), the strict comparison found 3 worse: smaller trees on california_housing (strict mean -0.036 in R2) and kin8nm (-0.131), and dropping a column on breast_cancer (-0.008 in F1). It confirmed most "simpler and not worse" verdicts (7 of 8 accepted), and accepted 5 of the Judge's 8 discards, among them class weights on balance-scale (strict +0.100 in macro F1, keep).
- The two methods do not answer the same question. The Judge compares the configurations that the Session's search found (refitted on 15 folds of 80% of the training part), while the strict comparison measures what a new, shorter search finds on one 80/20 split at a time; for changes of the search space, such as smaller trees, the result depends on how much search there is. The strict differences are also noisy (standard deviation over the 5 splits from 0.007 to 0.20).
- Against this, section 9.17 found that the Judge's `keep` steps did carry over to the outer test part in the development Sessions (+0.016 on average, 8 of 11 better). So the study does not show that the Judge is wrong; it shows that its verdicts are not reproduced by searching again, which limits how much a single verdict should be trusted.
- 20 pairs give wide intervals: a kappa anywhere from about -0.2 to 0.3 fits these data. A larger study, with strict searches as large as the Sessions', is future work.

### 9.19 What was not produced, and why

- The first benchmark run: stopped after 33 of 390 cells, 23 usable (section 9.16), and not continued (the user's decision, 31 October 2026). Instead, the whole grid was run again with the final code in its own folders on 12 and 13 November 2026 (section 9.20), so the two runs are never mixed.
- The judge agreement study: 20 of the 57 available pairs (section 9.18).
- The improvements of 31 October 2026 (Algorithms 9 and 18) are tested but not measured on new data.
- Sections 9.2 to 9.6 (Phase 1) ran in temporary folders that were deleted; only their numbers remain. Re-running them with the final code would give raw files for the report (open item, Appendix D).

### 9.20 Second benchmark run, with the final code (12 and 13 November 2026)

Setup. Code of commit af8f14d (all optional tasks included). No tracked file changed during the run (`git diff af8f14d` was empty at the end); the 390 grid rows still say `git_dirty` true, only because the run's own new folder `results/benchmark-v2/` was not in git yet. Laptop on mains power throughout: no wait for power and no excluded cell. All cores (`n_jobs` -1, 16 logical CPUs). Docker Desktop and the Ollama server were stopped before the start. Started by `scripts/run_benchmark_v2.cmd` (memory build, freeze, smoke setting, grid, each with `--label v2`); logs and run notes in `experiments/benchmark-v2-logs/`.

Frozen memory (Requirement 22.3, the procedure of section 9.15): 30 Sessions on the 10 development datasets (seeds 1 to 3, 300 s each, random selector, starting from an empty file), 12 November 19:46:21 to 21:33:27, all completed. 95 records from 25 of the Sessions (the other 5 ended after their first Experiment; stop reasons of the 30: 16 no change left, 8 time, 4 three discards in a row, 2 the Experiment limit): 37 discard, 45 keep_simpler, 13 keep; 17 had no effect on any fold score, so 43 count as kept. 21 arms (task type, Strategy, diagnosis kind), of which 7 have at least 3 records from at least 2 datasets and can move (the first frozen memory: 97 records, 20 arms, 6 that can move). SHA-256 `d678080aee96877dd65da31ae380bada1d87b138372a7605bffeaf3218a39f5d`. The optional Strategies in it: target_encode 4 times and token_features 3 times, all on titanic and all discarded; smote 3 times on wilt, 2 kept. Files: `results/benchmark-v2/frozen_memory.jsonl`, `frozen_memory_summary.csv`, `memory_sessions.csv`; records in `experiments/benchmark-v2-memory/`.

Smoke setting (one Session per system on titanic, seed 1, 120 s; a check, not a result): all 8 systems completed (`results/benchmark-v2/smoke/benchmark_cells.csv`).

Grid: all 390 cells (6 evaluation datasets, seeds 1 to 5, the 5 main systems at 120 and 300 s and 3 ablations at 300 s) completed, 12 November 21:46:25 to 13 November 14:01:55, 16.25 h of cell time, no failure. Rows in `results/benchmark-v2/benchmark_cells.csv`, analysis tables and figures in `results/benchmark-v2/` (`analysis_*.csv`, `figures/`), Session records in `experiments/benchmark-v2/`, models in `models/benchmark-v2/`. Every comparison has 6 datasets and 30 (dataset, seed) blocks.

Pairs (difference of signed test scores, each dataset's mean over the 5 seeds; the 95% bootstrap interval of the median over the 6 datasets; wins, ties and losses over the datasets; Wilcoxon p and Holm-corrected p within the family):

| Budget | Pair | Median difference | 95% interval | W / T / L | p | Holm |
|---|---|---|---|---|---|---|
| 120 s | autopilot_full vs flaml | -0.0137 | -0.0183 to -0.0022 | 0 / 0 / 6 | 0.031 | 0.156 |
| 120 s | autopilot_full vs manual | -0.0024 | -0.0181 to 0.0292 | 2 / 0 / 4 | 0.563 | 1.000 |
| 120 s | autopilot_full vs autogluon | -0.0003 | -0.0177 to 0.0206 | 3 / 0 / 3 | 1.000 | 1.000 |
| 120 s | autopilot_full vs autopilot_rules | 0.0001 | -0.0013 to 0.0182 | 3 / 1 / 2 | 0.625 | 1.000 |
| 120 s | autopilot_rules vs flaml | -0.0141 | -0.0363 to -0.0009 | 0 / 0 / 6 | 0.031 | 0.156 |
| 300 s | autopilot_full vs flaml | -0.0010 | -0.0051 to 0.0205 | 2 / 0 / 4 | 1.000 | 1.000 |
| 300 s | autopilot_full vs manual | 0.0055 | -0.0013 to 0.0384 | 5 / 0 / 1 | 0.094 | 0.469 |
| 300 s | autopilot_full vs autogluon | 0.0025 | -0.0081 to 0.0432 | 4 / 0 / 2 | 0.438 | 1.000 |
| 300 s | autopilot_full vs autopilot_rules | 0.0015 | -0.0017 to 0.0078 | 4 / 0 / 2 | 0.313 | 1.000 |
| 300 s | autopilot_rules vs flaml | -0.0020 | -0.0122 to 0.0210 | 2 / 0 / 4 | 1.000 | 1.000 |
| 300 s | autopilot_full vs random_choice | -0.0001 | -0.0023 to 0.0137 | 2 / 1 / 3 | 0.813 | 1.000 |
| 300 s | autopilot_full vs no_critic | -0.0002 | -0.0024 to 0.0137 | 2 / 0 / 4 | 1.000 | 1.000 |
| 300 s | autopilot_full vs no_resource_manager | 0.0057 | -0.0018 to 0.0218 | 4 / 0 / 2 | 0.219 | 0.875 |

(The pair autopilot_full against autopilot_rules at 300 s is in both families; its Holm value is 0.938 in the ablation family.)

Friedman test over the 6 datasets: main systems at 120 s chi-squared 7.97, p 0.093 (Iman and Davenport p 0.077); at 300 s p 0.355; ablations at 300 s p 0.434. The Nemenyi critical difference is 2.49 ranks for 5 systems, and at every budget all systems fall in one group. Average ranks (1 is best): at 120 s flaml 1.50, manual 2.83, autogluon 3.33, autopilot_full 3.58, autopilot_rules 3.75; at 300 s flaml 2.17, autopilot_full 2.50, autopilot_rules 3.00, manual 3.67, autogluon 3.67; ablations no_critic 2.00, autopilot_full 2.92, random_choice 3.08, autopilot_rules 3.33, no_resource_manager 3.67.

Time-to-target (target: 99% of the best score of any main system at either budget, per dataset and seed; 30 pairs): FLAML reached it by 120 s in 18 pairs and by 300 s in 22, the full system in 13 and 16 (never in 14); median time-to-target 120 s for FLAML and AutoGluon, 300 s for both AutoPilot systems, never for manual. In a pair-by-pair count the full system reached the target at an earlier budget than FLAML in 2 pairs, the same in 18 and later in 10.

Mean test score over the 5 seeds at 300 s (F1, R2 for energy_efficiency and solar_flare):

| Dataset | manual | flaml | autogluon | autopilot_rules | autopilot_full | random_choice | no_critic | no_resource_manager |
|---|---|---|---|---|---|---|---|---|
| MiceProtein | 0.9855 | 0.9989 | 0.9947 | 0.9866 | 0.9921 | 0.9954 | 0.9956 | 0.9956 |
| cylinder-bands | 0.8485 | 0.8547 | 0.8375 | 0.8426 | 0.8529 | 0.8400 | 0.8438 | 0.8420 |
| energy_efficiency | 0.9972 | 0.9989 | 0.9987 | 0.9988 | 0.9988 | 0.9989 | 0.9990 | 0.9989 |
| pc4 | 0.6125 | 0.5897 | 0.5507 | 0.6194 | 0.6218 | 0.6074 | 0.6035 | 0.5892 |
| segment | 0.9814 | 0.9805 | 0.9723 | 0.9766 | 0.9771 | 0.9784 | 0.9784 | 0.9766 |
| solar_flare | 0.0457 | 0.1043 | 0.1268 | 0.1166 | 0.1132 | 0.1132 | 0.1134 | 0.1023 |

At 120 s the full system's means were 0.9837, 0.8454, 0.9989, 0.5840, 0.9737 and 0.1023 in the same order, and FLAML's 0.9956, 0.8610, 0.9989, 0.6038, 0.9780 and 0.1192 (all per-dataset means and spreads: `analysis_per_dataset.csv`).

Time used (mean wall seconds per cell, 120 s / 300 s): flaml 121.1 / 300.9, autopilot_full 96.3 / 212.3, autopilot_rules 97.4 / 215.9, autogluon 65.8 / 98.5 (its `medium_quality` preset stops early), manual 0.7 / 0.6; at 300 s random_choice 215.1, no_resource_manager 233.2 and no_critic 292.1. The Session systems ran 4.5 Experiments on average at 300 s (no_critic 5.2).

What the Sessions changed (all Session systems of the grid, steps after the first): drop_columns was tried 472 times and kept 279 times, restrict_model_families 104 and 86, regularised_search_space 40 and 16, more_search_time 26 and 3, group_rare_categories 14 and 9, class_weight_balanced 17 and 1; frequency_encode (11), smoother_search_metric (10) and missing_indicator (7) were never kept. Of the optional Strategies, token_features was tried 26 times on cylinder-bands and kept 3 times; target_encode 7 times (cylinder-bands and solar_flare) and never kept; smote 4 times, kept twice on pc4.

Outcome against the success criteria of chapter 3 (written on 28 October 2026, before any of these cells ran):

- H4 (not worse than FLAML at the same budget): not shown. At 120 s the interval (-0.0183 to -0.0022) reaches below the margin of -0.005, so the hypothesis is not supported, but it does not lie wholly below it, so by the criteria it is not contradicted either; the full system was behind FLAML on all 6 datasets at 120 s. At 300 s the interval (-0.0051 to 0.0205) just reaches below -0.005.
- H1 (reaches the target sooner than FLAML): contradicted. FLAML reached the target at an earlier budget in 10 pairs and the full system in 2, and FLAML's median time-to-target (120 s) is earlier than the full system's (300 s).
- H2 (without the memory the loop does worse): not shown (interval -0.0017 to 0.0078 at 300 s, -0.0013 to 0.0182 at 120 s).
- H3 (without strategy selection the loop does worse): not shown (-0.0023 to 0.0137).
- RQ3 and the Critic (without the Resource_Manager, or without the Critic, the loop does worse): not shown (-0.0018 to 0.0218 and -0.0024 to 0.0137).

Reading. With the same budget, one FLAML search was the best or level system. At 120 s the loop was behind on every dataset; a likely reason, not measured separately, is that its first search gets only 35% of the time and the 15-fold Assessments take part of the rest. At 300 s the loop was level with FLAML within the intervals, better on pc4 (0.6218 against 0.5897 F1) and solar_flare, and worse on MiceProtein and segment. The Session systems also left about 30% of a 300 s budget unused (212 s of 300 s for the full system): of the 150 Session cells at 300 s, 79 stopped because no change was left to try, 27 after three discards in a row, 39 for time and 5 at the Experiment limit (at 120 s: 30 no change left, 8 three discards, 22 time, of 60). None of the components measured by the ablations (memory, strategy choice, Critic, Resource_Manager) made a difference larger than the intervals over 6 datasets; the run without the Critic even had the best average rank. With 6 datasets the Wilcoxon test cannot reach 0.05 after Holm's correction (Algorithm 21), so these are the results of this grid, not general claims.

- Charts available: the critical-difference diagrams and the score-against-budget chart of the 23 benchmark cells (`results/benchmark/figures/`, 2 datasets, descriptive only), and the Phase 1 search charts (`docs/imp/images/`). Per-seed difference and bootstrap charts were not made, since the benchmark has one seed.

---

## 10. Conclusion and future scope

Conclusions so far (Phase 1):

1. A complete, tested and fully offline baseline exists. It handles binary, multiclass and regression problems from a CSV file on a laptop CPU; an experiment takes its time budget plus a few seconds (for example 32 to 40 s with a 30 s budget).
2. Independent test evaluation, trivial baselines and train vs test scores make the results honest and easy to read.
3. The first experiments show why the planned decision layer needs careful comparison: a sensible change gave a small average gain, and a single split could not tell it apart from noise.
4. Every experiment is recorded with the information the critic, strategy engine and memory will need.

Conclusions so far (Phase 2, 27 October 2026):

1. The rule-based loop runs unattended within its budget, decides only on the training part, and records every Diagnosis, change, Verdict and cost so that each decision can be replayed (sections 9.11 and 9.14).
2. Its two decision components were measured on data with known answers: the Critic's data-level Diagnoses had no false alarm, overfitting was found with precision 0.88 and recall 0.91, and the unreliable-validation Diagnosis is weak (section 9.12); the Judge kept 0 of 18 harmful changes and 10 of 12 helpful ones on the reporting seeds (section 9.13).
3. Against FLAML alone at the same 300 s budget on the four sample datasets, the final test scores were about the same on average (4 better, 4 equal, 4 worse) with two thirds of the time used. The loop helped where a data problem FLAML cannot fix was present (Titanic) and cost a little where its changes were neutral or Assessments were expensive (section 9.14). Whether this holds on more datasets and budgets is the Phase 3 question.

Conclusions so far (Phase 3, 31 October 2026; the benchmark was stopped early, so these are observations, not tested claims):

1. The learned strategy choice, the ablation switches, the comparison systems, the resumable benchmark with its statistics, the frozen memory and the judge agreement study are built and tested (541 tests); the benchmark ran 23 usable cells on 2 evaluation datasets with 1 seed before the laptop's standby cut it short (section 9.16).
2. On those cells the full system was never below FLAML alone (pc4: 0.600 and 0.613 against 0.590; cylinder-bands: equal at 120 s, 0.872 against 0.870 at 300 s), and on pc4 above the rule order; with 2 datasets no test can confirm this.
3. Across the 42 development Sessions, only changes the Judge kept as significantly better raised the test score (+0.016 on average, 8 of 11 Sessions better); "simpler and not worse" changes left it about where it was (section 9.17). The loop's value so far is efficiency and explained decisions more than higher scores.
4. The cheap Judge's verdicts were not reproduced by searching again: on 20 real pairs it agreed with a strict comparison (new 25-trial searches on 5 inner splits) on 8 verdicts (kappa 0.05), and 3 of its 4 significant gains were judged worse there (section 9.18). Together with point 3, a single verdict should be read as evidence from fixed refits, not as a proven gain. The strict comparison cost 2.4 times as much even with its smaller searches.
5. Reading the records showed three flaws, now fixed and tested: rounding noise counted as a gain once (section 4.7), steps that changed nothing counted as successes in the memory, and arms seen on one dataset steered the learned order (Algorithms 9 and 18). Their effect on test scores is not measured.

Conclusions of the second, full benchmark run (13 November 2026; 390 cells on 6 evaluation datasets, 5 seeds, budgets of 120 and 300 s, code af8f14d with the fixes of 31 October and the optional tasks; section 9.20):

1. The whole system ran the memory build (30 Sessions) and all 390 cells unattended in about 18 hours, with no failure.
2. At the same budget, one FLAML search was as good as the loop or better. At 120 s FLAML was ahead on all 6 datasets (median difference 0.0137); at 300 s the two were level within the intervals (median -0.0010). By the criteria of chapter 3, H1 is contradicted and H4 is not shown.
3. The memory, the learned strategy choice, the Critic and the Resource_Manager did not change the test score by more than the intervals over 6 datasets (H2, H3 and RQ3 not shown); the run without the Critic had the best average rank of the ablations.
4. With the judge agreement study (point 4 of the Phase 3 conclusions) and the development Sessions (point 3), the evidence says that the loop's decisions are recorded, replayable and explained, but that on these datasets its changes rarely add to the test score beyond what FLAML finds alone. Spending part of the budget on Assessments is the likely price, most at short budgets (not measured separately).
5. Ideas from this run, not tested: give the first search more than 35% of a short budget, or skip the loop below some budget; use the time the Sessions left unused (about 30% at 300 s) for a last, longer search from the accepted Setup; and measure the optional Strategies on more datasets (in this grid only token_features, 3 of 26 tries kept on cylinder-bands, and smote, 2 of 4 on pc4, were ever kept; target_encode never).

Scope for improvement found in the results (31 October 2026; ranked by how directly the data points to it; the first three are built, the others are future work):

| No. | Improvement | Evidence | Effort |
|---|---|---|---|
| 1 | Keep long runs on mains power with the screen on, and exclude cells measured on battery | 10 of 33 benchmark cells measured partly in Modern Standby (section 9.16) | Done on 31 October 2026 (harness) |
| 2 | Let a learned selector move an arm only when its records come from at least 2 datasets | 7 of the 13 arms with 3 or more records come from one dataset; the class-weight arm's 3 binary records (all wilt) kept class weights out of reach on pc4, where they were the only significant gain (sections 9.16, 9.17) | Done on 31 October 2026 (Algorithm 18); 6 of 20 arms can still move. Not measured on new data |
| 3 | Do not reward steps that changed nothing; reuse the Accepted Assessment when the candidate refits the same model | 18 of 97 memory-build steps gave the identical model (7.7% of Session time); the memory counts them as kept, so `restrict_model_families` looks like the best arm (21 of 24 kept, 16 with a difference of 0) (section 9.17) | Done on 31 October 2026: such records count as not kept (34 of 97 records count as kept, 53 before; Algorithm 18), and identical refits reuse the fold scores (2.4% of the memory build's time; Algorithm 9). Not measured on new data |
| 4 | After a discard, try a different kind of change before another of the same kind | On pc4 the rule order spent its patience on "drop one of a redundant pair" and never reached class weights. This comes from an evaluation dataset, so its effect must be measured on datasets not seen yet | Small: the ranking in `strategy_engine.py` |
| 5 | Stronger changes for imbalanced binary F1 (decision-threshold tuning on the folds, SMOTE: optional task 29) and for high-cardinality text columns (target encoding: optional task 30) | Class weights kept 1 of 6 times but gained +0.085 when kept; frequency encoding 0 of 5 and grouping rare values 0 of 2 kept (frozen memory, section 9.15) | Medium: new Strategies with fold-safe fitting |
| 6 | Use the time left when no change remains, for example one longer search of the Accepted Setup | 20 of 30 memory-build Sessions stopped with about 118 s of 300 unused; but FLAML with 300 s was not better than with 120 s on the 2 usable datasets, so the gain is uncertain | Small, as an option |
| 7 | Check LightGBM's library defaults as a cheap candidate (one refit on the folds) | The defaults beat FLAML's search on titanic (smoke run) and pc4, not on cylinder-bands | Small to medium |
| 8 | Fairer AutoGluon reference: a stronger preset at the same budget | `medium_quality` stopped after 31 to 74 s of 120 or 300 s | Small (harness setting), but longer runs |

Built after this list was first written (optional tasks 29 to 35, 1 to 12 November 2026; section 5.5): target suggestion and a warning-only leakage check; word features (for example titles from Titanic's `Name`) and cross-fitted target encoding of high-cardinality columns; SMOTE; SHAP explanations of the final model, for the whole model and for one row; group-aware splits and folds; plain-language summaries by a local language model through Ollama, which no decision uses; a local REST API (FastAPI) and a Docker image. None of them has been measured in the benchmark yet.

Future scope (beyond the planned phases):

- A judge agreement study with strict searches as large as the Sessions', on the pairs of the second memory build (the full benchmark grid was run on 12 and 13 November 2026, section 9.20)
- More evaluation datasets, so that the Wilcoxon tests can reach significance after Holm's correction, and a measurement of the optional Strategies on them
- The ideas of point 5 of the second run's conclusions (the share of the first search, the unused time)
- Feature engineering beyond single columns, for example family size from two Titanic columns
- Warm starts after "smaller trees": move the Accepted_Setup's best configurations into the limited search space instead of leaving them out, and measure whether that helps (section 4.7)
- The improvements 4 to 8 of the table above
- A database backend and a login for several users of the API
- Time-series and text features, once the tabular loop is shown to work

---

## Appendix A. References

### A.1 Master list

`PROJECT.md` section 26 (27 entries plus the datasets). Keep it as the master list and add the sources below when the report cites them.

### A.2 Additional sources used in these notes

- Cawley, G. C., and Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. Journal of Machine Learning Research, 11, 2079-2107.
- Kaufman, S., Rosset, S., Perlich, C., and Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. ACM Transactions on Knowledge Discovery from Data, 6(4).
- Fisher, A., Rudin, C., and Dominici, F. (2019). All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously. Journal of Machine Learning Research, 20(177).
- Kayali, M., and Wang, C. (2022). Mining Robust Default Configurations for Resource-constrained AutoML. arXiv:2202.09927. The basis of FLAML's zero-shot AutoML (checked online on 12 October 2026).
- Yang, C., Akimoto, Y., Kim, D. W., and Udell, M. (2019). OBOE: Collaborative Filtering for AutoML Model Selection. KDD 2019. arXiv:1808.03233 (checked online on 12 October 2026).
- Thompson, W. R. (1933). On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples. Biometrika, 25(3/4), 285-294.
- Efron, B., and Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.
- Friedman, M. (1937). The Use of Ranks to Avoid the Assumption of Normality Implicit in the Analysis of Variance. Journal of the American Statistical Association, 32(200), 675-701.
- Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. Biometrics Bulletin, 1(6), 80-83.
- Nemenyi, P. B. (1963). Distribution-free Multiple Comparisons. PhD thesis, Princeton University.
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics, 6(2), 65-70.
- Iman, R. L., and Davenport, J. M. (1980). Approximations of the Critical Region of the Friedman Statistic. Communications in Statistics: Theory and Methods, 9(6), 571-595.
- Kerby, D. S. (2014). The Simple Difference Formula: An Approach to Teaching Nonparametric Correlation. Comprehensive Psychology, 3, article 1.
- Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
- Chen, T., and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

### A.3 To read (check the details before citing)

- Salinas, D., and Erickson, N. TabRepo: a large repository of tabular model evaluations and its AutoML uses (portfolios built offline).
- Domhan, T., Springenberg, J. T., and Hutter, F. (IJCAI 2015). Speeding up hyperparameter optimisation by extrapolating learning curves; relevant to stopping rules.
- Grinsztajn, L., Oyallon, E., and Varoquaux, G. (NeurIPS 2022, Datasets and Benchmarks). Why tree-based models still outperform deep learning on tabular data; supports the statement in chapter 1.

---

## Appendix B. Work log

| Date | Work | Commits |
|---|---|---|
| 2026-10-07 | Checked tools (Windows 11, Git 2.55, conda 26.5.3); created conda env `automl` with Python 3.11.16; installed pinned libraries (SHAP installed and removed again as not needed yet) | |
| 2026-10-07 | Project skeleton, settings and logging; task detector; profiler; evaluator; MLflow tracking; FLAML integration; sample datasets; Streamlit dashboard | b29514e to 80d7fbb |
| 2026-10-07 | README, PROJECT.md, architecture, roadmap, research direction | 4a3b013 |
| 2026-10-07 | Fixed the flood of FLAML unseen-category warnings | 2fddc57 |
| 2026-10-07 | Documents rewritten in plain language; demo guide; demo rehearsed (23 checks passed) | 95d582a |
| 2026-10-07 | EXP-001 and EXP-002 (Titanic, 60 s) run at 16:55 and 17:02 | |
| 2026-10-07 | Five-seed paired comparison; `scripts/compare_seeds.py`; results added to the demo guide | ae040b3, e8a5492 |
| 2026-10-07 | Removed the decorative rocket emoji from the dashboard | c88f9ea |
| 2026-10-07 | Fixed warnings returning when scikit-learn resets the warning filters | ca536c7 |
| 2026-10-07 | `Project_Presentation.md` for the Phase 1 review | e3ec86e |
| October 2026 | Phase 1 review presented and completed; project title and description posted for the department | |
| 2026-10-12 | Documents split into `docs/imp` (maintained) and `docs/unimp` (archived); links updated; roadmap updated after Phase 1 | e090039, 28a52cc |
| 2026-10-12 | Report notes started; ticket-overlap check, paired bootstrap and paired tests; shared-ticket explanation corrected; roadmap step for fair comparison of changes | e3dd52d |
| 2026-10-22 | FLAML 2.7.0 internals read in the source and checked against the EXP-001 and EXP-002 logs (section 5.2, 9.10); `scripts/plot_search_history.py` and three EXP-001 charts; matplotlib pinned; plain-language guide `how-it-works.md` with 10 diagrams, all parsed with Mermaid 11.15.0 | 5722435, afa3b7c |
| 2026-10-26 | Kiro spec written for Phases 2 and 3 (`.kiro/specs/autopilot-ml/`: 32 requirements with 160 acceptance criteria, design with decisions D1 to D8, 35 tasks plus 21 sub-tasks); links and coverage checked by a script; design diagrams parsed with Mermaid 11.15.0; roadmap rewritten as a summary of the spec; section 5.3 updated; steering rule updated to follow the spec | 7d099ba, 9618e6b |
| 2026-10-26 | Spec task 1: FLAML capability checks (8 tests in `tests/test_flaml_capabilities.py`, all passing; 55 tests in total); findings in section 5.2; design updated (refit through `max_iter=1`, our own check of starting points) | da4dadc |
| 2026-10-26 | Spec task 2: `app/services/splits.py` (outer split and session folds with fingerprints); the runner now uses the shared split rule; all 55 tests still pass | 70ec049 |
| 2026-10-26 | Spec task 2.1: 15 split tests in `tests/test_splits.py`, including a check that the new rule gives exactly the test rows, in the same order, saved by EXP-001 (Titanic, seed 42); 70 tests in total, all passing | 49c5aab |
| 2026-10-26 | Spec task 3: engine and runner extensions (starting points, search-space limits, class weights, separate search metric, `max_iter`, `fit_config`, `run_experiment` options for Sessions, `decision_view`, new record fields; section 8.2); found that FLAML's integer upper bounds are exclusive (section 5.2); a throwaway run on Titanic (10 s, seed 42, `max_iter=8`, temporary folder) showed every option reaching FLAML; all 70 tests still pass | 77ed822 |
| 2026-10-26 | Spec task 3.1: 28 tests in `tests/test_engine_extensions.py` and one more FLAML check (integer upper bound never searched); 99 tests in total, all passing in 64.8 s (section 8.6) | e54e69c |
| 2026-10-26 | Spec task 4: `research/` package and the synthetic data generator `research/synthetic.py` with 8 problem kinds and a fresh sample (section 5.4, Algorithm 8); section 5.4 started with Algorithm 7 from task 3 | d5bd0ac |
| 2026-10-26 | Spec task 4.1: 32 generator tests in `tests/test_synthetic.py`; 131 tests in total, all passing in 65.8 s. A planned check that another seed is a different problem was dropped: logistic regression trained on the seed-10 binary dataset (1,000 rows) scored accuracy 0.758 on its own fresh sample (20,000 rows) and still 0.715 on seed 11's (5,000 rows) | c786048 |
| 2026-10-26 | Spec task 5: `app/services/assessor.py` (Algorithm 9) and `app/services/search_summary.py` (Algorithm 10); the runner saves the search summary; cache key widened to setup key plus configuration (design updated); 5 search-summary tests, two of them on the saved EXP-001 and EXP-002 histories; throwaway Assessor check on Titanic (section 5.4) | ee11363 |
| 2026-10-26 | Spec task 5.1: 9 Assessor tests in `tests/test_assessor.py`, including two separate Assessments with real FLAML refits (LightGBM, 12 trees, 240 rows, 15 folds each) that agree within 1e-12; 145 tests in total, all passing in 83.9 s | 10a172f |
| 2026-10-26 | Guard: the runner refuses datasets with repeated row labels (section 4.7), with one new test in `tests/test_automl_runner.py` | 6fa465d |
| 2026-10-26 | Spec task 6: thin Critic (`app/services/critic.py`, Algorithm 11) with thresholds file `critic_thresholds.toml` (`v0-start`); one threshold added to the design (`min_ratio_for_recall`); throwaway check on Titanic gave the expected Diagnoses (section 5.4) | 1d49fe4 |
| 2026-10-26 | Spec task 6.1: 44 Critic tests in `tests/test_critic.py`; 190 tests in total, all passing in 86.7 s | 6b1ac70 |
| 2026-10-26 | Spec task 7: Strategy_Catalogue (`app/services/strategies.py`: drop_columns, class_weight_balanced, regularised_search_space) and the rule-based Strategy_Engine (`app/services/strategy_engine.py`), Algorithm 12 in section 5.4 | 32db280 |
| 2026-10-26 | Spec task 7.1: 21 Strategy tests and 8 Strategy_Engine tests; 219 tests in total, all passing in 86.7 s | a8629f3 |
| 2026-10-26 | Spec task 8: Judge with the corrected resampled t-test (`app/services/judge.py`, Algorithm 13 with a worked example in section 5.4) | 064396a |
| 2026-10-26 | Fix found by the Judge tests: equal differences are detected from the values, not from a rounded variance (section 4.7) | 032674e |
| 2026-10-26 | Spec task 8.1: 25 Judge tests in `tests/test_judge.py` (the worked example by hand, every verdict, error metrics, equal differences, records, errors); 244 tests in total, all passing in 87.6 s | 00954da |
| 2026-10-26 | Spec task 9: Resource_Manager (`app/services/resource_manager.py`, Algorithm 14): deadlines per step for measuring overruns, reserve from the previous step, stop rules in order | 733c86e |
| 2026-10-26 | Spec task 9.1: 41 Resource_Manager tests with a fake clock, including 20 simulated Sessions; 285 tests in total, all passing in 88.4 s | 47b55b7 |
| 2026-10-26 | Spec task 10: records and tracking for Sessions (session IDs, atomic JSON writes, step files, Phase 1 records read as version 1, Session column, session history, MLflow parent and child runs; section 8.2) | 3612e12 |
| 2026-10-26 | Spec task 10.1: 8 tracking tests in `tests/test_tracking.py`; 293 tests in total, all passing in 95.3 s | 847621f |
| 2026-10-26 | Spec task 11: Planner, the autonomous Session loop and `scripts/run_autopilot.py` (Algorithm 15 in section 5.4); two trial Sessions in a temporary folder ran end to end (not kept) | 62dc010 |
| 2026-10-26 | Spec task 11.1: 12 Session tests in `tests/test_autopilot.py` (Information_Barrier, stops, failures, replay); `Assessment.from_dict` and `DatasetProfile.from_dict` for replaying from the records; 305 tests in total, all passing in 103.8 s | 489c048 |
| 2026-10-26 | Spec task 11.2: first real Session on Titanic, SES-001 (300 s budget, at most 8 Experiments, seed 42; EXP-003 to EXP-006): stopped with no Proposals left after 206.0 s; refit validation F1 0.706 to 0.763; test F1 of the first and final model 0.684 and 0.724; records checked by hand (section 9.11); two weaknesses noted (section 4.7) | 49cb088 |
| 2026-10-26 | Dashboard sidebar: the first version of the autonomous layer moved from "planned" to "implemented" (section 8.3); the app renders without errors; 305 tests passing in 114.1 s | 173454a |
| 2026-10-26 | Spec task 12: Data_Steps (`app/services/transforms.py`, Algorithm 16) used by the runner (fitted on the Training_Part, saved with the model) and by the Assessor (fitted on each fold) | 8f4dc5f |
| 2026-10-26 | Spec task 12.1: 20 Data_Step tests in `tests/test_transforms.py`; 326 tests in total, all passing in 108.9 s | c82a09d |
| 2026-10-26 | Spec task 13: the full Critic (all twelve Diagnoses, section 5.4) and the profiler's near-constant columns and correlated pairs; checked on the saved SES-001 records | 00ef933 |
| 2026-10-26 | Spec task 13.1: 68 more Critic tests (112 in `tests/test_critic.py`) and 2 profiler tests; test counts in section 8.6 corrected from pytest's list (the runner file has 21 tests and the engine file 29); 396 tests in total, all passing in 114.2 s | 39c5569 |
| 2026-10-26 | Spec task 14: the full Strategy_Catalogue (nine Strategies) and rule table (section 5.4); a throwaway 90 s Session on Titanic used the new data Strategies end to end; existing tests updated for the longer Proposal lists | b7869fe |
| 2026-10-26 | Spec task 14.1: 30 more Strategy tests and 8 more Strategy_Engine tests; 434 tests in total, all passing in 112.0 s | 4fe539c |
| 2026-10-26 | Spec task 15: Experiment_Memory version 1 (`app/services/experiment_memory.py`, Algorithm 17); the Session adds one record per Verdict; Proposals record the severities of the Diagnoses they answer | 4500f7f |
| 2026-10-26 | Spec task 15.1: 14 memory tests in `tests/test_experiment_memory.py`; 448 tests in total, all passing in 115.8 s | 0ddc314 |
| 2026-10-26 | Spec task 16: critic study (`research/critic_study.py`, `scripts/run_study.py`; 270 synthetic datasets, 43.3 min); results and limits in section 9.12; thresholds frozen as `v1-critic-study`; judge study code and smoke tests written for task 17; slow test marker added; 457 tests passing in 139.7 s | 5ef7ca7 |
| 2026-10-26 | Spec task 17: judge study (84 pairs of Setups, 32.4 min; one fix found by a trial run first); results and limits in section 9.13; by the rule fixed before the run the defaults became 5 fold repeats and alpha 0.20 (were 3 and 0.10); the choice of 5 over 3 repeats rests on one pair and is open (section 4.7); 458 tests passing in 150.4 s | bad02ec |
| 2026-10-26 | Spec task 18: autonomous mode in the dashboard (`app/components/autopilot_viewer.py`, mode switch in `app/main.py`; section 8.3); the Session hands its record to the page after every save (`on_update`); sidebar: the full Critic and Strategies and the Experiment_Memory moved to built; two Phase 1 captions that called the Critic planned now point to the autonomous mode; checked with a throwaway AppTest run on Titanic in a temporary folder | fdb5243 |
| 2026-10-26 | Spec task 18.1: 5 dashboard tests with Streamlit's `AppTest` (`tests/test_dashboard.py`, marked slow) and 2 tests of the Session's update callback; the sidebar now lists the autonomous mode as built; 465 tests passing in 175.8 s (456 without the slow ones in 127.6 s) | b15bcd2 |
| 2026-10-26 | Judge defaults revised: the default number of fold repeats goes back to 3 (alpha stays 0.20), a recorded departure from the judge study's rule (sections 4.7 and 9.13); spec and documents updated; 465 tests passing | 7beb436 |
| 2026-10-26 | Decided the two open points of section 4.7: the ranking tie-break and moving warm starts into a limited search space are not applied (reasons in section 4.7; the second is listed as future work) | 4200033 |
| 2026-10-27 | Spec task 19, Phase 2 measurement tool: `research/phase2.py`, `scripts/run_phase2.py` and 3 tests | 940225a |
| 2026-10-27 | Spec task 19: Phase 2 measurement (24 runs, 300 s each, 1 h 42 min; section 9.14, Phase 2 conclusions in chapter 10); new section 9 of `how-it-works.md`; `architecture.md`, `PROJECT.md`, the README and the roadmap describe the built autonomous layer; report notes: objectives, method, flowchart 6.6, code size, commands, data locations | e3f7924 |
| 2026-10-27 | Spec task 20: development and evaluation datasets (`research/datasets.py`, `scripts/prepare_datasets.py`, registry `research/datasets.json`; section 4.9): 6 OpenML datasets for each list, drawn with seed 2026 from six groups of the laptop-sized CC18 and CTR23 datasets, all downloaded and checked; 4 tests; 472 tests passing in 180.7 s | 528e9cd |
| 2026-10-27 | Spec task 21: learned strategy choice (Algorithm 18): `random`, `thompson` and `ucb1` selectors next to `rules`, the `use_memory` switch, the selection of each step in `session.json`, a read-only memory never written; command-line flags and a dashboard switch | c6151fa |
| 2026-10-27 | Spec task 21.1: 10 selector tests in `tests/test_selectors.py`; the sidebar lists the learned strategy choice as built; documents updated; 482 tests passing in 173.2 s | 5b73142 |
| 2026-10-27 | Spec task 22: ablation switches (Algorithm 19): `use_critic` (random valid changes, no Diagnoses) and `use_resource_manager` (equal search budgets, only the time and Experiment limits stop), next to `use_memory` and the random order; recorded in `session.json`; command-line flags | 1e840e9 |
| 2026-10-27 | Spec task 22.1: 4 ablation tests in `tests/test_ablations.py`; 486 tests passing in 182.0 s | 73ae9f5 |
| 2026-10-27 | Spec task 23: comparison systems (Algorithm 20, `research/systems.py`): manual LightGBM, FLAML alone, AutoGluon through a worker in the new `autogluon` conda environment (decision D4: it needs pandas 2.3), and the AutoPilot variants, all on the same rows and scored by the same Evaluator; a throwaway run of every system | e81474d |
| 2026-10-27 | Spec task 23.1: 7 tests of the comparison systems in `tests/test_systems.py`, the AutoGluon bridge tested with stand-in worker scripts; 493 tests passing in 221.9 s | bc4d575 |
| 2026-10-27 | Spec task 24: benchmark harness (`research/benchmark.py`, `scripts/run_benchmark.py run / status`), the reduced grid agreed with the user (390 cells: budgets 120 and 300 s, ablations at 300 s; 600 s dropped), append-only rows with restart, memory guards and a smoke setting (section 4.10); a throwaway trial of all 8 systems | 1e05dc3 |
| 2026-10-27 | Spec task 24.1: 11 harness tests in `tests/test_benchmark.py` with stand-in systems; 504 tests passing in 218.6 s | f523358 |
| 2026-10-27 | Documents brought in line with tasks 22 to 24 (README, `PROJECT.md` with a new section 8.8, `architecture.md`, `how-it-works.md`, `development-roadmap.md`, `research-direction.md`): the ablation switches, comparison systems and harness described as built, the benchmark as not run; the reduced budgets; CPU seconds marked as not measured; the AutoGluon environment's setup in the README | d04e6db |
| 2026-10-27 | Spec task 25: benchmark statistics (Algorithm 21, `research/stat_tests.py`, renamed from `statistics.py` because that name broke AutoGluon's imports): Friedman with Iman and Davenport's F, Nemenyi critical difference and CD-diagram groups, Wilcoxon with Holm, rank-biserial correlation, bootstrap intervals of the median difference, time-to-target; the power limits with 6 datasets worked out | c428dc2 |
| 2026-10-27 | Spec task 25.1: 13 statistics tests in `tests/test_stat_tests.py` against SciPy's and Demšar's reference values and hand-worked ones; 517 tests passing in 222.0 s | 673d2ef |
| 2026-10-27 | Spec task 26, first part: the frozen memory's build (`research/memory_build.py`, `scripts/build_memory.py`; 30 Sessions on the development datasets with a random order, only finished Sessions frozen; section 4.10) and 4 tests; a throwaway trial (titanic, 40 s) ran, froze and summarised; 521 tests passing in 217.1 s | 94714d2 |
| 2026-10-27 | Spec task 26: the benchmark analysis (`research/benchmark_analysis.py`, `scripts/run_benchmark.py analyse`; section 4.10): complete blocks only, three families of comparisons with their own pairs, per-dataset tables, time-to-target, critical-difference diagrams and score-against-budget charts; fixed before any benchmark cell ran; 6 tests | 6c7ad8e |
| 2026-10-27 | Spec task 27: the judge agreement study (`research/judge_agreement.py`, `scripts/run_study.py agreement`; design in section 4.11, fixed before any strict comparison ran) and 6 tests | 26f7718 |
| 2026-10-27 | Spec task 26: built and froze the benchmark memory (30 Sessions, 17:10 to 18:54, none failed; 97 records, 13 of 20 arms with at least 3; SHA-256 4baeb012...; section 9.15) | 41737c8 |
| 2026-10-27 | Spec task 26: smoke run with the frozen memory (all 8 systems completed on titanic, 12 min 50 s; section 4.10) and a check of every system type on a synthetic boolean target | ec2be03 |
| 2026-10-28 | Started the 390 benchmark cells at 14:02, with the judge agreement study (57 pairs) queued after them; documents updated to say what is built and what is running | 7dfe4da |
| 2026-10-28 | Analysis fix (section 4.10): the median time-to-target counts runs that never reached the target, and pair counts of who reached it earlier were added; made after the first 6 cells had run. Only the analysis tests were run (7 passed, one core); the full suite runs after the benchmark so as not to share the CPU with it | d9c34bf |
| 2026-10-28 | Success criteria for H1 to H4 (chapter 3), written after 8 of the 390 cells; objectives O3, O5 and O7 updated | 2d93103 |
| 2026-10-31 | The benchmark stopped after 33 cells: the laptop switched to battery at 14:57 on 28 October, went into Modern Standby and rebooted uncleanly on 31 October. 10 cells excluded (`excluded_cells.csv`); the user decided not to run the rest. Harness fix: keep the screen on, wait while on battery, exclude cells that end on battery (2 tests). Analysis of the 23 usable cells (section 9.16), of the 42 development Sessions (section 9.17), and the scope for improvement (chapter 10) | cd34e35 |
| 2026-10-31 | Judge fix: a mean difference within the refit tolerance is no gain (one `keep` of the memory build came from rounding noise; section 4.7). Replaying the 20 study pairs' saved Assessments with the fixed Judge gives the recorded verdicts for 19 and `keep_simpler` instead of `keep` for that one | fabac45 |
| 2026-10-31 | Assessment reuse (improvement 3 of chapter 10, first half): the Assessor's cache key is now what the refits depend on (columns, Data_Steps, class weights, learner, configuration), so a candidate that refits like an earlier Setup reuses its fold scores (`reused_from`); in the memory build this would have saved the Assessments of 18 of 97 steps, 149.7 s of 6,203.6 s (Algorithm 9) | 9cc1054 |
| 2026-10-31 | Learned choice (improvements 2 and 3 of chapter 10): an arm moves only with records from 2 or more datasets, and a kept step that changed no fold score counts as not kept; on the frozen memory 6 of 20 arms can move (13 before) and 34 of 97 records count as kept (53 before) (Algorithm 18); 541 tests passing in 208.8 s | 376a62f |
| 2026-10-31 | Spec task 27: judge agreement study on 20 pairs (22:14 to 22:47): agreement 8 of 20, kappa 0.048; accepted against discarded 11 of 20, kappa 0.043; the strict comparison cost 2.38 times the Judge's Assessments (section 9.18) | 812eec4 |
| 2026-10-31 | Dashboard sidebar: the benchmark and the studies listed as built research tools, the optional extras as planned | d6cdf1c |
| 2026-10-31 | Spec task 28: README (results, AI tools used), `PROJECT.md`, `architecture.md`, `how-it-works.md`, `development-roadmap.md` and `research-direction.md` brought in line with the finished system; report notes completed where the material exists (organisation of the report, code size after Phase 3, code listings, outcome of the hypotheses, conclusions, open items); final check: 541 tests passing in 214.6 s, a smoke benchmark of all 8 systems, and the dashboard started and answering its health check (section 8.8) | 966a8f7 |
| 2026-10-31 | The final smoke run's rows kept in `results/benchmark/smoke/final_check_cells.csv` | 562f6cf |
| 2026-11-01 | The scikit-learn parallel-configuration warning (37,800 lines in the final smoke log) dropped as search noise; likely cause recorded, not confirmed (section 4.7) | 4bc8052 |
| 2026-11-01 | Optional task 29: SMOTE through FLAML's resampler (Algorithm 22): `imbalanced-learn==0.14.2` pinned, `FoldSafeSmote`, the `smote` Strategy as the third answer to class imbalance, the resampler in the runner, the engine and every refit of the Assessor; 9 tests in `tests/test_resampling.py`, rule-table and replay tests updated; 550 tests | 717e454 |
| 2026-11-12 | Optional task 30: cross-fitted target encoding and word features (Algorithms 23 and 24): Data_Steps `target_encode` and `token_features`, the Strategies of the same names, the profile's `frequent_words`, the Critic's `unused_words` Diagnosis (which can bring back a text column the first Setup left out, such as Titanic's `Name`, using only its words), the runner and the Assessor training on `fit_transform`; a throwaway Titanic check with real FLAML; 11 tests in `tests/test_text_steps.py`; 561 tests | 7b3ecd2 |
| 2026-11-12 | Optional task 31: group-aware splits (Algorithm 25): `SessionSettings.group_column`, the test part made of whole groups, Session_Folds that keep each group inside one fold, the column left out of the features, recorded in `session.json` and MLflow; `--group-column` and a dashboard option; a throwaway Titanic Session grouped by `Ticket`; 7 tests in `tests/test_group_folds.py`; 568 tests | 4020e33 |
| 2026-11-12 | Optional task 32: SHAP explanations of a Session's final model on Training_Part rows (Algorithm 26): `shap==0.51.0` pinned, `app/services/explainer.py` (tree SHAP, permutation estimate otherwise), a dashboard section and `scripts/explain_session.py`; timings of both explainers on three sample datasets; 5 tests in `tests/test_explainer.py` and a dashboard test step; 573 tests | cbeb26d |
| 2026-11-12 | Optional task 33: target suggestion with reasons and the single-column leakage check (Algorithms 27 and 28): `suggest_targets`, `single_column_scores` on the Training_Part, the warning-only `possible_leakage` Diagnosis with an optional threshold (0.99), dashboard warnings, `scripts/check_single_columns.py` run on all 16 datasets (no false alarm; `results/leakage_check.csv`); 5 tests in `tests/test_leakage.py`; 578 tests | 38182ad |
| 2026-11-12 | Optional task 34: plain-language Session summaries by a local language model through Ollama (Algorithm 29): `app/services/narrator.py` (facts from the record only, this machine only, labelled, unsupported numbers listed, saved apart), a dashboard section and `scripts/summarise_session.py`; two live runs with `qwen2.5:7b-instruct`, the first of which misread a verdict code, so outcomes are now given in words; 5 tests in `tests/test_narrator.py` and a dashboard test step; 583 tests | cf6c3a9 |
| 2026-11-12 | Optional task 35: the local HTTP API (`app/api.py`, `scripts/run_api.py`; fastapi, uvicorn and httpx2 pinned) and the Docker image (`Dockerfile`, `scripts/docker_entry.sh`, `.dockerignore`, `.gitattributes`); a live API Session with real FLAML; the image built (2.12 GB) and checked with a short Session inside it; 3 tests in `tests/test_api.py`; 586 tests | 5650aa1 |
| 2026-11-12 | A run label for the benchmark and the memory build (`--label v2`): a later run writes to `experiments/benchmark-v2*/`, `models/benchmark-v2*/` and `results/benchmark-v2/`, so it can never touch the first run's rows or its frozen memory; tests for both folder layouts | 3d66e8c |
| 2026-11-12 | Documents brought in line with optional tasks 29 to 35: README, PROJECT.md, how-it-works, architecture, roadmap, research direction, the services package docstring and two stale module docstrings, the dashboard sidebar (only the second benchmark run is planned), and in these notes the objectives table, the future scope and the sidebar description | 40c868f |
| 2026-11-12 | Code size after the optional tasks (section 8.1): 20,235 lines in 97 Python files at 40c868f | 4d479df |
| 2026-11-12 | `scripts/run_benchmark_v2.cmd` chains the second run; started at 19:46 from af8f14d with a clean tree, Docker Desktop and Ollama stopped | af8f14d |
| 2026-11-12 | Second frozen memory: 30 development Sessions (19:46 to 21:33), 95 records frozen, SHA-256 d678080a...39f5d; smoke setting: all 8 systems completed (section 9.20) | e995480 (data) |
| 2026-11-13 | Second benchmark run: all 390 cells completed (21:46 on 12 November to 14:01), no failure, no battery wait; analysis written to `results/benchmark-v2/`; H1 contradicted, H2 to H4 and the ablations not shown (chapter 3, section 9.20, chapter 10) | e995480 (data) |
| 2026-11-13 | Second run recorded: section 9.20, the outcome in chapter 3, the conclusions and future scope in chapter 10, objectives O3, O5 and O7, Appendix E; README, PROJECT.md, architecture, roadmap, research direction and how-it-works updated; the sidebar's planned items are now the judge agreement study with full-length searches and more datasets | 6d2cd7c |
| 2026-11-13 | Final report in the department's Word format: builder `scripts/report/build_report.py` (Times New Roman, 1.15 spacing, A4 with equal margins, numbered figures, tables and equations, IEEE references; Word fills the page numbers and exports the PDF) and its format checks; 15 figures drawn by `scripts/report/report_figures.py` from `results/benchmark-v2/`; text in `docs/imp/final-report/chapters/`. Built: 62 pages, 9 chapters, 25 tables, 14 equations, 3 algorithms, 79 references, 0 format problems. FLAML is described as the external engine it is (section 2.4 explains its search from its paper and source code, with citations). Demo guide: `docs/imp/demo-guide.md` | 19fab4f and the report commits after it |
| 2026-11-13 | Final report condensed and led by what was delivered: abstract and chapters 3, 4 and 6 to 9 rewritten, no dates or timeline in the chapters, 9 tables (from 25) and 9 figures (from 15), with two new tables (the steps of SES-001 from its `session.json`, and the average ranks at both budgets from `results/benchmark-v2/analysis_ranks.csv`) and a one-panel wall-time figure. The secondary studies (ablation variants, time-to-target, Judge agreement, hypothesis outcomes) are left out of the report and stay in these notes and in `results/`. Same measured numbers; every chapter inside the template's page range; 52 pages, 0 format problems. Demo guide updated to the new numbering, with prepared answers on the 2-minute budget and the ablation variants | f520d78, 43567aa |
| 2026-11-14 | Final report checked against every rule of the department's guidelines: roman page numbers on the front pages, Times New Roman also in the code listing (equations stay in Cambria Math, since Times New Roman broke roots, primes and floor brackets in a test), subheadings 14 pt and no third heading level, minus signs that do not break from their numbers, IEEE en-dash ranges and access dates, only component names capitalised, and every figure and table introduced by number in the paragraph next to it with what it shows. References cut from 77 to 34, all cited, with every paragraph of chapters 1 and 2 still cited. Diagrams redrawn with the same terms and tighter layouts, and long tables allowed to break with a repeated header, so no blank space above about 2 cm is left inside a chapter. New measured facts from `results/benchmark-v2/benchmark_cells.csv` and `experiments/benchmark-v2/SES-172/session.json`: at 300 s the full system took less time than FLAML alone on every dataset (16% less on MiceProtein to 64% less on solar_flare), ran 4.5 experiments per session on average, and on pc4 with seed 5 raised the test F1 from 0.5672 to 0.6875 (FLAML alone 0.5672). 47 pages, 0 format problems | this commit and the report commits before it |
| 2026-11-14 | Formal project explanation in `docs/imp/explanation.md`, self-contained with all 9 figures and 9 tables of the report: a 10-minute outline built on the Titanic session SES-001 (report Table 7), a demonstration appendix for use only on request, and answers to anticipated questions. Demo rehearsal from the command line (Titanic, 150 s, at most 6 experiments, seed 42, MLflow off, run in a temporary folder so the real records were not touched): it kept dropping `Cabin` (keep_simpler, +0.0082) and `Ticket` (keep, +0.0586, p(better) 0.008) and smaller trees (keep_simpler), discarded word features from `Name` (+0.0013, p(better) 0.467), stopped for time after 141 s, and raised the test F1 from 0.6949 to 0.7143 and the ROC AUC from 0.8026 to 0.8276. Record: `results/demo-rehearsal/session.json` | see the commit of `explanation.md` |
| 2026-11-15 | Presentation for the 5-minute review in `docs/imp/presentation/` (`.pptx` and a PDF copy), 10 slides in the department's prescribed order: title, introduction, literature review, problem statement, proposed methodology, system diagram with the flowchart and the Judge's rule, implementation, results, conclusion and future work, references. Built by `scripts/report/build_slides.py` (python-pptx); `scripts/report/ppt_finalize.ps1` adds fade transitions of 0.6 s and 43 fade or wipe entrance effects that each start after the previous one (one click per slide), and exports the PDF. The flowchart of report Figure 6 is redrawn with PowerPoint shapes so that its text stays legible; a new chart of the average ranks at 2 and 5 minutes comes from `results/benchmark-v2/analysis_ranks.csv`. Speaker notes: 608 words, about 4 minutes 20 seconds at 140 words a minute. All numbers are the report's measured ones | b81506b |
| 2026-11-15 | Results slide reworked, report unchanged: a slope chart of the average ranks from 2 to 5 minutes (report Table 9) with AutoPilot-ML full highlighted (3.58 to 2.50, the largest gain of the five systems; FLAML alone 1.50 to 2.17 stays first), and a card for the datasets whose mean test score rose from 120 to 300 s (full system 5 of 6, rules 4, FLAML alone 2, AutoGluon and the LightGBM baseline 0; from `results/benchmark-v2/analysis_per_dataset.csv`). Full test suite run again: 586 passed in 367.5 s (section 8.6) | e744653 |
| 2026-11-15 | Front-page details filled in. The work is the major project: MAJOR PROJECT REPORT on the report's cover, Major Project on the title slide, in the slide footers and the speaker notes, and B.Tech Major Project in `explanation.md` (the spec already said major project). The supervisor, Dr. Gagan Vishwakarma, Assistant Professor, Department of CSE, IIIT Bhopal, is on the cover, the certificate, the declaration, the acknowledgement and the title slide, which now reads him from `front.toml`. The Director's name, printed once in the acknowledgement, was already spelled correctly and is now written Dr. Ashutosh Kumar Singh (before: Prof.). Report rebuilt: 47 pages, 0 format problems, no placeholder left; presentation rebuilt: 10 slides, 43 entrance effects. Left for the team: the signatures and the certificate's date, the plagiarism report, and a list of publications only if any (Appendix D) | ab8f810, 41d2bc5, 4b1732a, 928b8ea, 2da8ee9 |
| 2026-11-15 | The Department Coordinator's name is spelled Dr. Rahul Chaurasia (before: Chourasiya, as in the department's template). It is printed twice, in the certificate's signature block and in the acknowledgement, and is not on the slides. Report rebuilt: 47 pages, 0 format problems | f8a9383 |
| 2026-11-15 | Final report revised at the department's request. The tools table (Table 5) no longer lists the development environment (Kiro) or python-docx; the statement that Kiro, an AI-assisted development environment, was used to write the specification, the code and the documents moved into section 4.1, so the report still declares it. Chapter 7 now says that everything except Docker Desktop is open source. All the text (abstract, acknowledgement, area of work and chapters 1 to 9) was rewritten in a more natural voice, with the team as "we" and shorter, more varied sentences; a script compared every chapter with 7e82880 and found the same numbers, citations, cross-references, tables, figures, equations, algorithms and headings. Rebuilt: 48 pages (40 numbered), 0 format problems, every figure and table referred to next to it, a citation in every paragraph of chapters 1 and 2. The explanation's copy of the tools table matches the report. No similarity or AI-detection score was measured: that needs the institute's tool | 7e82880, 6e7fb7c, 8c222df, cc6e20b |

---

## Appendix C. Figures and screenshots checklist

Diagrams (chapter 6): current block diagram, proposed block diagram, DFD level 0, DFD level 1, flowchart of one experiment, flowchart of the planned loop, sequence diagram (`architecture.md`).

Screenshots to take (not taken yet unless marked):

- [ ] Dashboard section 1: cards and the data quality tab (Titanic)
- [ ] Section 2: target, detected problem, class distribution
- [ ] Section 3 and 4: configuration and progress bar
- [ ] Section 5: result cards, train vs test table, leaderboard, search progress chart
- [ ] Section 6: permutation importance, confusion matrix, ROC curve (Titanic); residual plots (California housing)
- [ ] Section 7: experiment history table and chart
- [ ] MLflow experiment list with EXP-001 and EXP-002 (a screenshot was taken on 7 October 2026; keep a copy with the report material)
- [ ] MLflow run page: parameters, metrics, artifacts
- [ ] Terminal: `47 passed`
- [ ] Terminal: output of `scripts/compare_seeds.py`
- [ ] Autonomous run, settings and the live view while a Session runs (timeline, score chart, Stop)
- [ ] Autonomous run, a finished Session: the first and final model side by side ("Not used for decisions") and the Diagnosis cards
- [ ] Autonomous run: the memory records of the dataset
- [ ] MLflow: a Session's parent run with its Experiments as child runs
- [ ] Terminal: output of `scripts/run_autopilot.py` (SES-001) and of `scripts/run_phase2.py analyse`

Charts to make for chapter 9: test score against budget (9.5), per-seed differences (9.6), bootstrap intervals (9.3).

Charts made from the Phase 3 benchmark (in `results/benchmark/figures/`, 31 October 2026, from the 23 usable cells; descriptive only, section 9.16):

- [x] Critical-difference diagrams: `cd_main_120s.png`, `cd_main_300s.png`, `cd_ablations_300s.png`
- [x] Test score against budget per dataset: `score_against_budget.png`

Charts already made (in `docs/imp/images`, from `scripts/plot_search_history.py`, 22 October 2026):

- [x] EXP-001: every trial over time with the best so far (`exp001_search_timeline.png`)
- [x] EXP-001: search time per learner family (`exp001_time_per_learner.png`)
- [x] EXP-001: XGBoost's size settings and scores, trial by trial (`exp001_best_learner_settings.png`)

---

## Appendix D. Open items for the report

- Before submission (15 November 2026): the signatures and the certificate's date on the printed copy; the plagiarism report from the institute's tool, below 12% as the template asks, placed after the references; a "List of publications" page only if a paper is published, accepted or submitted; the guide's comments on the report. The guide's name, department and institute are filled in, and the template has no group number
- Exact date of the Phase 1 review, and the guide's feedback
- Final title (use the submitted one)
- Check the comparison table in chapter 2 against the papers
- Re-run the result tables with the final code, same seeds, and keep the raw files
- Render the diagrams as images
- Back up `experiments/` and `mlruns/` regularly (they are not in git)
- Check whether the department wants AI-assisted tools declared. The README now has a section on the AI tools used (task 28); use the department's wording if it has one
- Take the screenshots of Appendix C (they need the app open in a browser; none were taken by the tools that built it)
- The scikit-learn warning that filled the final smoke run's log is now dropped (section 4.7, 1 November 2026); its exact cause is not confirmed
- Read the papers of Appendix A.3 before citing them (chapter 2's "still to add")
- Design decisions D2 to D4 (spec `design.md`) were confirmed on 26 October 2026. D5 (splitting the work) is not needed: Priyanshu Verma builds all tasks alone. Credit Salil Kushwah (23U02003) as the second group member in the final report
- The spec in `.kiro/specs/autopilot-ml/` is outside the git repository, like the steering rule. Keep a copy with the report material, or move `.kiro` into the repository if the repository folder becomes the Kiro workspace
- Find out which warnings pytest listed once on 22 October: one of three identical test runs ended with "47 passed, 350 warnings", the other two with no warnings. The output of that run was not kept, so the warning types are unknown. All tests passed every time

---

## Appendix E. Where the data and documents are

- Raw results of EXP-001 and EXP-002: `experiments/EXP-001/`, `experiments/EXP-002/` (summary, predictions, leaderboard, trial history, importances, profile, FLAML log), `models/EXP-00x/model.pkl`, and the MLflow store in `mlruns/`. These folders are not in git, so back them up.
- Second benchmark run (section 9.20, 12 and 13 November 2026, code af8f14d): rows, frozen memory, analysis tables and figures in `results/benchmark-v2/` (in git); the Sessions' records in `experiments/benchmark-v2/` (grid), `experiments/benchmark-v2-memory/` (memory build) and `experiments/benchmark-v2-smoke/`, their models under `models/benchmark-v2*/`, and the logs with the run notes in `experiments/benchmark-v2-logs/` (not in git, so back them up).
- The development runs (9.2, 9.5), the rehearsal (9.4) and the seed checks (9.6) ran in temporary folders that were deleted afterwards. Only the numbers in this file remain; re-run them for the final report.
- Phase 2 Sessions: `experiments/SES-###/` (`session.json`, `train_profile.json`) and their Experiments `experiments/EXP-###/` (with `assessment.json`, `diagnoses.json`, `decision.json`), models in `models/EXP-###/`, MLflow parent and child runs in `mlruns/`, and the memory in `experiments/memory/memory.jsonl`. SES-001 is section 9.11; the Sessions of the Phase 2 measurement are listed in `results/phase2/phase2_runs.csv` (column `run_id`). Not in git: back them up.
- Component studies: summary CSV files in `results/studies/` (in git); the critic study's saved inputs in `experiments/studies/critic_study_inputs.jsonl` (not in git). Phase 2 measurement: `results/phase2/` (in git).
- Frozen memory of the benchmark: `results/benchmark/frozen_memory.jsonl`, `frozen_memory_summary.csv` and the build's rows `memory_sessions.csv` (in git); the build's Sessions and the growing memory in `experiments/benchmark-memory/` and `models/benchmark-memory/` (not in git: back them up).
- Judge agreement study (section 9.18): `results/studies/judge_agreement_pairs.csv`, `judge_agreement_summary.csv` and `judge_agreement_confusion.csv` (in git); the 200 searches in `experiments/studies/judge_agreement/` with `run.log`, and their models in `models/studies/judge_agreement/` (not in git: back them up).
- Benchmark (stopped after 33 cells, section 9.16): the rows that must not be scored, with the reason, in `results/benchmark/excluded_cells.csv`; the analysis of the 23 usable cells in `results/benchmark/analysis_*.csv` and `figures/`; the run's log in `experiments/benchmark/run.log`.
- Benchmark: one row per cell in `results/benchmark/benchmark_cells.csv` and the smoke run in `results/benchmark/smoke/` (in git); the Sessions and Experiments behind the rows in `experiments/benchmark/` and `models/benchmark/` (`benchmark-smoke/` for the smoke run), found by each row's `run_id` (not in git: back them up).
- Maintained documents: `docs/imp/` (this file, `PROJECT.md`, `how-it-works.md`, `architecture.md`, `development-roadmap.md`, `research-direction.md`) and the charts in `docs/imp/images/`.
- Archived Phase 1 material: `docs/unimp/PRESENTATION.md` (demo guide; section 7 concepts, section 15 viva questions) and `docs/unimp/Project_Presentation.md` (text presented at the review).
