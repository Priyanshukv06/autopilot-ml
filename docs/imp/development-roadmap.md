# Development roadmap

This page is the short version of the plan. The detailed, task-by-task plan is the Kiro spec in the workspace folder `.kiro/specs/autopilot-ml/` (one level above this repository):

- `requirements.md`: what each part must do, as numbered acceptance criteria (Requirements 1 to 32).
- `design.md`: how it is built: components, code shapes, rules, thresholds, data formats, and the decisions D1 to D8.
- `tasks.md`: the ordered tasks, each one commit, with tests, requirement links and a dependency graph.

If this page and the spec disagree, the spec is right, and this page should be fixed. There are no dates: the order of the tasks is what matters.

## Phase 1: Baseline (done)

- [x] Conda environment, pinned dependencies, project structure
- [x] `DatasetProfiler`: profile without any UI dependency, quality flags, class balance, meta-features
- [x] `TaskDetector`: binary, multiclass and regression detection, checking overrides, target candidates
- [x] `AutoMLRunner` + `FLAMLEngine`: split, search, leaderboard, trial history, saving the model
- [x] `Evaluator`: task metrics, choice of primary metric, trivial baselines, imbalance warning, permutation importance
- [x] `ExperimentTracker`: local MLflow (SQLite) runs and JSON records
- [x] Streamlit dashboard with seven sections, including the experiment history
- [x] pytest tests for the profiler, task detector, runner and evaluator

Done when: a CSV can go from upload to a tracked, explained model that can be reloaded, both from the UI and from Python, and the tests pass. This is met, and the Phase 1 review was completed in October 2026.

## Phase 2: Rule-based autonomous loop (done)

Goal: given a dataset, a target and a budget, the system runs the loop by itself: diagnose, propose a change, test it fairly on 15 folds of the training data, keep or drop it, and stop within the budget. The test part is used only once, at the end.

| Milestone | Tasks in `tasks.md` | Result |
|---|---|---|
| M0 Groundwork | 1 to 4.1 | FLAML features checked; session splits; engine extensions (warm start, search-space limits, class weights, refit of one configuration); synthetic data generator |
| M1 Thin loop | 5 to 11.2 | Assessor, Critic (3 diagnoses), 3 strategies with the rule engine, Judge, Resource_Manager, Session records, Planner and loop; a first real Session on Titanic from the command line |
| M2 Full Phase 2 | 12 to 17 | Data changes learned inside folds, the full Critic and catalogue, Experiment_Memory v1, the critic study (precision and recall) and the judge study (false keeps and power) |
| M3 Show and measure | 18 to 19 | Dashboard autonomous mode; the loop against FLAML alone at the same budget; documents updated |

Done when: the loop runs unattended on the development datasets within the budget; every decision is recorded with its evidence, effect and cost; nothing in the decision path reads the test part (checked by a test); the critic study, the judge study and the Phase 2 measurement are in the report notes; all tests pass.

Status (27 October 2026): done. All milestones are built and tested (468 tests). Phase 2 measurement, 4 sample datasets x 3 seeds at 300 s: the loop's final test scores matched FLAML alone on average (4 better, 4 equal, 4 worse) using about two thirds of the time (report notes, section 9.14).

## Phase 3: Learning and evaluation (done, with shortened runs)

Tasks 20 to 28: development and evaluation datasets (fixed in advance and kept apart), learned strategy choice (Thompson sampling and UCB1 over past outcomes from similar datasets), ablation switches, comparison systems (manual baseline, FLAML alone, AutoGluon), a resumable benchmark (5 seeds; budgets of 2 and 5 minutes, since the 10-minute budget was dropped on 27 October 2026 to fit the laptop), the statistics (Friedman and Nemenyi, Wilcoxon with Holm's correction, effect sizes, time-to-target), the judge agreement study, and the final documents.

Status (31 October 2026): tasks 20 to 28 are done (541 tests). The benchmark stopped after 33 of its 390 cells, when the laptop went into standby on battery; 23 cells are usable, and the user decided not to run the rest. The judge agreement study ran on 20 decisions instead of 57. Three flaws found in the records were fixed afterwards (report notes, chapter 10). Next, if wanted: the full benchmark on mains power, and the optional tasks below (built by 12 November 2026).

## Optional (after Phase 3, done)

Tasks 29 to 35, each one commit with its tests:

- [x] Task 29: SMOTE through FLAML's resampler, as the `smote` Strategy (training rows only)
- [x] Task 30: cross-fitted target encoding and word features, as Data_Steps and Strategies (`target_encode`, `token_features`), and the Critic's `unused_words` Diagnosis, which can bring back Titanic's `Name` using only its words
- [x] Task 31: group-aware outer split and Session_Folds (`SessionSettings.group_column`, `--group-column`, a dashboard option)
- [x] Task 32: SHAP explanations of a Session's final model on training-part rows (a dashboard section, `scripts/explain_session.py`)
- [x] Task 33: target suggestion with reasons, and the single-column leakage check with the warning-only `possible_leakage` Diagnosis (`scripts/check_single_columns.py`)
- [x] Task 34: plain-language Session summaries by a local language model through Ollama, labelled as generated text and never used for decisions
- [x] Task 35: a local HTTP API (FastAPI, on 127.0.0.1) and a Docker image

Status (12 November 2026): done (586 tests passed in 276.1 s). The catalogue now has 12 Strategies and the Critic 14 kinds of Diagnoses. Each part was checked with short runs (report notes, section 5.5); the benchmark ran before them, so none of its results include them.

Next: a second, full benchmark run with the final code. `scripts/build_memory.py` and `scripts/run_benchmark.py` take `--label v2`, so that run writes to its own folders and cannot touch the first run's rows or its frozen memory. It has not been run yet.

## Decisions

Confirmed on 26 October 2026 (details in `design.md`):

- D2: the Judge refits fixed configurations on the Session's folds and uses the corrected resampled t-test. The judge study (task 17) set alpha to 0.20 (0.10 before). The default stays 3 repeats of 5 folds (15 folds): 5 repeats accepted the same changes at five thirds of the cost.
- D3: small public OpenML datasets are downloaded once; `imbalanced-learn` is added only for the optional SMOTE task (done in task 29: `imbalanced-learn==0.14.2`).
- D4: AutoGluon gets its own conda environment if it does not install next to our pinned libraries. Checked at the start of task 23: it would replace pandas 3.0.6 by 2.3.3, so it runs from the environment `autogluon` as a separate process.
- D5: not needed. Priyanshu Verma builds all tasks; Salil Kushwah is credited in the final report.

## Commits

One logical change per commit, with conventional commit messages, for example:

```text
feat: add session splits
feat: add the judge
test: cover the information barrier
docs: describe the critic rules
```
