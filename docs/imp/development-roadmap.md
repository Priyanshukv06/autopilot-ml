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

## Phase 3: Learning and evaluation (next)

Tasks 20 to 28: development and evaluation datasets (fixed in advance and kept apart), learned strategy choice (Thompson sampling and UCB1 over past outcomes from similar datasets), ablation switches, comparison systems (manual baseline, FLAML alone, AutoGluon), a resumable benchmark (budgets of 2, 5 and 10 minutes, at least 5 seeds), the statistics (Friedman and Nemenyi, Wilcoxon with Holm's correction, effect sizes, time-to-target), the judge agreement study, and the final documents.

## Optional (after Phase 3)

Tasks 29 to 35: SMOTE through FLAML's resampler, target encoding and token features, group-aware folds, SHAP, target suggestion and a leakage check, local language-model summaries (never used for decisions), an API and a Docker image.

## Decisions

Confirmed on 26 October 2026 (details in `design.md`):

- D2: the Judge refits fixed configurations on the Session's folds and uses the corrected resampled t-test. The judge study (task 17) set alpha to 0.20 (0.10 before). The default stays 3 repeats of 5 folds (15 folds): 5 repeats accepted the same changes at five thirds of the cost.
- D3: small public OpenML datasets are downloaded once; `imbalanced-learn` is added only for the optional SMOTE task.
- D4: AutoGluon gets its own conda environment if it does not install next to our pinned libraries (checked at the start of task 23).
- D5: not needed. Priyanshu Verma builds all tasks; Salil Kushwah is credited in the final report.

## Commits

One logical change per commit, with conventional commit messages, for example:

```text
feat: add session splits
feat: add the judge
test: cover the information barrier
docs: describe the critic rules
```
