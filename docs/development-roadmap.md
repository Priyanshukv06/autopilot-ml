# Development Roadmap

Each phase ends with a measurable exit criterion. Phases after Phase 1 are **Planned** and nothing in them is implemented yet.

## Phase 1 — Baseline (Done)

- [x] Conda environment, pinned dependencies, project skeleton
- [x] `DatasetProfiler` (UI-independent profile, quality flags, class balance, meta-features)
- [x] `TaskDetector` (binary / multiclass / regression, override validation, target candidates)
- [x] `AutoMLRunner` + `FLAMLEngine` (split, search, leaderboard, history, model saving)
- [x] `Evaluator` (task metrics, configurable primary metric, trivial baselines, imbalance warning, permutation importance)
- [x] `ExperimentTracker` (local MLflow SQLite runs + JSON index)
- [x] Streamlit dashboard with seven sections, including experiment history
- [x] pytest suite for profiler, task detector, runner and evaluator

Exit criterion (met): a CSV can be taken from upload to a tracked, explained, reloadable model through the UI and from Python; tests pass.

## Phase 2 — Rule-based autonomous loop (Planned)

| Step | Deliverable | Exit criterion |
|---|---|---|
| 2.1 | `critic.py`: structured diagnoses (overfitting, underfitting, imbalance, missingness, high cardinality, low variance, redundancy, excessive time) with thresholds in config | Unit tests with synthetic datasets that trigger each diagnosis; diagnoses stored in `summary.json` |
| 2.2 | Strategy catalogue: `class_weight`, resampling (SMOTE, after adding `imbalanced-learn`), feature dropping/selection, high-cardinality encoding, regularisation-biased search spaces, model-family restriction | Each strategy is a pure transformation of `ExperimentConfig` (or of the data) with a test |
| 2.3 | `strategy_engine.py` v1: rule table diagnosis → candidate strategies, ranked by a fixed priority | Deterministic ranking for a given diagnosis; documented rule table |
| 2.4 | `experiment_planner.py` + loop: baseline run → critic → strategy → next run, until budget or no improvement | Loop runs unattended on all sample datasets within the user budget |
| 2.5 | `experiment_memory.py` v1: store (meta-features, diagnoses, strategy, model, config, Δscore, cost) | Records survive restarts; retrieval returns the k most similar past datasets |
| 2.6 | `resource_manager.py` v1: total time, max experiments, CPU cores (`n_jobs`), per-experiment budget split | Budget is never exceeded by more than the retraining overhead; overruns logged |

## Phase 3 — Learning and evaluation (Planned)

| Step | Deliverable | Exit criterion |
|---|---|---|
| 3.1 | Strategy selection as a contextual multi-armed bandit (reward = Δscore per compute second) | Regret / time-to-target compared against the fixed-priority rules |
| 3.2 | Memory-based warm start: prioritise strategies that helped on similar datasets | Ablation "with vs without memory" on held-out datasets |
| 3.3 | `AutoGluonEngine` implementing `AutoMLEngine` | Same benchmark harness runs FLAML and AutoGluon |
| 3.4 | Benchmark harness: dataset suite, budgets, seeds, result tables, statistical tests | Reproducible script; raw results committed as CSV |
| 3.5 | Ablation study and write-up | Tables for the six baselines; see `research-direction.md` |

## Phase 4 — Future Work

- Optional local LLM (Ollama) to explain diagnoses and strategy choices in plain language. It must not replace the measurable decision logic.
- SHAP explanations, automatic target detection, leakage checks.
- FastAPI service, Docker image, PostgreSQL-backed tracking.

## Commit conventions

Conventional commits, one logical change per commit, for example:

```text
feat: add autonomous critic
feat: add strategy selection
feat: add experiment memory
test: cover overfitting diagnosis
docs: describe strategy rule table
```
