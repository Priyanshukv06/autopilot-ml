# Development roadmap

Each phase ends with something we can check. Only Phase 1 is done. Everything after it is planned.

## Phase 1: Baseline (done)

- [x] Conda environment, pinned dependencies, project structure
- [x] `DatasetProfiler`: profile without any UI dependency, quality flags, class balance, meta-features
- [x] `TaskDetector`: binary, multiclass and regression detection, checking overrides, target candidates
- [x] `AutoMLRunner` + `FLAMLEngine`: split, search, leaderboard, trial history, saving the model
- [x] `Evaluator`: task metrics, choice of primary metric, trivial baselines, imbalance warning, permutation importance
- [x] `ExperimentTracker`: local MLflow (SQLite) runs and JSON records
- [x] Streamlit dashboard with seven sections, including the experiment history
- [x] pytest tests for the profiler, task detector, runner and evaluator

Done when: a CSV can go from upload to a tracked, explained model that can be reloaded, both from the UI and from Python, and the tests pass. This is met.

## Phase 2: Rule-based autonomous loop (next)

1. **Critic** (`critic.py`): structured diagnoses for overfitting, underfitting, imbalance, missingness, high cardinality, low variance, redundancy and excessive time, with thresholds kept in config.
   Done when: tests on synthetic datasets trigger each diagnosis on purpose, and the diagnoses are saved in `summary.json`.
2. **Strategy catalogue**: class weights, resampling (SMOTE, which needs `imbalanced-learn` added), dropping or selecting columns, encoding high-cardinality columns, a search space biased towards regularisation, a restricted model list.
   Done when: each strategy is a tested change to `ExperimentConfig` or to the data.
3. **Strategy engine v1** (`strategy_engine.py`): a rule table from diagnosis to candidate strategies, ranked by a fixed priority.
   Done when: the same diagnosis always gives the same ranking, and the rule table is documented.
4. **Planner and loop** (`experiment_planner.py`): baseline run, then critic, then strategy, then the next run, until the budget is used up or nothing improves.
   Done when: the loop runs on all four sample datasets without any input, and stays within the budget.
5. **Experiment memory v1** (`experiment_memory.py`): stores meta-features, diagnoses, strategy, model, config, change in score and cost.
   Done when: records survive a restart, and the k most similar past datasets can be retrieved.
6. **Resource manager v1** (`resource_manager.py`): total time, maximum experiments, CPU cores (`n_jobs`), and the split of the budget between experiments.
   Done when: the budget is never exceeded by more than the retraining time, and any overrun is logged.

## Phase 3: Learning and evaluation

1. Strategy selection as a multi-armed bandit, with reward = change in score per second of compute. Compared against the fixed rules on regret and time to target.
2. Memory-based warm start: try the strategies that helped on similar datasets first. Checked with an ablation, with and without memory, on datasets the memory has not seen.
3. `AutoGluonEngine`, implementing the same engine interface, so the same benchmark can run FLAML and AutoGluon.
4. Benchmark setup: dataset suite, budgets, seeds, result tables and statistical tests, all from one script, with the raw results saved as CSV.
5. Ablation study and write-up (see `research-direction.md`).

## Phase 4: Later

- An optional local LLM (through Ollama) to explain diagnoses and strategy choices in plain language. It will not replace the measurable decision logic.
- SHAP explanations, automatic target detection, leakage checks.
- FastAPI service, Docker image, PostgreSQL-backed tracking.

## Commits

One logical change per commit, with conventional commit messages, for example:

```text
feat: add autonomous critic
feat: add strategy selection
feat: add experiment memory
test: cover overfitting diagnosis
docs: describe strategy rule table
```
