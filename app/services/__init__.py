"""Business logic of AutoPilot-ML, independent of the Streamlit UI.

Single experiment (Phase 1 baseline):
    profiler            - DatasetProfiler: dataset profile, quality flags, frequent words, leakage scores
    task_detector       - TaskDetector: binary / multiclass / regression detection, target suggestions
    automl_runner       - AutoMLRunner + FLAMLEngine: split, search, evaluate, save
    evaluator           - Evaluator: metrics, baselines, permutation importance
    experiment_tracker  - ExperimentTracker: MLflow logging, JSON records, experiment and Session IDs

Autonomous layer (Phases 2 and 3):
    autopilot           - AutopilotSession: the Session loop, its records and the Final_Evaluation
    splits              - outer split and Session_Folds (group-aware when a group column is given)
    experiment_planner  - Planner: the first Setup and each candidate Setup
    assessor            - Assessor: refits a Setup's best configuration on the Session_Folds
    search_summary      - summarize_search: how one FLAML search spent its time
    critic              - Critic: 14 kinds of Diagnoses with evidence (thresholds in critic_thresholds.toml)
    strategies          - Strategy_Catalogue: the 12 changes a Session may make to a Setup
    strategy_engine     - rule table and selectors (rules, random, thompson, ucb1) that rank Proposals
    transforms          - Data_Steps, learned on training rows only, and the model wrapper that applies them
    judge               - Judge: corrected resampled t-test; keep, keep_simpler or discard
    resource_manager    - ResourceManager: the budget of each step and the stop rules
    experiment_memory   - ExperimentMemory: one record per Verdict, nearest datasets, Strategy statistics

Optional extensions (tasks 29 to 35):
    resampling          - FoldSafeSmote: SMOTE on the training rows FLAML hands a resampler
    explainer           - SHAP explanations of a Session's final model on Training_Part rows
    narrator            - labelled plain-language Session summaries by a local model (Ollama)

The local HTTP API (task 35) is ``app/api.py``, outside this package.
"""
