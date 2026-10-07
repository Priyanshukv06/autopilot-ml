"""Business logic of AutoPilot-ML, independent of the Streamlit UI.

Current baseline modules:
    profiler            - DatasetProfiler: structured dataset profile
    task_detector       - TaskDetector: binary / multiclass / regression detection
    automl_runner       - AutoMLRunner + FLAMLEngine: split, search, evaluate, save
    evaluator           - Evaluator: metrics, baselines, permutation importance
    experiment_tracker  - ExperimentTracker: MLflow logging + local JSON index

Planned modules (not implemented yet): experiment_planner, critic,
strategy_engine, experiment_memory, resource_manager.
"""
