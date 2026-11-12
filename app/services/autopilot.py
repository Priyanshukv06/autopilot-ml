"""Autonomous Session: search, assess, diagnose, propose, plan, search the candidate, judge, repeat.

One Session (Requirement 12) makes one Training_Part and one Test_Part, and fixed Session_Folds of
the Training_Part. Every Experiment searches the Training_Part only and every decision is made from
Training_Part information: the Assessor refits on the folds, the Critic reads the Training_Part
profile and the Assessment, the Strategy_Engine reads the Diagnoses, and the Judge compares two
Assessments. The Test_Part is used once, at the end, by the Final_Evaluation, which scores the first
and the final model.

``session.json`` is rewritten after every step, so a crash keeps the finished steps. The inputs of
every decision are saved (the Training_Part profile in ``train_profile.json``, each Experiment's
``summary.json`` and ``assessment.json``), so the Diagnoses and Verdicts can be replayed.
"""

from __future__ import annotations

import copy
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.services.assessor import Assessment, Assessor
from app.services.automl_runner import (
    AutoMLEngine,
    AutoMLRunner,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    FLAMLEngine,
    to_jsonable,
    unseen_category_warnings,
)
from app.services.critic import Critic, CriticThresholds, Diagnosis, DiagnosisKind, ThresholdsError
from app.services.evaluator import Evaluator
from app.services.experiment_memory import ExperimentMemory, MemoryRecord, dataset_fingerprint
from app.services.experiment_planner import Planner
from app.services.experiment_tracker import ExperimentTracker, write_json_atomic
from app.services.judge import Judge, JudgeDecision, JudgeSettings
from app.services.profiler import DatasetProfile, DatasetProfiler
from app.services.resource_manager import KEPT, EqualShareManager, ResourceManager, SessionBudget, StopReason
from app.services.search_summary import SearchSummary
from app.services.splits import OuterSplit, SessionFolds, SplitError, make_outer_split, make_session_folds
from app.services.strategy_engine import (
    SELECTORS,
    Proposal,
    SelectionContext,
    StrategySelector,
    propose,
    random_proposals,
)
from app.services.task_detector import TaskType

logger = get_logger(__name__)

SESSION_SCHEMA_VERSION = 1
CODE_ROOT = Path(__file__).resolve().parents[2]  # the repository of this code, for the code version
Progress = Callable[[str, int, str], None]
RecordUpdate = Callable[[dict[str, Any]], None]  # receives a copy of session.json after every save
AssessorFactory = Callable[..., Assessor]
_LIBRARIES = ("flaml", "scikit-learn", "pandas", "numpy", "lightgbm", "xgboost", "mlflow")


@dataclass
class SessionSettings:
    base: ExperimentConfig  # target, task, primary metric, seed, test share, exclusions
    budget: SessionBudget
    n_splits: int = 5
    n_repeats: int = 3  # kept after the judge study (task 17): 5 accepted the same changes at 5/3 of the cost
    judge: JudgeSettings = field(default_factory=JudgeSettings)
    thresholds_path: Path | None = None
    selector: str = "rules"  # rules, random, thompson or ucb1
    use_memory: bool = False  # let the selector read the Experiment_Memory (Requirement 19)
    # Ablation switches (Requirement 20). Without the Critic, every change is a random valid one; without
    # the Resource_Manager, every search gets an equal share and only the time and Experiment limits stop.
    use_critic: bool = True
    use_resource_manager: bool = True
    # Optional task 31 (Requirement 28): rows with the same value of this column stay together, in the outer
    # split and in every Session_Fold; the column itself is never a feature.
    group_column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": to_jsonable(self.base.__dict__),
            "budget": self.budget.to_dict(),
            "n_splits": self.n_splits,
            "n_repeats": self.n_repeats,
            "judge": asdict(self.judge),
            "thresholds_path": str(self.thresholds_path) if self.thresholds_path else None,
            "selector": self.selector,
            "use_memory": self.use_memory,
            "use_critic": self.use_critic,
            "use_resource_manager": self.use_resource_manager,
            "group_column": self.group_column,
        }


@dataclass
class SessionStep:
    index: int
    experiment_id: str | None
    parent_experiment_id: str | None
    proposal: dict[str, Any] | None
    diagnoses: list[dict[str, Any]]  # Diagnoses of the Setup the proposal tried to improve
    assessment: dict[str, Any] | None  # brief: means and standard deviations of the primary metric
    decision: dict[str, Any] | None  # JudgeDecision as a dict; None for the first Experiment
    accepted: bool
    cost: dict[str, Any] | None  # StepCost as a dict
    status: Literal["completed", "failed"]
    setup_key: str | None = None
    error: str | None = None
    selection: dict[str, Any] | None = None  # how the selector placed this step's Proposal (arm, score, ranks)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class SessionResult:
    session_id: str
    status: Literal["completed", "failed"]
    settings: dict[str, Any]
    steps: list[SessionStep]
    first_experiment_id: str | None
    final_experiment_id: str | None
    stop_reason: StopReason
    final_evaluation: dict[str, Any] | None  # test results of the first and the final model
    total_seconds: float
    error: str | None = None


@dataclass
class _Accepted:
    """The Accepted_Setup with its search result, Assessment and Diagnoses."""

    config: ExperimentConfig
    result: ExperimentResult
    assessment: Assessment
    diagnoses: list[Diagnosis]

    @property
    def search(self) -> SearchSummary | None:
        return SearchSummary(**self.result.search_summary) if self.result.search_summary else None


@dataclass
class _Context:
    """What every step of one Session needs."""

    session_id: str
    dataframe: pd.DataFrame
    split: OuterSplit
    train_frame: pd.DataFrame
    y_train: pd.Series
    profile: DatasetProfile
    task_type: TaskType
    metric: str
    assessor: Assessor
    critic: Critic
    judge: Judge
    rm: ResourceManager
    notify: Progress
    dataset_fingerprint: str = ""
    use_critic: bool = True
    # the columns the first Setup left out by itself (recorded as first_setup_exclusions), never the user's
    restorable: tuple[str, ...] = ()

    def diagnose(self, config: ExperimentConfig, result: ExperimentResult, assessment: Assessment) -> list[Diagnosis]:
        if not self.use_critic:  # ablation: no Diagnoses at all
            return []
        search = SearchSummary(**result.search_summary) if result.search_summary else None
        return self.critic.diagnose(profile=self.profile, config=config, assessment=assessment, search=search,
                                    task_type=self.task_type, primary_metric=self.metric,
                                    restorable=self.restorable)


class AutopilotSession:
    def __init__(
        self,
        settings: Settings | None = None,
        engine: AutoMLEngine | None = None,
        tracker: ExperimentTracker | None = None,
        clock: Callable[[], float] = time.monotonic,
        enable_mlflow: bool = True,
        assessor_factory: AssessorFactory = Assessor,
        memory: ExperimentMemory | None = None,
    ) -> None:
        """``memory``: where every Verdict is recorded; by default ``experiments/memory/memory.jsonl``."""
        self.settings = settings or get_settings()
        self.engine = engine or FLAMLEngine()
        self.tracker = tracker or ExperimentTracker.from_settings(self.settings)
        self.clock = clock
        self.enable_mlflow = enable_mlflow
        self.assessor_factory = assessor_factory
        self.memory = memory or ExperimentMemory(self.settings.experiments_dir / "memory" / "memory.jsonl")
        self.runner = AutoMLRunner(settings=self.settings, engine=self.engine, tracker=self.tracker,
                                   enable_mlflow=enable_mlflow)
        self.planner = Planner()

    # ------------------------------------------------------------------ run
    def run(
        self,
        dataframe: pd.DataFrame,
        session: SessionSettings,
        progress: Progress | None = None,
        stop_requested: Callable[[], bool] | None = None,
        on_update: RecordUpdate | None = None,
    ) -> SessionResult:
        """``on_update`` is called with a copy of the session record each time ``session.json`` is
        written (for the dashboard's live view); an error in it is logged and never stops the Session."""
        notify = progress or (lambda stage, step, message: None)
        stop_requested = stop_requested or (lambda: False)
        thresholds, selector = self._check(session)

        # 1. Data, the outer split and the Session_Folds. Errors here mean nothing is written.
        notify("preparing", 0, "Checking the data and making the splits")
        base = replace(copy.deepcopy(session.base), n_jobs=session.budget.n_jobs)
        group = session.group_column
        if group is not None:  # the grouping column is never a feature, so it becomes one of the user's exclusions
            if group not in dataframe.columns:
                raise ExperimentError(f"Grouping column '{group}' not found in the dataset.")
            if group == base.target_column:
                raise ExperimentError("The grouping column cannot be the target column.")
            if group not in base.excluded_columns:
                base = replace(base, excluded_columns=[*base.excluded_columns, group])
        _, y, task_type, metric, notes = self.runner.prepare(dataframe, base)
        base = replace(base, task_type=task_type, primary_metric=metric)
        try:
            groups = dataframe.loc[y.index, group] if group is not None else None
            split, split_notes = make_outer_split(y, task_type, base.test_size, base.random_seed, groups=groups)
            labelled = dataframe.loc[y.index]
            train_frame, y_train = labelled.iloc[split.train_positions], y.iloc[split.train_positions]
            folds, fold_notes = make_session_folds(
                y_train, task_type, session.n_splits, session.n_repeats, base.random_seed + 1,
                groups=None if groups is None else groups.iloc[split.train_positions])
        except SplitError as exc:
            raise ExperimentError(str(exc)) from exc
        if group is not None:
            notes = [*notes, f"Rows are grouped by '{group}', which is left out of the features."]
        # with the single-column check of the Training_Part, for the possible_leakage warning (optional task 33)
        profile = DatasetProfiler().profile(train_frame, base.target_column, base.dataset_name, task_type,
                                            check_leakage=True)

        # 2. Records.
        session_id = self.tracker.allocate_session_id()
        rm = (ResourceManager if session.use_resource_manager else EqualShareManager)(session.budget, self.clock)
        first_exclusions = self.planner.safe_exclusions(base, profile)
        state = _State(self, session_id, session, base, thresholds, split, folds,
                       [*notes, *split_notes, *fold_notes], first_exclusions, on_update=on_update)
        write_json_atomic(self.tracker.session_dir(session_id) / "train_profile.json", to_jsonable(profile.to_dict()))
        state.save()
        state.start_mlflow(base)
        notify("started", 0, f"{session_id}: {split.n_train} training rows, {split.n_test} test rows, "
                             f"{len(folds)} folds")
        ctx = _Context(
            session_id=session_id, dataframe=dataframe, split=split, train_frame=train_frame, y_train=y_train,
            profile=profile, task_type=task_type, metric=metric,
            assessor=self.assessor_factory(self.engine, folds, task_type, metric, session.budget.n_jobs,
                                           base.random_seed),
            critic=Critic(thresholds), judge=Judge(session.judge, test_train_ratio=folds.test_train_ratio),
            rm=rm, notify=notify, dataset_fingerprint=dataset_fingerprint(dataframe, base.target_column),
            use_critic=session.use_critic, restorable=tuple(first_exclusions),
        )

        # 3-4. The first Experiment, its Assessment and its Diagnoses.
        first = self._first_step(ctx, state, replace(self.planner.first_setup(base, profile),
                                                     time_budget=rm.first_budget()))
        if first is None:
            return state.result()
        accepted = first

        # 5. The loop: propose, budget, plan, search, assess, judge, update.
        selection = SelectionContext(
            memory=self.memory if session.use_memory else None, meta_features=profile.meta_features(),
            task_type=task_type, fingerprint=ctx.dataset_fingerprint, rng=np.random.default_rng(base.random_seed),
        )
        tried: set[str] = set()
        while True:
            if stop_requested():
                rm.request_stop()
            if session.use_critic:
                proposals = selector.rank(propose(accepted.diagnoses, accepted.config, profile, task_type, tried),
                                          selection)
            else:  # ablation: a random valid change, whatever the data looks like
                families = list(accepted.config.estimator_list or accepted.result.best_config_per_estimator)
                proposals = random_proposals(accepted.config, accepted.result.feature_columns, families, profile,
                                             task_type, tried, selection.rng)
            stop = rm.check_stop(len(proposals))
            if stop is not None:
                break
            proposal = proposals[0]
            placed = selector.explain(proposal) if session.use_critic else {"selector": "random (no Critic)", "rank": 0}
            wasted = any(d.kind is DiagnosisKind.WASTED_BUDGET for d in accepted.diagnoses)
            budget = rm.next_budget(proposal, accepted.search, wasted)
            if isinstance(budget, StopReason):
                stop = budget
                break
            tried.add(proposal.spec.key())
            accepted = self._candidate_step(ctx, state, len(state.steps), proposal, budget, accepted, placed)

        # 6. The Final_Evaluation: the only use of the Test_Part.
        notify("final_evaluation", len(state.steps), "Scoring the first and the final model on the test part")
        final_evaluation = self._final_evaluation(labelled, y, split, train_frame, first.result, accepted.result,
                                                  task_type, metric, base.random_seed)
        state.finish(stop, final_evaluation, first, accepted)
        notify("done", len(state.steps), f"{session_id} finished: {stop.value}")
        return state.result()

    # ---------------------------------------------------------------- steps
    def _first_step(self, ctx: _Context, state: _State, setup: ExperimentConfig) -> _Accepted | None:
        """Search, assess and diagnose the first Setup. On failure the Session stops (returns None)."""
        ctx.notify("experiment", 0, f"First Experiment ({setup.time_budget} s search)")
        started = self.clock()
        result = None
        try:
            result = self._search(ctx, setup, parent=None, proposal=None)
            searched = self.clock()
            ctx.notify("assessing", 0, f"Assessing {result.experiment_id} on {len(ctx.assessor.folds)} folds")
            assessment = ctx.assessor.assess(ctx.train_frame[result.feature_columns], ctx.y_train, setup, result)
            if assessment.failed:
                raise ExperimentError(f"The Assessment failed: {assessment.error}")
            assessed = self.clock()
            diagnoses = ctx.diagnose(setup, result, assessment)
        except Exception as exc:
            logger.exception("The first Experiment of %s failed", ctx.session_id)
            experiment_id = result.experiment_id if result is not None else getattr(exc, "experiment_id", None)
            state.fail(f"{type(exc).__name__}: {exc}", experiment_id, ctx.rm)
            return None
        cost = ctx.rm.record(budget_seconds=setup.time_budget, search_wall_seconds=searched - started,
                             assessment_seconds=assessed - searched, other_seconds=self.clock() - assessed,
                             gain=None, verdict=None)
        self._save_step_files(result.experiment_id, assessment, diagnoses, None)
        state.add_step(SessionStep(
            index=0, experiment_id=result.experiment_id, parent_experiment_id=None, proposal=None,
            diagnoses=[d.to_dict() for d in diagnoses], assessment=_brief(assessment, ctx.metric), decision=None,
            accepted=True, cost=cost.to_dict(), status="completed", setup_key=result.setup_key,
        ))
        state.record["first_experiment_id"] = result.experiment_id
        ctx.notify("step", 0, _step_text(state.steps[-1]))
        return _Accepted(setup, result, assessment, diagnoses)

    def _candidate_step(self, ctx: _Context, state: _State, index: int, proposal: Proposal, budget: int,
                        accepted: _Accepted, selection: dict[str, Any] | None = None) -> _Accepted:
        """Plan, search, assess and judge one candidate; return the Accepted_Setup after the step."""
        ctx.notify("experiment", index, f"Trying: {_change(proposal.to_dict())} ({budget} s search)")
        started = self.clock()
        searched = assessed = None
        result = assessment = None
        try:
            candidate = self.planner.candidate(accepted.config, accepted.result, proposal, budget)
            result = self._search(ctx, candidate, parent=accepted.result.experiment_id, proposal=proposal.to_dict())
            searched = self.clock()
            ctx.notify("assessing", index, f"Assessing {result.experiment_id} on {len(ctx.assessor.folds)} folds")
            assessment = ctx.assessor.assess(ctx.train_frame[result.feature_columns], ctx.y_train, candidate, result)
            assessed = self.clock()
            if assessment.failed:
                raise ExperimentError(f"The Assessment failed: {assessment.error}")
            decision = ctx.judge.compare(accepted.assessment, assessment, proposal.complexity == "simpler")
        except Exception as exc:  # the candidate is discarded and the Session continues (Requirement 12.5)
            logger.warning("Step %d of %s failed: %s", index, ctx.session_id, exc)
            now = self.clock()
            cost = ctx.rm.record(
                budget_seconds=budget, search_wall_seconds=(searched or now) - started,
                assessment_seconds=((assessed or now) - searched) if searched else 0.0,
                other_seconds=(now - assessed) if assessed else 0.0, gain=0.0, verdict="discard",
            )
            if result is not None and assessment is not None:
                self._save_step_files(result.experiment_id, assessment, None, None)
            state.add_step(SessionStep(
                index=index,
                experiment_id=result.experiment_id if result is not None else getattr(exc, "experiment_id", None),
                parent_experiment_id=accepted.result.experiment_id, proposal=proposal.to_dict(),
                diagnoses=[d.to_dict() for d in accepted.diagnoses],
                assessment=_brief(assessment, ctx.metric) if assessment is not None else None, decision=None,
                accepted=False, cost=cost.to_dict(), status="failed",
                setup_key=result.setup_key if result is not None else None, error=f"{type(exc).__name__}: {exc}",
                selection=selection,
            ))
            ctx.notify("step", index, _step_text(state.steps[-1]))
            return accepted

        kept = decision.verdict in KEPT
        new_diagnoses = ctx.diagnose(candidate, result, assessment) if kept else None
        cost = ctx.rm.record(budget_seconds=budget, search_wall_seconds=searched - started,
                             assessment_seconds=assessed - searched, other_seconds=self.clock() - assessed,
                             gain=decision.mean_difference if kept else 0.0, verdict=decision.verdict)
        self._save_step_files(result.experiment_id, assessment, new_diagnoses, decision)
        state.add_step(SessionStep(
            index=index, experiment_id=result.experiment_id, parent_experiment_id=accepted.result.experiment_id,
            proposal=proposal.to_dict(), diagnoses=[d.to_dict() for d in accepted.diagnoses],
            assessment=_brief(assessment, ctx.metric), decision=decision.to_dict(), accepted=kept,
            cost=cost.to_dict(), status="completed", setup_key=result.setup_key, selection=selection,
        ))
        self._remember(ctx, proposal, result, decision, cost.wall_seconds, cost.utility)
        ctx.notify("step", index, _step_text(state.steps[-1]))
        return _Accepted(candidate, result, assessment, new_diagnoses) if kept else accepted

    def _remember(self, ctx: _Context, proposal: Proposal, result: ExperimentResult, decision: JudgeDecision,
                  cost_seconds: float, utility: float | None) -> None:
        """One Experiment_Memory record per Verdict (Requirement 14.1). A read-only memory (the frozen memory
        of the benchmark, Requirement 22.3) is read by the selectors but never written."""
        if self.memory.read_only:
            return
        record = MemoryRecord(
            record_id=f"{ctx.session_id}/{result.experiment_id}",
            created_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            session_id=ctx.session_id,
            experiment_id=result.experiment_id,
            dataset_name=result.dataset_name,
            dataset_fingerprint=ctx.dataset_fingerprint,
            task_type=ctx.task_type.value,
            primary_metric=ctx.metric,
            meta_features=to_jsonable(ctx.profile.meta_features()),
            diagnoses=[{"kind": kind, "severity": severity} for kind, severity in proposal.answered],
            strategy=proposal.spec.to_dict(),
            verdict=decision.verdict,
            mean_difference=decision.mean_difference,
            std_difference=decision.std_difference,
            n_folds=decision.n_folds,
            cost_seconds=round(cost_seconds, 3),
            utility=utility,
            engine=f"{result.engine}=={result.engine_version}",
        )
        try:
            self.memory.add(record)
        except Exception:  # the memory must never stop a Session
            logger.exception("Could not add a record to the Experiment_Memory")

    def _search(self, ctx: _Context, config: ExperimentConfig, parent: str | None,
                proposal: dict[str, Any] | None) -> ExperimentResult:
        result = self.runner.run_experiment(
            ctx.dataframe, config, split=ctx.split, evaluate_test=False, compute_importance=False,
            profile_rows="train", session_id=ctx.session_id, parent_experiment_id=parent, proposal=proposal,
        )
        if result.test_evaluated:  # never true with evaluate_test=False; a guard for the Information_Barrier
            raise RuntimeError("An Experiment inside a Session used the test part.")
        return result

    def _save_step_files(self, experiment_id: str, assessment: Assessment, diagnoses: list[Diagnosis] | None,
                         decision: JudgeDecision | None) -> None:
        self.tracker.save_step_file(experiment_id, "assessment.json", assessment.to_dict())
        if diagnoses is not None:
            self.tracker.save_step_file(experiment_id, "diagnoses.json", [d.to_dict() for d in diagnoses])
        if decision is not None:
            self.tracker.save_step_file(experiment_id, "decision.json", to_jsonable(decision.to_dict()))

    # ------------------------------------------------------- final evaluation
    def _final_evaluation(self, labelled: pd.DataFrame, y: pd.Series, split: OuterSplit, train_frame: pd.DataFrame,
                          first: ExperimentResult, final: ExperimentResult, task_type: TaskType, metric: str,
                          seed: int) -> dict[str, Any]:
        started = time.perf_counter()
        test_frame, y_test = labelled.iloc[split.test_positions], y.iloc[split.test_positions]
        evaluator = Evaluator(task_type, metric)

        def score(result: ExperimentResult, importance: bool) -> dict[str, Any]:
            bundle = AutoMLRunner.load_model(result.model_path)
            X_test = test_frame[result.feature_columns]
            evaluation = evaluator.evaluate(bundle["model"], X_test, y_test, bundle.get("classes"))
            entry: dict[str, Any] = {
                "experiment_id": result.experiment_id,
                "test_score": evaluation.primary_score,
                "test_metrics": evaluation.metrics,
                "notes": [*evaluation.notes, *unseen_category_warnings(train_frame[result.feature_columns], X_test)],
            }
            if importance:
                try:
                    table = evaluator.feature_importance(bundle["model"], X_test, y_test, random_state=seed)
                    entry["feature_importance"] = table.to_dict(orient="records")
                except Exception as exc:
                    entry["notes"].append(f"Feature importance could not be computed: {exc}")
            return entry

        same = first.experiment_id == final.experiment_id
        first_entry = score(first, importance=same)
        final_entry = first_entry if same else score(final, importance=True)
        return to_jsonable({
            "first": first_entry,
            "final": final_entry,
            "baseline_metrics": evaluator.baseline_metrics(y.iloc[split.train_positions], y_test),
            "n_test": len(y_test),
            "seconds": round(time.perf_counter() - started, 2),
        })

    # ---------------------------------------------------------------- checks
    @staticmethod
    def _check(session: SessionSettings) -> tuple[CriticThresholds, StrategySelector]:
        """Settings errors are raised before any work, so nothing is written."""
        if session.selector not in SELECTORS:
            raise ExperimentError(f"Unknown selector '{session.selector}'. Available now: {', '.join(SELECTORS)}.")
        if session.n_splits < 2 or session.n_repeats < 1:
            raise ExperimentError("A Session needs at least 2 folds and 1 repeat.")
        try:
            thresholds = CriticThresholds.load(session.thresholds_path)
        except ThresholdsError as exc:
            raise ExperimentError(str(exc)) from exc
        return thresholds, SELECTORS[session.selector]()


# --------------------------------------------------------------------------- helpers
def _finite_stats(values: np.ndarray) -> tuple[float | None, float | None]:
    values = values[np.isfinite(values)]
    mean = float(values.mean()) if len(values) else None
    std = float(values.std(ddof=1)) if len(values) > 1 else None
    return mean, std


def _brief(assessment: Assessment, metric: str) -> dict[str, Any]:
    """Means and standard deviations of the primary metric over the folds, for session.json."""
    validation_mean, validation_std = _finite_stats(assessment.values(metric, "validation"))
    train_mean, train_std = _finite_stats(assessment.values(metric, "train"))
    baseline_mean, _ = _finite_stats(assessment.values(metric, "baseline"))
    return {"metric": metric, "validation_mean": validation_mean, "validation_std": validation_std,
            "train_mean": train_mean, "train_std": train_std, "baseline_mean": baseline_mean,
            "n_folds": len(assessment.folds), "seconds": round(assessment.seconds, 2),
            "failed": assessment.failed, "error": assessment.error,
            **({"reused_from": assessment.reused_from} if assessment.reused_from else {})}


def _change(proposal: dict[str, Any] | None) -> str:
    return proposal["reason"].split(". Answers:")[0] if proposal else "first Setup"


def _step_text(step: SessionStep) -> str:
    text = f"Step {step.index} ({_change(step.proposal)}): {step.experiment_id or 'no experiment'}"
    if step.status == "failed":
        return f"{text}, failed and counted as a discard: {step.error}"
    score = (step.assessment or {}).get("validation_mean")
    if score is not None:
        text += f", validation {score:.4f}"
    if step.decision:
        p = step.decision["p_better"]
        text += (f", {step.decision['verdict']} (mean difference {step.decision['mean_difference']:+.4f}, "
                 f"p(better) {'n/a' if p is None else f'{p:.3f}'})")
    return text


def code_version(project_root: Path) -> dict[str, Any]:
    """The git commit (if the project is a git repository) and the versions of the main libraries."""
    version: dict[str, Any] = {"git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True,
                                timeout=10)
        if commit.returncode == 0:
            version["git_commit"] = commit.stdout.strip()
            status = subprocess.run(["git", "status", "--porcelain"], cwd=project_root, capture_output=True,
                                    text=True, timeout=10)
            version["git_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    for library in _LIBRARIES:
        try:
            version[library] = metadata.version(library)
        except metadata.PackageNotFoundError:
            version[library] = None
    return version


class _State:
    """The growing record of one Session; ``save()`` rewrites session.json."""

    def __init__(self, owner: AutopilotSession, session_id: str, session: SessionSettings, base: ExperimentConfig,
                 thresholds: CriticThresholds, split: OuterSplit, folds: SessionFolds, notes: list[str],
                 first_exclusions: list[str], on_update: RecordUpdate | None = None) -> None:
        """``base`` is the user's base Setup with the task type, primary metric and CPU cores resolved."""
        self.owner = owner
        self.session_id = session_id
        self.session = session
        self.on_update = on_update
        self.steps: list[SessionStep] = []
        self.start = owner.clock()
        self.record: dict[str, Any] = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": session_id,
            "status": "running",
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "code_version": code_version(CODE_ROOT),
            "settings": {**session.to_dict(), "base": to_jsonable(base.__dict__),
                         "requested": {"task_type": to_jsonable(session.base.task_type),
                                       "primary_metric": session.base.primary_metric},
                         "thresholds_version": thresholds.version},
            "splits": {
                "outer": split.fingerprint(), "folds": folds.fingerprint(), "n_train": split.n_train,
                "n_test": split.n_test, "outer_seed": split.seed, "folds_seed": folds.seed,
                "outer_stratified": split.stratified, "folds_stratified": folds.stratified, "n_folds": len(folds),
                "grouped": split.grouped and folds.grouped,  # optional task 31
            },
            "first_setup_exclusions": first_exclusions,
            "notes": list(dict.fromkeys(notes)),
            "steps": [],
            "stop_reason": None,
            "final_evaluation": None,
            "first_experiment_id": None,
            "final_experiment_id": None,
            "mlflow_run_id": None,
            "total_seconds": None,
            "error": None,
        }

    def save(self) -> None:
        self.record["steps"] = [step.to_dict() for step in self.steps]
        self.record["total_seconds"] = round(self.owner.clock() - self.start, 2)
        record = to_jsonable(self.record)  # new containers: what on_update gets shares nothing with self.record
        self.owner.tracker.save_session(self.session_id, record)
        if self.on_update is not None:
            try:
                self.on_update(record)
            except Exception:  # a viewer must never stop a Session
                logger.exception("The update callback of %s failed", self.session_id)

    def add_step(self, step: SessionStep) -> None:
        self.steps.append(step)
        self.save()

    def start_mlflow(self, base: ExperimentConfig) -> None:
        if not self.owner.enable_mlflow:
            return
        budget = self.session.budget
        params = {"dataset": base.dataset_name, "target": base.target_column, "primary_metric": base.primary_metric,
                  "total_seconds": budget.total_seconds, "max_experiments": budget.max_experiments,
                  "n_jobs": budget.n_jobs, "seed": base.random_seed, "selector": self.session.selector,
                  "use_memory": self.session.use_memory, "use_critic": self.session.use_critic,
                  "use_resource_manager": self.session.use_resource_manager,
                  "judge_alpha": self.session.judge.alpha, "split_fingerprint": self.record["splits"]["outer"],
                  "group_column": self.session.group_column or "none"}
        try:
            self.record["mlflow_run_id"] = self.owner.tracker.start_session_run(
                self.session_id, params, {"dataset": base.dataset_name})
        except Exception as exc:  # tracking must never stop a Session
            logger.exception("MLflow parent run could not be started")
            self.record["notes"].append(f"MLflow logging failed: {exc}")

    def fail(self, message: str, experiment_id: str | None, rm: ResourceManager) -> None:
        rm.report_error(message)
        self.steps.append(SessionStep(
            index=0, experiment_id=experiment_id, parent_experiment_id=None, proposal=None, diagnoses=[],
            assessment=None, decision=None, accepted=False, cost=None, status="failed", error=message,
        ))
        self.record.update({"status": "failed", "stop_reason": StopReason.ERROR.value, "error": message})
        self.save()
        self._finish_mlflow(failed=True)

    def finish(self, stop: StopReason, final_evaluation: dict[str, Any], first: _Accepted, final: _Accepted) -> None:
        self.record.update({"status": "completed", "stop_reason": stop.value, "final_evaluation": final_evaluation,
                            "final_experiment_id": final.result.experiment_id})
        self.save()
        self._finish_mlflow(failed=False, metrics={
            "first_validation": _brief(first.assessment, first.assessment.primary_metric)["validation_mean"],
            "final_validation": _brief(final.assessment, final.assessment.primary_metric)["validation_mean"],
            "first_test": final_evaluation["first"]["test_score"],
            "final_test": final_evaluation["final"]["test_score"],
            "n_experiments": len(self.steps), "total_seconds": self.record["total_seconds"],
        })

    def result(self) -> SessionResult:
        r = self.record
        return SessionResult(
            session_id=self.session_id, status=r["status"], settings=r["settings"], steps=list(self.steps),
            first_experiment_id=r["first_experiment_id"], final_experiment_id=r["final_experiment_id"],
            stop_reason=StopReason(r["stop_reason"]), final_evaluation=r["final_evaluation"],
            total_seconds=r["total_seconds"], error=r["error"],
        )

    def _finish_mlflow(self, failed: bool, metrics: dict[str, Any] | None = None) -> None:
        if not self.owner.enable_mlflow or self.record.get("mlflow_run_id") is None:
            return
        folder = self.owner.tracker.session_dir(self.session_id)
        try:
            self.owner.tracker.finish_session_run(
                self.session_id, metrics or {},
                [(folder / "session.json", "session"), (folder / "train_profile.json", "session")], failed=failed)
        except Exception:
            logger.exception("MLflow parent run could not be finished")
