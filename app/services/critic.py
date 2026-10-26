"""Critic: finds what is wrong with the data or with an Experiment, with evidence (Requirement 5).

The Critic reads only Training_Part information: the profile of the Training_Part, the Setup, the
Assessment on the Session_Folds and the search summary. It is a pure function of these inputs, so
the Diagnoses of a finished Session can be recomputed from its records. All thresholds come from
``critic_thresholds.toml``, and every Diagnosis records that file's version label.

All twelve kinds of Requirement 5.2 (the thin version of task 6 had overfitting, class_imbalance
and high_cardinality; task 13 added the others).
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from app.services.assessor import Assessment
from app.services.automl_runner import ExperimentConfig
from app.services.evaluator import METRICS
from app.services.profiler import DatasetProfile
from app.services.search_summary import SearchSummary
from app.services.task_detector import TaskType
from app.services.transforms import as_specs

DEFAULT_THRESHOLDS_PATH = Path(__file__).with_name("critic_thresholds.toml")

# Metrics that are bounded and higher-is-better; any other primary metric is replaced by F1
# (macro F1 for multiclass) or R2 for the model-level rules, so that gaps have one scale.
_BOUNDED_METRICS = ("f1", "accuracy", "roc_auc", "r2")


class DiagnosisKind(str, Enum):
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    BUDGET_LIMITED = "budget_limited"
    WASTED_BUDGET = "wasted_budget"
    CLASS_IMBALANCE = "class_imbalance"
    MISSING_VALUES = "missing_values"
    HIGH_CARDINALITY = "high_cardinality"
    IDENTIFIER_COLUMN = "identifier_column"
    CONSTANT_COLUMN = "constant_column"
    REDUNDANT_FEATURES = "redundant_features"
    UNRELIABLE_VALIDATION = "unreliable_validation"
    WEAK_MODEL_FAMILIES = "weak_model_families"


class Severity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass(frozen=True)
class Diagnosis:
    kind: DiagnosisKind
    severity: Severity
    columns: tuple[str, ...]
    evidence: dict[str, float | str]
    message: str
    thresholds_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "columns": list(self.columns),
            "evidence": dict(self.evidence),
            "message": self.message,
            "thresholds_version": self.thresholds_version,
        }


class ThresholdsError(ValueError):
    """The thresholds file is missing or invalid."""


# Every threshold the rules read, as "section.name".
REQUIRED_THRESHOLDS = (
    "overfitting.moderate_gap",
    "overfitting.high_gap",
    "underfitting.max_gap",
    "underfitting.moderate_skill",
    "underfitting.high_skill",
    "budget_limited.moderate_gain",
    "budget_limited.high_gain",
    "wasted_budget.moderate_share",
    "wasted_budget.high_share",
    "class_imbalance.moderate_ratio",
    "class_imbalance.high_ratio",
    "class_imbalance.moderate_recall",
    "class_imbalance.high_recall",
    "class_imbalance.min_ratio_for_recall",
    "missing_values.moderate_share",
    "missing_values.high_share",
    "high_cardinality.moderate_distinct",
    "high_cardinality.moderate_unseen",
    "high_cardinality.high_unseen",
    "constant_column.near_constant_share",
    "redundant_features.moderate_correlation",
    "redundant_features.high_correlation",
    "unreliable_validation.moderate_std",
    "unreliable_validation.high_std",
    "weak_model_families.min_gap",
    "weak_model_families.min_time_share",
)
# Pairs (less severe, more severe) that must be in this order: (a, b) means a <= b.
_ORDERED = (
    ("overfitting.moderate_gap", "overfitting.high_gap"),
    ("underfitting.high_skill", "underfitting.moderate_skill"),  # for skill, high is the lower one
    ("budget_limited.moderate_gain", "budget_limited.high_gain"),
    ("wasted_budget.moderate_share", "wasted_budget.high_share"),
    ("class_imbalance.moderate_ratio", "class_imbalance.high_ratio"),
    ("class_imbalance.high_recall", "class_imbalance.moderate_recall"),
    ("missing_values.moderate_share", "missing_values.high_share"),
    ("high_cardinality.moderate_unseen", "high_cardinality.high_unseen"),
    ("redundant_features.moderate_correlation", "redundant_features.high_correlation"),
    ("unreliable_validation.moderate_std", "unreliable_validation.high_std"),
)


@dataclass(frozen=True)
class CriticThresholds:
    version: str
    values: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> CriticThresholds:
        """Read and check a thresholds file (the default one next to this module if not given)."""
        path = Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ThresholdsError(f"Critic thresholds file not found: {path}") from exc
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ThresholdsError(f"Critic thresholds file {path} cannot be read: {exc}") from exc
        return cls.from_mapping(data, source=str(path))

    @classmethod
    def from_mapping(cls, data: dict[str, Any], source: str = "thresholds") -> CriticThresholds:
        version = data.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ThresholdsError(f"{source}: 'version' must be a non-empty text label.")
        values: dict[str, float] = {}
        for section, table in data.items():
            if section == "version":
                continue
            if not isinstance(table, dict):
                raise ThresholdsError(f"{source}: '{section}' must be a table of thresholds.")
            for name, value in table.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ThresholdsError(f"{source}: '{section}.{name}' must be a number.")
                values[f"{section}.{name}"] = float(value)
        missing = [key for key in REQUIRED_THRESHOLDS if key not in values]
        unknown = sorted(set(values) - set(REQUIRED_THRESHOLDS))
        if missing:
            raise ThresholdsError(f"{source}: missing thresholds {missing}.")
        if unknown:
            raise ThresholdsError(f"{source}: unknown thresholds {unknown} (a typing mistake?).")
        for low, high in _ORDERED:
            if values[low] > values[high]:
                raise ThresholdsError(f"{source}: '{low}' ({values[low]:g}) must not be above '{high}' "
                                      f"({values[high]:g}).")
        return cls(version=version.strip(), values=values)

    def __getitem__(self, key: str) -> float:
        return self.values[key]


def diagnostic_metric(task_type: TaskType, primary_metric: str) -> str:
    """The metric of the model-level rules: the primary metric if it is bounded and higher is
    better, otherwise F1 (macro F1 for multiclass) or R2."""
    if primary_metric in _BOUNDED_METRICS:
        return primary_metric
    return "f1" if task_type.is_classification else "r2"


class Critic:
    def __init__(self, thresholds: CriticThresholds) -> None:
        self.thresholds = thresholds

    def diagnose(
        self,
        *,
        profile: DatasetProfile,
        config: ExperimentConfig,
        assessment: Assessment,
        search: SearchSummary | None,
        task_type: TaskType,
        primary_metric: str,
    ) -> list[Diagnosis]:
        """Diagnoses in a fixed order: by rule (the order of design.md's table), then by column name.

        ``profile`` must describe the Training_Part only, and ``assessment`` must be complete.
        ``search`` is read by the budget rules of the full version.
        """
        if assessment.failed or not assessment.folds:
            raise ValueError("The Critic needs a completed Assessment.")
        used = self._used_columns(profile, config)
        metric = diagnostic_metric(task_type, primary_metric)
        search_metric = config.search_metric or primary_metric
        treated = {c for step in as_specs(config.data_steps)
                   if step.kind in ("frequency_encode", "group_rare_categories") for c in step.columns}
        found: list[Diagnosis | None] = [
            self._overfitting(assessment, metric),
            self._underfitting(assessment, metric),
            self._budget_limited(search, search_metric),
            self._wasted_budget(search),
        ]
        if task_type.is_classification:
            found.append(self._class_imbalance(profile, assessment))
        found.extend(self._missing_values(profile, column) for column in used)
        found.extend(self._high_cardinality(profile, assessment, column) for column in used
                     if column in profile.categorical_features and column not in treated)
        found.extend(self._identifier_column(profile, column) for column in used)
        found.extend(self._constant_column(profile, column) for column in used)
        found.extend(self._redundant_features(pair) for pair in profile.correlated_pairs
                     if all(c in used for c in pair["columns"]))
        found.append(self._unreliable_validation(assessment, metric))
        found.append(self._weak_model_families(search, search_metric))
        return [d for d in found if d is not None]

    # ------------------------------------------------------------------ rules
    def _overfitting(self, assessment: Assessment, metric: str) -> Diagnosis | None:
        train, validation = assessment.values(metric, "train"), assessment.values(metric, "validation")
        both = np.isfinite(train) & np.isfinite(validation)
        if not both.any():
            return None
        gap = float(np.mean(train[both] - validation[both]))
        severity = self._level(gap >= self.thresholds["overfitting.high_gap"],
                               gap >= self.thresholds["overfitting.moderate_gap"])
        if severity is None:
            return None
        train_mean, validation_mean = float(train[both].mean()), float(validation[both].mean())
        return self._make(
            DiagnosisKind.OVERFITTING, severity, (),
            {"metric": metric, "gap": gap, "train_mean": train_mean, "validation_mean": validation_mean,
             "folds": int(both.sum())},
            f"The model scores {train_mean:.3f} on the rows it was trained on but {validation_mean:.3f} on "
            f"validation rows ({metric}, mean over {int(both.sum())} folds), a gap of {gap:.3f}: it fits "
            "details of the training rows that do not carry over to new rows.",
        )

    def _class_imbalance(self, profile: DatasetProfile, assessment: Assessment) -> Diagnosis | None:
        balance = profile.class_balance
        if balance is None:
            return None
        ratio = float(balance.imbalance_ratio)
        recalls = [fold.class_recall.get(balance.minority_class) for fold in assessment.folds if fold.class_recall]
        recalls = [float(r) for r in recalls if r is not None]
        recall = float(np.mean(recalls)) if recalls else None
        use_recall = recall is not None and ratio >= self.thresholds["class_imbalance.min_ratio_for_recall"]
        severity = self._level(
            ratio >= self.thresholds["class_imbalance.high_ratio"]
            or (use_recall and recall < self.thresholds["class_imbalance.high_recall"]),
            ratio >= self.thresholds["class_imbalance.moderate_ratio"]
            or (use_recall and recall < self.thresholds["class_imbalance.moderate_recall"]),
        )
        if severity is None:
            return None
        evidence: dict[str, float | str] = {"imbalance_ratio": ratio, "minority_class": balance.minority_class,
                                            "minority_share": float(balance.minority_ratio)}
        message = (f"The smallest class '{balance.minority_class}' has 1 training row for every {ratio:.1f} rows "
                   "of the largest class")
        if recall is not None:
            evidence.update({"minority_recall": recall, "folds": len(recalls)})
            message += f", and the model finds {recall:.0%} of its rows in validation (mean over {len(recalls)} folds)"
        return self._make(DiagnosisKind.CLASS_IMBALANCE, severity, (str(profile.target_column),), evidence,
                          message + ".")

    def _high_cardinality(self, profile: DatasetProfile, assessment: Assessment, column: str) -> Diagnosis | None:
        distinct = int(profile.unique_counts.get(column, 0))
        shares = [fold.unseen_share[column] for fold in assessment.folds if column in fold.unseen_share]
        unseen = float(np.mean(shares)) if shares else None
        severity = self._level(
            unseen is not None and unseen >= self.thresholds["high_cardinality.high_unseen"],
            distinct > self.thresholds["high_cardinality.moderate_distinct"]
            or (unseen is not None and unseen >= self.thresholds["high_cardinality.moderate_unseen"]),
        )
        if severity is None:
            return None
        evidence: dict[str, float | str] = {"distinct_values": distinct}
        message = f"'{column}' has {distinct} different values in the training part."
        if unseen is not None:
            evidence.update({"unseen_share": unseen, "folds": len(shares)})
            message += (f" On average {unseen:.0%} of validation rows have a value that the fold's training rows "
                        f"never had (mean over {len(shares)} folds); the model gets no information from this "
                        "column for those rows.")
        return self._make(DiagnosisKind.HIGH_CARDINALITY, severity, (column,), evidence, message)

    def _underfitting(self, assessment: Assessment, metric: str) -> Diagnosis | None:
        """Low skill over the trivial baseline while the training score is not higher than the validation score."""
        t = self.thresholds
        train, validation = assessment.values(metric, "train"), assessment.values(metric, "validation")
        baseline = assessment.values(metric, "baseline")
        if metric == "roc_auc":  # the baseline gives no probabilities; a constant scores 0.5
            baseline = np.where(np.isfinite(baseline), baseline, 0.5)
        both = np.isfinite(train) & np.isfinite(validation) & np.isfinite(baseline)
        if not both.any():
            return None
        gap = float(np.mean(train[both] - validation[both]))
        v, b = float(validation[both].mean()), float(baseline[both].mean())
        if gap >= t["underfitting.max_gap"] or 1 - b <= 1e-9:
            return None
        skill = (v - b) / (1 - b)
        severity = self._level(skill < t["underfitting.high_skill"], skill < t["underfitting.moderate_skill"])
        if severity is None:
            return None
        return self._make(
            DiagnosisKind.UNDERFITTING, severity, (),
            {"metric": metric, "skill": skill, "validation_mean": v, "baseline_mean": b, "gap": gap},
            f"The model scores {v:.3f} on validation rows against {b:.3f} for the trivial baseline ({metric}); it "
            f"closes only {skill:.0%} of the distance to a perfect score, and it does no better on its own training "
            f"rows (gap {gap:+.3f}), so it is too simple or has too little time.",
        )

    def _budget_limited(self, search: SearchSummary | None, search_metric: str) -> Diagnosis | None:
        """The search was still improving at the end. Only for bounded metrics, where the gain has a fixed scale."""
        if search is None or search.gain_last_quarter is None or search_metric not in _BOUNDED_METRICS:
            return None
        t, gain = self.thresholds, float(search.gain_last_quarter)
        severity = self._level(gain >= t["budget_limited.high_gain"], gain >= t["budget_limited.moderate_gain"])
        if severity is None:
            return None
        return self._make(
            DiagnosisKind.BUDGET_LIMITED, severity, (),
            {"metric": search_metric, "gain_last_quarter": gain, "search_seconds": search.search_seconds},
            f"The search was still improving at the end: its best {search_metric} rose by {gain:.3f} in the last "
            f"quarter of its {search.search_seconds:.0f} s, so more time may help.",
        )

    def _wasted_budget(self, search: SearchSummary | None) -> Diagnosis | None:
        if search is None or search.search_seconds <= 0:
            return None
        t, share = self.thresholds, float(search.share_after_best)
        severity = self._level(share >= t["wasted_budget.high_share"], share >= t["wasted_budget.moderate_share"])
        if severity is None:
            return None
        return self._make(
            DiagnosisKind.WASTED_BUDGET, severity, (),
            {"share_after_best": share, "time_to_best": search.time_to_best, "search_seconds": search.search_seconds},
            f"The search found its best configuration after {search.time_to_best:.1f} s of {search.search_seconds:.1f} s "
            f"and found nothing better in the remaining {share:.0%} of the time.",
        )

    def _missing_values(self, profile: DatasetProfile, column: str) -> Diagnosis | None:
        t, share = self.thresholds, float(profile.missing_ratio.get(column, 0.0))
        severity = self._level(share >= t["missing_values.high_share"], share >= t["missing_values.moderate_share"])
        if severity is None:
            return None
        return self._make(DiagnosisKind.MISSING_VALUES, severity, (column,), {"missing_share": share},
                          f"'{column}' is empty in {share:.0%} of the training rows.")

    def _identifier_column(self, profile: DatasetProfile, column: str) -> Diagnosis | None:
        if column not in profile.id_like_columns:
            return None
        distinct = int(profile.unique_counts.get(column, 0))
        return self._make(DiagnosisKind.IDENTIFIER_COLUMN, Severity.HIGH, (column,),
                          {"distinct_values": distinct, "rows": int(profile.rows)},
                          f"'{column}' looks like an identifier ({distinct} different values in {profile.rows} training "
                          "rows); a model can only memorise it, it cannot learn from it.")

    def _constant_column(self, profile: DatasetProfile, column: str) -> Diagnosis | None:
        distinct = int(profile.unique_counts.get(column, 0))
        share = float(profile.main_value_share.get(column, 1.0 if distinct <= 1 else 0.0))
        if distinct > 1 and share < self.thresholds["constant_column.near_constant_share"]:
            return None
        text = "has one value only" if distinct <= 1 else f"has the same value in {share:.1%} of the training rows"
        return self._make(DiagnosisKind.CONSTANT_COLUMN, Severity.LOW, (column,),
                          {"distinct_values": distinct, "main_value_share": share},
                          f"'{column}' {text}, so it carries almost no information.")

    def _redundant_features(self, pair: dict[str, Any]) -> Diagnosis | None:
        t, r = self.thresholds, float(pair["correlation"])
        severity = self._level(abs(r) >= t["redundant_features.high_correlation"],
                               abs(r) >= t["redundant_features.moderate_correlation"])
        if severity is None:
            return None
        a, b = pair["columns"]
        return self._make(DiagnosisKind.REDUNDANT_FEATURES, severity, (str(a), str(b)), {"correlation": r},
                          f"'{a}' and '{b}' carry nearly the same information (correlation {r:+.3f} on the training "
                          "rows); one of them is enough.")

    def _unreliable_validation(self, assessment: Assessment, metric: str) -> Diagnosis | None:
        values = assessment.values(metric, "validation")
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return None
        t, sd = self.thresholds, float(values.std(ddof=1))
        severity = self._level(sd >= t["unreliable_validation.high_std"], sd >= t["unreliable_validation.moderate_std"])
        if severity is None:
            return None
        return self._make(
            DiagnosisKind.UNRELIABLE_VALIDATION, severity, (),
            {"metric": metric, "std": sd, "mean": float(values.mean()), "folds": len(values)},
            f"The validation {metric} varies a lot between folds (standard deviation {sd:.3f} over {len(values)} "
            "folds), so a single score says little and small differences between Setups are hard to see.",
        )

    def _weak_model_families(self, search: SearchSummary | None, search_metric: str) -> Diagnosis | None:
        """Learner families clearly behind the best one that still took a real share of the search time."""
        if search is None or search_metric not in _BOUNDED_METRICS or len(search.best_per_family) < 2:
            return None
        t = self.thresholds
        sign = 1.0 if METRICS[search_metric].greater_is_better else -1.0
        best_family = max(search.best_per_family, key=lambda f: (sign * search.best_per_family[f], f))
        best = search.best_per_family[best_family]
        total = sum(search.seconds_per_family.values()) or 1.0
        weak = {}
        for family in sorted(search.best_per_family):
            gap = sign * (best - search.best_per_family[family])
            time_share = search.seconds_per_family.get(family, 0.0) / total
            if gap >= t["weak_model_families.min_gap"] and time_share >= t["weak_model_families.min_time_share"]:
                weak[family] = (gap, time_share)
        if not weak:
            return None
        # tried_families: every family the search tried, in the order it first tried them (the order of the
        # summary), so that restricting the list can keep FLAML's order.
        evidence: dict[str, float | str] = {"families": ",".join(weak), "best_family": best_family,
                                            "metric": search_metric,
                                            "tried_families": ",".join(search.seconds_per_family)}
        for family, (gap, time_share) in weak.items():
            evidence[f"gap.{family}"] = gap
            evidence[f"time_share.{family}"] = time_share
        names = ", ".join(f"{f} ({g:.3f} behind, {s:.0%} of the time)" for f, (g, s) in weak.items())
        return self._make(DiagnosisKind.WEAK_MODEL_FAMILIES, Severity.LOW, (), evidence,
                          f"These learner families were far behind {best_family} ({search_metric}) but still used "
                          f"search time: {names}.")

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _used_columns(profile: DatasetProfile, config: ExperimentConfig) -> list[str]:
        """Feature columns of the Setup, sorted by name: not the target, not excluded, not dropped by a Data_Step."""
        dropped = {c for step in as_specs(config.data_steps) if step.kind == "drop_columns" for c in step.columns}
        skipped = {profile.target_column, *config.excluded_columns, *dropped}
        return sorted(str(c) for c in profile.column_names if c not in skipped)

    @staticmethod
    def _level(high: bool, moderate: bool) -> Severity | None:
        if high:
            return Severity.HIGH
        if moderate:
            return Severity.MODERATE
        return None

    def _make(self, kind: DiagnosisKind, severity: Severity, columns: tuple[str, ...],
              evidence: dict[str, float | str], message: str) -> Diagnosis:
        return Diagnosis(kind, severity, columns, evidence, message, self.thresholds.version)
