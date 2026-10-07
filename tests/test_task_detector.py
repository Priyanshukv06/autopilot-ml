import numpy as np
import pandas as pd
import pytest

from app.services.task_detector import TaskDetectionError, TaskDetector, TaskType


@pytest.fixture
def detector() -> TaskDetector:
    return TaskDetector()


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["yes", "no", "yes", "no"], TaskType.BINARY),
        ([0, 1, 1, 0, 1], TaskType.BINARY),
        ([True, False, True], TaskType.BINARY),
        (["cat", "dog", "bird", "cat"], TaskType.MULTICLASS),
        (list(np.tile([0, 1, 2], 100)), TaskType.MULTICLASS),
        (list(np.random.default_rng(0).normal(size=200)), TaskType.REGRESSION),
        (list(np.arange(500) % 137), TaskType.REGRESSION),
    ],
)
def test_detect(detector, values, expected):
    assert detector.detect(pd.Series(values, name="y")).task_type is expected


def test_detect_reports_classes_and_confidence(detector):
    result = detector.detect(pd.Series(["a", "b", "a"]))
    assert result.n_classes == 2
    assert result.confidence == "high"
    assert result.to_dict()["task_type"] == "binary"
    integer_codes = detector.detect(pd.Series(np.tile([1, 2, 3, 4], 50)))
    assert integer_codes.task_type is TaskType.MULTICLASS
    assert integer_codes.confidence == "medium"


def test_detect_ignores_missing_values(detector):
    assert detector.detect(pd.Series(["a", None, "b", "a"])).task_type is TaskType.BINARY


@pytest.mark.parametrize("values", [[1, 1, 1], [np.nan, np.nan]])
def test_detect_rejects_unusable_targets(detector, values):
    with pytest.raises(TaskDetectionError):
        detector.detect(pd.Series(values, name="y"))


def test_validate_rejects_impossible_overrides(detector):
    with pytest.raises(TaskDetectionError):
        detector.validate(pd.Series(["a", "b", "c"]), TaskType.REGRESSION)
    with pytest.raises(TaskDetectionError):
        detector.validate(pd.Series(["a", "b", "c"]), TaskType.BINARY)
    with pytest.raises(TaskDetectionError):
        detector.validate(pd.Series([0, 1, 0, 1]), TaskType.MULTICLASS)


def test_validate_allows_overrides_with_warnings(detector):
    # Integer codes detected as multiclass may legitimately be treated as regression.
    assert detector.validate(pd.Series(np.tile([1, 2, 3], 30)), TaskType.REGRESSION) == []
    warnings = detector.validate(pd.Series(["a"] * 10 + ["b"] * 10 + ["c"] + [None]), TaskType.MULTICLASS)
    assert any("missing target" in w for w in warnings)
    assert any("fewer than 2 samples" in w for w in warnings)


def test_target_candidates(detector):
    df = pd.DataFrame(
        {
            "id": np.arange(50),
            "constant": [1] * 50,
            "feature": np.random.default_rng(0).normal(size=50),
            "label": ["x", "y"] * 25,
        }
    )
    candidates = {c.column: c for c in detector.target_candidates(df)}
    assert not candidates["id"].eligible
    assert not candidates["constant"].eligible
    assert candidates["label"].eligible and candidates["label"].inferred_task == "binary"
    assert candidates["feature"].inferred_task == "regression"


def test_suggest_target(detector):
    assert detector.suggest_target(pd.DataFrame(columns=["a", " Target ", "b"])) == " Target "
    assert detector.suggest_target(pd.DataFrame(columns=["a", "b", "c"])) == "c"
    assert detector.suggest_target(pd.DataFrame()) is None


def test_task_type_properties():
    assert TaskType.BINARY.is_classification and TaskType.MULTICLASS.is_classification
    assert not TaskType.REGRESSION.is_classification
    assert TaskType("multiclass").display_name == "Multiclass Classification"
