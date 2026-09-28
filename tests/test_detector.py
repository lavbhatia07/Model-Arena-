"""
Unit tests for Problem Type Detector module.
"""
import pytest
import pandas as pd
import numpy as np

from core.detector import detect_problem_type, ProblemDetectionResult


def test_detect_categorical_target_as_classification():
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "target": ["cat", "dog", "cat", "dog", "bird", "dog"]
    })
    res = detect_problem_type(df, "target")
    assert isinstance(res, ProblemDetectionResult)
    assert res.problem_type == "Classification"
    assert res.num_classes == 3
    assert res.confidence == "High"


def test_detect_boolean_target_as_classification():
    df = pd.DataFrame({
        "feature1": [10, 20, 30, 40, 50],
        "target": [True, False, True, True, False]
    })
    res = detect_problem_type(df, "target")
    assert res.problem_type == "Classification"
    assert res.num_classes == 2


def test_detect_low_cardinality_numeric_as_classification():
    df = pd.DataFrame({
        "feature1": np.random.randn(50),
        "target": np.random.choice([0, 1, 2], size=50)
    })
    res = detect_problem_type(df, "target")
    assert res.problem_type == "Classification"
    assert res.num_classes == 3


def test_detect_continuous_float_target_as_regression():
    np.random.seed(42)
    df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "target": np.random.randn(100) * 100.0 + 5.0
    })
    res = detect_problem_type(df, "target")
    assert res.problem_type == "Regression"
    assert res.num_classes is None
    assert res.num_unique_targets > 50


def test_detect_missing_target_raises_error():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        detect_problem_type(df, "non_existent_column")
