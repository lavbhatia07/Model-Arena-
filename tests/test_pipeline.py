"""
End-to-end integration tests for ModelArena Benchmarking Pipeline & Prediction.
"""
import pytest
import pandas as pd
import numpy as np

from core.benchmark import run_benchmark, BenchmarkResult
from core.model_manager import export_model_package, load_model_package
from core.predictor import predict_on_dataframe


@pytest.fixture
def classification_df():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_num1": np.random.randn(n),
        "feature_num2": np.random.uniform(0, 100, size=n),
        "feature_cat": np.random.choice(["Red", "Green", "Blue"], size=n),
        "target": np.random.choice(["Pass", "Fail"], size=n)
    })
    # Add a few missing values
    df.loc[5:10, "feature_num1"] = np.nan
    df.loc[15:18, "feature_cat"] = np.nan
    return df


@pytest.fixture
def regression_df():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "city": np.random.choice(["NY", "LA", "Chicago"], size=n),
        "target_val": np.random.randn(n) * 50.0 + 10.0
    })
    return df


def test_classification_benchmark_pipeline(classification_df):
    res = run_benchmark(
        df=classification_df,
        target_col="target",
        problem_type="Classification",
        selected_model_keys=["Logistic Regression", "Random Forest Classifier"],
        test_size=0.25,
        cv_folds=3,
        random_state=42
    )
    
    assert isinstance(res, BenchmarkResult)
    assert res.problem_type == "Classification"
    assert len(res.model_results) == 2
    assert not res.leaderboard_df.empty
    assert "F1" in res.leaderboard_df.columns
    assert "Accuracy" in res.leaderboard_df.columns
    assert res.recommended_model_name in res.model_results


def test_regression_benchmark_pipeline(regression_df):
    res = run_benchmark(
        df=regression_df,
        target_col="target_val",
        problem_type="Regression",
        selected_model_keys=["Linear Regression", "Random Forest Regressor"],
        test_size=0.20,
        cv_folds=3,
        random_state=42
    )
    
    assert isinstance(res, BenchmarkResult)
    assert res.problem_type == "Regression"
    assert len(res.model_results) == 2
    assert not res.leaderboard_df.empty
    assert "RMSE" in res.leaderboard_df.columns
    assert "R²" in res.leaderboard_df.columns


def test_model_export_and_predict(classification_df):
    bench_res = run_benchmark(
        df=classification_df,
        target_col="target",
        problem_type="Classification",
        selected_model_keys=["Random Forest Classifier"],
        test_size=0.20,
        cv_folds=3,
        random_state=42
    )
    
    best_res = bench_res.model_results[bench_res.recommended_model_name]
    
    # Export package
    package_bytes = export_model_package(
        model_result=best_res,
        target_col="target",
        feature_cols=bench_res.feature_cols
    )
    assert len(package_bytes) > 0
    
    # Load package
    pkg = load_model_package(package_bytes)
    assert pkg["model_name"] == bench_res.recommended_model_name
    assert pkg["problem_type"] == "Classification"
    
    # Predict on new test sample
    new_sample = pd.DataFrame({
        "feature_num1": [0.5, np.nan],
        "feature_num2": [50.0, 75.0],
        "feature_cat": ["Red", "UnknownCity"]
    })
    
    preds_df, summary = predict_on_dataframe(pkg, new_sample)
    assert not preds_df.empty
    assert "Predicted_target" in preds_df.columns
    assert len(preds_df) == 2
