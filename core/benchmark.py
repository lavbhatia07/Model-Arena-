"""
Benchmarking Engine Module for ModelArena.
Executes multi-model training, cross-validation, metric collection, leaderboard creation, and recommendation.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from core.preprocessing import identify_feature_types, create_preprocessor
from core.models import get_available_models
from core.trainer import train_single_model, SingleModelResult
from utils.helpers import get_logger

logger = get_logger("Benchmark")


@dataclass
class BenchmarkResult:
    problem_type: str
    target_col: str
    feature_cols: List[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    leaderboard_df: pd.DataFrame
    model_results: Dict[str, SingleModelResult]
    recommended_model_name: str
    recommendation_reason: str
    primary_metric: str
    failed_models: List[Dict[str, str]]
    test_size: float
    cv_folds: int
    random_state: int


def run_benchmark(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    selected_model_keys: Optional[List[str]] = None,
    test_size: float = 0.20,
    cv_folds: int = 5,
    primary_metric: Optional[str] = None,
    random_state: int = 42,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> BenchmarkResult:
    """
    Executes the full ModelArena benchmarking suite.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    feature_cols = X.columns.tolist()
    num_cols, cat_cols = identify_feature_types(X)
    
    # 1. Train / Test Split
    if problem_type == "Classification":
        # Check if stratified split is possible
        class_counts = y.value_counts()
        can_stratify = (class_counts.min() >= 2) and (len(class_counts) > 1)
        
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
    # 2. Get Available Models
    registry = get_available_models(problem_type)
    if selected_model_keys is None:
        target_models = list(registry.keys())
    else:
        target_models = [k for k in selected_model_keys if k in registry]
        
    if not target_models:
        raise ValueError("No valid models selected for benchmarking.")
        
    # 3. Create Preprocessor Template
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    # 4. Train Models Loop
    model_results: Dict[str, SingleModelResult] = {}
    failed_models: List[Dict[str, str]] = []
    
    total_count = len(target_models)
    for idx, model_name in enumerate(target_models):
        if progress_callback:
            progress_callback(idx, total_count, model_name)
            
        model_info = registry[model_name]
        res = train_single_model(
            model_name=model_name,
            model_info=model_info,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            problem_type=problem_type,
            cv_folds=cv_folds,
            random_state=random_state
        )
        
        if res.success:
            model_results[model_name] = res
        else:
            failed_models.append({
                "model_name": model_name,
                "error": res.error_message or "Unknown error"
            })
            
    if progress_callback:
        progress_callback(total_count, total_count, "Completed")
        
    if not model_results:
        raise RuntimeError("All selected models failed to train. Check your dataset and missing values.")
        
    # 5. Build Leaderboard DataFrame
    leaderboard_rows = []
    
    # Determine default primary metric if not specified
    if primary_metric is None:
        primary_metric = "F1" if problem_type == "Classification" else "RMSE"
        
    for name, res in model_results.items():
        m = res.metrics
        if problem_type == "Classification":
            leaderboard_rows.append({
                "Model": name,
                "Accuracy": round(m.accuracy, 4),
                "Precision": round(m.precision, 4),
                "Recall": round(m.recall, 4),
                "F1": round(m.f1, 4),
                "ROC-AUC": round(m.roc_auc, 4) if m.roc_auc is not None else np.nan,
                "CV Score": round(res.cv_mean, 4),
                "CV Std": round(res.cv_std, 4),
                "Training Time (s)": round(res.train_time_sec, 4)
            })
        else:
            leaderboard_rows.append({
                "Model": name,
                "MAE": round(m.mae, 4),
                "MSE": round(m.mse, 4),
                "RMSE": round(m.rmse, 4),
                "R²": round(m.r2, 4),
                "CV Score": round(res.cv_mean, 4),
                "CV Std": round(res.cv_std, 4),
                "Training Time (s)": round(res.train_time_sec, 4)
            })
            
    df_leaderboard = pd.DataFrame(leaderboard_rows)
    
    # Sort Leaderboard based on primary metric
    ascending_sort = False
    if problem_type == "Regression":
        if primary_metric in ["MAE", "MSE", "RMSE"]:
            ascending_sort = True
            
    if primary_metric in df_leaderboard.columns:
        df_leaderboard = df_leaderboard.sort_values(by=primary_metric, ascending=ascending_sort).reset_index(drop=True)
    else:
        # Fallback sorting
        fallback_col = "F1" if problem_type == "Classification" else "RMSE"
        df_leaderboard = df_leaderboard.sort_values(by=fallback_col, ascending=(problem_type == "Regression")).reset_index(drop=True)
        primary_metric = fallback_col
        
    best_model_name = df_leaderboard.iloc[0]["Model"]
    best_val = df_leaderboard.iloc[0][primary_metric]
    
    recommendation_reason = (
        f"Selected as the top-performing model based on the primary metric **{primary_metric}** "
        f"(Score: **{best_val}**). ModelArena evaluated {len(model_results)} models using {cv_folds}-fold cross-validation."
    )
    
    return BenchmarkResult(
        problem_type=problem_type,
        target_col=target_col,
        feature_cols=feature_cols,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        leaderboard_df=df_leaderboard,
        model_results=model_results,
        recommended_model_name=best_model_name,
        recommendation_reason=recommendation_reason,
        primary_metric=primary_metric,
        failed_models=failed_models,
        test_size=test_size,
        cv_folds=cv_folds,
        random_state=random_state
    )
