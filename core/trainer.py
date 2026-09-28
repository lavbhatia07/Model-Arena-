"""
Single Model Trainer Module for ModelArena.
Handles model pipeline fitting, cross-validation, feature importance extraction, and error handling.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from core.preprocessing import get_feature_names
from core.evaluator import evaluate_classification, evaluate_regression, ClassificationMetrics, RegressionMetrics
from utils.helpers import Timer, get_logger

logger = get_logger("Trainer")


@dataclass
class SingleModelResult:
    model_name: str
    problem_type: str
    fitted_pipeline: Optional[Pipeline]
    target_encoder: Optional[LabelEncoder]
    train_time_sec: float
    cv_mean: float
    cv_std: float
    cv_scores: List[float]
    metrics: Optional[Any]  # ClassificationMetrics or RegressionMetrics
    y_pred: Optional[np.ndarray]
    y_proba: Optional[np.ndarray]
    feature_names: List[str]
    feature_importances: Optional[pd.DataFrame]
    hyperparameters: Dict[str, Any]
    supports_importance: bool
    success: bool = True
    error_message: Optional[str] = None


def train_single_model(
    model_name: str,
    model_info: Dict[str, Any],
    preprocessor: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
    cv_folds: int = 5,
    random_state: int = 42
) -> SingleModelResult:
    """
    Trains a single model inside a leak-free sklearn Pipeline.
    Performs cross-validation on train split, fits on full train split, and evaluates on test split.
    Catches model training errors gracefully.
    """
    model_cls = model_info["class"]
    params = dict(model_info["params"])
    model_instance = model_cls(**params)
    supports_imp = model_info.get("supports_importance", False)
    
    target_encoder = None
    y_tr_encoded = y_train.copy()
    y_te_encoded = y_test.copy()
    
    # Target Encoding for Classification if non-numeric
    if problem_type == "Classification":
        if not pd.api.types.is_numeric_dtype(y_train) or isinstance(y_train.iloc[0], (str, bool)):
            target_encoder = LabelEncoder()
            y_tr_encoded = target_encoder.fit_transform(y_train)
            y_te_encoded = target_encoder.transform(y_test)
            
    # Assemble full Pipeline: Preprocessor + Estimator
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model_instance)
    ])
    
    timer = Timer()
    try:
        # 1. Perform Leak-Free Cross-Validation on Training Split
        if problem_type == "Classification":
            scoring_metric = "f1_weighted"
            # Handle stratified folds gracefully
            min_class_samples = pd.Series(y_tr_encoded).value_counts().min()
            actual_cv = min(cv_folds, max(2, min_class_samples))
            cv_splitter = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=random_state)
        else:
            scoring_metric = "r2"
            actual_cv = min(cv_folds, len(y_tr_encoded))
            cv_splitter = KFold(n_splits=actual_cv, shuffle=True, random_state=random_state)
            
        with timer:
            cv_results = cross_validate(
                full_pipeline, 
                X_train, 
                y_tr_encoded, 
                cv=cv_splitter, 
                scoring=scoring_metric,
                return_train_score=False
            )
            
        cv_scores = cv_results["test_score"].tolist()
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        
        # 2. Fit Full Pipeline on Training Data
        with timer:
            full_pipeline.fit(X_train, y_tr_encoded)
            
        train_time = timer.elapsed
        
        # 3. Model Predictions on Test Split
        y_pred = full_pipeline.predict(X_test)
        
        y_proba = None
        if problem_type == "Classification":
            if hasattr(full_pipeline, "predict_proba"):
                try:
                    y_proba = full_pipeline.predict_proba(X_test)
                except Exception:
                    y_proba = None
            elif hasattr(full_pipeline, "decision_function"):
                try:
                    dfunc = full_pipeline.decision_function(X_test)
                    if dfunc.ndim == 1:
                        # Convert to 2D probability proxy via sigmoid
                        p1 = 1.0 / (1.0 + np.exp(-dfunc))
                        y_proba = np.vstack([1.0 - p1, p1]).T
                except Exception:
                    y_proba = None
                    
        # 4. Evaluate Metrics
        if problem_type == "Classification":
            unique_classes = target_encoder.classes_ if target_encoder is not None else np.unique(y_tr_encoded)
            metrics = evaluate_classification(
                y_true=y_te_encoded,
                y_pred=y_pred,
                y_proba=y_proba,
                class_names=unique_classes.tolist() if isinstance(unique_classes, np.ndarray) else unique_classes
            )
        else:
            metrics = evaluate_regression(
                y_true=y_te_encoded,
                y_pred=y_pred
            )
            
        # 5. Extract Feature Names and Feature Importance
        fitted_prep = full_pipeline.named_steps["preprocessor"]
        fitted_estimator = full_pipeline.named_steps["model"]
        
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_train.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
        
        feature_names = get_feature_names(fitted_prep, num_cols, cat_cols)
        
        importance_df = None
        if supports_imp:
            try:
                importances = None
                if hasattr(fitted_estimator, "feature_importances_"):
                    importances = fitted_estimator.feature_importances_
                elif hasattr(fitted_estimator, "coef_"):
                    coef = fitted_estimator.coef_
                    importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
                    
                if importances is not None and len(importances) == len(feature_names):
                    imp_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False)
                    importance_df = imp_df.head(20)  # Top 20 features
            except Exception as e:
                logger.warning(f"Could not extract feature importances for {model_name}: {e}")
                importance_df = None
                
        return SingleModelResult(
            model_name=model_name,
            problem_type=problem_type,
            fitted_pipeline=full_pipeline,
            target_encoder=target_encoder,
            train_time_sec=train_time,
            cv_mean=cv_mean,
            cv_std=cv_std,
            cv_scores=cv_scores,
            metrics=metrics,
            y_pred=y_pred,
            y_proba=y_proba,
            feature_names=feature_names,
            feature_importances=importance_df,
            hyperparameters=params,
            supports_importance=supports_imp,
            success=True,
            error_message=None
        )
        
    except Exception as err:
        logger.error(f"Error training model {model_name}: {err}", exc_info=True)
        return SingleModelResult(
            model_name=model_name,
            problem_type=problem_type,
            fitted_pipeline=None,
            target_encoder=None,
            train_time_sec=timer.elapsed,
            cv_mean=0.0,
            cv_std=0.0,
            cv_scores=[],
            metrics=None,
            y_pred=None,
            y_proba=None,
            feature_names=[],
            feature_importances=None,
            hyperparameters=params,
            supports_importance=supports_imp,
            success=False,
            error_message=str(err)
        )
