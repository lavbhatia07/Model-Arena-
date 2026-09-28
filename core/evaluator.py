"""
Evaluation Metrics Module for ModelArena.
Computes comprehensive evaluation metrics for Classification and Regression models.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    labels: List[Any]
    roc_auc: Optional[float] = None
    roc_curve_data: Optional[Dict[str, Any]] = None


@dataclass
class RegressionMetrics:
    mae: float
    mse: float
    rmse: float
    r2: float
    residuals: np.ndarray


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[Any]] = None
) -> ClassificationMetrics:
    """
    Computes classification metrics including Accuracy, Precision, Recall, F1, 
    Confusion Matrix, and ROC-AUC / ROC Curve.
    """
    # Identify unique class values present in data/predictions
    unique_present = np.unique(np.concatenate([y_true, y_pred]))
    
    acc = float(accuracy_score(y_true, y_pred))
    
    is_binary = len(unique_present) == 2
    avg_strategy = "binary" if is_binary else "weighted"
    
    prec = float(precision_score(y_true, y_pred, average=avg_strategy, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=avg_strategy, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=avg_strategy, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_present)
    
    # Format labels for display
    if class_names is not None and len(class_names) == len(unique_present):
        display_labels = [str(lbl) for lbl in class_names]
    else:
        display_labels = [str(lbl) for lbl in unique_present]
    
    roc_auc_val: Optional[float] = None
    roc_curve_data: Optional[Dict[str, Any]] = None
    
    if y_proba is not None:
        try:
            if is_binary and y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                proba_pos = y_proba[:, 1]
                roc_auc_val = float(roc_auc_score(y_true, proba_pos))
                fpr, tpr, thresholds = roc_curve(y_true, proba_pos, pos_label=unique_present[1])
                roc_curve_data = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist()
                }
            elif not is_binary and y_proba.ndim == 2:
                # Multiclass ROC-AUC (OVR)
                roc_auc_val = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            roc_auc_val = None
            roc_curve_data = None
            
    return ClassificationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        confusion_matrix=cm,
        labels=display_labels,
        roc_auc=roc_auc_val,
        roc_curve_data=roc_curve_data
    )


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> RegressionMetrics:
    """
    Computes regression metrics: MAE, MSE, RMSE, R², and Residuals.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    residuals = y_true - y_pred
    
    return RegressionMetrics(
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,
        residuals=residuals
    )
