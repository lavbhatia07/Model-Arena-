"""
Model Serialization and Package Management for ModelArena.
Saves and loads trained pipelines with full metadata via Joblib.
"""
from datetime import datetime
import io
from typing import Dict, Any, Union, Optional
import joblib

from core.trainer import SingleModelResult
from utils.helpers import get_logger

logger = get_logger("ModelManager")


def export_model_package(
    model_result: SingleModelResult,
    target_col: str,
    feature_cols: list
) -> bytes:
    """
    Bundles the fitted model pipeline, target encoder, and metadata into a Joblib binary payload.
    Returns bytes suitable for Streamlit download.
    """
    if not model_result.success or model_result.fitted_pipeline is None:
        raise ValueError("Cannot export an unsuccessful or un-fitted model result.")
        
    metrics_dict = {}
    if model_result.metrics:
        if model_result.problem_type == "Classification":
            metrics_dict = {
                "Accuracy": model_result.metrics.accuracy,
                "Precision": model_result.metrics.precision,
                "Recall": model_result.metrics.recall,
                "F1": model_result.metrics.f1,
                "ROC-AUC": model_result.metrics.roc_auc
            }
        else:
            metrics_dict = {
                "MAE": model_result.metrics.mae,
                "MSE": model_result.metrics.mse,
                "RMSE": model_result.metrics.rmse,
                "R²": model_result.metrics.r2
            }
            
    package = {
        "pipeline": model_result.fitted_pipeline,
        "target_encoder": model_result.target_encoder,
        "model_name": model_result.model_name,
        "problem_type": model_result.problem_type,
        "target_col": target_col,
        "expected_features": feature_cols,
        "transformed_feature_names": model_result.feature_names,
        "metrics_summary": metrics_dict,
        "cv_mean": model_result.cv_mean,
        "cv_std": model_result.cv_std,
        "hyperparameters": model_result.hyperparameters,
        "created_at": datetime.now().isoformat(),
        "platform": "ModelArena v1.0"
    }
    
    buffer = io.BytesIO()
    joblib.dump(package, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def load_model_package(source: Union[str, bytes, io.BytesIO]) -> Dict[str, Any]:
    """
    Loads a ModelArena joblib model package from filepath or bytes buffer.
    """
    if isinstance(source, bytes):
        buffer = io.BytesIO(source)
        package = joblib.load(buffer)
    elif isinstance(source, io.BytesIO):
        package = joblib.load(source)
    else:
        package = joblib.load(source)
        
    if not isinstance(package, dict) or "pipeline" not in package:
        raise ValueError("Loaded file is not a valid ModelArena export package.")
        
    return package
