"""
Inference & Prediction Module for ModelArena.
Executes predictions on new tabular data using fitted model packages.
"""
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from utils.validation import validate_prediction_input
from utils.helpers import get_logger

logger = get_logger("Predictor")


def predict_on_dataframe(
    package: Dict[str, Any],
    df_new: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies a loaded ModelArena package to generate predictions on a new DataFrame.
    
    Returns:
        (results_dataframe, summary_info)
    """
    pipeline = package["pipeline"]
    target_encoder = package.get("target_encoder", None)
    expected_features = package["expected_features"]
    problem_type = package["problem_type"]
    target_col = package.get("target_col", "Target")
    
    # 1. Align features
    aligned_df, missing_cols, extra_cols = validate_prediction_input(df_new, expected_features)
    
    # 2. Run Pipeline Prediction
    raw_preds = pipeline.predict(aligned_df)
    
    # Inverse transform target if encoded
    if problem_type == "Classification" and target_encoder is not None:
        try:
            final_preds = target_encoder.inverse_transform(raw_preds)
        except Exception:
            final_preds = raw_preds
    else:
        final_preds = raw_preds
        
    # Build output dataframe
    results_df = df_new.copy()
    pred_col_name = f"Predicted_{target_col}"
    results_df[pred_col_name] = final_preds
    
    # Check for probability predictions in Classification
    proba_df = None
    if problem_type == "Classification" and hasattr(pipeline, "predict_proba"):
        try:
            probas = pipeline.predict_proba(aligned_df)
            classes = target_encoder.classes_ if target_encoder is not None else np.unique(raw_preds)
            
            for idx, cls_name in enumerate(classes):
                col_name = f"Probability_{cls_name}"
                results_df[col_name] = np.round(probas[:, idx], 4)
        except Exception as e:
            logger.warning(f"Could not compute prediction probabilities: {e}")
            
    summary_info = {
        "num_rows": len(df_new),
        "target_col": target_col,
        "prediction_column": pred_col_name,
        "missing_features_imputed": missing_cols,
        "ignored_extra_features": extra_cols,
        "model_used": package["model_name"]
    }
    
    return results_df, summary_info
