"""
Data and Target Validation Utilities for ModelArena.
"""
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np


class DatasetValidationError(Exception):
    """Custom exception raised when dataset validation fails."""
    pass


def validate_uploaded_file(file_obj) -> Tuple[bool, Optional[str]]:
    """
    Validates if an uploaded file object is non-empty and has a CSV extension/type.
    
    Returns:
        (is_valid, error_message)
    """
    if file_obj is None:
        return False, "No file was uploaded."
    
    name = getattr(file_obj, "name", "").lower()
    if not name.endswith(".csv"):
        return False, "Unsupported file format. Please upload a valid CSV file."
    
    size = getattr(file_obj, "size", None)
    if size is not None and size == 0:
        return False, "Uploaded file is empty (0 bytes)."
    
    return True, None


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validates raw dataframe structure.
    
    Checks:
    - Non-empty dataframe
    - Minimum rows (>= 5)
    - Minimum columns (>= 2)
    - Valid column names
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, "Input data is not a valid pandas DataFrame."
    
    if df.empty:
        return False, "The dataset contains no data (0 rows)."
    
    if df.shape[0] < 5:
        return False, f"Dataset has only {df.shape[0]} rows. At least 5 rows are required for benchmarking."
    
    if df.shape[1] < 2:
        return False, f"Dataset has only {df.shape[1]} column. At least 2 columns (1 feature + 1 target) are required."
    
    # Check for unnamed or duplicate column headers
    unnamed_cols = [str(c) for c in df.columns if "unnamed" in str(c).lower()]
    if len(unnamed_cols) == len(df.columns):
        return False, "Dataset columns appear to be unnamed or corrupted."
    
    return True, None


def validate_target_column(df: pd.DataFrame, target_col: str) -> Tuple[bool, Optional[str]]:
    """
    Validates the selected target column.
    
    Checks:
    - Target exists in dataframe
    - Target is not 100% missing
    - Target has at least 5 non-null values
    - Target has more than 1 distinct value
    - There is at least 1 feature column remaining
    """
    if target_col not in df.columns:
        return False, f"Selected target column '{target_col}' does not exist in the dataset."
    
    y = df[target_col].dropna()
    
    if len(y) == 0:
        return False, f"Target column '{target_col}' contains only missing values."
    
    if len(y) < 5:
        return False, f"Target column '{target_col}' has fewer than 5 non-null values ({len(y)} found)."
    
    unique_vals = y.nunique()
    if unique_vals < 2:
        return False, f"Target column '{target_col}' has only {unique_vals} unique value ('{y.iloc[0]}'). Learning requires at least 2 distinct target values."
    
    feature_cols = [col for col in df.columns if col != target_col]
    if len(feature_cols) == 0:
        return False, "No feature columns remaining after target selection."
    
    return True, None


def validate_prediction_input(
    df_pred: pd.DataFrame, 
    expected_features: List[str]
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Validates prediction input dataframe against expected model feature columns.
    
    Returns:
        (aligned_df, missing_cols, extra_cols)
    """
    current_cols = set(df_pred.columns)
    expected_set = set(expected_features)
    
    missing_cols = [col for col in expected_features if col not in current_cols]
    extra_cols = [col for col in df_pred.columns if col not in expected_set]
    
    aligned_df = df_pred.copy()
    
    # Fill missing columns with NaN so preprocessor handles imputation
    for col in missing_cols:
        aligned_df[col] = np.nan
        
    # Reorder to match exact trained feature order
    aligned_df = aligned_df[expected_features]
    
    return aligned_df, missing_cols, extra_cols
