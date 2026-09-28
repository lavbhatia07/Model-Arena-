"""
Problem Type Detector Module for ModelArena.
Automatically detects whether a target variable represents Classification or Regression.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class ProblemDetectionResult:
    problem_type: str  # "Classification" or "Regression"
    confidence: str    # "High" or "Medium"
    reasoning: str     # Explanation text
    num_unique_targets: int
    target_dtype: str
    num_classes: Optional[int] = None


def detect_problem_type(df: pd.DataFrame, target_col: str) -> ProblemDetectionResult:
    """
    Analyzes the target column and detects whether the problem is Classification or Regression.
    
    Rules:
    - Object, Category, String, Bool dtypes -> Classification
    - Float / Int dtypes:
      - Low cardinality (<= 10 unique values) or low unique ratio with integer values -> Classification
      - Continuous numerical values -> Regression
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
        
    s = df[target_col].dropna()
    num_unique = int(s.nunique())
    total_len = len(s)
    unique_ratio = num_unique / total_len if total_len > 0 else 0.0
    dtype = s.dtype
    dtype_str = str(dtype)
    
    # 1. Non-numeric types (object, string, category, bool) -> Classification
    if isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        return ProblemDetectionResult(
            problem_type="Classification",
            confidence="High",
            reasoning=f"Target column has categorical/text type ('{dtype_str}') containing {num_unique} discrete unique values.",
            num_unique_targets=num_unique,
            target_dtype=dtype_str,
            num_classes=num_unique
        )
        
    # 2. Check if values are floats with fractional parts
    if pd.api.types.is_float_dtype(dtype):
        # Check if all floats are actually integer values (e.g. 1.0, 2.0, 3.0)
        has_fractional = not np.all(np.equal(np.mod(s.values, 1), 0))
        if has_fractional and num_unique > 10:
            return ProblemDetectionResult(
                problem_type="Regression",
                confidence="High",
                reasoning=f"Target column is continuous float data type with {num_unique} unique numeric values.",
                num_unique_targets=num_unique,
                target_dtype=dtype_str,
                num_classes=None
            )
            
    # 3. Numeric (integer or integer-like float)
    if num_unique <= 10:
        return ProblemDetectionResult(
            problem_type="Classification",
            confidence="High" if num_unique <= 5 else "Medium",
            reasoning=f"Target column is numeric but low-cardinality discrete with only {num_unique} unique values.",
            num_unique_targets=num_unique,
            target_dtype=dtype_str,
            num_classes=num_unique
        )
    elif num_unique <= 20 and unique_ratio < 0.05:
        return ProblemDetectionResult(
            problem_type="Classification",
            confidence="Medium",
            reasoning=f"Target column is discrete integer numeric with low unique ratio ({unique_ratio:.1%}, {num_unique} unique classes).",
            num_unique_targets=num_unique,
            target_dtype=dtype_str,
            num_classes=num_unique
        )
    else:
        return ProblemDetectionResult(
            problem_type="Regression",
            confidence="High" if num_unique > 50 else "Medium",
            reasoning=f"Target column is continuous/high-cardinality numeric with {num_unique} unique values ({unique_ratio:.1%} ratio).",
            num_unique_targets=num_unique,
            target_dtype=dtype_str,
            num_classes=None
        )
