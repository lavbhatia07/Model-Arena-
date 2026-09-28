"""
Dataset Analyzer Module for ModelArena.
Performs non-destructive analysis and statistics on tabular datasets.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from utils.helpers import format_bytes


def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes a DataFrame and returns comprehensive dataset stats.
    Does NOT mutate the original DataFrame.
    """
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    memory_bytes = df.memory_usage(deep=True).sum()
    
    # Classify column data types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    
    # Missing values analysis
    missing_series = df.isnull().sum()
    total_missing_cells = int(missing_series.sum())
    total_cells = total_rows * total_cols
    missing_pct = (total_missing_cells / total_cells * 100.0) if total_cells > 0 else 0.0
    
    missing_per_col = {
        col: {
            "count": int(count),
            "percentage": float(count / total_rows * 100.0) if total_rows > 0 else 0.0
        }
        for col, count in missing_series.items() if count > 0
    }
    
    # Duplicates analysis
    duplicate_rows = int(df.duplicated().sum())
    
    # Column details table
    column_details = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        unique_cnt = int(df[col].nunique(dropna=True))
        missing_cnt = int(df[col].isnull().sum())
        missing_p = float(missing_cnt / total_rows * 100.0) if total_rows > 0 else 0.0
        
        column_details.append({
            "Column": col,
            "Type": dtype_str,
            "Unique Values": unique_cnt,
            "Missing Count": missing_cnt,
            "Missing %": f"{missing_p:.2f}%",
            "Kind": "Numerical" if col in num_cols else ("Categorical" if col in cat_cols else "Other")
        })
    df_col_details = pd.DataFrame(column_details)
    
    # Descriptive statistics
    num_summary = df.describe(include=[np.number]).T if len(num_cols) > 0 else pd.DataFrame()
    cat_summary = df.describe(include=["object", "category", "bool"]).T if len(cat_cols) > 0 else pd.DataFrame()
    
    return {
        "rows": total_rows,
        "cols": total_cols,
        "memory_bytes": memory_bytes,
        "memory_formatted": format_bytes(memory_bytes),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "datetime_cols": datetime_cols,
        "total_missing_cells": total_missing_cells,
        "missing_percentage": missing_pct,
        "missing_per_col": missing_per_col,
        "duplicate_rows": duplicate_rows,
        "column_details_df": df_col_details,
        "num_summary": num_summary,
        "cat_summary": cat_summary,
        "head": df.head(10).copy()
    }
