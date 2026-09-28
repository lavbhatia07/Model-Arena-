"""
Preprocessing Pipeline Module for ModelArena.
Builds robust, leak-free Scikit-Learn ColumnTransformer pipelines for tabular features.
"""
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numerical and categorical feature column names in X.
    
    Returns:
        (numerical_cols, categorical_cols)
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
    
    # Check for non-numeric/non-categorical columns if any remain
    remaining = [col for col in X.columns if col not in num_cols and col not in cat_cols]
    if remaining:
        # Default remaining columns to categorical
        cat_cols.extend(remaining)
        
    return num_cols, cat_cols


def create_preprocessor(
    num_cols: List[str], 
    cat_cols: List[str], 
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Creates a ColumnTransformer for feature preprocessing.
    
    Numerical Pipeline:
    - Median Imputation
    - StandardScaler (optional, default True)
    
    Categorical Pipeline:
    - Most Frequent Imputation
    - OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    transformers = []
    
    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(steps=num_steps)
        transformers.append(("num", num_pipeline, num_cols))
        
    if cat_cols:
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"  # Drop unlisted columns if any
    )
    
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, original_num_cols: List[str], original_cat_cols: List[str]) -> List[str]:
    """
    Safely retrieves feature names out of a fitted ColumnTransformer.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out fails
        output_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                output_names.extend([f"num__{c}" for c in cols])
            elif name == "cat":
                if hasattr(trans, "named_steps") and "encoder" in trans.named_steps:
                    try:
                        cat_out = trans.named_steps["encoder"].get_feature_names_out(cols)
                        output_names.extend([f"cat__{c}" for c in cat_out])
                    except Exception:
                        output_names.extend([f"cat__{c}" for c in cols])
                else:
                    output_names.extend([f"cat__{c}" for c in cols])
        return output_names
