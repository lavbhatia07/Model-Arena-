"""
Unit tests for Preprocessing pipeline module.
"""
import pytest
import pandas as pd
import numpy as np

from core.preprocessing import identify_feature_types, create_preprocessor, get_feature_names


def test_identify_feature_types():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "num2": [10, 20, 30],
        "cat1": ["a", "b", "c"],
        "bool1": [True, False, True]
    })
    num_cols, cat_cols = identify_feature_types(df)
    assert set(num_cols) == {"num1", "num2"}
    assert set(cat_cols) == {"cat1", "bool1"}


def test_preprocessor_imputation_and_scaling():
    X_train = pd.DataFrame({
        "age": [25.0, np.nan, 35.0, 45.0, 55.0],
        "gender": ["M", "F", np.nan, "M", "F"]
    })
    
    num_cols, cat_cols = identify_feature_types(X_train)
    preprocessor = create_preprocessor(num_cols, cat_cols, scale_numeric=True)
    
    # Fit only on train data
    X_trans = preprocessor.fit_transform(X_train)
    
    # Check transformed array shape
    # age -> 1 column, gender (M, F) -> 2 one-hot columns = total 3 columns
    assert X_trans.shape[0] == 5
    assert X_trans.shape[1] == 3
    assert not np.isnan(X_trans).any()


def test_preprocessor_handle_unknown_categories():
    X_train = pd.DataFrame({
        "num": [1, 2, 3],
        "city": ["Paris", "Tokyo", "London"]
    })
    X_test = pd.DataFrame({
        "num": [4],
        "city": ["New York"]  # Unseen category
    })
    
    preprocessor = create_preprocessor(["num"], ["city"])
    preprocessor.fit(X_train)
    
    # Should not raise error on unseen category due to handle_unknown='ignore'
    X_test_trans = preprocessor.transform(X_test)
    assert X_test_trans.shape == (1, 4)
    # The one-hot columns for unseen category will be all zeros
    assert X_test_trans[0, 1] == 0.0
    assert X_test_trans[0, 2] == 0.0
    assert X_test_trans[0, 3] == 0.0


def test_get_feature_names():
    X = pd.DataFrame({"age": [20, 30], "group": ["A", "B"]})
    preprocessor = create_preprocessor(["age"], ["group"])
    preprocessor.fit(X)
    
    names = get_feature_names(preprocessor, ["age"], ["group"])
    assert len(names) == 3
    assert any("age" in n for n in names)
    assert any("group" in n for n in names)
