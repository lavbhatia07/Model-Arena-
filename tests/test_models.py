"""
Unit tests for Model Registry module.
"""
import pytest
from core.models import (
    get_available_models, 
    instantiate_model, 
    CLASSIFICATION_MODELS, 
    REGRESSION_MODELS
)


def test_classification_registry_contains_required_models():
    models = get_available_models("Classification")
    assert "Logistic Regression" in models
    assert "Random Forest Classifier" in models
    assert "K-Nearest Neighbors" in models
    assert "Support Vector Machine" in models
    assert "Decision Tree" in models
    assert "XGBoost Classifier" in models


def test_regression_registry_contains_required_models():
    models = get_available_models("Regression")
    assert "Linear Regression" in models
    assert "Random Forest Regressor" in models
    assert "K-Nearest Neighbors Regressor" in models
    assert "Support Vector Regressor" in models
    assert "Decision Tree Regressor" in models
    assert "XGBoost Regressor" in models


def test_instantiate_model():
    model_info = CLASSIFICATION_MODELS["Random Forest Classifier"]
    instance = instantiate_model(model_info)
    assert hasattr(instance, "fit")
    assert hasattr(instance, "predict")
    assert instance.n_estimators == 100


def test_invalid_problem_type():
    with pytest.raises(ValueError):
        get_available_models("Unsupervised")
