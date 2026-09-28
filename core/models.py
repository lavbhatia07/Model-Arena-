"""
Model Registry Module for ModelArena.
Defines model registries for Classification and Regression tasks.
"""
from typing import Dict, Any, Type, List, Tuple
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


CLASSIFICATION_MODELS: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {
        "key": "Logistic Regression",
        "class": LogisticRegression,
        "params": {"max_iter": 1000, "random_state": 42},
        "description": "Linear classifier for binary and multiclass problems.",
        "supports_importance": True
    },
    "Random Forest Classifier": {
        "key": "Random Forest Classifier",
        "class": RandomForestClassifier,
        "params": {"n_estimators": 100, "random_state": 42},
        "description": "Ensemble of decision trees using bagging.",
        "supports_importance": True
    },
    "K-Nearest Neighbors": {
        "key": "K-Nearest Neighbors",
        "class": KNeighborsClassifier,
        "params": {"n_neighbors": 5},
        "description": "Instance-based non-parametric classifier.",
        "supports_importance": False
    },
    "Support Vector Machine": {
        "key": "Support Vector Machine",
        "class": SVC,
        "params": {"probability": True, "random_state": 42},
        "description": "Maximum margin hyperplane separator.",
        "supports_importance": False
    },
    "Decision Tree": {
        "key": "Decision Tree",
        "class": DecisionTreeClassifier,
        "params": {"random_state": 42},
        "description": "Tree-based recursive splitting model.",
        "supports_importance": True
    }
}

if HAS_XGBOOST:
    CLASSIFICATION_MODELS["XGBoost Classifier"] = {
        "key": "XGBoost Classifier",
        "class": XGBClassifier,
        "params": {"random_state": 42, "eval_metric": "logloss"},
        "description": "Gradient boosted decision trees implementation.",
        "supports_importance": True
    }


REGRESSION_MODELS: Dict[str, Dict[str, Any]] = {
    "Linear Regression": {
        "key": "Linear Regression",
        "class": LinearRegression,
        "params": {},
        "description": "Standard ordinary least squares linear regression.",
        "supports_importance": True
    },
    "Random Forest Regressor": {
        "key": "Random Forest Regressor",
        "class": RandomForestRegressor,
        "params": {"n_estimators": 100, "random_state": 42},
        "description": "Ensemble decision tree regressor.",
        "supports_importance": True
    },
    "K-Nearest Neighbors Regressor": {
        "key": "K-Nearest Neighbors Regressor",
        "class": KNeighborsRegressor,
        "params": {"n_neighbors": 5},
        "description": "Distance-based regression using k-nearest neighbors.",
        "supports_importance": False
    },
    "Support Vector Regressor": {
        "key": "Support Vector Regressor",
        "class": SVR,
        "params": {},
        "description": "Support Vector Machine tuned for regression.",
        "supports_importance": False
    },
    "Decision Tree Regressor": {
        "key": "Decision Tree Regressor",
        "class": DecisionTreeRegressor,
        "params": {"random_state": 42},
        "description": "Recursive tree splitting for continuous targets.",
        "supports_importance": True
    }
}

if HAS_XGBOOST:
    REGRESSION_MODELS["XGBoost Regressor"] = {
        "key": "XGBoost Regressor",
        "class": XGBRegressor,
        "params": {"random_state": 42},
        "description": "Gradient boosted decision tree regressor.",
        "supports_importance": True
    }


def get_available_models(problem_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns the model registry for the selected problem type.
    """
    if problem_type == "Classification":
        return CLASSIFICATION_MODELS
    elif problem_type == "Regression":
        return REGRESSION_MODELS
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


def instantiate_model(model_info: Dict[str, Any], **custom_params) -> Any:
    """
    Instantiates a model object with default and optional custom parameters.
    """
    cls = model_info["class"]
    params = dict(model_info["params"])
    params.update(custom_params)
    return cls(**params)
