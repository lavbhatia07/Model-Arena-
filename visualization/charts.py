"""
Plotly Chart Generators for ModelArena.
Creates interactive, highly-polished visualizations for benchmarks, model details, and metrics.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Modern sleek dark theme configuration
THEME_TEMPLATE = "plotly_dark"
COLOR_PRIMARY = "#4FACFE"
COLOR_ACCENT = "#00F2FE"
COLOR_SUCCESS = "#00F260"
COLOR_WARNING = "#F7B733"
COLOR_DANGER = "#FF416C"
COLOR_PURPLE = "#9D50BB"


def plot_leaderboard(
    df_leaderboard: pd.DataFrame, 
    primary_metric: str, 
    problem_type: str
) -> go.Figure:
    """
    Creates a bar chart comparing models based on the primary metric.
    """
    df_sorted = df_leaderboard.sort_values(
        by=primary_metric, 
        ascending=(problem_type == "Regression" and primary_metric in ["MAE", "MSE", "RMSE"])
    )
    
    fig = px.bar(
        df_sorted,
        x=primary_metric,
        y="Model",
        orientation="h",
        text=primary_metric,
        title=f"Model Performance Leaderboard — Primary Metric: {primary_metric}",
        color=primary_metric,
        color_continuous_scale="Viridis" if problem_type == "Classification" or primary_metric == "R²" else "Cividis_r",
        template=THEME_TEMPLATE
    )
    
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        height=max(400, len(df_leaderboard) * 55),
        margin=dict(l=20, r=40, t=50, b=30),
        xaxis_title=primary_metric,
        yaxis_title="Model Algorithm",
        coloraxis_showscale=False
    )
    return fig


def plot_metrics_comparison(
    df_leaderboard: pd.DataFrame, 
    problem_type: str
) -> go.Figure:
    """
    Creates a grouped bar chart comparing multiple evaluation metrics across models.
    """
    if problem_type == "Classification":
        metrics_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1", "CV Score"] if c in df_leaderboard.columns]
    else:
        metrics_cols = [c for c in ["MAE", "RMSE", "R²", "CV Score"] if c in df_leaderboard.columns]
        
    df_melted = df_leaderboard.melt(
        id_vars=["Model"],
        value_vars=metrics_cols,
        var_name="Metric",
        value_name="Score"
    )
    
    fig = px.bar(
        df_melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Comprehensive Multi-Metric Comparison",
        template=THEME_TEMPLATE,
        color_discrete_sequence=[COLOR_PRIMARY, COLOR_ACCENT, COLOR_SUCCESS, COLOR_WARNING, COLOR_PURPLE]
    )
    
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=50, b=40),
        xaxis_title="Model Algorithm",
        yaxis_title="Metric Value",
        legend_title="Metric"
    )
    return fig


def plot_training_time(df_leaderboard: pd.DataFrame) -> go.Figure:
    """
    Creates a bar chart comparing model training speeds in seconds.
    """
    df_sorted = df_leaderboard.sort_values(by="Training Time (s)", ascending=True)
    
    fig = px.bar(
        df_sorted,
        x="Training Time (s)",
        y="Model",
        orientation="h",
        text="Training Time (s)",
        title="Training & Cross-Validation Execution Time (Seconds)",
        template=THEME_TEMPLATE,
        color="Training Time (s)",
        color_continuous_scale="Sunset"
    )
    
    fig.update_traces(texttemplate="%{text:.3f} s", textposition="outside")
    fig.update_layout(
        height=max(350, len(df_leaderboard) * 45),
        margin=dict(l=20, r=40, t=50, b=30),
        xaxis_title="Seconds",
        yaxis_title="Model Algorithm",
        coloraxis_showscale=False
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], model_name: str) -> go.Figure:
    """
    Creates an annotated heatmap for a classification confusion matrix.
    """
    z = cm.tolist()
    
    fig = px.imshow(
        z,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        title=f"Confusion Matrix — {model_name}",
        template=THEME_TEMPLATE,
        text_auto=True
    )
    
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=40),
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    return fig


def plot_roc_curve(roc_data: Dict[str, Any], model_name: str) -> go.Figure:
    """
    Creates a Receiver Operating Characteristic (ROC) curve.
    """
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    
    fig = go.Figure()
    
    # Model ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"{model_name}",
        line=dict(color=COLOR_ACCENT, width=3)
    ))
    
    # Diagonal random guess line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Guess",
        line=dict(color="gray", width=2, dash="dash")
    ))
    
    fig.update_layout(
        title=f"ROC Curve — {model_name}",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        template=THEME_TEMPLATE,
        height=420,
        margin=dict(l=20, r=20, t=50, b=40),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05])
    )
    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    """
    Creates an Actual vs Predicted scatter plot for regression models.
    """
    min_val = min(float(y_true.min()), float(y_pred.min()))
    max_val = max(float(y_true.max()), float(y_pred.max()))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        name="Predictions",
        marker=dict(color=COLOR_PRIMARY, opacity=0.7, size=7)
    ))
    
    # Ideal y = x reference line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Ideal Perfect Fit (y = x)",
        line=dict(color=COLOR_SUCCESS, width=2, dash="dash")
    ))
    
    fig.update_layout(
        title=f"Actual vs Predicted — {model_name}",
        xaxis_title="Actual Target Values",
        yaxis_title="Predicted Target Values",
        template=THEME_TEMPLATE,
        height=420,
        margin=dict(l=20, r=20, t=50, b=40)
    )
    return fig


def plot_residual_analysis(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    """
    Creates a dual-panel chart: Residuals vs Predicted scatter + Residual distribution histogram.
    """
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Residuals vs Predicted", "Residual Distribution")
    )
    
    # Subplot 1: Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(color=COLOR_WARNING, opacity=0.7, size=7),
            name="Residuals"
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # Subplot 2: Histogram of Residuals
    fig.add_trace(
        go.Histogram(
            x=residuals,
            marker_color=COLOR_PURPLE,
            name="Distribution"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"Residual Analysis — {model_name}",
        template=THEME_TEMPLATE,
        height=420,
        margin=dict(l=20, r=20, t=50, b=40),
        showlegend=False
    )
    fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals (Actual - Predicted)", row=1, col=1)
    fig.update_xaxes(title_text="Residual Error", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str) -> go.Figure:
    """
    Creates a horizontal bar chart displaying feature importances for a model.
    """
    df_sorted = importance_df.sort_values(by="Importance", ascending=True)
    
    fig = px.bar(
        df_sorted,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Feature Importances — {model_name}",
        color="Importance",
        color_continuous_scale="Tealgrn",
        template=THEME_TEMPLATE
    )
    
    fig.update_layout(
        height=max(350, len(importance_df) * 30),
        margin=dict(l=20, r=20, t=50, b=30),
        xaxis_title="Importance Weight",
        yaxis_title="Preprocessed Feature Name",
        coloraxis_showscale=False
    )
    return fig
