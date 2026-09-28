"""
ModelArena — "Where Machine Learning Models Compete."
Complete Production Streamlit Web Application for Supervised Classical ML Benchmarking.
"""
import io
import time
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import streamlit as st

# Core imports
from utils.validation import validate_uploaded_file, validate_dataset, validate_target_column
from utils.helpers import format_bytes, format_metric
from core.analyzer import analyze_dataset
from core.detector import detect_problem_type, ProblemDetectionResult
from core.models import get_available_models
from core.benchmark import run_benchmark, BenchmarkResult
from core.model_manager import export_model_package, load_model_package
from core.predictor import predict_on_dataframe

# Visualization imports
from visualization.charts import (
    plot_leaderboard,
    plot_metrics_comparison,
    plot_training_time,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_actual_vs_predicted,
    plot_residual_analysis,
    plot_feature_importance
)


# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="ModelArena | Classical ML Benchmarking Platform",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Glassmorphism & Cyberpunk Dark CSS Aesthetics
st.markdown("""
<style>
    /* Main container background and text */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Card */
    .brand-header {
        background: linear-gradient(90deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.8rem;
    }
    
    .brand-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #818CF8, #C7D2FE, #6366F1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .brand-tagline {
        font-size: 1.1rem;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }
    
    /* Metric Card Styling */
    .arena-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    
    /* Detector Banner */
    .detector-box {
        background: rgba(49, 46, 129, 0.5);
        border-left: 5px solid #6366F1;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    
    /* Winner Recommendation Box */
    .winner-box {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.3) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1.2rem 0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #090D16;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "df_raw" not in st.session_state:
    st.session_state["df_raw"] = None
if "dataset_filename" not in st.session_state:
    st.session_state["dataset_filename"] = ""
if "target_col" not in st.session_state:
    st.session_state["target_col"] = None
if "detection_result" not in st.session_state:
    st.session_state["detection_result"] = None
if "confirmed_problem_type" not in st.session_state:
    st.session_state["confirmed_problem_type"] = None
if "benchmark_result" not in st.session_state:
    st.session_state["benchmark_result"] = None
if "selected_model_name" not in st.session_state:
    st.session_state["selected_model_name"] = None


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚔️ MODELARENA")
    st.caption("Automated Supervised ML Benchmark")
    
    page = st.radio(
        "Navigation Menu",
        options=[
            "🏠 Home",
            "📁 Dataset Upload & Analysis",
            "🎯 Target & Task Selection",
            "⚡ Benchmark Engine",
            "🏆 Leaderboard",
            "📊 Visualizations",
            "🔍 Model Deep Dive",
            "💾 Model Export",
            "🔮 Prediction Engine",
            "ℹ️ About & Methodology"
        ],
        index=0
    )
    
    st.markdown("---")
    
    # Active Dataset Status Summary Widget
    if st.session_state["df_raw"] is not None:
        st.markdown("#### 📊 Active Dataset")
        st.write(f"**Source:** `{st.session_state['dataset_filename']}`")
        st.write(f"**Shape:** `{st.session_state['df_raw'].shape[0]} rows × {st.session_state['df_raw'].shape[1]} cols`")
        if st.session_state["target_col"]:
            st.write(f"**Target:** `{st.session_state['target_col']}`")
        if st.session_state["confirmed_problem_type"]:
            st.write(f"**Task:** `{st.session_state['confirmed_problem_type']}`")
    else:
        st.info("No dataset uploaded yet.")


# ==============================================================================
# HEADER BANNER
# ==============================================================================
st.markdown("""
<div class="brand-header">
    <h1 class="brand-title">MODELARENA</h1>
    <p class="brand-tagline">Where Machine Learning Models Compete.</p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: HOME
# ==============================================================================
if page == "🏠 Home":
    st.markdown("### Welcome to ModelArena")
    st.write(
        "ModelArena is an automated, production-grade benchmarking platform for **Supervised Classical Machine Learning**. "
        "Upload any tabular CSV dataset, specify your target column, and let ModelArena automatically detect the problem type, "
        "preprocess features without data leakage, evaluate competing algorithms, and deliver a clear model leaderboard."
    )
    
    st.markdown("#### 🚀 Quick Start — Try with Sample Data")
    st.write("Don't have a CSV file handy? Click below to instantly load a benchmark dataset:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Load Sample Classification Dataset (Loan Approval)", use_container_width=True):
            try:
                sample_df = pd.read_csv("data/classification_sample.csv")
                st.session_state["df_raw"] = sample_df
                st.session_state["dataset_filename"] = "classification_sample.csv"
                st.session_state["target_col"] = "loan_approved"
                st.session_state["detection_result"] = detect_problem_type(sample_df, "loan_approved")
                st.session_state["confirmed_problem_type"] = st.session_state["detection_result"].problem_type
                st.session_state["benchmark_result"] = None
                st.success("Successfully loaded Sample Classification Dataset! Proceed to 'Dataset Upload & Analysis' or 'Benchmark Engine'.")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample file: {e}")
                
    with col2:
        if st.button("🏡 Load Sample Regression Dataset (House Prices)", use_container_width=True):
            try:
                sample_df = pd.read_csv("data/regression_sample.csv")
                st.session_state["df_raw"] = sample_df
                st.session_state["dataset_filename"] = "regression_sample.csv"
                st.session_state["target_col"] = "house_price"
                st.session_state["detection_result"] = detect_problem_type(sample_df, "house_price")
                st.session_state["confirmed_problem_type"] = st.session_state["detection_result"].problem_type
                st.session_state["benchmark_result"] = None
                st.success("Successfully loaded Sample Regression Dataset! Proceed to 'Dataset Upload & Analysis' or 'Benchmark Engine'.")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample file: {e}")

    st.markdown("---")
    st.markdown("#### ⚡ Platform Features & Capabilities")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        ##### 1. Automated Detection
        - Smart problem type inference (Classification vs. Regression).
        - Confidence reasoning & user override support.
        - Non-destructive dataset profiling.
        """)
    with c2:
        st.markdown("""
        ##### 2. Leak-Free Preprocessing
        - Scikit-Learn `ColumnTransformer` + `Pipeline`.
        - Imputation, Scaling, & One-Hot Encoding fitted strictly on train split / CV folds.
        - Zero data leakage guarantee.
        """)
    with c3:
        st.markdown("""
        ##### 3. Multi-Model Arena
        - Logistic / Linear Regression, Random Forests, XGBoost, SVM/SVR, KNN, Decision Trees.
        - 5-Fold Cross-Validation with stratified splitting.
        - Exportable `.joblib` model bundles.
        """)


# ==============================================================================
# PAGE 2: DATASET UPLOAD & ANALYSIS
# ==============================================================================
elif page == "📁 Dataset Upload & Analysis":
    st.markdown("### 📁 Dataset Upload & Analysis")
    
    uploaded_file = st.file_uploader("Upload your labeled tabular CSV dataset:", type=["csv"])
    
    if uploaded_file is not None:
        is_valid, err_msg = validate_uploaded_file(uploaded_file)
        if not is_valid:
            st.error(f"❌ Upload Error: {err_msg}")
        else:
            try:
                df = pd.read_csv(uploaded_file)
                is_valid_ds, ds_err = validate_dataset(df)
                if not is_valid_ds:
                    st.error(f"❌ Dataset Error: {ds_err}")
                else:
                    st.session_state["df_raw"] = df
                    st.session_state["dataset_filename"] = uploaded_file.name
                    # Reset target selection on new upload
                    st.session_state["target_col"] = None
                    st.session_state["benchmark_result"] = None
                    st.success(f"✅ Dataset successfully uploaded: `{uploaded_file.name}` ({df.shape[0]} rows, {df.shape[1]} columns)")
            except Exception as e:
                st.error(f"❌ Failed to parse CSV file: {e}")
                
    if st.session_state["df_raw"] is not None:
        df = st.session_state["df_raw"]
        stats = analyze_dataset(df)
        
        st.markdown("#### 📊 Dataset Overview Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", stats["rows"])
        m2.metric("Columns", stats["cols"])
        m3.metric("Numerical Cols", len(stats["num_cols"]))
        m4.metric("Categorical Cols", len(stats["cat_cols"]))
        m5.metric("Duplicate Rows", stats["duplicate_rows"])
        
        st.markdown("---")
        st.markdown("#### 🔍 Dataset Preview (First 10 Rows)")
        st.dataframe(stats["head"], use_container_width=True)
        
        tab_summary, tab_cols, tab_missing = st.tabs(["Numerical Summary", "Column Details", "Missing Values Breakdown"])
        
        with tab_summary:
            if not stats["num_summary"].empty:
                st.dataframe(stats["num_summary"].style.format("{:.3f}"), use_container_width=True)
            else:
                st.info("No numerical columns found in this dataset.")
                
        with tab_cols:
            st.dataframe(stats["column_details_df"], use_container_width=True)
            
        with tab_missing:
            if stats["total_missing_cells"] == 0:
                st.success("🎉 No missing values detected in the entire dataset!")
            else:
                st.warning(f"Total Missing Cells: **{stats['total_missing_cells']}** ({stats['missing_percentage']:.2f}% of dataset)")
                missing_df = pd.DataFrame([
                    {"Column": col, "Missing Count": val["count"], "Missing %": f"{val['percentage']:.2f}%"}
                    for col, val in stats["missing_per_col"].items()
                ])
                st.dataframe(missing_df, use_container_width=True)


# ==============================================================================
# PAGE 3: TARGET & TASK SELECTION
# ==============================================================================
elif page == "🎯 Target & Task Selection":
    st.markdown("### 🎯 Target Column Selection & Problem Detection")
    
    if st.session_state["df_raw"] is None:
        st.warning("⚠️ Please upload a dataset first under 'Dataset Upload & Analysis' or load a sample from 'Home'.")
    else:
        df = st.session_state["df_raw"]
        cols = list(df.columns)
        
        default_idx = cols.index(st.session_state["target_col"]) if st.session_state["target_col"] in cols else len(cols) - 1
        
        selected_target = st.selectbox(
            "Select the Target (Output / Y) column from your dataset:",
            options=cols,
            index=default_idx,
            help="ModelArena will learn to predict this target column using remaining feature columns."
        )
        
        is_valid_target, target_err = validate_target_column(df, selected_target)
        
        if not is_valid_target:
            st.error(f"❌ Invalid Target Column: {target_err}")
        else:
            st.session_state["target_col"] = selected_target
            
            # Run automatic detection
            detection = detect_problem_type(df, selected_target)
            st.session_state["detection_result"] = detection
            
            if st.session_state["confirmed_problem_type"] is None:
                st.session_state["confirmed_problem_type"] = detection.problem_type
                
            st.markdown(f"""
            <div class="detector-box">
                <h4>🤖 Problem Type Detector Output</h4>
                <p><strong>Detected Task:</strong> <span style="font-size:1.2rem; color:#818CF8; font-weight:700;">{detection.problem_type}</span> (Confidence: <em>{detection.confidence}</em>)</p>
                <p><strong>Reasoning:</strong> {detection.reasoning}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### ⚙️ Task Confirmation & Manual Override")
            st.write(f"Current Active Task: **`{st.session_state['confirmed_problem_type']}`**")
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("✅ Confirm Classification", use_container_width=True):
                    st.session_state["confirmed_problem_type"] = "Classification"
                    st.session_state["benchmark_result"] = None
                    st.success("Task set to Classification.")
                    st.rerun()
            with col_b2:
                if st.button("🔄 Change to Regression", use_container_width=True):
                    st.session_state["confirmed_problem_type"] = "Regression"
                    st.session_state["benchmark_result"] = None
                    st.success("Task overridden to Regression.")
                    st.rerun()


# ==============================================================================
# PAGE 4: BENCHMARK ENGINE
# ==============================================================================
elif page == "⚡ Benchmark Engine":
    st.markdown("### ⚡ ModelArena Benchmark Runner")
    
    if st.session_state["df_raw"] is None or st.session_state["target_col"] is None:
        st.warning("⚠️ Please select a valid target column under 'Target & Task Selection' before benchmarking.")
    else:
        df = st.session_state["df_raw"]
        target_col = st.session_state["target_col"]
        problem_type = st.session_state["confirmed_problem_type"]
        
        st.markdown(f"**Target Column:** `{target_col}` | **Problem Type:** `{problem_type}`")
        
        available_registry = get_available_models(problem_type)
        all_model_names = list(available_registry.keys())
        
        st.markdown("#### 1. Select Models to Train")
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            select_all = st.checkbox("Select All Models", value=True)
            
        selected_models = []
        if select_all:
            selected_models = all_model_names
        else:
            selected_models = st.multiselect(
                "Choose specific algorithms:",
                options=all_model_names,
                default=all_model_names[:3]
            )
            
        st.markdown("#### 2. Benchmark Hyperparameters")
        c_p1, c_p2, c_p3 = st.columns(3)
        
        with c_p1:
            test_size_val = st.slider("Test Split Ratio", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
        with c_p2:
            cv_folds_val = st.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5, step=1)
        with c_p3:
            if problem_type == "Classification":
                metric_options = ["F1", "Accuracy", "Precision", "Recall", "ROC-AUC"]
            else:
                metric_options = ["RMSE", "MAE", "R²", "MSE"]
            primary_metric_val = st.selectbox("Primary Leaderboard Metric", options=metric_options, index=0)
            
        st.markdown("---")
        
        if st.button("🚀 Run ModelArena Benchmark", use_container_width=True, type="primary"):
            if not selected_models:
                st.error("Please select at least one algorithm to run.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, model_name):
                    pct = int((current / total) * 100) if total > 0 else 0
                    progress_bar.progress(min(pct, 100))
                    status_text.text(f"Training Model {current}/{total}: {model_name}...")
                    
                try:
                    with st.spinner("Executing leak-free preprocessing, model training, and cross-validation..."):
                        bench_result = run_benchmark(
                            df=df,
                            target_col=target_col,
                            problem_type=problem_type,
                            selected_model_keys=selected_models,
                            test_size=test_size_val,
                            cv_folds=cv_folds_val,
                            primary_metric=primary_metric_val,
                            random_state=42,
                            progress_callback=update_progress
                        )
                        st.session_state["benchmark_result"] = bench_result
                        st.session_state["selected_model_name"] = bench_result.recommended_model_name
                        
                    progress_bar.progress(100)
                    status_text.success("🎉 ModelArena Benchmark Completed Successfully!")
                    st.balloons()
                except Exception as ex:
                    st.error(f"❌ Benchmark Execution Error: {ex}")
                    
        # Display summary of completed run if available
        if st.session_state["benchmark_result"] is not None:
            res = st.session_state["benchmark_result"]
            st.markdown("---")
            st.markdown("### 🏆 Benchmark Summary")
            st.success(f"Benchmarked **{len(res.model_results)}** models. Top performer: **{res.recommended_model_name}**")
            
            if res.failed_models:
                st.warning(f"⚠️ {len(res.failed_models)} model(s) failed during execution.")
                with st.expander("View Failed Model Errors"):
                    for err in res.failed_models:
                        st.error(f"Model: **{err['model_name']}** — Error: {err['error']}")


# ==============================================================================
# PAGE 5: LEADERBOARD
# ==============================================================================
elif page == "🏆 Leaderboard":
    st.markdown("### 🏆 ModelArena Leaderboard")
    
    if st.session_state["benchmark_result"] is None:
        st.info("💡 Run the benchmark under '⚡ Benchmark Engine' to generate the leaderboard.")
    else:
        res: BenchmarkResult = st.session_state["benchmark_result"]
        
        st.markdown(f"""
        <div class="winner-box">
            <h3 style="margin:0; color:#10B981;">🥇 Top Recommended Model: {res.recommended_model_name}</h3>
            <p style="margin-top:0.4rem; color:#E2E8F0;">{res.recommendation_reason}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"#### 📊 Complete Model Standings (Sorted by `{res.primary_metric}`)")
        
        # Interactive metric sorting override
        metric_cols = list(res.leaderboard_df.columns[1:])
        selected_sort = st.selectbox("Sort Leaderboard by:", options=metric_cols, index=metric_cols.index(res.primary_metric) if res.primary_metric in metric_cols else 0)
        
        asc = (res.problem_type == "Regression" and selected_sort in ["MAE", "MSE", "RMSE", "Training Time (s)"])
        sorted_df = res.leaderboard_df.sort_values(by=selected_sort, ascending=asc).reset_index(drop=True)
        
        st.dataframe(
            sorted_df.style.highlight_max(subset=[col for col in ["Accuracy", "F1", "R²", "ROC-AUC"] if col in sorted_df.columns], color="#065F46")
                           .highlight_min(subset=[col for col in ["RMSE", "MAE", "MSE"] if col in sorted_df.columns], color="#065F46"),
            use_container_width=True
        )


# ==============================================================================
# PAGE 6: VISUALIZATIONS
# ==============================================================================
elif page == "📊 Visualizations":
    st.markdown("### 📊 Interactive Benchmark Visualizations")
    
    if st.session_state["benchmark_result"] is None:
        st.info("💡 Run the benchmark under '⚡ Benchmark Engine' first to unlock interactive charts.")
    else:
        res: BenchmarkResult = st.session_state["benchmark_result"]
        
        st.plotly_chart(plot_leaderboard(res.leaderboard_df, res.primary_metric, res.problem_type), use_container_width=True)
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.plotly_chart(plot_metrics_comparison(res.leaderboard_df, res.problem_type), use_container_width=True)
        with col_v2:
            st.plotly_chart(plot_training_time(res.leaderboard_df), use_container_width=True)
            
        st.markdown("---")
        st.markdown("#### 🔬 Detailed Model Diagnostics")
        
        model_names = list(res.model_results.keys())
        selected_viz_model = st.selectbox("Select model for diagnostic plots:", options=model_names)
        
        m_result = res.model_results[selected_viz_model]
        
        if res.problem_type == "Classification":
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.plotly_chart(plot_confusion_matrix(m_result.metrics.confusion_matrix, m_result.metrics.labels, selected_viz_model), use_container_width=True)
            with v_col2:
                if m_result.metrics.roc_curve_data:
                    st.plotly_chart(plot_roc_curve(m_result.metrics.roc_curve_data, selected_viz_model), use_container_width=True)
                else:
                    st.info("ROC Curve not available for multiclass without probability thresholds or single class test set.")
        else:
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.plotly_chart(plot_actual_vs_predicted(res.y_test, m_result.y_pred, selected_viz_model), use_container_width=True)
            with v_col2:
                st.plotly_chart(plot_residual_analysis(res.y_test, m_result.y_pred, selected_viz_model), use_container_width=True)


# ==============================================================================
# PAGE 7: MODEL DEEP DIVE
# ==============================================================================
elif page == "🔍 Model Deep Dive":
    st.markdown("### 🔍 Model Hyperparameters & Feature Importance")
    
    if st.session_state["benchmark_result"] is None:
        st.info("💡 Run the benchmark under '⚡ Benchmark Engine' to inspect individual model details.")
    else:
        res: BenchmarkResult = st.session_state["benchmark_result"]
        model_names = list(res.model_results.keys())
        
        selected_model = st.selectbox("Choose model to inspect:", options=model_names, index=model_names.index(st.session_state.get("selected_model_name", model_names[0])))
        
        m_res = res.model_results[selected_model]
        
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("CV Mean Score", f"{m_res.cv_mean:.4f}")
        d2.metric("CV Standard Dev", f"{m_res.cv_std:.4f}")
        d3.metric("Training Time", f"{m_res.train_time_sec:.3f} s")
        d4.metric("Features Used", len(m_res.feature_names))
        
        st.markdown("---")
        
        tab_imp, tab_params, tab_cv = st.tabs(["Feature Importance", "Hyperparameters", "Cross-Validation Folds"])
        
        with tab_imp:
            if m_res.feature_importances is not None and not m_res.feature_importances.empty:
                st.plotly_chart(plot_feature_importance(m_res.feature_importances, selected_model), use_container_width=True)
            else:
                st.info(f"Model '{selected_model}' does not natively provide tree-based or linear feature importances.")
                
        with tab_params:
            st.json(m_res.hyperparameters)
            
        with tab_cv:
            cv_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(m_res.cv_scores))],
                "Score": m_res.cv_scores
            })
            st.dataframe(cv_df, use_container_width=True)


# ==============================================================================
# PAGE 8: MODEL EXPORT
# ==============================================================================
elif page == "💾 Model Export":
    st.markdown("### 💾 Export Trained Model Package")
    
    if st.session_state["benchmark_result"] is None:
        st.info("💡 Train models under '⚡ Benchmark Engine' to export your best pipeline.")
    else:
        res: BenchmarkResult = st.session_state["benchmark_result"]
        model_names = list(res.model_results.keys())
        
        selected_export_model = st.selectbox("Select model to download:", options=model_names, index=model_names.index(res.recommended_model_name))
        
        m_res = res.model_results[selected_export_model]
        
        st.markdown("#### Package Metadata Summary")
        st.json({
            "model_name": m_res.model_name,
            "problem_type": m_res.problem_type,
            "target_column": res.target_col,
            "cv_mean_score": m_res.cv_mean,
            "features_count": len(res.feature_cols),
            "pipeline_steps": [name for name, _ in m_res.fitted_pipeline.steps]
        })
        
        try:
            package_bytes = export_model_package(m_res, res.target_col, res.feature_cols)
            
            st.download_button(
                label=f"📥 Download '{selected_export_model}' Pipeline (.joblib)",
                data=package_bytes,
                file_name=f"modelarena_{selected_export_model.lower().replace(' ', '_')}.joblib",
                mime="application/octet-stream",
                use_container_width=True
            )
        except Exception as err:
            st.error(f"Error packaging model: {err}")


# ==============================================================================
# PAGE 9: PREDICTION ENGINE
# ==============================================================================
elif page == "🔮 Prediction Engine":
    st.markdown("### 🔮 Inference & Prediction Engine")
    
    loaded_pkg = None
    
    # 1. Option to use benchmarked model or uploaded package
    pred_source = st.radio("Prediction Model Source:", options=["Use Active Benchmarked Model", "Upload Saved .joblib Package"])
    
    if pred_source == "Use Active Benchmarked Model":
        if st.session_state["benchmark_result"] is None:
            st.warning("⚠️ No active benchmark result available. Please run benchmark first or upload a .joblib package.")
        else:
            res: BenchmarkResult = st.session_state["benchmark_result"]
            model_names = list(res.model_results.keys())
            sel_m = st.selectbox("Select active model:", options=model_names, index=model_names.index(res.recommended_model_name))
            m_res = res.model_results[sel_m]
            
            # Construct package dictionary in-memory
            loaded_pkg = {
                "pipeline": m_res.fitted_pipeline,
                "target_encoder": m_res.target_encoder,
                "model_name": m_res.model_name,
                "problem_type": m_res.problem_type,
                "target_col": res.target_col,
                "expected_features": res.feature_cols
            }
    else:
        model_file = st.file_uploader("Upload ModelArena .joblib model file:", type=["joblib", "pkl"])
        if model_file is not None:
            try:
                loaded_pkg = load_model_package(model_file.getvalue())
                st.success(f"Loaded model package: `{loaded_pkg['model_name']}` ({loaded_pkg['problem_type']})")
            except Exception as e:
                st.error(f"Invalid model package: {e}")
                
    if loaded_pkg is not None:
        st.markdown("---")
        st.markdown("#### Make Predictions")
        
        tab_batch, tab_single = st.tabs(["Batch Prediction (CSV)", "Single Sample Interactive Form"])
        
        with tab_batch:
            pred_csv = st.file_uploader("Upload new feature CSV file for prediction:", type=["csv"], key="pred_csv_uploader")
            if pred_csv is not None:
                try:
                    df_new = pd.read_csv(pred_csv)
                    results_df, summary = predict_on_dataframe(loaded_pkg, df_new)
                    
                    st.success(f"Generated predictions for {summary['num_rows']} rows using `{summary['model_used']}`.")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results CSV
                    csv_bytes = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_bytes,
                        file_name="modelarena_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as ex:
                    st.error(f"Prediction Error: {ex}")
                    
        with tab_single:
            st.write("Enter values for each feature:")
            expected_feats = loaded_pkg["expected_features"]
            
            single_input = {}
            cols_per_row = st.columns(min(3, len(expected_feats)))
            
            for idx, feat in enumerate(expected_feats):
                col_target = cols_per_row[idx % len(cols_per_row)]
                single_input[feat] = col_target.text_input(f"Feature: {feat}", value="0")
                
            if st.button("🔮 Generate Single Prediction", use_container_width=True):
                try:
                    # Convert inputs safely
                    formatted_input = {}
                    for k, v in single_input.items():
                        try:
                            formatted_input[k] = float(v)
                        except ValueError:
                            formatted_input[k] = v
                            
                    df_single = pd.DataFrame([formatted_input])
                    results_single, _ = predict_on_dataframe(loaded_pkg, df_single)
                    
                    pred_val = results_single[f"Predicted_{loaded_pkg.get('target_col', 'Target')}"].iloc[0]
                    
                    st.markdown(f"""
                    <div style="background:rgba(16, 185, 129, 0.2); border:1px solid #10B981; padding:1.2rem; border-radius:10px;">
                        <h3 style="margin:0; color:#10B981;">Prediction Output</h3>
                        <p style="font-size:1.4rem; font-weight:700; margin:0.4rem 0 0 0;">{pred_val}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as err:
                    st.error(f"Single prediction failed: {err}")


# ==============================================================================
# PAGE 10: ABOUT & METHODOLOGY
# ==============================================================================
elif page == "ℹ️ About & Methodology":
    st.markdown("### ℹ️ About ModelArena & Methodology")
    
    st.markdown("""
    #### 🎯 Scope & Boundaries
    ModelArena is specifically engineered for **Supervised Classical Machine Learning** on labeled tabular data.
    
    **Supported Tasks:**
    - **Classification** (Binary & Multiclass)
    - **Regression** (Continuous numerical targets)
    
    *Out of scope for this platform:* Deep Learning (CNN, RNN, LSTM, Transformers), Computer Vision, and Unsupervised Learning.
    
    ---
    
    #### 🛡️ Data Leakage Prevention Guarantee
    ModelArena follows strict statistical safeguards to prevent data leakage during preprocessing and cross-validation:
    
    1. **Train/Test Splitting Before Preprocessing:** The dataset is split into training (80%) and testing (20%) sets *before* fitting any imputer, scaler, or one-hot encoder.
    2. **Encapsulated Pipelines:** Preprocessing steps are encapsulated inside Scikit-Learn `ColumnTransformer` and `Pipeline` objects.
    3. **Leak-Free Cross-Validation:** Cross-validation uses `sklearn.model_selection.cross_validate` on the entire pipeline. In each fold, imputation, scaling, and encoding are fitted *strictly* on the fold's training split and applied to the validation split.
    
    ---
    
    #### 🧮 Algorithms in the Arena
    - **Classification:** Logistic Regression, Random Forest Classifier, K-Nearest Neighbors, Support Vector Machine (SVM), Decision Tree, XGBoost Classifier.
    - **Regression:** Linear Regression, Random Forest Regressor, K-Nearest Neighbors Regressor, Support Vector Regressor (SVR), Decision Tree Regressor, XGBoost Regressor.
    """)
