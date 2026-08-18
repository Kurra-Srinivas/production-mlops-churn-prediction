#!/usr/bin/env python3
"""
E2E Training Pipeline Orchestrator.

Lifecycle:
    1. Data Ingestion
    2. Data Validation
    3. Cleaning / Preprocessing
    4. Stratified Train / Val / Test Split
    5. Fitting Preprocessing Pipeline (No data leakage)
    6. Baseline Models (Logistic Regression, Random Forest)
    7. Primary Model (XGBoost) + Optional Optuna Tuning
    8. Multi-metric Evaluation & Automatic Business Cost Threshold Optimization
    9. Artifact Persistence (Model + Pipeline + Schema)

Usage:
    python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn [--tune] [--n_trials 30]
"""

import os
import sys

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import argparse
import time

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Project root resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.pipeline.evaluate import (
    compute_business_cost,
    evaluate_predictions,
    run_threshold_analysis,
    select_optimal_threshold,
)
from src.pipeline.preprocess import (
    build_preprocessing_pipeline,
    get_feature_names,
    save_pipeline,
)
from src.pipeline.train import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from src.pipeline.tune import run_optuna_study
from src.utils.validate_data import validate_telco_data


def load_config():
    config_path = os.path.join(project_root, "configs", "model_config.yaml")
    model_cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            model_cfg = yaml.safe_load(f) or {}

    biz_path = os.path.join(project_root, "configs", "business_config.yaml")
    biz_cfg = {}
    if os.path.exists(biz_path):
        with open(biz_path) as f:
            biz_cfg = yaml.safe_load(f) or {}

    return model_cfg, biz_cfg


def main(args):
    config, biz_config = load_config()
    cost_fn = float(biz_config.get("business_cost", {}).get("cost_false_negative", 1200.0))
    cost_fp = float(biz_config.get("business_cost", {}).get("cost_false_positive", 50.0))

    mlruns_path = (
        args.mlflow_uri
        or f"file:///{os.path.abspath(os.path.join(project_root, 'mlruns')).replace(os.sep, '/')}"
    )
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    # 1. Ingestion
    print("\n[1/7] Loading data...")
    df = load_data(args.input)
    print(f"   Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. Validation
    print("\n[2/7] Validating data quality...")
    is_valid, failed = validate_telco_data(df)
    if not is_valid:
        print(f"[ERROR] Data validation failed on checks: {failed}")
        sys.exit(1)

    # 3. Clean target & drop customerID
    print("\n[3/7] Preprocessing dataset...")
    df = preprocess_data(df)

    # 4. Stratified Splits (Train: 70%, Val: 10%, Test: 20%)
    print("\n[4/7] Creating Stratified Splits (Train / Val / Test)...")
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Split 80% train+val, 20% test
    X_train_val, X_test_raw, y_train_val, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Split 87.5% train (70% total), 12.5% val (10% total)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
    )

    print(f"   Train: {len(X_train_raw):,} | Val: {len(X_val_raw):,} | Test: {len(X_test_raw):,}")

    # 5. Preprocessing Pipelines
    print("\n[5/7] Fitting Preprocessing Pipelines...")
    tree_pipeline = build_preprocessing_pipeline(scale_numerics=False)
    X_train_tree = tree_pipeline.fit_transform(X_train_raw)
    X_val_tree = tree_pipeline.transform(X_val_raw)
    X_test_tree = tree_pipeline.transform(X_test_raw)

    scaled_pipeline = build_preprocessing_pipeline(scale_numerics=True)
    X_train_scaled = scaled_pipeline.fit_transform(X_train_raw)
    X_test_scaled = scaled_pipeline.transform(X_test_raw)

    feature_names = get_feature_names(tree_pipeline)
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    pipeline_path = os.path.join(artifacts_dir, "pipeline.pkl")
    save_pipeline(tree_pipeline, pipeline_path)

    feature_cols_path = os.path.join(artifacts_dir, "feature_columns.txt")
    with open(feature_cols_path, "w") as f:
        f.write("\n".join(feature_names))

    # 6. Train Baselines & Primary Models
    print("\n[6/7] Training Models & Tracking with MLflow...")

    eval_threshold = args.threshold if args.threshold is not None else 0.50

    # --- Run 1: Logistic Regression Baseline ---
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        mlflow.set_tag("model_family", "baseline_linear")
        lr_model = train_logistic_regression(X_train_scaled, y_train)
        lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
        lr_metrics = evaluate_predictions(y_test, lr_proba, threshold=eval_threshold)
        mlflow.log_metrics(lr_metrics)
        mlflow.sklearn.log_model(lr_model, name="model")
        print(
            f"   [LR Baseline]  PR-AUC: {lr_metrics['pr_auc']:.3f} | ROC-AUC: {lr_metrics['roc_auc']:.3f} | F1: {lr_metrics['f1']:.3f}"
        )

    # --- Run 2: Random Forest Baseline ---
    with mlflow.start_run(run_name="Random_Forest_Baseline"):
        mlflow.set_tag("model_family", "baseline_ensemble")
        rf_model = train_random_forest(X_train_tree, y_train)
        rf_proba = rf_model.predict_proba(X_test_tree)[:, 1]
        rf_metrics = evaluate_predictions(y_test, rf_proba, threshold=eval_threshold)
        mlflow.log_metrics(rf_metrics)
        mlflow.sklearn.log_model(rf_model, name="model")
        print(
            f"   [RF Baseline]  PR-AUC: {rf_metrics['pr_auc']:.3f} | ROC-AUC: {rf_metrics['roc_auc']:.3f} | F1: {rf_metrics['f1']:.3f}"
        )

    # --- Run 3: XGBoost (Primary Model with optional Optuna & Automatic Threshold Selection) ---
    with mlflow.start_run(run_name="XGBoost_Primary"):
        mlflow.set_tag("model_family", "gradient_boosting")
        xgb_params = config.get("xgboost", {})

        if args.tune:
            print(f"\n   [OPTUNA] Running Optuna HPO ({args.n_trials} trials)...")
            best_optuna_params = run_optuna_study(
                X_train_tree, y_train, X_val_tree, y_val, n_trials=args.n_trials
            )
            xgb_params.update(best_optuna_params)
            mlflow.set_tag("tuned", "true")
        else:
            mlflow.set_tag("tuned", "false")

        # --- Automatic Threshold Optimization on Validation Data ---
        # 1. Fit model on Train only to generate validation probabilities
        val_model = train_xgboost(X_train_tree, y_train, params=xgb_params)
        val_proba = val_model.predict_proba(X_val_tree)[:, 1]

        # 2. Run threshold sweep and compute business costs on validation set
        val_cost_df = run_threshold_analysis(
            y_true=y_val,
            proba=val_proba,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )

        if args.threshold is not None:
            chosen_threshold = args.threshold
            print(f"\n   [THRESHOLD] Using manually specified threshold: {chosen_threshold:.3f}")
        else:
            chosen_threshold = select_optimal_threshold(val_cost_df, strategy="business_cost")
            print(
                f"\n   [THRESHOLD] Automatically selected cost-optimal threshold from validation set: {chosen_threshold:.3f}"
            )

        # Compute validation performance at chosen threshold
        val_cost_metrics = compute_business_cost(
            y_true=y_val,
            proba=val_proba,
            threshold=chosen_threshold,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )
        val_eval_metrics = evaluate_predictions(y_val, val_proba, threshold=chosen_threshold)

        print(f"   [Validation Metrics @ Threshold {chosen_threshold:.3f}]:")
        print(
            f"      Precision: {val_eval_metrics['precision']:.4f} | Recall: {val_eval_metrics['recall']:.4f} | F1: {val_eval_metrics['f1']:.4f} | PR-AUC: {val_eval_metrics['pr_auc']:.4f}"
        )
        print(
            f"      FP: {val_eval_metrics['fp']} | FN: {val_eval_metrics['fn']} | Total Business Cost: ${val_cost_metrics['total_cost']:,.2f}"
        )

        mlflow.log_params(xgb_params)
        mlflow.log_param("threshold", chosen_threshold)
        mlflow.log_param("cost_fn", cost_fn)
        mlflow.log_param("cost_fp", cost_fp)
        mlflow.log_metric("val_optimal_threshold", chosen_threshold)
        mlflow.log_metric("val_total_business_cost", val_cost_metrics["total_cost"])

        # --- Final Model Retraining on Train + Val ---
        t0 = time.time()
        X_train_val_tree = tree_pipeline.transform(X_train_val)
        xgb_model = train_xgboost(X_train_val_tree, y_train_val, params=xgb_params)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)

        # --- Final Test Evaluation on Untouched Test Set using Frozen Threshold ---
        xgb_proba = xgb_model.predict_proba(X_test_tree)[:, 1]
        test_metrics = evaluate_predictions(y_test, xgb_proba, threshold=chosen_threshold)
        test_cost_metrics = compute_business_cost(
            y_true=y_test,
            proba=xgb_proba,
            threshold=chosen_threshold,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )

        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("test_total_business_cost", test_cost_metrics["total_cost"])
        mlflow.log_metric("test_cost_per_customer", test_cost_metrics["cost_per_customer"])

        print(f"\n   [Final TEST Metrics @ Threshold {chosen_threshold:.3f}]:")
        print(
            f"      PR-AUC: {test_metrics['pr_auc']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f}"
        )
        print(
            f"      Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}"
        )
        print(
            f"      TP: {test_metrics['tp']} | FP: {test_metrics['fp']} | TN: {test_metrics['tn']} | FN: {test_metrics['fn']}"
        )
        print(
            f"      Total Business Cost: ${test_cost_metrics['total_cost']:,.2f} | Cost/Customer: ${test_cost_metrics['cost_per_customer']:.2f}"
        )

        # Save model to local artifacts directory for direct fast serving
        model_dir = os.path.join(artifacts_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(xgb_model, os.path.join(model_dir, "model.pkl"))

        # Artifacts
        mlflow.log_artifact(pipeline_path)
        mlflow.log_artifact(feature_cols_path)
        mlflow.xgboost.log_model(xgb_model, name="model")

        y_pred = (xgb_proba >= chosen_threshold).astype(int)
        report = classification_report(y_test, y_pred, digits=3)
        mlflow.log_text(report, artifact_file="classification_report.txt")

    print("\n[7/7] Pipeline execution complete! View runs with: mlflow ui")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Telco Churn Training Pipeline")
    p.add_argument("--input", type=str, required=True, help="Path to raw dataset CSV")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (if omitted, automatically optimized on validation set)",
    )
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--tune", action="store_true", help="Enable Optuna hyperparameter tuning")
    p.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None)
    args = p.parse_args()
    main(args)
