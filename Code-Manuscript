# ==========================
# VIRTUAL TOURISM MANUSCRIPT 
# ==========================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import optuna
import shap


# ============================================================
# 1. GENERAL CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS = 50
TEST_SIZE = 0.20

DATA_PATH = "virtual_tourism.csv"   # Replace with the real dataset file
TARGET_COL = "INT1"
REDUNDANT_INT_COLS = ["INT2", "INT3", "INT4"]

OUTPUT_DIR = "tourism_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_dataset(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")
    return df


# ============================================================
# 3. CATEGORICAL VARIABLE ENCODING
# ============================================================

def encode_categorical_columns(df):
    df_encoded = df.copy()

    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = le.fit_transform(df_encoded[col])

    return df_encoded


# ============================================================
# 4. CORRELATION ANALYSIS BETWEEN INT1, INT2, INT3, AND INT4
# ============================================================

def correlation_analysis(df):
    cols = [TARGET_COL] + [c for c in REDUNDANT_INT_COLS if c in df.columns]
    corr = df[cols].corr(method="pearson")
    corr.to_csv(f"{OUTPUT_DIR}/correlation_INT_variables.csv")

    print("\nPearson correlations among intention variables:")
    print(corr)

    return corr


# ============================================================
# 5. FEATURE MATRIX AND TARGET VARIABLE DEFINITION
# ============================================================

def prepare_features_target(df):
    df = df.copy()

    # Remove INT2, INT3, and INT4 due to their high correlation with INT1
    redundant = [c for c in REDUNDANT_INT_COLS if c in df.columns]
    df = df.drop(columns=redundant)

    if TARGET_COL not in df.columns:
        raise ValueError("INT1 was not found in the dataset.")

    y_raw = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Binary classification:
    # 1 = high travel intention
    # 0 = low or medium travel intention
    y = (y_raw >= 4).astype(int)

    return X, y


# ============================================================
# 6. TARGET VARIABLE DISTRIBUTION
# ============================================================

def plot_target_distribution(y):
    counts = y.value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Distribution of the Target Variable INT1")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/target_distribution.png", dpi=300)
    plt.close()


# ============================================================
# 7. DATA PREPROCESSING
# ============================================================

def preprocess_data(X):
    preprocessing = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    X_processed = preprocessing.fit_transform(X)

    return X_processed, preprocessing


# ============================================================
# 8. BASELINE MODELS
# ============================================================

def get_baseline_models():
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.10,
            num_leaves=31,
            min_child_samples=20,
            random_state=RANDOM_STATE,
            verbose=-1
        ),

        "MLP": MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            learning_rate_init=0.001,
            batch_size=32,
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }

    return models


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, cv):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = np.nan

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "ROC-AUC": roc_auc,
        "CV Accuracy Mean": cv_scores.mean(),
        "CV Accuracy Std": cv_scores.std()
    }

    return metrics, y_pred, cv_scores


def run_baseline_experiments(X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = get_baseline_models()
    results = []
    fitted_models = {}

    for name, model in models.items():
        metrics, y_pred, cv_scores = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            cv
        )

        metrics["Model"] = name
        results.append(metrics)
        fitted_models[name] = model

        cm = confusion_matrix(y_test, y_pred)
        pd.DataFrame(cm).to_csv(
            f"{OUTPUT_DIR}/{name}_confusion_matrix.csv",
            index=False
        )

        pd.DataFrame({"CV_Accuracy": cv_scores}).to_csv(
            f"{OUTPUT_DIR}/{name}_cv_scores.csv",
            index=False
        )

    results_df = pd.DataFrame(results)
    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "CV Accuracy Mean",
            "CV Accuracy Std"
        ]
    ]

    results_df.to_csv(f"{OUTPUT_DIR}/baseline_model_results.csv", index=False)

    return results_df, fitted_models


# ============================================================
# 10. XGBOOST HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ============================================================

def optimize_xgboost(X_train, y_train):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE
        }

        model = XGBClassifier(**params)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy"
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"{OUTPUT_DIR}/optuna_xgboost_trials.csv", index=False)

    best_params = study.best_params

    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE
    })

    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    return best_model, study


# ============================================================
# 11. LIGHTGBM HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ============================================================

def optimize_lightgbm(X_train, y_train):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
            "num_leaves": trial.suggest_int("num_leaves", 15, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": RANDOM_STATE,
            "verbose": -1
        }

        model = LGBMClassifier(**params)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy"
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"{OUTPUT_DIR}/optuna_lightgbm_trials.csv", index=False)

    best_params = study.best_params
    best_params.update({
        "random_state": RANDOM_STATE,
        "verbose": -1
    })

    best_model = LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)

    return best_model, study


# ============================================================
# 12. MLP HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ============================================================

def optimize_mlp(X_train, y_train):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    def objective(trial):
        hidden_units = trial.suggest_categorical(
            "hidden_units",
            [32, 64, 128]
        )

        params = {
            "hidden_layer_sizes": (hidden_units,),
            "activation": "relu",
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init",
                0.0001,
                0.01,
                log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size",
                [16, 32, 64]
            ),
            "max_iter": 1000,
            "random_state": RANDOM_STATE
        }

        model = MLPClassifier(**params)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy"
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"{OUTPUT_DIR}/optuna_mlp_trials.csv", index=False)

    best_params = study.best_params
    hidden_units = best_params["hidden_units"]

    best_model = MLPClassifier(
        hidden_layer_sizes=(hidden_units,),
        activation="relu",
        learning_rate_init=best_params["learning_rate_init"],
        batch_size=best_params["batch_size"],
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    best_model.fit(X_train, y_train)

    return best_model, study


# ============================================================
# 13. OPTUNA PROGRESS PLOT
# ============================================================

def plot_optuna_progress(study, filename):
    values = [trial.value for trial in study.trials if trial.value is not None]
    best_values = np.maximum.accumulate(values)

    plt.figure(figsize=(7, 4))
    plt.plot(values, marker="o", label="Objective Value")
    plt.plot(best_values, label="Best Value")
    plt.title("Progression of Model Accuracy During Optuna Tuning")
    plt.xlabel("Trial")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300)
    plt.close()


# ============================================================
# 14. FINAL EVALUATION OF OPTIMIZED MODELS
# ============================================================

def evaluate_optimized_models(models, X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = {}

    for name, model in models.items():
        metrics, y_pred, cv_scores = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            cv
        )

        results[name] = metrics

        report = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )

        with open(f"{OUTPUT_DIR}/{name}_classification_report.txt", "w") as f:
            f.write(report)

        cm = confusion_matrix(y_test, y_pred)
        pd.DataFrame(cm).to_csv(
            f"{OUTPUT_DIR}/{name}_optimized_confusion_matrix.csv",
            index=False
        )

    results_df = pd.DataFrame(results).T.reset_index()
    results_df = results_df.rename(columns={"index": "Model"})

    results_df.to_csv(
        f"{OUTPUT_DIR}/optimized_model_results.csv",
        index=False
    )

    return results_df


# ============================================================
# 15. GLOBAL AND INDIVIDUAL SHAP ANALYSIS
# ============================================================

def run_shap_analysis(model, X_test, feature_names):
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/shap_summary_plot.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": mean_abs_shap
    })

    shap_importance = shap_importance.sort_values(
        by="MeanAbsSHAP",
        ascending=False
    )

    shap_importance.to_csv(
        f"{OUTPUT_DIR}/shap_feature_importance.csv",
        index=False
    )

    base_value = explainer.expected_value

    individual_cases = [25, 61]

    for idx in individual_cases:
        if idx < X_test.shape[0]:
            force_plot = shap.force_plot(
                base_value,
                shap_values[idx, :],
                X_test[idx, :],
                feature_names=feature_names,
                matplotlib=False
            )

            shap.save_html(
                f"{OUTPUT_DIR}/shap_force_tourist_{idx}.html",
                force_plot
            )

    return shap_importance


# ============================================================
# 16. MAIN PIPELINE
# ============================================================

def main():

    df = load_dataset(DATA_PATH)

    print("\nInitial dataset dimensions:")
    print(df.shape)

    df = encode_categorical_columns(df)

    if all(col in df.columns for col in [TARGET_COL] + REDUNDANT_INT_COLS):
        correlation_analysis(df)

    X, y = prepare_features_target(df)

    print("\nPredictor variables:")
    print(X.columns.tolist())

    print("\nTarget variable distribution:")
    print(y.value_counts())

    plot_target_distribution(y)

    feature_names = X.columns.tolist()

    X_processed, preprocessing = preprocess_data(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    baseline_results, fitted_models = run_baseline_experiments(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nBaseline model results:")
    print(baseline_results)

    print("\nOptimizing XGBoost...")
    xgb_best, xgb_study = optimize_xgboost(X_train, y_train)

    print("\nOptimizing LightGBM...")
    lgbm_best, lgbm_study = optimize_lightgbm(X_train, y_train)

    print("\nOptimizing MLP...")
    mlp_best, mlp_study = optimize_mlp(X_train, y_train)

    plot_optuna_progress(
        xgb_study,
        "optuna_xgboost_progress.png"
    )

    plot_optuna_progress(
        lgbm_study,
        "optuna_lightgbm_progress.png"
    )

    plot_optuna_progress(
        mlp_study,
        "optuna_mlp_progress.png"
    )

    optimized_models = {
        "XGBoost_Optimized": xgb_best,
        "LightGBM_Optimized": lgbm_best,
        "MLP_Optimized": mlp_best
    }

    optimized_results = evaluate_optimized_models(
        optimized_models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nOptimized model results:")
    print(optimized_results)

    print("\nRunning SHAP analysis...")

    shap_importance = run_shap_analysis(
        xgb_best,
        X_test,
        feature_names
    )

    print("\nTop 10 features according to SHAP:")
    print(shap_importance.head(10))

    print("\nProcess completed.")
    print(f"Results saved in the folder: {OUTPUT_DIR}")


# ============================================================
# 17. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
