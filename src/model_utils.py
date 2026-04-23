# src/model_utils.py
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import cross_val_score
from config import *


def train_model(X_train, y_train, X_test, y_test):
    """Train LightGBM model"""
    print("=" * 50)
    print("   🌟 TRAINING LIGHTGBM MODEL")
    print("=" * 50)

    model = LGBMClassifier(
        n_estimators      = 3000,
        max_depth         = 10,
        learning_rate     = 0.02,
        num_leaves        = 127,
        min_child_samples = 10,
        subsample         = 0.85,
        subsample_freq    = 1,
        colsample_bytree  = 0.85,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbose           = -1
    )

    model.fit(
        X_train[FEATURES_13], y_train,
        eval_set  = [(X_test[FEATURES_13], y_test)],
        callbacks = [early_stopping(100), log_evaluation(300)]
    )

    print(f"\n   Best iteration : {model.best_iteration_}")
    joblib.dump(model, f"{MODEL_PATH}/best_model.pkl")
    print(f"💾 Saved → models/best_model.pkl")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    le_target = joblib.load(f"{MODEL_PATH}/le_target.pkl")
    y_pred    = model.predict(X_test[FEATURES_13])

    train_acc = accuracy_score(y_test,  y_pred)
    bal_acc   = balanced_accuracy_score(y_test, y_pred)

    print("=" * 50)
    print("   🏆 MODEL EVALUATION")
    print("=" * 50)
    print(f"\n   Test  Accuracy   : {train_acc*100:.2f}%")
    print(f"   Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"\n📊 Per Class Performance:\n")
    print(classification_report(y_test, y_pred,
          target_names=le_target.classes_))
    return y_pred


def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix"""
    le_target = joblib.load(f"{MODEL_PATH}/le_target.pkl")
    cm        = confusion_matrix(y_test, y_pred)
    cm_pct    = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_,
                ax=axes[0])
    axes[0].set_title("Confusion Matrix — Counts", fontweight="bold")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_,
                ax=axes[1])
    axes[1].set_title("Confusion Matrix — %", fontweight="bold")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")

    plt.suptitle("Confusion Matrix — LightGBM",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/confusion_matrix.png", dpi=150)
    plt.show()
    print("💾 Saved → confusion_matrix.png")


def plot_feature_importance(model):
    """Plot feature importance"""
    importance_df = pd.DataFrame({
        "Feature"   : FEATURES_13,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"],
             importance_df["Importance"],
             color=["#e94560" if i < 5 else "#3498db"
                    for i in range(len(importance_df))])
    plt.title("Feature Importance — LightGBM", fontweight="bold")
    plt.xlabel("Importance Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/feature_importance.png", dpi=150)
    plt.show()
    print("💾 Saved → feature_importance.png")


def predict_unlabeled(model):
    """Predict on unlabeled data"""
    from sklearn.preprocessing import LabelEncoder

    le_target    = joblib.load(f"{MODEL_PATH}/le_target.pkl")
    df_unlabeled = pd.read_csv(UNLABELED_FILE, low_memory=False)

    # Drop columns
    cols = [c for c in COLUMNS_TO_DROP if c in df_unlabeled.columns]
    df_unlabeled.drop(columns=cols, inplace=True)

    # Encode categoricals
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_unlabeled[col] = le.fit_transform(
                            df_unlabeled[col].astype(str))

    # Impute
    df_unlabeled[NUMERICAL_FEATURES] = df_unlabeled\
        .groupby("WELL")[NUMERICAL_FEATURES]\
        .transform(lambda x: x.fillna(x.median()))
    df_unlabeled[NUMERICAL_FEATURES] = df_unlabeled[NUMERICAL_FEATURES]\
        .fillna(df_unlabeled[NUMERICAL_FEATURES].median())

    # Predict
    predictions   = model.predict(df_unlabeled[FEATURES_13])
    probabilities = model.predict_proba(df_unlabeled[FEATURES_13])

    df_unlabeled["PREDICTED_LITHOLOGY"] = le_target\
        .inverse_transform(predictions)
    df_unlabeled["CONFIDENCE"]          = probabilities\
        .max(axis=1).round(3)

    # Save
    df_unlabeled.to_csv(f"{OUTPUT_PATH}/predictions.csv", index=False)
    print(f"✅ Predictions saved → data/output/predictions.csv")
    print(f"   Total predicted : {len(df_unlabeled):,}")
    return df_unlabeled


if __name__ == "__main__":
    # Quick test
    X_train, X_test, y_train, y_test = joblib.load(
        f"{MODEL_PATH}/train_test_split.pkl")
    model  = joblib.load(f"{MODEL_PATH}/best_model.pkl")
    y_pred = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model)