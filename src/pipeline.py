# src/pipeline.py
import sys
import os
sys.path.append("D:/Python/Lithofacies_Classification_NMM/src")

import joblib
import pandas as pd
from config import *
from preprocessing import run_preprocessing
from model_utils import (train_model, evaluate_model,
                         plot_confusion_matrix,
                         plot_feature_importance,
                         predict_unlabeled)


def run_pipeline(mode="full"):
    """
    Master pipeline
    mode = 'full'    → preprocess + train + evaluate
    mode = 'train'   → train + evaluate only
    mode = 'predict' → predict unlabeled only
    """

    print("=" * 60)
    print("   🛢️  LITHOFACIES CLASSIFICATION PIPELINE")
    print("   FORCE 2020 — NMM")
    print("=" * 60)

    # ── Step 1: Preprocessing ────────────────────────────
    if mode in ["full"]:
        print("\n📦 STEP 1 — Preprocessing...")
        X_train, X_test, y_train, y_test = run_preprocessing()
    else:
        print("\n📦 STEP 1 — Loading saved splits...")
        X_train, X_test, y_train, y_test = joblib.load(
            f"{MODEL_PATH}/train_test_split.pkl")
        print(f"   ✅ Train : {len(X_train):,}")
        print(f"   ✅ Test  : {len(X_test):,}")

    # ── Step 2: Training ─────────────────────────────────
    if mode in ["full", "train"]:
        print("\n🌟 STEP 2 — Training Model...")
        model = train_model(X_train, y_train, X_test, y_test)
    else:
        print("\n🌟 STEP 2 — Loading saved model...")
        model = joblib.load(f"{MODEL_PATH}/best_model.pkl")
        print("   ✅ Model loaded!")

    # ── Step 3: Evaluation ───────────────────────────────
    print("\n🏆 STEP 3 — Evaluating Model...")
    y_pred = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model)

    # ── Step 4: Predict Unlabeled ────────────────────────
    print("\n🔮 STEP 4 — Predicting Unlabeled Data...")
    df_predictions = predict_unlabeled(model)

    # ── Step 5: Summary ──────────────────────────────────
    from sklearn.metrics import accuracy_score
    test_acc = accuracy_score(y_test, y_pred)

    print(f"""
{'=' * 60}
   ✅ PIPELINE COMPLETE!

   Test Accuracy     : {test_acc*100:.2f}%
   Target (96%)      : {'✅ ACHIEVED!' if test_acc >= 0.96 else '❌ Not reached'}
   Predictions saved : data/output/predictions.csv
   Best model saved  : models/best_model.pkl
{'=' * 60}
    """)

    return model, y_pred, df_predictions


if __name__ == "__main__":
    # Usage:
    # python pipeline.py          → full pipeline
    # python pipeline.py train    → train + evaluate
    # python pipeline.py predict  → predict only

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    run_pipeline(mode=mode)