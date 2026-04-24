# src/config.py
# Central configuration for the entire project

import os

# ── Paths ────────────────────────────────────────────────
BASE_PATH       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH        = os.path.join(BASE_PATH, "data/raw")
PROCESSED_PATH  = os.path.join(BASE_PATH, "data/processed")
OUTPUT_PATH     = os.path.join(BASE_PATH, "data/output")
MODEL_PATH      = os.path.join(BASE_PATH, "models")
FIGURES_PATH    = os.path.join(BASE_PATH, "reports/figures")

# ── Files ────────────────────────────────────────────────
LABELED_FILE    = os.path.join(PROCESSED_PATH, "labeled_data.csv")
CLEANED_FILE    = os.path.join(PROCESSED_PATH, "cleaned_data.csv")
UNLABELED_FILE  = os.path.join(PROCESSED_PATH, "unlabeled_data.csv")

# ── Features ─────────────────────────────────────────────
FINAL_FEATURES = [
    "DEPTH_MD", "GR", "DTC", "CALI", "RDEP",
    "NPHI", "RHOB", "DRHO", "RMED", "PEF",
    "WELL", "FORMATION", "GROUP"
]

CATEGORICAL_FEATURES = ["WELL", "FORMATION", "GROUP"]

NUMERICAL_FEATURES = [
    "DEPTH_MD", "GR", "DTC", "CALI", "RDEP",
    "NPHI", "RHOB", "DRHO", "RMED", "PEF"
]

LOG_TRANSFORM_FEATURES = ["RMED", "RDEP", "PEF", "DRHO"]

TARGET = "FORCE_2020_LITHOFACIES_LITHOLOGY"
TARGET_CLEAN = "LITHO_FINAL"

# ── Class Mapping ─────────────────────────────────────────
LITHO_MAP = {
    30000 : "Sandstone",
    65000 : "Shale",
    65030 : "Shale",
    70000 : "Limestone",
    70032 : "Limestone",
    74000 : "Limestone",
    80000 : "Marl",
    86000 : "Coal",
    88000 : "Anhydrite",
    90000 : "Tuff",
    99000 : "Igneous"
}

FINAL_CLASSES = [
    "Shale", "Sandstone", "Limestone",
    "Anhydrite", "Marl", "Igneous",
    "Coal", "Tuff"
]

# ── Columns to Drop ───────────────────────────────────────
COLUMNS_TO_DROP = [
    "MUDWEIGHT", "SGR", "RXO", "RMIC", "RSHA",
    "DCAL", "SP", "ROPA", "DTS", "ROP", "BS",
    "Unnamed: 0", "DEPT", "Z_LOC", "X_LOC", "Y_LOC",
    "SOURCE_FILE", "FORCE_2020_LITHOFACIES_CONFIDENCE"
]

# ── Model Settings ────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
CV_FOLDS        = 5

print("✅ Config loaded!")
# ── Data Loader Helper ────────────────────────────────────
import pandas as pd

def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    print(f"✅ Loaded → {path}")
    print(f"   Shape  : {df.shape}")
    return df

# ── Feature Sets for Comparison ──────────────────────────
FEATURES_10 = [
    "DEPTH_MD", "GR", "DTC", "CALI", "RDEP",
    "NPHI", "RHOB", "DRHO", "RMED", "PEF"
]

FEATURES_13 = [
    "DEPTH_MD", "GR", "DTC", "CALI", "RDEP",
    "NPHI", "RHOB", "DRHO", "RMED", "PEF",
    "WELL", "FORMATION", "GROUP"
]