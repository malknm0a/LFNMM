# 🛢️ Lithofacies Classification — FORCE 2020

A machine learning pipeline for automatic lithofacies classification
from well log data using the FORCE 2020 dataset.

---

## 🏆 Results

| Metric | Value |
|---|---|
| Algorithm | LightGBM |
| Test Accuracy | **97.11%** |
| Target | 96.00% |
| Classes | 8 lithofacies |
| Features | 13 well log features |

---

## 📁 Project Structure
```
Lithofacies_Classification_NMM/
│
├── data/
│   ├── raw/                ← Original CSV files (21 wells)
│   ├── processed/          ← Cleaned & combined data
│   └── output/             ← Final predictions
│
├── notebooks/
│   ├── 00_Data_Combination.ipynb
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_EDA_Visualization.ipynb
│   ├── 04_Model_Training.ipynb
│   └── 05_Model_Evaluation.ipynb
│
├── src/
│   ├── config.py           ← Paths, features, class mappings
│   ├── preprocessing.py    ← Data cleaning pipeline
│   ├── model_utils.py      ← Train, evaluate, predict
│   └── pipeline.py         ← Master pipeline
│
├── models/
│   ├── best_model.pkl      ← Trained LightGBM model
│   ├── le_target.pkl       ← Label encoder
│   └── train_test_split.pkl
│
├── reports/figures/        ← All generated plots
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <repo>
cd Lithofacies_Classification_NMM
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Data
```
Place your CSV files in data/raw/
```

### 3. Run Full Pipeline
```bash
python src/pipeline.py
```

### 4. Train Only
```bash
python src/pipeline.py train
```

### 5. Predict Only
```bash
python src/pipeline.py predict
```

---

## 📊 Dataset

| Item | Value |
|---|---|
| Source | FORCE 2020 Machine Learning Competition |
| Total files | 21 CSV files |
| Labeled wells | 11 |
| Unlabeled wells | 10 |
| Total rows | 121,424 (after cleaning) |
| Corrupted rows removed | 19,160 (Unknown well) |

---

## 🪨 Lithofacies Classes

| Code | Class | Train Count |
|---|---|---|
| 65000/65030 | Shale | 66,582 |
| 70000/70032/74000 | Limestone | 11,278 |
| 30000 | Sandstone | 10,566 |
| 88000 | Anhydrite | 6,498 |
| 80000 | Marl | 4,198 |
| 99000 | Igneous | 895 |
| 86000 | Coal | 597 |
| 90000 | Tuff | 179 |

---

## 🎯 Features Used (13)

| # | Feature | Description | Importance |
|---|---|---|---|
| 1 | GR | Gamma Ray | 11.8% |
| 2 | DEPTH_MD | Measured Depth | 10.9% |
| 3 | DTC | Compressional Sonic | 10.4% |
| 4 | CALI | Caliper | 9.9% |
| 5 | RHOB | Bulk Density | 9.4% |
| 6 | DRHO | Density Correction | 9.2% |
| 7 | RDEP | Deep Resistivity | 8.6% |
| 8 | RMED | Medium Resistivity | 8.1% |
| 9 | NPHI | Neutron Porosity | 8.0% |
| 10 | PEF | Photoelectric Factor | 7.7% |
| 11 | FORMATION | Formation name | 3.0% |
| 12 | GROUP | Group name | 1.6% |
| 13 | WELL | Well name | 1.3% |

---

## 📈 Model Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Anhydrite | 99.5% | 99.8% | 99.6% |
| Shale | 98.2% | 98.6% | 98.4% |
| Igneous | 95.6% | 96.6% | 96.1% |
| Sandstone | 94.1% | 93.9% | 94.0% |
| Marl | 92.2% | 93.0% | 92.6% |
| Limestone | 94.3% | 91.8% | 93.1% |
| Tuff | 94.3% | 91.7% | 93.0% |
| Coal | 94.7% | 89.9% | 92.2% |

---

## 🔑 Key Insight

> **Data quality was more important than model complexity.**
> Removing 19,160 corrupted rows from the Unknown well
> jumped accuracy from **91.5% → 97.1%** — a gain of 5.6%
> that no amount of hyperparameter tuning could achieve.

---

## 👤 Author

**NMM** | 2026

---

## 📄 License

This project uses the FORCE 2020 dataset.
Please refer to the original competition for data licensing terms.