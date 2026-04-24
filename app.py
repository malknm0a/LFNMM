import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import *

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title            = "Automated Lithofacies Classification System",
    page_icon             = "🪨",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# ════════════════════════════════════════════════════════════════
# STYLING
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@300;400;600;700&family=Roboto:wght@300;400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"], p, span, div, label {
    font-family: 'Roboto', sans-serif !important;
}
h1, h2, h3, h4, h5, h6,
.main-title, .step-title {
    font-family: 'Roboto Slab', serif !important;
}

/* ── Page background ── */
.stApp                          { background-color: #FFFFFF !important; }
.main .block-container          { background-color: #FFFFFF !important; padding-top: 1rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"]                { background-color: #F4F7FB !important; border-right: 1px solid #D8E2F0; }
section[data-testid="stSidebar"] *              { color: #1A2B4A !important; }
section[data-testid="stSidebar"] .stMarkdown p  { color: #2C3E50 !important; }

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: #F7F9FC;
    border: 1px solid #D8E2F0;
    border-top: 3px solid #1565C0;
    border-radius: 8px;
    padding: 14px 16px;
}
div[data-testid="stMetricLabel"] p {
    font-family: 'Roboto Slab', serif !important;
    color: #1565C0 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetricValue"] {
    font-family: 'Roboto Slab', serif !important;
    color: #1A2B4A !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
div[data-testid="stMetricDelta"] { color: #2E7D32 !important; }

/* ── Buttons ── */
div[data-testid="stButton"] button {
    font-family: 'Roboto Slab', serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    background-color: #90A4AE !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}
div[data-testid="stButton"] button:hover {
    background-color: #78909C !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 8px rgba(0,0,0,0.15) !important;
}
div[data-testid="stDownloadButton"] button {
    font-family: 'Roboto Slab', serif !important;
    font-weight: 600 !important;
    background-color: #90A4AE !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
}
div[data-testid="stDownloadButton"] button:hover {
    background-color: #78909C !important;
    color: #FFFFFF !important;
}
div[data-testid="stDownloadButton"] button {
    font-family: 'Roboto Slab', serif !important;
    font-weight: 600 !important;
    background-color: #2E7D32 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
}
div[data-testid="stDownloadButton"] button:hover {
    background-color: #1B5E20 !important;
}

/* ── Step boxes ── */
.step-box {
    background: #F7F9FC;
    border: 1px solid #D8E2F0;
    border-top: 3px solid #1565C0;
    border-radius: 8px;
    padding: 22px 16px;
    text-align: center;
    height: 170px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* ── Tag pills ── */
.tag-container { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.tag-required {
    background-color: #FFFFFF; color: #C0392B;
    padding: 5px 12px; border-radius: 6px;
    font-size: 12px; font-weight: 700;
    border: 2px solid #C0392B;
    font-family: 'Roboto', sans-serif;
    display: inline-block;
}
.tag-optional {
    background-color: #FFFFFF; color: #2E7D32;
    padding: 5px 12px; border-radius: 6px;
    font-size: 12px; font-weight: 700;
    border: 2px solid #2E7D32;
    font-family: 'Roboto', sans-serif;
    display: inline-block;
}

/* ── Log description box ── */
.log-desc-box {
    background: #F7F9FC;
    padding: 18px 24px;
    border-radius: 8px;
    border-left: 4px solid #1565C0;
    border-top: 1px solid #D8E2F0;
    border-right: 1px solid #D8E2F0;
    border-bottom: 1px solid #D8E2F0;
}

/* ── Section headers ── */
h3 { color: #1A2B4A !important; font-family: 'Roboto Slab', serif !important; }

/* ── Divider ── */
hr { border-color: #E8EDF5 !important; margin: 1.2rem 0 !important; }

/* ── Dataframe ── */
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Expander ── */
div[data-testid="stExpander"] {
    border: 1px solid #D8E2F0 !important;
    border-radius: 8px !important;
}

/* ── Success / warning / error / info boxes ── */
div[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed #90CAF9 !important;
    border-radius: 8px !important;
    background: #F7F9FC !important;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════
CLASS_COLORS = {
    "Shale"    : "#4A7FB5", "Sandstone": "#E8A838",
    "Limestone": "#3DAD78", "Anhydrite": "#E05555",
    "Marl"     : "#9575CD", "Igneous"  : "#E07040",
    "Coal"     : "#37474F", "Tuff"     : "#26A69A"
}
LOG_COLORS = {
    "GR"  : "#27AE60", "RHOB": "#E74C3C",
    "NPHI": "#3498DB", "DTC" : "#8E44AD", "RDEP": "#E67E22",
}
GEO_PATTERNS = {
    "Shale"    : dict(shape="/",  size=4, solidity=0.7),
    "Sandstone": dict(shape=".",  size=5, solidity=0.6),
    "Limestone": dict(shape="+",  size=6, solidity=0.5),
    "Anhydrite": dict(shape="x",  size=6, solidity=0.7),
    "Marl"     : dict(shape="\\", size=4, solidity=0.5),
    "Igneous"  : dict(shape="-",  size=4, solidity=0.8),
    "Coal"     : dict(shape="",   size=8, solidity=1.0),
    "Tuff"     : dict(shape=".",  size=3, solidity=0.4),
}
GEO_SYMBOLS = {
    "Shale"    : "////", "Sandstone": "....",
    "Limestone": "++++", "Anhydrite": "xxxx",
    "Marl"     : "\\\\", "Igneous"  : "----",
    "Coal"     : "████", "Tuff"     : "····",
}
DEPTH_COL = "DEPTH_MD"
WELL_COL  = "WELL"

# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════
for k, v in {
    "show_results": False, "df_results": None,
    "well_name": "",       "active_tab": "home"
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    try:
        model     = joblib.load(f"{MODEL_PATH}/best_model.pkl")
        le_target = joblib.load(f"{MODEL_PATH}/le_target.pkl")
        return model, le_target, True
    except Exception as e:
        return None, None, str(e)

model, le_target, status = load_artifacts()

# ════════════════════════════════════════════════════════════════
# PREDICT FUNCTION
# ════════════════════════════════════════════════════════════════
def predict_well(df):
    df_proc = df.copy()
    cols = [c for c in COLUMNS_TO_DROP if c in df_proc.columns]
    df_proc.drop(columns=cols, inplace=True)
    for col in CATEGORICAL_FEATURES:
        if col in df_proc.columns:
            le = LabelEncoder()
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))
        else:
            df_proc[col] = 0
    df_proc[NUMERICAL_FEATURES] = df_proc[NUMERICAL_FEATURES]\
        .fillna(df_proc[NUMERICAL_FEATURES].median())
    preds  = model.predict(df_proc[FEATURES_13])
    probas = model.predict_proba(df_proc[FEATURES_13])
    names  = le_target.inverse_transform(preds)
    conf   = probas.max(axis=1)
    return names, conf, probas

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
# Logo — centered using columns
logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo1.jpg")
if os.path.exists(logo_path):
    from PIL import Image as PILImage
    logo      = PILImage.open(logo_path)
    _, lc, _  = st.columns([2, 1, 2])
    with lc:
        st.image(logo, use_container_width=True)
else:
    st.warning("⚠️ Logo not found — place log.png in project root")

st.markdown("""
<div style='text-align:center; padding: 8px 0 4px 0;'>
    <h1 style='font-family:"Roboto Slab",serif; font-size:2.4rem;
               font-weight:700; color:#C0392B; letter-spacing:-0.5px;
               margin-bottom:4px;'>
        🪨 Automated Lithofacies Classification System
    </h1>
    <h3 style='font-family:"Roboto Slab",serif; font-size:1.2rem;
               font-weight:600; color:#1565C0; margin:4px 0;'>
        Automated Lithofacies Classification from Well Log Data —
        A Comparative Machine Learning Framework
    </h3>
    <p style='font-family:"Roboto",sans-serif; font-size:1rem;
              color:#444; margin:6px 0 2px 0;'>
        CPG-555 Project — AI Applications for Petroleum Engineers
        and Geoscientists — Dr. Abdulazeez Abdulraheem
    </p>
    <p style='font-family:"Roboto Slab",serif; font-size:1rem;
              font-weight:600; color:#1565C0; margin:2px 0;'>
        Done by: Naif Almalki
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<h2 style='font-family:"Roboto Slab",serif; color:#1A2B4A;
           font-size:1.1rem; font-weight:700; margin-bottom:8px;'>
    ⚙️ Model Information
</h2>""", unsafe_allow_html=True)

if status is True:
    st.sidebar.success("✅ Model loaded successfully")
    st.sidebar.markdown("""
<div style='font-family:"Roboto",sans-serif; color:#2C3E50;
            font-size:0.88rem; line-height:2.0;
            background:#FFFFFF; padding:12px 14px;
            border-radius:8px; border:1px solid #D8E2F0;'>
    <b>Model:</b> LightGBM<br>
    <b>Test Accuracy:</b> 97.11%<br>
    <b>Balanced Accuracy:</b> 94.41%<br>
    <b>Features:</b> 13 well log features<br>
    <b>Classes:</b> 8 lithofacies<br>
    <b>Dataset:</b> FORCE 2020 — North Sea
</div>
""", unsafe_allow_html=True)
else:
    st.sidebar.error(f"❌ Model failed: {status}")

st.sidebar.markdown("---")

st.sidebar.markdown(
    '<p style="font-family:\'Roboto Slab\',serif; color:#C0392B; '
    'font-weight:700; font-size:0.9rem; margin-bottom:6px;">Required Log Columns:</p>',
    unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="tag-container">' +
    ''.join([f'<span class="tag-required">{l}</span>'
             for l in ["DEPTH_MD","GR","RHOB","NPHI","DTC","RDEP","CALI"]]) +
    '</div>', unsafe_allow_html=True)

st.sidebar.markdown(
    '<p style="font-family:\'Roboto Slab\',serif; color:#2E7D32; '
    'font-weight:700; font-size:0.9rem; margin:10px 0 6px 0;">Optional:</p>',
    unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="tag-container">' +
    ''.join([f'<span class="tag-optional">{l}</span>'
             for l in ["DRHO","RMED","PEF","WELL","FORMATION","GROUP"]]) +
    '</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background:#90A4AE; border-radius:8px; padding:8px 12px; margin-bottom:8px;'>
    <p style='font-family:"Roboto Slab",serif; color:#FFFFFF;
              font-weight:700; font-size:1rem; margin:0;'>🧭 Navigation</p>
</div>""", unsafe_allow_html=True)

if st.sidebar.button("🏠 Home",              use_container_width=True):
    st.session_state.active_tab   = "home"
    st.session_state.show_results = False
    st.session_state.df_results   = None
    st.rerun()
if st.sidebar.button("📂 Batch Prediction",  use_container_width=True):
    st.session_state.active_tab   = "batch"
    st.session_state.show_results = False
    st.rerun()
if st.sidebar.button("🔬 Single Prediction", use_container_width=True):
    st.session_state.active_tab   = "single"
    st.session_state.show_results = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.download_button(
    "📥 Download CSV Template",
    data = pd.DataFrame([{
        "DEPTH_MD":2100, "GR":75,   "DTC":100,  "CALI":12.5,
        "RDEP":1.2,      "NPHI":0.30,"RHOB":2.35,"DRHO":0.01,
        "RMED":1.1,      "PEF":2.5, "WELL":"15/9-23",
        "FORMATION":"Draupne", "GROUP":"VIKING"
    }]).to_csv(index=False),
    file_name="template.csv", mime="text/csv",
    use_container_width=True
)

# ════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "home":

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy",       "97.11%",  "+5.54% vs baseline")
    c2.metric("Balanced Accuracy",   "94.41%",  "All 8 classes")
    c3.metric("Training Samples",    "80,634",  "Clean data only")
    c4.metric("Lithofacies Classes", "8",        "FORCE 2020")
    st.markdown("---")

    st.markdown("### 📋 How to Use")
    s1, s2, s3, s4 = st.columns(4)
    for col, icon, step, color, text in [
        (s1, "📁", "Step 1", "#1565C0",
         "Choose <b>Batch Prediction</b> to upload a CSV or <b>Single Prediction</b> to enter values manually"),
        (s2, "📊", "Step 2", "#2E7D32",
         "File is validated — required: GR, RHOB, NPHI, DTC, RDEP, CALI, DEPTH_MD"),
        (s3, "🚀", "Step 3", "#E65100",
         "Click <b>Predict</b> — LightGBM classifies each depth interval with confidence scores"),
        (s4, "📥", "Step 4", "#6A1B9A",
         "View multi-track well log plot with geological patterns and download full CSV results"),
    ]:
        with col:
            st.markdown(f"""
            <div class='step-box'>
                <div style='font-size:2rem; margin-bottom:6px;'>{icon}</div>
                <h4 style='font-family:"Roboto Slab",serif; color:{color};
                           margin:0 0 8px 0; font-size:1rem;'>{step}</h4>
                <p style='font-size:13px; color:#444; margin:0;'>{text}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Well Log Descriptions")
    st.markdown("""
    <div class='log-desc-box'>
        <p style='font-size:14.5px; margin:10px 0;'>
            <strong style='color:#1565C0;'>☢️ GR — Gamma Ray</strong>
            <span style='color:#888; font-size:12px;'> [gAPI] </span>
            Detects natural radioactivity.
            <b>High GR → Shale</b> &nbsp;|&nbsp; <b>Low GR → Sandstone / Carbonate</b>
        </p>
        <p style='font-size:14.5px; margin:10px 0;'>
            <strong style='color:#1565C0;'>⚡ RDEP — Deep Resistivity</strong>
            <span style='color:#888; font-size:12px;'> [ohm.m] </span>
            Fluid conductivity.
            <b>High → Hydrocarbons / Tight rock</b> &nbsp;|&nbsp; <b>Low → Brine</b>
        </p>
        <p style='font-size:14.5px; margin:10px 0;'>
            <strong style='color:#1565C0;'>🔊 DTC — Compressional Sonic</strong>
            <span style='color:#888; font-size:12px;'> [us/ft] </span>
            Acoustic travel time — primary porosity indicator.
            <b>High DTC → High Porosity</b>
        </p>
        <p style='font-size:14.5px; margin:10px 0;'>
            <strong style='color:#1565C0;'>⚖️ RHOB — Bulk Density</strong>
            <span style='color:#888; font-size:12px;'> [g/cm³] </span>
            Essential for lithology ID and <b>total porosity</b> calculation.
        </p>
        <p style='font-size:14.5px; margin:10px 0;'>
            <strong style='color:#1565C0;'>⚛️ NPHI — Neutron Porosity</strong>
            <span style='color:#888; font-size:12px;'> [v/v] </span>
            Used with RHOB to detect <b>Gas</b> and differentiate
            <b>Limestone from Dolomite</b>.
        </p>
        <p style='font-size:14.5px; margin:10px 0;'>
            <strong style='color:#1565C0;'>📏 CALI — Caliper</strong>
            <span style='color:#888; font-size:12px;'> [in] </span>
            Borehole diameter — critical <b>QC tool</b>,
            identifies washouts invalidating RHOB/NPHI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Supported Lithofacies Classes")
    cols = st.columns(8)
    for i, (cls, color) in enumerate(CLASS_COLORS.items()):
        sym = GEO_SYMBOLS.get(cls, "")
        with cols[i]:
            st.markdown(f"""
            <div style='background:{color}; border-radius:8px;
                        padding:14px 4px; text-align:center;
                        box-shadow:0 2px 4px rgba(0,0,0,0.12);'>
                <p style='color:white; font-family:"Roboto",monospace;
                          font-weight:700; font-size:11px; margin:0;
                          letter-spacing:1px;'>{sym}</p>
                <p style='color:white; font-family:"Roboto Slab",serif;
                          font-weight:600; font-size:12px;
                          margin:4px 0 0 0;'>{cls}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Performance by Class")
    st.dataframe(pd.DataFrame({
        "Lithofacies" : ["Anhydrite","Shale","Igneous","Sandstone",
                         "Marl","Limestone","Tuff","Coal"],
        "Pattern"     : ["xxxx","////","----","....","\\\\\\\\","++++","····","████"],
        "Precision %" : [99.5,98.2,95.6,94.1,92.2,94.3,94.3,94.7],
        "Recall %"    : [99.8,98.6,96.6,93.9,93.0,91.8,91.7,89.9],
        "F1 %"        : [99.6,98.4,96.1,94.0,92.6,93.1,93.0,92.2],
        "Support"     : [1300,13316,179,2113,840,2256,36,119],
    }), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICTION
# ════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "batch":

    st.markdown("## 📂 Batch Prediction — Upload CSV")
    st.markdown("Upload a well log CSV file to classify lithofacies across all depth intervals.")
    st.markdown("---")

    if not st.session_state.show_results:

        uploaded_file = st.file_uploader(
            "Upload your well log CSV", type=["csv"],
            help="Must contain: DEPTH_MD, GR, RHOB, NPHI, DTC, RDEP, CALI"
        )

        if uploaded_file is not None:
            try:
                raw   = uploaded_file.read()
                text  = raw.decode("utf-8", errors="replace")
                lines = text.split("\n")
                sep   = "," if lines[0].count(",") >= lines[0].count(";") else ";"
                df    = pd.read_csv(BytesIO(raw), sep=sep)

                for col in df.columns:
                    if col not in [WELL_COL,"GROUP","FORMATION"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                if DEPTH_COL not in df.columns:
                    cands = [c for c in df.columns
                             if "DEPTH" in c.upper() or "MD" in c.upper()]
                    if cands:
                        df = df.rename(columns={cands[0]: DEPTH_COL})

                well_name = uploaded_file.name.replace(".csv","")
                if WELL_COL not in df.columns:
                    df[WELL_COL] = well_name

                st.success(f"✅ **{uploaded_file.name}** — {len(df):,} rows loaded")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Samples",    f"{len(df):,}")
                mc2.metric("Columns",    len(df.columns))
                mc3.metric("Depth Top",
                    f"{df[DEPTH_COL].min():.0f} m" if DEPTH_COL in df.columns else "N/A")
                mc4.metric("Depth Base",
                    f"{df[DEPTH_COL].max():.0f} m" if DEPTH_COL in df.columns else "N/A")

                required = ["GR","RHOB","NPHI","DTC","RDEP","CALI"]
                missing  = [l for l in required if l not in df.columns]
                if missing:
                    st.warning(f"⚠️ Missing: {missing} — will impute with medians")
                else:
                    st.success("✅ All required logs present")

                with st.expander("📊 Preview — first 10 rows"):
                    st.dataframe(df.head(10), use_container_width=True)

                if status is True:
                    if st.button("🚀 Predict Lithofacies",
                                 type="primary", use_container_width=True):
                        with st.spinner("Running LightGBM model..."):
                            try:
                                names, conf, probas = predict_well(df)
                                df["PREDICTED_LITHOLOGY"] = names
                                df["CONFIDENCE"] = (conf * 100).round(2)
                                st.session_state.df_results   = df
                                st.session_state.show_results = True
                                st.session_state.well_name    = well_name
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Prediction error: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                else:
                    st.error("❌ Model not loaded")

            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

    # ── RESULTS ──────────────────────────────────────────────────────
    else:
        df        = st.session_state.df_results
        well_name = st.session_state.well_name
        conf_arr  = df["CONFIDENCE"].values / 100
        depth     = df[DEPTH_COL].values if DEPTH_COL in df.columns \
                    else np.arange(len(df))
        pred_arr  = df["PREDICTED_LITHOLOGY"].values

        st.success("✅ Prediction complete!")

        tc1, tc2, _ = st.columns([1, 1, 2])
        with tc1:
            if st.button("🏠 New Prediction", use_container_width=True):
                st.session_state.show_results = False
                st.session_state.df_results   = None
                st.rerun()
        with tc2:
            st.download_button(
                "📥 Download Results CSV",
                data                = df.to_csv(index=False).encode("utf-8"),
                file_name           = f"pred_{well_name}.csv",
                mime                = "text/csv",
                use_container_width = True
            )

        st.markdown("---")
        st.markdown(
            f"### 📊 Well: `{well_name}` &nbsp;—&nbsp; "
            f"`{depth.min():.0f} m` to `{depth.max():.0f} m`"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Samples",   f"{len(df):,}")
        m2.metric("Mean Confidence", f"{conf_arr.mean()*100:.1f}%")
        m3.metric("High Conf >80%",
                  f"{(conf_arr>0.8).sum():,} ({(conf_arr>0.8).mean()*100:.0f}%)")
        m4.metric("Low Conf <50%",
                  f"{(conf_arr<0.5).sum():,} ({(conf_arr<0.5).mean()*100:.0f}%)")

        st.markdown("---")
        st.markdown("### 🪨 Well Log Plot with Predicted Lithofacies")

        # ── Downsample for speed ──────────────────────────────────
        MAX_PLOT = 2000
        if len(df) > MAX_PLOT:
            step    = len(df) // MAX_PLOT
            df_plot = df.iloc[::step].reset_index(drop=True)
            st.info(f"Plot shows {len(df_plot):,} of {len(df):,} "
                    f"points for speed. Full data in download.")
        else:
            df_plot = df

        depth_p = df_plot[DEPTH_COL].values \
                  if DEPTH_COL in df_plot.columns \
                  else np.arange(len(df_plot))
        pred_p  = df_plot["PREDICTED_LITHOLOGY"].values
        conf_p  = df_plot["CONFIDENCE"].values / 100

        available_logs = [l for l in ["GR","RHOB","NPHI","DTC","RDEP"]
                          if l in df_plot.columns]
        n_tracks = len(available_logs) + 2

        fig = make_subplots(
            rows=1, cols=n_tracks, shared_yaxes=True,
            subplot_titles=available_logs + ["Confidence","Lithofacies"],
            horizontal_spacing=0.02
        )

        # Log tracks
        for i, log in enumerate(available_logs):
            fig.add_trace(go.Scatter(
                x=df_plot[log].values, y=depth_p,
                mode="lines", name=log,
                line=dict(color=LOG_COLORS.get(log,"#1565C0"), width=1),
                showlegend=False,
            ), row=1, col=i+1)

        # Confidence track
        conf_col = len(available_logs) + 1
        fig.add_trace(go.Scatter(
            x=conf_p*100, y=depth_p,
            mode="lines", name="Confidence",
            line=dict(color="#1565C0", width=1.5),
            fill="tozerox", fillcolor="rgba(21,101,192,0.12)",
            showlegend=False,
        ), row=1, col=conf_col)
        fig.add_vline(x=50, line_dash="dash",
                      line_color="#E74C3C", line_width=1,
                      col=conf_col, row=1)

        # ── Geological pattern lithofacies track ──────────────────
        lith_col = len(available_logs) + 2

        segments = []
        if len(depth_p) > 1:
            curr_cls  = pred_p[0]
            seg_start = depth_p[0]
            for i in range(1, len(depth_p)):
                if pred_p[i] != curr_cls or i == len(depth_p) - 1:
                    seg_end = depth_p[i]
                    segments.append((curr_cls, seg_start, seg_end))
                    curr_cls  = pred_p[i]
                    seg_start = depth_p[i]

        for cls, y0, y1 in segments:
            color   = CLASS_COLORS.get(cls, "#CCCCCC")
            pattern = GEO_PATTERNS.get(cls, dict(shape="", size=4, solidity=0.5))
            fig.add_trace(go.Bar(
                x           = [1],
                y           = [abs(y1 - y0)],
                base        = [min(y0, y1)],
                orientation = "v",
                name        = cls,
                marker      = dict(
                    color   = color,
                    opacity = 0.88,
                    pattern = dict(
                        shape    = pattern["shape"],
                        size     = pattern["size"],
                        solidity = pattern["solidity"],
                        fgcolor  = "rgba(0,0,0,0.28)",
                        bgcolor  = color,
                    ),
                    line    = dict(width=0),
                ),
                showlegend    = False,
                hovertemplate = f"<b>{cls}</b><br>Depth: %{{base:.0f}} m<extra></extra>",
            ), row=1, col=lith_col)

        fig.update_xaxes(range=[0,1], showticklabels=False,
                         showgrid=False, row=1, col=lith_col)
        fig.update_layout(
            height=820, paper_bgcolor="white",
            plot_bgcolor="#FAFBFD", barmode="stack",
            font=dict(family="Roboto", size=11, color="#1A2B4A"),
            margin=dict(l=60, r=20, t=50, b=20),
        )
        fig.update_yaxes(autorange="reversed", title_text="Depth (m)",
                         col=1, row=1, gridcolor="#E8EDF5")
        for i in range(2, n_tracks+1):
            fig.update_yaxes(autorange="reversed", showgrid=True,
                             gridcolor="#E8EDF5", row=1, col=i)

        st.plotly_chart(fig, use_container_width=True)

        # Legend
        present = [c for c in CLASS_COLORS if c in pred_arr]
        lc = st.columns(len(present))
        for i, cls in enumerate(present):
            cnt = (pred_arr == cls).sum()
            pct = cnt / len(pred_arr) * 100
            sym = GEO_SYMBOLS.get(cls, "")
            with lc[i]:
                st.markdown(f"""
                <div style='background:{CLASS_COLORS[cls]}; border-radius:8px;
                    padding:10px 6px; text-align:center; margin-bottom:6px;
                    box-shadow:0 2px 4px rgba(0,0,0,0.1);'>
                    <p style='color:white; font-weight:700; font-size:12px;
                        margin:0; font-family:"Roboto",monospace;
                        letter-spacing:1px;'>{sym}</p>
                    <p style='color:white; font-family:"Roboto Slab",serif;
                        font-weight:600; font-size:12px; margin:3px 0 0 0;'>{cls}</p>
                    <p style='color:rgba(255,255,255,0.9); font-size:11px;
                        margin:1px 0 0 0;'>{cnt:,} ({pct:.1f}%)</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📈 Lithofacies Distribution")

        dist_df = pd.DataFrame({"Lithofacies": pred_arr})\
            .value_counts().reset_index()
        dist_df.columns = ["Lithofacies","Count"]
        dist_df["Pct"]  = (dist_df["Count"]/len(pred_arr)*100).round(1)
        dist_df["Color"]= dist_df["Lithofacies"].map(CLASS_COLORS)

        dc1, dc2 = st.columns([1, 2])
        with dc1:
            st.dataframe(dist_df[["Lithofacies","Count","Pct"]],
                         use_container_width=True, hide_index=True)
        with dc2:
            pie = go.Figure(go.Pie(
                labels=dist_df["Lithofacies"], values=dist_df["Count"],
                marker=dict(colors=dist_df["Color"].tolist(),
                            line=dict(color="white", width=2)),
                hole=0.38, textinfo="label+percent",
                textfont=dict(size=12, family="Roboto"),
            ))
            pie.update_layout(paper_bgcolor="white", showlegend=False,
                               margin=dict(l=10,r=10,t=10,b=10), height=300)
            st.plotly_chart(pie, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📉 Confidence Distribution")

        conf_hist = go.Figure()
        conf_hist.add_trace(go.Histogram(
            x=conf_arr*100, nbinsx=50,
            marker=dict(color="#1976D2", line=dict(color="white", width=0.5)),
        ))
        conf_hist.add_vline(x=50, line_dash="dash", line_color="#E74C3C",
                            annotation_text="50% threshold",
                            annotation_font_color="#E74C3C")
        conf_hist.add_vline(x=conf_arr.mean()*100, line_dash="dot",
                            line_color="#0D47A1",
                            annotation_text=f"Mean = {conf_arr.mean()*100:.1f}%",
                            annotation_font_color="#0D47A1")
        conf_hist.update_layout(
            xaxis_title="Confidence (%)", yaxis_title="Count",
            paper_bgcolor="white", plot_bgcolor="#FAFBFD",
            height=280, margin=dict(l=40,r=20,t=20,b=40),
            font=dict(family="Roboto", size=11, color="#1A2B4A")
        )
        st.plotly_chart(conf_hist, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Detailed Predictions")
        display_cols = (
            [c for c in [WELL_COL, DEPTH_COL] if c in df.columns] +
            [l for l in ["GR","RHOB","NPHI","DTC","RDEP"] if l in df.columns] +
            ["PREDICTED_LITHOLOGY","CONFIDENCE"]
        )
        st.dataframe(df[display_cols].head(100),
                     use_container_width=True, hide_index=True)
        if len(df) > 100:
            st.info(f"Showing first 100 of {len(df):,} rows. "
                    f"Download full results above.")

        st.markdown("---")
        _, bc2, _ = st.columns([1, 2, 1])
        with bc2:
            if st.button("🏠 Return Home & Upload New Well",
                         type="primary", use_container_width=True,
                         key="bottom_return"):
                st.session_state.show_results = False
                st.session_state.df_results   = None
                st.rerun()

# ════════════════════════════════════════════════════════════════
# PAGE: SINGLE SAMPLE
# ════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "single":

    st.markdown("## 🔬 Single Sample Prediction")
    st.markdown("Enter well log values manually to predict lithofacies at a single depth point.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📏 Depth & Acoustic")
        depth_v     = st.number_input("DEPTH_MD (m)",   0.0,  6000.0, 2100.0, 0.5)
        gr_v        = st.number_input("GR (API)",        0.0,   300.0,   75.0, 0.1)
        dtc_v       = st.number_input("DTC (us/ft)",    40.0,   200.0,  100.0, 0.1)
        cali_v      = st.number_input("CALI (inches)",   5.0,    25.0,   12.5, 0.1)
    with col2:
        st.markdown("#### 🧲 Resistivity & Porosity")
        rdep_v      = st.number_input("RDEP (ohm.m)",   0.1,  1000.0,    1.2, 0.1)
        rmed_v      = st.number_input("RMED (ohm.m)",   0.1,  1000.0,    1.1, 0.1)
        nphi_v      = st.number_input("NPHI (v/v)",     0.0,     1.0,   0.30, 0.01)
        pef_v       = st.number_input("PEF (b/e)",      0.5,    10.0,    2.5, 0.1)
    with col3:
        st.markdown("#### ⚖️ Density & Geology")
        rhob_v      = st.number_input("RHOB (g/cc)",    1.0,     3.5,   2.35, 0.01)
        drho_v      = st.number_input("DRHO (g/cc)",   -0.5,     0.5,   0.01, 0.01)
        well_v      = st.selectbox("WELL",
            ["15/9-23","16/2-7","16/7-6","17/4-1","25/10-9","Other"])
        formation_v = st.selectbox("FORMATION",
            ["Draupne","Skagerrak","Statfjord","Brent","Dunlin",
             "Heather","Sleipner","Tor","Ekofisk","Other"])
        group_v     = st.selectbox("GROUP",
            ["VIKING","STATFJORD","BRENT","DUNLIN",
             "ZECHSTEIN","ROTLIEGEND","Other"])

    st.markdown("---")

    if st.button("🚀 Predict Lithofacies", type="primary", use_container_width=True):
        sample = pd.DataFrame([{
            "DEPTH_MD":depth_v, "GR":gr_v,    "DTC":dtc_v,
            "CALI":cali_v,      "RDEP":rdep_v, "NPHI":nphi_v,
            "RHOB":rhob_v,      "DRHO":drho_v, "RMED":rmed_v,
            "PEF":pef_v,        "WELL":well_v, "FORMATION":formation_v,
            "GROUP":group_v
        }])
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            sample[col] = le.fit_transform(sample[col].astype(str))

        pred   = model.predict(sample[FEATURES_13])
        probas = model.predict_proba(sample[FEATURES_13])[0]
        litho  = le_target.inverse_transform(pred)[0]
        conf   = probas.max() * 100
        sym    = GEO_SYMBOLS.get(litho, "")
        color  = CLASS_COLORS.get(litho, "#1565C0")

        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Lithofacies", f"{sym} {litho}")
        r2.metric("Confidence",            f"{conf:.1f}%")
        r3.metric("Depth",                 f"{depth_v:.1f} m")

        # Result badge
        st.markdown(f"""
        <div style='background:{color}; border-radius:10px; padding:16px 24px;
                    margin:12px 0; display:inline-block; box-shadow:0 3px 8px rgba(0,0,0,0.15);'>
            <span style='color:white; font-family:"Roboto Slab",serif;
                         font-size:1.3rem; font-weight:700;'>
                {sym}&nbsp;&nbsp;{litho}&nbsp;&nbsp;—&nbsp;&nbsp;{conf:.1f}% confidence
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📊 All Class Probabilities")
        proba_df = pd.DataFrame({
            "Lithofacies": le_target.classes_,
            "Probability": probas * 100
        }).sort_values("Probability", ascending=True)

        bar_fig = go.Figure(go.Bar(
            x=proba_df["Probability"],
            y=[f"{GEO_SYMBOLS.get(c,'')} {c}" for c in proba_df["Lithofacies"]],
            orientation="h",
            marker_color=[CLASS_COLORS.get(c,"#1565C0")
                          for c in proba_df["Lithofacies"]],
            text=[f"{p:.1f}%" for p in proba_df["Probability"]],
            textposition="outside"
        ))
        bar_fig.update_layout(
            xaxis=dict(title="Probability (%)", range=[0,115]),
            yaxis=dict(title=""),
            height=380, paper_bgcolor="white",
            plot_bgcolor="#FAFBFD", showlegend=False,
            font=dict(family="Roboto", size=12, color="#1A2B4A"),
            margin=dict(l=20, r=40, t=20, b=20)
        )
        st.plotly_chart(bar_fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:10px 0;
            background:#F7F9FC; border-radius:8px;
            border:1px solid #E8EDF5; margin-top:8px;'>
    <p style='font-family:"Roboto Slab",serif; color:#1565C0;
              font-size:0.85rem; margin:0; font-weight:600;'>
        FORCE 2020 Lithofacies Classification System &nbsp;|&nbsp;
        LightGBM 97.11% &nbsp;|&nbsp; Naif Almalki &nbsp;|&nbsp; 2026
    </p>
</div>
""", unsafe_allow_html=True)