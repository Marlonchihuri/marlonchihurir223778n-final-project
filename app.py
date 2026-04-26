# ─────────────────────────────────────────────────────────────────────────────
# Constants & Config
# ─────────────────────────────────────────────────────────────────────────────
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, date

FARMER_FEATURES = [
    "Nitrogen Applied (kg/ha)",       # COLUMN: N_kg_ha renamed
    "Phosphorus Applied (kg/ha)",     # COLUMN: P_kg_ha renamed
    "Planting Density (plants/m²)",   # COLUMN: pl_m2 renamed
    "Planting Day of Year",           # COLUMN: plant_doy renamed
    "Management Score",               # COLUMN: mgmt_score renamed
]

FEATURE_META = {
    "Nitrogen Applied (kg/ha)":      {"min": 0.0,  "max": 276.5, "default": 28.8,  "step": 1.0,  "unit": "kg/ha",      "warn_below": 20,  "warn_msg": "⚠️ Nitrogen below 20 kg/ha — yields may suffer significantly"},
    "Phosphorus Applied (kg/ha)":    {"min": 0.0,  "max": 120.0, "default": 16.0,  "step": 0.5,  "unit": "kg/ha",      "warn_below": 10,  "warn_msg": "⚠️ Phosphorus below recommended level (10 kg/ha)"},
    "Planting Density (plants/m²)":  {"min": 0.3,  "max": 12.0,  "default": 3.0,   "step": 0.1,  "unit": "plants/m²",  "warn_below": 1.5, "warn_msg": "⚠️ Very low density — consider planting more closely"},
    "Planting Day of Year":          {"min": 1.0,  "max": 365.0, "default": 92.0,  "step": 1.0,  "unit": "day",        "warn_below": 50,  "warn_msg": "⚠️ Very early planting — check for frost or drought risk"},
    "Management Score":              {"min": 0.0,  "max": 1.0,   "default": 0.5,   "step": 0.05, "unit": "",           "warn_below": 0.3, "warn_msg": "⚠️ Low management score — timely weeding and pest control matter"},
}

COUNTRY_AVGS = {
    "burundi": 2603.8, "kenya": 4651.4, "nigeria": 3951.7,
    "rwanda": 3074.3,  "tanzania": 4242.5, "uganda": 3674.9, "zambia": 4728.4,
}
YEAR_AVGS = {2016: 4088.9, 2017: 4426.3, 2018: 4279.5,
             2019: 4160.0, 2020: 4238.5, 2022: 3951.7}
DISTRICT_AVG = 4205.7

AGRONOMIC_TIPS = [
    ("🌱", "Nitrogen Application",
     "Apply 50–80 kg N/ha in two split doses: ⅓ at planting, ⅔ at 30–40 days "
     "after emergence. Split applications cut leaching losses by 20–30%."),
    ("🔬", "Phosphorus Application",
     "Apply 15–25 kg P/ha in-furrow at planting. Phosphorus is immobile in soil, "
     "so placement near the seed maximises root uptake at emergence."),
    ("📐", "Planting Density",
     "Target 3–5 plants/m² (75 cm rows × 25–30 cm spacing). Higher density "
     "increases intra-crop competition; lower density wastes potential yield."),
    ("📅", "Planting Timing",
     "Plant within 2 weeks of onset of rains (typically Day 75–130 for East Africa). "
     "Late planting by 2 weeks can cost 5–15% of final yield."),
    ("⭐", "Management Score",
     "High management = timely weeding (2–3×/season), proper soil prep, pest "
     "monitoring, and careful post-harvest storage. Each practice adds incremental gains."),
    ("💧", "Drought & Water Stress",
     "Drought stress during tasseling and grain fill is the #1 yield killer. "
     "Supplement with 30–50 mm irrigation at this critical 2-week window if possible."),
]

# ─────────────────────────────────────────────────────────────────────────────
# Model Loading (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────────────────
def find_artifact(*names):
    for name in names:
        if os.path.exists(name):
            return name
    raise FileNotFoundError(f"None of {names} exist in {os.getcwd()}")


def build_shap_explainer(model):
    try:
        return shap.TreeExplainer(model)
    except Exception:
        try:
            return shap.Explainer(model, algorithm="auto")
        except Exception:
            return None


@st.cache_resource
def load_all_artefacts():
    """
    Load model artefacts using available .joblib or .pkl filenames.
    Returns a dict of artefacts or an error string.
    """
    artefacts = {}

    try:
        artefacts["model"] = joblib.load(find_artifact("model.joblib", "xgb_model.pkl", "model.pkl"))
        artefacts["scaler"] = joblib.load(find_artifact("scaler.joblib", "scaler.pkl"))
        artefacts["feature_names"] = joblib.load(find_artifact("feature_names.joblib", "feature_names.pkl"))
    except FileNotFoundError as e:
        return str(e)

    if os.path.exists("metrics.joblib"):
        artefacts["metrics"] = joblib.load("metrics.joblib")
    elif os.path.exists("metrics.pkl"):
        artefacts["metrics"] = joblib.load("metrics.pkl")
    else:
        artefacts["metrics"] = {}

    try:
        shap_path = find_artifact("shap_explainer.joblib", "shap_explainer.pkl")
        if shap_path.endswith(".joblib"):
            artefacts["shap_explainer"] = joblib.load(shap_path)
        else:
            with open(shap_path, "rb") as f:
                artefacts["shap_explainer"] = pickle.load(f)
    except Exception:
        artefacts["shap_explainer"] = None

    try:
        artefacts["lime_explainer"] = joblib.load(find_artifact("lime_explainer.joblib", "lime_explainer.pkl"))
    except Exception:
        try:
            X_train = pd.read_csv("X_train.csv")
            artefacts["lime_explainer"] = LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=list(X_train.columns),
                mode="regression",
                random_state=42,
            )
        except Exception:
            artefacts["lime_explainer"] = None

    if os.path.exists("xai_metrics.joblib"):
        artefacts["xai_metrics"] = joblib.load("xai_metrics.joblib")
    elif os.path.exists("xai_metrics.pkl"):
        artefacts["xai_metrics"] = joblib.load("xai_metrics.pkl")
    else:
        artefacts["xai_metrics"] = {}

    return artefacts

# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_input_row(
    farmer_vals: dict,
    feature_names: list,
    country: str,
    season: str,
) -> pd.DataFrame:
    """
    Construct a single-row DataFrame that matches the model's expected
    feature columns, filling non-farmer features with dataset medians.
    """
    # Medians for non-farmer features (from Phase 2 cleaning)
    default_vals = {
        "Year": 2022.0,
        "Latitude": 0.0,
        "Longitude": 34.0,
        "Avg Season Temp (°C)": 20.5,
        "Season Precipitation (mm)": 850.0,
        "Early Season Precip (mm)": 420.0,
        "Late Season Precip (mm)": 430.0,
        "Aridity Index": 1.0,
        "Heat Stress Index": 0.3,
        "Drought Stress Index": 0.4,
        "Water Stress Index": 0.3,
        "Climate Score": 0.55,
        "Soil Organic Carbon (%)": 1.2,
        "Soil Water-Holding Capacity": 120.0,
        "Soil Clay Content (%)": 35.0,
        "Soil pH": 6.2,
        "Soil Fertility Index": 0.5,
        "Soil Quality Index": 0.55,
        "Nitrogen Applied (kg/ha)": 28.8,
        "Phosphorus Applied (kg/ha)": 16.0,
        "Potassium Applied (kg/ha)": 8.0,
        "Total NPK (kg/ha)": 52.8,
        "Fertilizer Adequacy": 0.6,
        "Management Score": 0.5,
        "Planting Density (plants/m²)": 3.0,
        "Planting Day of Year": 92.0,
        "Weeding (0/1)": 1.0,
        "Hybrid Maturity": 120.0,
        "Elevation (m)": 1400,
        # country dummies
        "country_burundi": 0, "country_kenya": 0, "country_nigeria": 0,
        "country_rwanda": 0,  "country_tanzania": 0, "country_uganda": 0,
        "country_zambia": 0,
        # season dummies
        "season_first season": 0, "season_second season": 0,
    }

    # Override with farmer inputs
    for k, v in farmer_vals.items():
        default_vals[k] = v

    # Update derived features that depend on farmer inputs
    N = farmer_vals.get("Nitrogen Applied (kg/ha)", default_vals["Nitrogen Applied (kg/ha)"])
    P = farmer_vals.get("Phosphorus Applied (kg/ha)", default_vals["Phosphorus Applied (kg/ha)"])
    K = default_vals["Potassium Applied (kg/ha)"]
    default_vals["Total NPK (kg/ha)"]     = N + P + K          # COLUMN: NPK_total
    default_vals["Fertilizer Adequacy"]   = min((N + P) / 55.0, 1.0)  # COLUMN: fert_adequacy

    # Set country dummy
    country_key = f"country_{country.lower()}"                  # COLUMN: country one-hot
    if country_key in default_vals:
        default_vals[country_key] = 1

    # Set season dummy
    season_key = f"season_{season.lower()}"                     # COLUMN: season one-hot
    if season_key in default_vals:
        default_vals[season_key] = 1

    row = pd.DataFrame([{col: default_vals.get(col, 0.0) for col in feature_names}])
    return row


def predict_with_ci(model, scaler, row: pd.DataFrame) -> tuple[float, float, float]:
    """
    Return (prediction, lower_95ci, upper_95ci).
    CI uses 2 × test RMSE as a practical uncertainty estimate.
    """
    try:
        if os.path.exists("metrics.joblib"):
            metrics = joblib.load("metrics.joblib")
        elif os.path.exists("metrics.pkl"):
            metrics = joblib.load("metrics.pkl")
        else:
            metrics = {}
        rmse = metrics.get("test_rmse", 379.5)
    except Exception:
        rmse = 379.5
    pred = float(model.predict(row)[0])
    return pred, max(991.0, pred - 2 * rmse), min(8614.0, pred + 2 * rmse)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP explanation for one sample
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def compute_shap_for_row(_model, _explainer, row_tuple: tuple, feature_names: list):
    """
    Compute SHAP values for a single input row.
    Row passed as tuple for hashability (Streamlit caching requirement).
    """
    row_df = pd.DataFrame([row_tuple], columns=feature_names)
    explainer = _explainer if _explainer is not None else build_shap_explainer(_model)
    if explainer is None:
        return np.zeros(len(feature_names), dtype=float), 0.0

    try:
        if hasattr(explainer, "shap_values"):
            shap_vals = explainer.shap_values(row_df)
            sv = shap_vals[0] if isinstance(shap_vals, (list, tuple)) else shap_vals[0]
            base_value = getattr(explainer, "expected_value", 0.0)
            base_value = float(base_value[0] if isinstance(base_value, (list, tuple, np.ndarray)) else base_value)
        else:
            explanation = explainer(row_df)
            values = np.array(explanation.values)
            sv = values[0] if values.ndim == 2 else values
            base_value = explanation.base_values
            base_value = float(base_value[0] if isinstance(base_value, (list, tuple, np.ndarray)) else base_value)
        return sv, base_value
    except Exception:
        return np.zeros(len(feature_names), dtype=float), 0.0


def plot_shap_waterfall_plotly(
    shap_vals: np.ndarray,
    feature_names: list,
    base_value: float,
    prediction: float,
    top_n: int = 12,
) -> go.Figure:
    """
    Interactive Plotly waterfall chart for a single prediction.
    All feature labels are human-readable (never raw column names).
    """
    # Sort by absolute SHAP, take top N
    order  = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    feats  = [feature_names[i] for i in order]
    vals   = [shap_vals[i] for i in order]

    # Truncate long feature names for display
    feats_short = [f[:35] + "…" if len(f) > 35 else f for f in feats]

    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=feats_short,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}" for v in vals],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="white", line_width=1)
    fig.update_layout(
        title=f"SHAP Feature Contributions  (base: {base_value:.0f} → pred: {prediction:.0f} kg/ha)",
        xaxis_title="SHAP value (kg/ha)",
        yaxis_title="",
        plot_bgcolor="#162816",
        paper_bgcolor="#0f1a0f",
        font_color="#e8f5e9",
        height=420,
        margin=dict(l=20, r=80, t=60, b=40),
    )
    return fig


def plot_lime_bar_plotly(lime_contrib: dict) -> go.Figure:
    """
    Interactive Plotly bar chart for LIME contributions.
    All feature labels are human-readable.
    """
    items = sorted(lime_contrib.items(), key=lambda x: abs(x[1]), reverse=True)
    feats = [f[:35] + "…" if len(f) > 35 else f for f, _ in items]
    vals  = [v for _, v in items]
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=feats,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}" for v in vals],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="white", line_width=1)
    fig.update_layout(
        title="LIME Local Feature Contributions",
        xaxis_title="LIME Contribution (kg/ha)",
        plot_bgcolor="#162816",
        paper_bgcolor="#0f1a0f",
        font_color="#e8f5e9",
        height=360,
        margin=dict(l=20, r=80, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plain-language explanation
# ─────────────────────────────────────────────────────────────────────────────
def plain_explanation(
    prediction: float,
    country_avg: float,
    shap_vals: np.ndarray,
    feature_names: list,
) -> str:
    """Generate 3–5 sentence farmer-friendly explanation from SHAP values."""
    diff = prediction - country_avg
    tier = "HIGH" if prediction > 5000 else "MODERATE" if prediction > 3500 else "LOW"

    ranked = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    top_pos = [(f, v) for f, v in ranked if v > 30][:2]
    top_neg = [(f, v) for f, v in ranked if v < -30][:2]

    lines = [
        f"Your predicted yield is **{prediction:,.0f} kg/ha** — a **{tier}** result "
        f"({'above' if diff >= 0 else 'below'} the {country_avg:,.0f} kg/ha country average "
        f"by {abs(diff):,.0f} kg/ha).",
    ]
    if top_pos:
        feat, val = top_pos[0]
        lines.append(f"The biggest yield booster is **{feat}**, contributing approximately "
                     f"+{val:,.0f} kg/ha to your prediction.")
    if top_neg:
        feat, val = top_neg[0]
        lines.append(f"Your yield is being held back mainly by **{feat}**, "
                     f"reducing it by ~{abs(val):,.0f} kg/ha.")
        if "Nitrogen" in feat:
            lines.append("Adjusting **Nitrogen Applied** toward 50–80 kg/ha could increase "
                         "your yield by approximately 200–400 kg/ha.")
        elif "Management" in feat:
            lines.append("Improving the **Management Score** through timely weeding and "
                         "pest control could add 100–300 kg/ha.")
    if tier == "LOW":
        lines.append("Overall, there is significant room for improvement — focus on "
                     "fertiliser adequacy and management practices first.")
    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PDF Export
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(
    prediction: float,
    ci_lo: float,
    ci_hi: float,
    country: str,
    country_avg: float,
    farmer_vals: dict,
    explanation: str,
    shap_path: str = "outputs/shap_waterfall.png",
    lime_path: str = "outputs/lime_bar.png",
) -> bytes:
    """
    Generate a downloadable PDF report with prediction, explanation,
    and SHAP/LIME charts.  Returns raw bytes.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "MaizeIQ — Yield Prediction Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(6)

    # Prediction summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Predicted Yield", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"  {prediction:,.0f} kg/ha  ({prediction/1000:.2f} t/ha)", ln=True)
    pdf.cell(0, 7, f"  95% CI: {ci_lo:,.0f} – {ci_hi:,.0f} kg/ha", ln=True)
    pdf.cell(0, 7, f"  Country average ({country.title()}): {country_avg:,.0f} kg/ha", ln=True)
    pdf.ln(4)

    # Farmer inputs
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Input Values", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for feat, val in farmer_vals.items():
        pdf.cell(0, 6, f"  {feat}: {val}", ln=True)
    pdf.ln(4)

    # Plain-language explanation
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Explanation", ln=True)
    pdf.set_font("Helvetica", "", 10)
    # Strip markdown bold markers for plain text
    clean = explanation.replace("**", "")
    pdf.multi_cell(0, 6, clean)
    pdf.ln(4)

    # SHAP chart (if file exists)
    if os.path.exists(shap_path):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "SHAP Feature Contributions", ln=True)
        pdf.image(shap_path, w=170)
        pdf.ln(4)

    # LIME chart
    if os.path.exists(lime_path):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "LIME Local Explanation", ln=True)
        pdf.image(lime_path, w=170)

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit page layout helpers
# ─────────────────────────────────────────────────────────────────────────────
def yield_color(y: float) -> str:
    return "#22c55e" if y > 5000 else "#f59e0b" if y > 3500 else "#ef4444"


def yield_tier(y: float) -> str:
    return "🟢 HIGH" if y > 5000 else "🟡 MODERATE" if y > 3500 else "🔴 LOW"


# ─────────────────────────────────────────────────────────────────────────────
# ── APP ENTRY POINT ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaizeIQ — Yield Predictor",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS
st.markdown("""
<style>
    body, .stApp { background: #0f1a0f; color: #e8f5e9; font-family: Georgia, serif; }
    .stSidebar { background: #162816; }
    .metric-card {
        background: #162816; border: 2px solid; border-radius: 14px;
        padding: 24px; text-align: center; margin-bottom: 16px;
    }
    .yield-number { font-size: 56px; font-weight: 900; line-height: 1.1; }
    .section-title { color: #86efac; font-size: 14px; font-weight: 700;
                     text-transform: uppercase; letter-spacing: 1.5px;
                     border-bottom: 1px solid #2d5a27; padding-bottom: 6px; margin-bottom: 12px; }
    .tip-card { background: #162816; border: 1px solid #2d5a27; border-radius: 10px;
                padding: 14px; margin-bottom: 10px; }
    .explain-box { background: #0f2a1a; border-left: 3px solid #22c55e;
                   padding: 14px 18px; border-radius: 6px; font-size: 14px; line-height: 1.75; }
</style>
""", unsafe_allow_html=True)

# ── Load artefacts ────────────────────────────────────────────────────────────
artefacts = load_all_artefacts()

if isinstance(artefacts, str):
    st.error(f"❌ Could not load model artefacts. Run phases 1–6 first.\n\n{artefacts}")
    st.stop()

model         = artefacts["model"]
scaler        = artefacts["scaler"]
feature_names = artefacts["feature_names"]
shap_exp      = artefacts["shap_explainer"]
lime_exp      = artefacts["lime_explainer"]
xai_metrics   = artefacts["xai_metrics"]
eval_metrics  = artefacts.get("metrics", {})

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []        # list of past prediction dicts
if "last_shap" not in st.session_state:
    st.session_state.last_shap = None    # (shap_vals, expected_value, row, prediction)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Maize_ear.jpg/240px-Maize_ear.jpg",
                 width=120, use_column_width=False)
st.sidebar.markdown("## 🌽 MaizeIQ")
st.sidebar.markdown("*One Acre Fund · East Africa · 2016–2022*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🔮 Predict Yield", "🔍 Explain Prediction", "📋 Past Forecasts", "💡 Tips"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown(f"R²: `{eval_metrics.get('test_r2', 0.879):.4f}`")
st.sidebar.markdown(f"RMSE: `{eval_metrics.get('test_rmse', 379.5):.1f}` kg/ha")
st.sidebar.markdown(f"MAE: `{eval_metrics.get('test_mae', 291.9):.1f}` kg/ha")
st.sidebar.markdown("Dataset: `74,967 samples`")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Predict Yield
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔮 Predict Yield":
    st.title("🔮 Maize Yield Prediction")
    st.caption("Adjust your agronomic inputs to get a real-time yield estimate.")

    col_inputs, col_results = st.columns([1, 1.6], gap="large")

    # ── Left: inputs ──────────────────────────────────────────────────────────
    with col_inputs:
        st.markdown('<p class="section-title">🌍 Location & Season</p>', unsafe_allow_html=True)
        country = st.selectbox(
            "Country",
            list(COUNTRY_AVGS.keys()),
            format_func=str.title,
        )
        season = st.selectbox("Season", ["first season", "second season"])
        planting_date = st.date_input("Planting Date", value=date(2024, 4, 1))
        plant_doy = planting_date.timetuple().tm_yday   # convert to day-of-year

        st.markdown('<p class="section-title" style="margin-top:18px">🌾 Agronomic Inputs</p>', unsafe_allow_html=True)

        farmer_vals = {}
        for feat in FARMER_FEATURES:
            if feat == "Planting Day of Year":
                # Driven by the date picker above
                farmer_vals[feat] = float(plant_doy)
                st.info(f"📅 Planting Day of Year: **{plant_doy}** (from date picker)")
                continue
            meta = FEATURE_META[feat]
            val = st.slider(
                f"{feat}  [{meta['unit']}]" if meta["unit"] else feat,
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(meta["default"]),
                step=float(meta["step"]),
            )
            farmer_vals[feat] = val
            if val < meta["warn_below"]:
                st.warning(meta["warn_msg"])

        farmer_vals["Planting Day of Year"] = float(plant_doy)  # COLUMN: plant_doy

        st.markdown('<p class="section-title" style="margin-top:18px">💧 What-If: Drought</p>', unsafe_allow_html=True)
        drought_sim = st.slider(
            "Drought Severity (0 = none, 50 = severe)", 0, 50, 0, 1,
            help="Simulates the additional yield penalty from drought stress beyond your current inputs."
        )
        drought_penalty = drought_sim * -18.0   # ~18 kg/ha per unit of drought stress
        if drought_sim > 20:
            st.warning(f"⚠️ Severe drought: estimated penalty of {abs(drought_penalty):,.0f} kg/ha")

        predict_btn = st.button("🔮 Predict & Save Forecast", use_container_width=True, type="primary")

    # ── Build input row & predict ─────────────────────────────────────────────
    row_df = build_input_row(farmer_vals, feature_names, country, season)
    prediction_raw, ci_lo, ci_hi = predict_with_ci(model, scaler, row_df)
    prediction = max(991.0, prediction_raw + drought_penalty)
    ci_lo = max(991.0, ci_lo + drought_penalty)
    ci_hi = min(8614.0, ci_hi + drought_penalty)
    country_avg = COUNTRY_AVGS.get(country, DISTRICT_AVG)
    pct_vs_avg  = (prediction - country_avg) / country_avg * 100

    # Precompute SHAP for this row (used on Explain page too)
    row_tuple = tuple(row_df.iloc[0].values)
    sv, ev = compute_shap_for_row(model, shap_exp, row_tuple, feature_names)
    st.session_state.last_shap = {
        "sv": sv, "ev": ev, "row_df": row_df,
        "prediction": prediction, "country_avg": country_avg,
        "farmer_vals": farmer_vals, "country": country,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
    }

    # Save to history
    if predict_btn:
        st.session_state.history.insert(0, {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Country": country.title(),
            "Season": season.title(),
            "Prediction (kg/ha)": round(prediction),
            **{k: round(v, 2) for k, v in farmer_vals.items()},
            "Drought Severity": drought_sim,
            "Tier": yield_tier(prediction).split()[-1],
        })
        st.success("✅ Forecast saved to Past Forecasts!")

    # ── Right: results ────────────────────────────────────────────────────────
    with col_results:
        col = yield_color(prediction)
        st.markdown(f"""
        <div class="metric-card" style="border-color:{col}">
            <div style="color:{col};font-size:11px;font-weight:700;
                        text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">
                {yield_tier(prediction)}
            </div>
            <div class="yield-number" style="color:{col}">{prediction:,.0f}</div>
            <div style="color:#86efac;font-size:16px">kg/ha &nbsp;·&nbsp; {prediction/1000:.2f} t/ha</div>
            <div style="color:#6b7280;font-size:12px;margin-top:6px">
                95% CI: {ci_lo:,.0f} – {ci_hi:,.0f} kg/ha
            </div>
            <div style="color:{'#22c55e' if pct_vs_avg >= 0 else '#ef4444'};font-size:13px;margin-top:4px;font-weight:600">
                {'▲' if pct_vs_avg >= 0 else '▼'} {abs(pct_vs_avg):.1f}% vs {country.title()} average
                ({country_avg:,.0f} kg/ha)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Yield scale progress bar
        st.markdown('<p class="section-title">Yield Performance Scale</p>', unsafe_allow_html=True)
        pct = min(100, max(0, (prediction - 991) / (8614 - 991) * 100))
        st.progress(int(pct))
        st.caption(f"Position: {prediction:,.0f} kg/ha on scale 991–8,614 kg/ha")

        # Year trend
        st.markdown('<p class="section-title" style="margin-top:18px">Historical Yield Trend</p>', unsafe_allow_html=True)
        trend_df = pd.DataFrame({"Year": list(YEAR_AVGS.keys()), "Avg Yield (kg/ha)": list(YEAR_AVGS.values())})
        fig_trend = px.line(
            trend_df, x="Year", y="Avg Yield (kg/ha)", markers=True,
            color_discrete_sequence=["#22c55e"],
        )
        fig_trend.add_hline(y=prediction, line_color=col, line_dash="dash",
                            annotation_text="Your Prediction", annotation_font_color=col)
        fig_trend.update_layout(
            plot_bgcolor="#162816", paper_bgcolor="#0f1a0f",
            font_color="#e8f5e9", height=220,
            margin=dict(l=10, r=10, t=30, b=30),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Country comparison
        st.markdown('<p class="section-title">Country Yield Comparison</p>', unsafe_allow_html=True)
        cc_df = pd.DataFrame({
            "Country": [c.title() for c in COUNTRY_AVGS],
            "Avg Yield": list(COUNTRY_AVGS.values()),
            "Selected": [c == country for c in COUNTRY_AVGS],
        })
        fig_cc = px.bar(
            cc_df, x="Country", y="Avg Yield", color="Selected",
            color_discrete_map={True: "#22c55e", False: "#2d5a27"},
        )
        fig_cc.update_layout(
            plot_bgcolor="#162816", paper_bgcolor="#0f1a0f",
            font_color="#e8f5e9", showlegend=False, height=200,
            margin=dict(l=10, r=10, t=20, b=40),
        )
        st.plotly_chart(fig_cc, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Explain Prediction
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Explain Prediction":
    st.title("🔍 Explanation")
    st.caption("Understand what's driving the model's prediction — in plain language and charts.")

    if st.session_state.last_shap is None:
        st.info("👉 Go to **Predict Yield** first to generate a prediction, then come back here.")
        st.stop()

    state = st.session_state.last_shap
    sv, ev        = state["sv"], state["ev"]
    row_df        = state["row_df"]
    prediction    = state["prediction"]
    country_avg   = state["country_avg"]
    farmer_vals   = state["farmer_vals"]
    country       = state["country"]
    ci_lo, ci_hi  = state["ci_lo"], state["ci_hi"]

    # Model metrics banner
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("R² (Test)", f"{eval_metrics.get('test_r2', 0.879):.4f}")
    m_col2.metric("RMSE", f"{eval_metrics.get('test_rmse', 379.5):.1f} kg/ha")
    m_col3.metric("MAE",  f"{eval_metrics.get('test_mae', 291.9):.1f} kg/ha")
    m_col4.metric("Training samples", "59,973")

    st.markdown("---")

    # ── SHAP waterfall ────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🔬 SHAP Feature Contributions</p>', unsafe_allow_html=True)
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows how each input feature pushes the prediction "
        "above (green) or below (red) the average yield of 4,205 kg/ha."
    )
    fig_shap = plot_shap_waterfall_plotly(sv, feature_names, ev, prediction)
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")

    # ── LIME ──────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🔬 LIME Local Feature Contributions</p>', unsafe_allow_html=True)
    st.caption(
        "LIME builds a simple local linear model around your specific input to explain the "
        "XGBoost prediction. All labels are human-readable feature names."
    )
    if lime_exp is not None:
        try:
            lime_exp_result = lime_exp.explain_instance(
                row_df.iloc[0].values, model.predict, num_features=10
            )
            lime_contrib = dict(lime_exp_result.as_list())
            fig_lime = plot_lime_bar_plotly(lime_contrib)
            st.plotly_chart(fig_lime, use_container_width=True)
        except Exception as e:
            st.warning(f"LIME explanation unavailable: {e}")
    else:
        st.info("LIME explainer not loaded. Run phase6_xai.py to generate it.")

    st.markdown("---")

    # ── Plain-language explanation ────────────────────────────────────────────
    st.markdown('<p class="section-title">💬 Plain-Language Summary</p>', unsafe_allow_html=True)
    explanation = plain_explanation(prediction, country_avg, sv, feature_names)
    st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Explanation Quality Metrics ───────────────────────────────────────────
    if xai_metrics:
        st.markdown('<p class="section-title">📏 Explanation Quality Metrics</p>', unsafe_allow_html=True)
        eq1, eq2, eq3 = st.columns(3)
        eq1.metric(
            "Fidelity (LIME↔XGBoost r)",
            f"{xai_metrics.get('fidelity', 0):.4f}",
            help="Correlation between LIME's local model and XGBoost on perturbed samples. Closer to 1 = better."
        )
        eq2.metric(
            "Stability (rank std, 10 runs)",
            f"{xai_metrics.get('stability_std', 0):.4f}",
            help="Standard deviation of LIME feature ranks across 10 repeated runs. Lower = more stable."
        )
        eq3.metric(
            "Sparsity (90% variance)",
            f"{xai_metrics.get('sparsity_n', '—')} features",
            help="Number of top features needed to explain 90% of cumulative |SHAP| variance."
        )

    st.markdown("---")

    # ── PDF export ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">📄 Export Report</p>', unsafe_allow_html=True)
    if st.button("📥 Generate PDF Report"):
        try:
            pdf_bytes = generate_pdf_report(
                prediction, ci_lo, ci_hi,
                country, country_avg,
                farmer_vals, explanation,
            )
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"maize_yield_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.warning(f"PDF generation requires fpdf2: pip install fpdf2\n\nError: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Past Forecasts
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Past Forecasts":
    st.title("📋 Past Forecasts")
    st.caption("All predictions made in this session. Data is stored in memory and resets on refresh.")

    if not st.session_state.history:
        st.info("No forecasts yet — make a prediction on the **Predict Yield** page and click *Predict & Save Forecast*.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)

        # Colour-code the Tier column
        def tier_color(val):
            colors = {"HIGH": "color: #22c55e", "MODERATE": "color: #f59e0b", "LOW": "color: #ef4444"}
            return colors.get(val, "")

        st.dataframe(
            hist_df.style.map(tier_color, subset=["Tier"]),
            use_container_width=True,
            height=400,
        )

        # CSV download
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=f"maize_forecasts_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

        # Quick summary stats
        st.markdown("---")
        st.markdown('<p class="section-title">Summary Statistics</p>', unsafe_allow_html=True)
        pred_col = "Prediction (kg/ha)"
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Forecasts saved", len(hist_df))
        s2.metric("Mean prediction", f"{hist_df[pred_col].mean():,.0f} kg/ha")
        s3.metric("Best prediction", f"{hist_df[pred_col].max():,.0f} kg/ha")
        s4.metric("% above avg", f"{(hist_df[pred_col] > DISTRICT_AVG).mean()*100:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Tips
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💡 Tips":
    st.title("💡 Agronomic Tips")
    st.caption("Evidence-based guidance for each farmer-controllable feature.")

    for icon, title, content in AGRONOMIC_TIPS:
        st.markdown(f"""
        <div class="tip-card">
            <div style="font-size:15px;font-weight:700;color:#86efac;margin-bottom:6px">
                {icon} {title}
            </div>
            <div style="font-size:13px;color:#a7f3d0;line-height:1.65">{content}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Model Performance (XGBoost — Real Metrics)</p>',
                unsafe_allow_html=True)
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("R² (Test)",   f"{eval_metrics.get('test_r2',   0.879):.4f}", help="Fraction of yield variance explained")
    tc2.metric("RMSE",        f"{eval_metrics.get('test_rmse', 379.5):.1f} kg/ha")
    tc3.metric("MAE",         f"{eval_metrics.get('test_mae',  291.9):.1f} kg/ha")
    tc4.metric("Train R²",    f"{eval_metrics.get('train_r2',  0.929):.4f}")

    st.markdown("---")
    st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)
    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Total samples",  "74,967")
    dc2.metric("Countries",      "7")
    dc3.metric("Years",          "2016–2022")
    dc4.metric("Features",       "38")

    st.markdown("---")
    st.caption(
        "Model: XGBoost with Optuna hyperparameter tuning (50 trials). "
        "Stratified 80/20 train/test split by yield quartile. "
        "Explanations via SHAP TreeExplainer and LIME LimeTabularExplainer."
    )
