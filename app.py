"""
Phase 7 — Streamlit Farmer-Facing Maize Yield Prediction App
One Acre Fund Maize Yield Prediction System

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import shap
import lime
import lime.lime_tabular
import joblib
import json
import os
import warnings
from datetime import date, datetime
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaizeIQ — Yield Predictor",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET             = "Yield (kg/ha)"
DISTRICT_AVG       = 3500.0  # Baseline district average kg/ha
HISTORY_FILE       = "prediction_history.csv"

# Top 5 farmer-controllable features (from Phase 1 MI analysis)
FARMER_FEATURES = [
    "Phosphorus Applied (kg/ha)",    # COLUMN: verify rename matches Phase 2
    "Management Score",              # COLUMN: verify rename matches Phase 2
    "Nitrogen Applied (kg/ha)",      # COLUMN: verify rename matches Phase 2
    "Planting Density (plants/m²)",  # COLUMN: verify rename matches Phase 2
    "Planting Day of Year",          # COLUMN: verify rename matches Phase 2
]

FEATURE_UNITS = {
    "Phosphorus Applied (kg/ha)":   "kg/ha",
    "Management Score":             "score (0–10)",
    "Nitrogen Applied (kg/ha)":     "kg/ha",
    "Planting Density (plants/m²)": "plants/m²",
    "Planting Day of Year":         "day (1–365)",
}

FEATURE_RANGES = {
    "Phosphorus Applied (kg/ha)":   (0.0,  60.0, 20.0),   # (min, max, default)
    "Management Score":             (0.0,  10.0, 5.0),
    "Nitrogen Applied (kg/ha)":     (0.0, 150.0, 50.0),
    "Planting Density (plants/m²)": (1.0,  10.0, 5.0),
    "Planting Day of Year":         (1.0, 365.0, 80.0),
}

TIPS = {
    "Phosphorus Applied (kg/ha)":   "Applying 20–30 kg/ha of phosphorus at planting significantly improves root development and early growth. Use diammonium phosphate (DAP) for best results.",
    "Management Score":             "Good farm management — timely weeding, pest control, and proper spacing — can raise yields by up to 30%. Aim for a management score above 7.",
    "Nitrogen Applied (kg/ha)":     "Nitrogen is the #1 yield driver. Split applications (1/3 at planting, 2/3 at V6) improve uptake. Optimal range is 60–90 kg/ha for most regions.",
    "Planting Density (plants/m²)": "5–6 plants/m² is optimal for most maize hybrids. Too dense causes competition; too sparse wastes sunlight.",
    "Planting Day of Year":         "Planting within the optimal window (day 60–100 in most E. Africa regions) maximises use of early season rains. Late planting reduces yields by ~5% per week.",
}


# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    """Load XGBoost model, scaler, and feature names. Cache across sessions."""
    try:
        model       = joblib.load("model.joblib")
        scaler      = joblib.load("scaler.joblib")
        feat_names  = joblib.load("feature_names.joblib")
        return model, scaler, feat_names
    except FileNotFoundError as e:
        st.error(f"⚠️ Model artifacts not found. Run phase4_5_train.py first.\n{e}")
        st.stop()


@st.cache_resource
def load_training_data(feat_names):
    """Load a background sample for LIME/SHAP. Cached to avoid reloading."""
    try:
        df = pd.read_csv("cleaned_data.csv")
        df.drop(columns=["lon"], inplace=True, errors="ignore")
        cats = [c for c in ["country", "season"] if c in df.columns]
        if cats:
            df = pd.get_dummies(df, columns=cats, drop_first=False)
        for c in df.select_dtypes("object").columns:
            df.drop(columns=[c], inplace=True)
        X = df.drop(columns=[TARGET])[feat_names]
        y = df[TARGET]
        sample = X.sample(min(500, len(X)), random_state=42)
        return X, y, sample
    except Exception as e:
        st.warning(f"Background data not found — LIME unavailable: {e}")
        return None, None, None


model, scaler, feat_names = load_model_artifacts()
X_full, y_full, bg_sample = load_training_data(feat_names)

shap_explainer = None
shap_error = None
if bg_sample is not None:
    try:
        # Use TreeExplainer directly for XGBoost models
        shap_explainer = shap.TreeExplainer(model, data=scaler.transform(bg_sample))
    except Exception as e:
        shap_error = str(e)
else:
    shap_error = "Background data not loaded"

lime_explainer = None
if bg_sample is not None:
    try:
        bg_s = scaler.transform(bg_sample)
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=bg_s,
            feature_names=feat_names,
            mode="regression",
            random_state=42,
        )
    except Exception as e:
        pass


# ── Helpers ────────────────────────────────────────────────────────────────────
def build_input_vector(user_inputs: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame from user inputs.
    Non-specified features are filled with training medians.
    """
    if X_full is None:
        base = {f: 0.0 for f in feat_names}
    else:
        base = X_full.median().to_dict()
    base.update(user_inputs)
    return pd.DataFrame([base])[feat_names]


def predict_yield(row_df: pd.DataFrame) -> tuple:
    """Return (prediction_kg_ha, confidence_interval_tuple)."""
    row_s = scaler.transform(row_df)
    pred  = float(model.predict(row_s)[0])
    # Approximate 90% CI from model residual std (from evaluation: ~388 kg/ha RMSE)
    std_err = 388.0
    ci_lo   = max(0, pred - 1.645 * std_err)
    ci_hi   = pred + 1.645 * std_err
    return pred, (ci_lo, ci_hi)


def get_shap_values(row_s: np.ndarray):
    """Return SHAP values for a row as 1D array, or None if SHAP is unavailable."""
    if shap_explainer is None:
        return None
    try:
        # TreeExplainer.shap_values() returns shape (n_samples, n_features)
        sv = shap_explainer.shap_values(row_s)
        if isinstance(sv, list):
            sv = sv[0] if len(sv) > 0 else None
        # Flatten to 1D if 2D
        if sv is not None and isinstance(sv, np.ndarray) and sv.ndim > 1:
            sv = sv[0]
        return sv
    except Exception as e:
        return None


def yield_color(kg_ha: float) -> str:
    """Return color string based on yield level."""
    if kg_ha >= 4500:
        return "#27ae60"   # green
    elif kg_ha >= 3000:
        return "#f39c12"   # amber
    return "#e74c3c"       # red


def yield_label(kg_ha: float) -> str:
    """Return descriptive yield level."""
    if kg_ha >= 4500:
        return "🟢 High Yield"
    elif kg_ha >= 3000:
        return "🟡 Medium Yield"
    return "🔴 Low Yield"


def plain_language(pred: float, shap_top: list) -> str:
    """Generate farmer-friendly plain-language explanation from SHAP values."""
    level = "HIGH" if pred > 4500 else ("MEDIUM" if pred > 3000 else "LOW")
    pct   = ((pred - DISTRICT_AVG) / DISTRICT_AVG) * 100
    dir_  = "above" if pct >= 0 else "below"
    top_pos = [(f, v) for f, v in shap_top if v > 0][:2]
    top_neg = [(f, v) for f, v in shap_top if v < 0][:2]

    lines = [
        f"**Predicted yield: {pred:,.0f} kg/ha ({pred/1000:.2f} t/ha)** — {level} yield.",
        f"This is **{abs(pct):.1f}% {dir_}** the district average of {DISTRICT_AVG:,.0f} kg/ha.",
        "",
    ]
    if top_pos:
        factors = " and ".join([f"**{f[0]}**" for f in top_pos])
        lines.append(f"✅ Your yield is boosted mainly by {factors}.")
    if top_neg:
        for f, v in top_neg:
            lines.append(f"⚠️ **{f}** is pulling your yield down by ~{abs(v):.0f} kg/ha.")
            lines.append(f"   Improving this factor could increase your yield by approximately **{abs(v):.0f} kg/ha**.")
    return "\n".join(lines)


def save_prediction(inputs: dict, pred: float) -> None:
    """Append prediction to history CSV."""
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "predicted_kg_ha": round(pred, 1)}
    row.update({k: round(v, 2) if isinstance(v, float) else v for k, v in inputs.items()})
    df_new = pd.DataFrame([row])
    if os.path.exists(HISTORY_FILE):
        existing = pd.read_csv(HISTORY_FILE)
        df_new = pd.concat([existing, df_new], ignore_index=True)
    df_new.to_csv(HISTORY_FILE, index=False)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Space+Mono&display=swap');
  html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
  .metric-card { 
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
      border-radius: 16px; padding: 28px 32px; text-align: center;
      box-shadow: 0 8px 32px rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.08);
  }
  .metric-value { font-size: 3.5rem; font-weight: 700; font-family: 'Space Mono'; }
  .metric-unit  { font-size: 1.1rem; color: #aaa; margin-top: -8px; }
  .metric-label { font-size: 0.9rem; color: #ccc; margin-top: 8px; }
  .section-header { font-size: 1.4rem; font-weight: 700; margin-top: 16px;
                    border-left: 4px solid #2ecc71; padding-left: 12px; }
  .warn-box { background:#fff3cd; border-left:4px solid #f39c12;
              padding:10px 14px; border-radius:6px; font-size:0.88rem; }
  .success-box { background:#d4edda; border-left:4px solid #27ae60;
                 padding:10px 14px; border-radius:6px; font-size:0.88rem; }
  div[data-testid="metric-container"] { background: #f0f4ff; border-radius:10px; padding:12px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🌽 MaizeIQ")
    st.caption("AI-powered yield prediction for smallholder farmers")
    st.divider()

    page = st.radio("Navigate", ["🌾 Predict Yield", "🔍 Explain Prediction",
                                  "📋 Past Forecasts", "💡 Tips"], label_visibility="collapsed")

    st.divider()
    st.markdown("### 🎛️ Farm Inputs")
    st.caption("Adjust your farm conditions below:")

    user_inputs = {}
    warnings_list = []

    for feat in FARMER_FEATURES:
        lo, hi, default = FEATURE_RANGES[feat]
        unit = FEATURE_UNITS.get(feat, "")
        val = st.slider(f"{feat}", min_value=float(lo), max_value=float(hi),
                        value=float(default), step=0.5 if hi <= 20 else 1.0,
                        help=f"Unit: {unit}")
        user_inputs[feat] = val

    # Input validation warnings
    if user_inputs["Nitrogen Applied (kg/ha)"] < 20:                    # COLUMN: threshold
        warnings_list.append("⚠️ Nitrogen below 20 kg/ha — likely insufficient for good yield.")
    if user_inputs["Phosphorus Applied (kg/ha)"] < 10:                  # COLUMN: threshold
        warnings_list.append("⚠️ Phosphorus below 10 kg/ha — consider adding starter fertilizer.")
    if user_inputs["Management Score"] < 3:                             # COLUMN: threshold
        warnings_list.append("⚠️ Low management score — improve practices to unlock yield potential.")

    if warnings_list:
        for w in warnings_list:
            st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)
    st.divider()

    planting_date = st.date_input("🗓️ Planting Date", value=date(2024, 3, 15))
    drought_slider = st.slider("💧 Drought Impact Simulation", 0.0, 1.0, 0.3,
                                help="0 = no drought, 1 = severe drought. Adjusts prediction.")
    country_sel = st.selectbox("🌍 Country", ["kenya","tanzania","rwanda","burundi","zambia","uganda","nigeria"])
    season_sel  = st.selectbox("🌦️ Season", ["first season", "second season"])
    st.divider()
    predict_btn = st.button("🚀 Generate Prediction", type="primary", use_container_width=True)


# ── Compute prediction ─────────────────────────────────────────────────────────
input_row = build_input_vector(user_inputs)

# Inject country & season one-hot columns if present
for col in feat_names:
    if col.startswith("country_"):
        input_row[col] = 1.0 if col == f"country_{country_sel}" else 0.0
    if col.startswith("season_"):
        input_row[col] = 1.0 if col == f"season_{season_sel}" else 0.0

# Apply drought simulation scaling
input_row = input_row.copy()
drought_scale = 1.0 - 0.35 * drought_slider  # linear penalty
pred_kg_ha, (ci_lo, ci_hi) = predict_yield(input_row)
pred_adj = pred_kg_ha * drought_scale
ci_lo_adj, ci_hi_adj = ci_lo * drought_scale, ci_hi * drought_scale

if predict_btn:
    save_prediction(user_inputs, pred_adj)
    st.session_state["last_pred"]   = pred_adj
    st.session_state["last_inputs"] = user_inputs.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Predict Yield
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🌾 Predict Yield":
    st.title("🌽 Maize Yield Prediction")
    st.caption("Enter your farm conditions in the sidebar to generate a personalised yield forecast.")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        color = yield_color(pred_adj)
        label = yield_label(pred_adj)
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{color}">{pred_adj:,.0f}</div>
          <div class="metric-unit">kg / hectare</div>
          <div class="metric-label">{label}</div>
          <div class="metric-label" style="margin-top:10px">
            {pred_adj/1000:.2f} t/ha &nbsp;|&nbsp;
            90% CI: {ci_lo_adj:,.0f} – {ci_hi_adj:,.0f} kg/ha
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        pct_vs_avg = ((pred_adj - DISTRICT_AVG) / DISTRICT_AVG) * 100
        st.metric("vs District Average", f"{pct_vs_avg:+.1f}%",
                  delta=f"{pred_adj - DISTRICT_AVG:+.0f} kg/ha")
        st.metric("Confidence Interval", f"±{(ci_hi_adj-ci_lo_adj)/2:,.0f} kg/ha")

    with col3:
        st.metric("Planting Day", int(user_inputs["Planting Day of Year"]))
        st.metric("Drought Risk",
                  "High" if drought_slider > 0.6 else ("Medium" if drought_slider > 0.3 else "Low"))

    st.divider()

    # Yield driver chart
    st.markdown('<p class="section-header">📊 Yield Driver Importance</p>', unsafe_allow_html=True)

    row_s = scaler.transform(input_row)
    sv = get_shap_values(row_s)
    if sv is None:
        st.warning("SHAP explanations are unavailable in this environment.")
    else:
        sv_df = pd.DataFrame({"Feature": feat_names, "SHAP Value": sv})
        sv_df = sv_df.reindex(sv_df["SHAP Value"].abs().sort_values(ascending=False).index).head(10)

        fig = go.Figure(go.Bar(
            x=sv_df["SHAP Value"],
            y=sv_df["Feature"],
            orientation="h",
            marker_color=["#27ae60" if v >= 0 else "#e74c3c" for v in sv_df["SHAP Value"]],
            text=[f"{v:+.0f}" for v in sv_df["SHAP Value"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="SHAP Values — Feature Contributions to Your Prediction",
            xaxis_title="Contribution to Yield (kg/ha)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=420,
            font=dict(size=12),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Growth stage progress bar
    doy = int(user_inputs["Planting Day of Year"])
    days_since_plant = max(0, date.today().timetuple().tm_yday - doy)
    growth_pct = min(1.0, days_since_plant / 120)
    stage = ("🌱 Germination" if growth_pct < 0.15 else
             "🌿 Vegetative" if growth_pct < 0.45 else
             "🌾 Tasseling" if growth_pct < 0.70 else
             "🌽 Grain Fill" if growth_pct < 0.90 else
             "✅ Harvest Ready")
    st.markdown(f"**Estimated Crop Stage:** {stage}")
    st.progress(growth_pct)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Explain Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Explain Prediction":
    st.title("🔍 Explain Your Yield Prediction")
    st.caption("Understand *why* the model predicted this yield using SHAP and LIME.")

    row_s = scaler.transform(input_row)
    sv = get_shap_values(row_s)
    if sv is None:
        st.warning("SHAP explanations are unavailable in this environment.")
        shap_top = []
    else:
        shap_top = sorted(zip(feat_names, sv), key=lambda x: abs(x[1]), reverse=True)

    # Plain-language summary
    st.markdown("### 📝 Plain-Language Summary")
    summary = plain_language(pred_adj, shap_top)
    for line in summary.split("\n"):
        st.markdown(line)
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if sv is None:
            st.info("SHAP waterfall is unavailable when SHAP explanations cannot be computed.")
        else:
            st.markdown("### 🧮 SHAP Waterfall")
            st.markdown("""
            <div style="background:#f8f9fa;border-radius:8px;padding:10px 14px;font-size:0.82rem;color:#555">
            <b>What is SHAP?</b> SHAP shows how each feature pushes the prediction
            above or below the model's baseline. Green bars = positive impact,
            red bars = negative impact.
            </div>""", unsafe_allow_html=True)
            top_n = 12
            top_idx = np.argsort(np.abs(sv))[-top_n:][::-1]
            fig, ax = plt.subplots(figsize=(7, 5))
            colors = ["#27ae60" if sv[i] >= 0 else "#e74c3c" for i in top_idx]
            ax.barh(range(top_n), sv[top_idx], color=colors, alpha=0.85)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([feat_names[i][:35] for i in top_idx][::-1], fontsize=9) # Corrected indexing for yticklabels
            ax.axvline(0, color="black", linewidth=0.7)
            ax.set_xlabel("SHAP contribution (kg/ha)")
            ax.set_title("SHAP Feature Contributions", fontweight="bold", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.markdown("### 🔬 LIME Local Explanation")
        st.markdown("""
        <div style="background:#f8f9fa;border-radius:8px;padding:10px 14px;font-size:0.82rem;color:#555">
        <b>What is LIME?</b> LIME explains individual predictions by fitting a simple
        linear model in the neighbourhood of your input. It shows which features
        matter most *locally* for your specific farm conditions.
        </div>""", unsafe_allow_html=True)

        if lime_explainer is not None:
            try:
                x_s = scaler.transform(input_row.values)[0]
                exp = lime_explainer.explain_instance(
                    x_s, lambda arr: model.predict(arr), num_features=10)
                lime_pairs = sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True)
                lime_feats = [p[0] for p in lime_pairs]
                lime_vals  = [p[1] for p in lime_pairs]
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                colors2 = ["#27ae60" if v >= 0 else "#e74c3c" for v in lime_vals]
                ax2.barh(range(len(lime_feats)), lime_vals[::-1], color=colors2[::-1], alpha=0.85)
                ax2.set_yticks(range(len(lime_feats)))
                ax2.set_yticklabels(lime_feats[::-1], fontsize=8)
                ax2.axvline(0, color="black", linewidth=0.7)
                ax2.set_xlabel("LIME contribution (kg/ha)")
                ax2.set_title("LIME Local Feature Contributions", fontweight="bold", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            except Exception as e:
                st.info(f"LIME unavailable for this input: {e}")
        else:
            st.info("LIME requires cleaned_data.csv to be present.")

    # What-if actionable insights
    st.divider()
    st.markdown("### 💡 Actionable What-If Insights")
    cols = st.columns(len(FARMER_FEATURES))
    for i, feat in enumerate(FARMER_FEATURES):
        with cols[i]:
            lo, hi, _ = FEATURE_RANGES[feat]
            improved = input_row.copy()
            improved[feat] = min(hi, input_row[feat].values[0] * 1.25 + 5)
            new_pred, _ = predict_yield(improved)
            new_pred_adj = new_pred * (1.0 - 0.35 * drought_slider)
            delta = new_pred_adj - pred_adj
            st.metric(
                label=feat[:28],
                value=f"{delta:+, .0f} kg/ha", # Corrected f-string
                delta=f"If +25% {FEATURE_UNITS.get(feat,'')}",
                delta_color="normal",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Past Forecasts
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Past Forecasts":
    st.title("📋 Past Forecast History")
    st.caption("All predictions made in this session and previous runs.")

    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        # Human-readable column headers already stored
        st.dataframe(hist.sort_values("timestamp", ascending=False)
                        .rename(columns={"predicted_kg_ha": "Predicted Yield (kg/ha)",
                                         "timestamp": "Timestamp"}),
                     use_container_width=True)

        fig_hist = px.line(hist, x="timestamp", y="predicted_kg_ha",
                           markers=True, title="Predicted Yield Over Time",
                           labels={"predicted_kg_ha": "Yield (kg/ha)", "timestamp": "Date"})
        fig_hist.add_hline(y=DISTRICT_AVG, line_dash="dash",
                           annotation_text="District Avg", line_color="orange")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Download button
        csv_bytes = hist.to_csv(index=False).encode()
        st.download_button("⬇️ Download History CSV", csv_bytes,
                           "prediction_history.csv", "text/csv")
    else:
        st.info("No predictions yet. Make a prediction on the 'Predict Yield' page first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Tips
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Tips":
    st.title("💡 Agronomic Tips for Higher Yields")
    st.caption("Evidence-based guidance on the top controllable yield drivers.")

    for feat, tip in TIPS.items():
        lo, hi, default = FEATURE_RANGES[feat]
        unit = FEATURE_UNITS.get(feat, "")
        current_val = user_inputs.get(feat, default)
        with st.expander(f"🌿 {feat}", expanded=True):
            col_tip, col_gauge = st.columns([3, 1])
            with col_tip:
                st.markdown(tip)
            with col_gauge:
                pct = (current_val - lo) / (hi - lo) if hi != lo else 0.5
                st.metric("Your current value", f"{current_val:.1f} {unit}")
                st.progress(float(pct))
                if pct < 0.3:
                    st.markdown('<div class="warn-box">Below recommended range</div>',
                                unsafe_allow_html=True)
                elif pct > 0.85:
                    st.markdown('<div class="warn-box">Above recommended range</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">Within optimal range ✓</div>',
                                unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📚 Additional Resources")
    st.markdown("""
    - [One Acre Fund Agronomic Training Guides](https://oneacrefund.org)
    - [CIMMYT Maize Production Guide](https://cimmyt.org)
    - [FAO Crop Calendar East Africa](https://www.fao.org/agriculture/seed/cropcalendar)
    """)
