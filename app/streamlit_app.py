"""
Streamlit Front-End — Mobile Phone Price Prediction
=====================================================
Sri Lanka Mobile Phone Price Prediction using XGBoost

Clean, professional UI with:
    - Sidebar for user inputs
    - Card-style layout
    - Model performance summary
    - Global SHAP explanation
    - Local SHAP explanation for individual predictions
    - Smart phone spec detection (handles basic phones without storage/RAM)

Usage:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Mobile Phone Price Predictor | Sri Lanka",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📱</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Force ALL text to be dark on white background
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* ============================================================
       GLOBAL — force white bg + dark text everywhere
       ============================================================ */
    .stApp,
    .stApp > header,
    .main,
    .main .block-container,
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }

    /* --- Force all text dark --- */
    .stApp, .stApp *,
    .stMarkdown, .stMarkdown *,
    p, span, div, label, li, td, th, a {
        color: #1E293B !important;
    }

    /* --- Headings --- */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #1E293B !important;
    }

    /* --- Main container --- */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* ============================================================
       SIDEBAR
       ============================================================ */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] * {
        color: #1E293B !important;
    }
    [data-testid="stSidebar"] label {
        color: #334155 !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stMarkdown hr {
        border-color: #CBD5E1 !important;
    }

    /* ============================================================
       FORM WIDGETS
       ============================================================ */
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] div,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #1E293B !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
    }
    ul[role="listbox"] li,
    ul[role="listbox"] li span,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li span,
    div[data-baseweb="menu"] li,
    div[data-baseweb="menu"] li span {
        color: #1E293B !important;
        background-color: #FFFFFF !important;
    }
    ul[role="listbox"] li:hover,
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="menu"] li:hover {
        background-color: #EFF6FF !important;
    }
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stSlider"] div,
    [data-testid="stSlider"] span,
    .stSlider div, .stSlider span {
        color: #1E293B !important;
    }
    .stCheckbox label span {
        color: #1E293B !important;
    }
    .stSelectbox label,
    .stSlider label,
    .stCheckbox label,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p {
        color: #334155 !important;
        font-weight: 500 !important;
    }
    [data-testid="stTooltipIcon"] svg {
        color: #94A3B8 !important;
        fill: #94A3B8 !important;
    }

    /* ============================================================
       METRIC CARDS
       ============================================================ */
    .metric-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        text-align: center;
        transition: box-shadow 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.10);
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #64748B !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-card .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1E293B !important;
    }

    /* ============================================================
       PREDICTION CARD (gradient — white text is correct here)
       ============================================================ */
    .prediction-card {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(30, 64, 175, 0.25);
    }
    .prediction-card .pred-label {
        font-size: 1rem;
        font-weight: 500;
        color: #FFFFFF !important;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    .prediction-card .pred-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
    }

    /* ============================================================
       SECTION HEADER
       ============================================================ */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E293B !important;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #3B82F6;
        margin-bottom: 1.5rem;
        margin-top: 2rem;
    }

    /* ============================================================
       INFO BOX
       ============================================================ */
    .info-box {
        background: #F0F9FF !important;
        border-left: 4px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #1E3A5F !important;
        line-height: 1.6;
    }
    .info-box * {
        color: #1E3A5F !important;
    }

    /* ============================================================
       SPEC NOTICE
       ============================================================ */
    .spec-notice {
        background: #FFFBEB !important;
        border-left: 4px solid #F59E0B;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.82rem;
        color: #92400E !important;
        line-height: 1.5;
    }
    .spec-notice * {
        color: #92400E !important;
    }

    /* ============================================================
       FOOTER / CAPTIONS / MISC
       ============================================================ */
    .footer-text {
        color: #94A3B8 !important;
        font-size: 0.85rem;
        text-align: center;
        padding: 1rem 0;
    }
    .footer-text * { color: #94A3B8 !important; }

    .stImage > div > div > p,
    [data-testid="caption"] {
        color: #64748B !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stAlert p, .stAlert span, .stAlert div {
        color: #1E293B !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #1E293B !important;
    }
    hr {
        border-color: #E2E8F0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load model and metadata
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    """Load all saved model artifacts."""
    try:
        model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))
        encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.joblib"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

        with open(os.path.join(MODELS_DIR, "feature_meta.json"), "r") as f:
            feature_meta = json.load(f)

        with open(os.path.join(MODELS_DIR, "column_info.json"), "r") as f:
            column_info = json.load(f)

        metrics_path = os.path.join(OUTPUTS_DIR, "metrics.json")
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

        # Load SHAP explainer
        explainer = None
        explainer_path = os.path.join(MODELS_DIR, "shap_explainer.joblib")
        if os.path.exists(explainer_path):
            explainer = joblib.load(explainer_path)

        # Load brand -> model mapping
        brand_model_map = {}
        bm_path = os.path.join(MODELS_DIR, "brand_model_mapping.json")
        if os.path.exists(bm_path):
            with open(bm_path, "r") as f:
                brand_model_map = json.load(f)

        # Load phone specs mapping (for auto-detecting storage/RAM)
        phone_specs = {}
        specs_path = os.path.join(MODELS_DIR, "phone_specs_mapping.json")
        if os.path.exists(specs_path):
            with open(specs_path, "r") as f:
                phone_specs = json.load(f)

        return model, encoders, scaler, feature_meta, column_info, metrics, explainer, brand_model_map, phone_specs

    except FileNotFoundError as e:
        st.error(f"Model files not found. Run the training pipeline first.\n\nMissing: {e}")
        st.stop()


# ---------------------------------------------------------------------------
# Lookup phone specs from the mapping
# ---------------------------------------------------------------------------

def get_phone_specs(brand, phone_model, phone_specs):
    """
    Look up typical storage/RAM for a brand+model combination.
    Returns (storage_gb, ram_gb, has_storage, has_ram).
    """
    key = f"{brand}||{phone_model}"
    if key in phone_specs:
        info = phone_specs[key]
        return info.get("storage"), info.get("ram"), info.get("has_storage", True), info.get("has_ram", True)
    return None, None, True, True


# ---------------------------------------------------------------------------
# Sidebar — User Inputs (with brand-filtered model dropdown + smart specs)
# ---------------------------------------------------------------------------

def render_sidebar(column_info, encoders, brand_model_map, phone_specs):
    """Render sidebar. Model dropdown filters by brand; storage/RAM auto-detects."""
    st.sidebar.markdown("## Phone Specifications")
    st.sidebar.markdown("---")

    inputs = {}

    # --- Brand ---
    brand_classes = column_info.get("brand_classes", {}).get("brand_clean", ["Apple", "Samsung", "Xiaomi"])
    brand_classes_sorted = sorted([b for b in brand_classes if b != "Other"]) + ["Other"]
    inputs["brand"] = st.sidebar.selectbox(
        "Brand",
        options=brand_classes_sorted,
        index=0,
        help="Select the phone manufacturer",
    )

    # --- Phone Model (filtered by selected brand) ---
    selected_brand = inputs["brand"]
    if brand_model_map and selected_brand in brand_model_map:
        available_models = brand_model_map[selected_brand]
    else:
        available_models = column_info.get("brand_classes", {}).get("phone_model_clean", ["Unknown"])

    models_sorted = sorted([m for m in available_models if m != "Other" and m != "Unknown"])
    model_options = models_sorted + ["Other"]
    inputs["phone_model"] = st.sidebar.selectbox(
        "Phone Model",
        options=model_options,
        index=0 if models_sorted else 0,
        help="Models shown are filtered by the selected brand",
    )

    # --- Condition ---
    condition_classes = column_info.get("brand_classes", {}).get("condition", ["Used", "Brand New"])
    inputs["condition"] = st.sidebar.selectbox(
        "Condition",
        options=sorted(condition_classes),
        index=0,
        help="New or used phone",
    )

    # --- Smart Storage & RAM (auto-detect based on phone model) ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Storage and Memory")

    # Look up if this phone typically has storage/RAM
    spec_storage, spec_ram, has_storage, has_ram = get_phone_specs(
        inputs["brand"], inputs["phone_model"], phone_specs
    )

    is_basic_phone = not has_storage and not has_ram

    if is_basic_phone:
        # Basic phone (like Nokia 105) — no meaningful storage/RAM
        st.sidebar.markdown("""
        <div class="spec-notice">
            This is a basic phone model. Storage and RAM are not applicable
            and will be set automatically for accurate prediction.
        </div>
        """, unsafe_allow_html=True)

        # Use the model's trained median values (what the scaler expects)
        # These will be set to the dataset median during preprocessing
        inputs["storage_gb"] = spec_storage if spec_storage and spec_storage > 0 else 0
        inputs["ram_gb"] = spec_ram if spec_ram and spec_ram > 0 else 0
        inputs["auto_specs"] = True

        st.sidebar.markdown(
            f"**Storage:** {'N/A' if inputs['storage_gb'] == 0 else str(int(inputs['storage_gb'])) + ' GB'}  \n"
            f"**RAM:** {'N/A' if inputs['ram_gb'] == 0 else str(int(inputs['ram_gb'])) + ' GB'}"
        )
    else:
        inputs["auto_specs"] = False

        # Determine default values from the phone specs mapping
        default_storage = 128
        default_ram = 6
        if spec_storage and spec_storage > 0:
            # Find the closest option
            storage_options = [16, 32, 64, 128, 256, 512, 1024]
            default_storage = min(storage_options, key=lambda x: abs(x - spec_storage))
        if spec_ram and spec_ram > 0:
            ram_options = [1, 2, 3, 4, 6, 8, 12, 16]
            default_ram = min(ram_options, key=lambda x: abs(x - spec_ram))

        # Storage
        storage_options = [16, 32, 64, 128, 256, 512, 1024]
        inputs["storage_gb"] = st.sidebar.select_slider(
            "Storage (GB)",
            options=storage_options,
            value=default_storage,
            help="Internal storage capacity" + (f" (typical for this model: ~{int(spec_storage)} GB)" if spec_storage else ""),
        )

        # RAM
        ram_options = [1, 2, 3, 4, 6, 8, 12, 16]
        inputs["ram_gb"] = st.sidebar.select_slider(
            "RAM (GB)",
            options=ram_options,
            value=default_ram,
            help="Random Access Memory" + (f" (typical for this model: ~{int(spec_ram)} GB)" if spec_ram else ""),
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Seller Information")

    # Location
    location_classes = column_info.get("brand_classes", {}).get("location_clean", ["Colombo"])
    location_sorted = sorted([l for l in location_classes if l != "Other"]) + ["Other"]
    inputs["location"] = st.sidebar.selectbox(
        "Location",
        options=location_sorted,
        index=0,
        help="Seller location in Sri Lanka",
    )

    # Is member
    inputs["is_member"] = st.sidebar.checkbox(
        "Verified Seller (Member)",
        value=False,
        help="Seller has a verified ikman.lk membership",
    )

    return inputs


# ---------------------------------------------------------------------------
# Prepare input features
# ---------------------------------------------------------------------------

def prepare_features(inputs, encoders, scaler, feature_meta):
    """Transform user inputs into model-ready feature array."""
    feature_cols = feature_meta["feature_cols"]
    features = {}

    # Categorical encoding
    cat_mapping = {
        "brand_clean_encoded": ("brand_clean", inputs["brand"]),
        "condition_encoded": ("condition", inputs["condition"]),
        "location_clean_encoded": ("location_clean", inputs["location"]),
        "phone_model_clean_encoded": ("phone_model_clean", inputs["phone_model"]),
    }

    for feat_col, (encoder_key, value) in cat_mapping.items():
        if feat_col in feature_cols and encoder_key in encoders:
            le = encoders[encoder_key]
            if value in le.classes_:
                features[feat_col] = le.transform([value])[0]
            else:
                if "Other" in le.classes_:
                    features[feat_col] = le.transform(["Other"])[0]
                else:
                    features[feat_col] = 0

    # Numeric features — scale using the same scaler from training
    numeric_raw = np.array([[inputs["storage_gb"], inputs["ram_gb"], int(inputs["is_member"])]])
    numeric_scaled = scaler.transform(numeric_raw)[0]

    num_feature_names = ["storage_gb", "ram_gb", "is_member_flag"]
    for i, name in enumerate(num_feature_names):
        if name in feature_cols:
            features[name] = numeric_scaled[i]

    feature_array = np.array([[features.get(col, 0) for col in feature_cols]])
    return feature_array


# ---------------------------------------------------------------------------
# Beautiful Custom SHAP Feature Contribution Chart
# ---------------------------------------------------------------------------

def create_beautiful_shap_chart(shap_vals, feature_names, base_value, predicted_value):
    """
    Create a custom, beautiful horizontal bar chart showing feature contributions.
    Replaces the default SHAP waterfall plot with a cleaner, smaller-font design.
    """
    # Pair features with their SHAP values
    pairs = list(zip(feature_names, shap_vals))
    # Sort by absolute value (largest impact first)
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    names = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    # Colours: positive impact (increases price) = blue, negative = coral
    colors = ["#2563EB" if v >= 0 else "#F43F5E" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.55)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = range(len(names))
    bars = ax.barh(
        y_pos, values,
        color=colors,
        height=0.55,
        edgecolor="none",
        zorder=3,
    )

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        label_x = bar.get_width()
        ha = "left" if val >= 0 else "right"
        offset = max(abs(max(values, key=abs)) * 0.02, 200)
        ax.text(
            label_x + (offset if val >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{'+'if val>=0 else ''}Rs {val:,.0f}",
            va="center", ha=ha,
            fontsize=8, fontweight="500",
            color="#374151",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.5, color="#374151")
    ax.invert_yaxis()

    ax.set_xlabel("Impact on Predicted Price (LKR)", fontsize=9, color="#64748B", labelpad=8)
    ax.set_title(
        "Feature Contributions to This Prediction",
        fontsize=11, fontweight="bold", color="#1E293B", pad=12,
    )

    # Subtle grid
    ax.grid(axis="x", alpha=0.15, linewidth=0.5, color="#94A3B8", zorder=0)
    ax.axvline(x=0, color="#CBD5E1", linewidth=0.8, zorder=2)

    # Clean up spines
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#E2E8F0")
    ax.spines["left"].set_linewidth(0.5)

    ax.tick_params(axis="x", labelsize=7.5, colors="#94A3B8")
    ax.tick_params(axis="y", length=0)

    # Add base value and prediction annotation
    annotation_text = (
        f"Base price: Rs {base_value:,.0f}  →  "
        f"Predicted: Rs {predicted_value:,.0f}"
    )
    ax.text(
        0.5, -0.12, annotation_text,
        transform=ax.transAxes,
        fontsize=7.5, color="#94A3B8",
        ha="center", style="italic",
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2563EB", label="Increases Price"),
        Patch(facecolor="#F43F5E", label="Decreases Price"),
    ]
    leg = ax.legend(
        handles=legend_elements, loc="lower right",
        fontsize=7.5, frameon=True, framealpha=0.9,
        edgecolor="#E2E8F0", fancybox=True,
    )
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout(pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    artifacts = load_artifacts()
    model, encoders, scaler, feature_meta, column_info, metrics, explainer, brand_model_map, phone_specs = artifacts

    # --- Header ---
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2rem; font-weight: 800; margin-bottom: 0.25rem; color: #1E293B !important;">
            Mobile Phone Price Predictor
        </h1>
        <p style="font-size: 1.05rem; color: #64748B !important; margin: 0;">
            Sri Lanka Market Analysis — Powered by XGBoost Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar inputs ---
    inputs = render_sidebar(column_info, encoders, brand_model_map, phone_specs)

    # --- Predict button ---
    st.sidebar.markdown("---")
    predict_clicked = st.sidebar.button(
        "Predict Price",
        use_container_width=True,
        type="primary",
    )

    # ===================================================================
    # Main content
    # ===================================================================

    if predict_clicked:
        # Prepare features and predict
        feature_array = prepare_features(inputs, encoders, scaler, feature_meta)
        predicted_price = model.predict(feature_array)[0]
        predicted_price = max(predicted_price, 0)

        # --- Prediction Card ---
        st.markdown(f"""
        <div class="prediction-card">
            <div class="pred-label">Estimated Market Price</div>
            <div class="pred-value">Rs {predicted_price:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Input summary ---
        is_basic = inputs.get("auto_specs", False)

        if is_basic:
            # Basic phone — show 2 cards only
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Brand & Model</div>
                    <div class="metric-value" style="font-size: 1.25rem;">{inputs["brand"]} {inputs["phone_model"]}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Condition</div>
                    <div class="metric-value" style="font-size: 1.25rem;">{inputs["condition"]}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Brand</div>
                    <div class="metric-value" style="font-size: 1.25rem;">{inputs["brand"]}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Storage / RAM</div>
                    <div class="metric-value" style="font-size: 1.25rem;">{inputs["storage_gb"]}GB / {inputs["ram_gb"]}GB</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Condition</div>
                    <div class="metric-value" style="font-size: 1.25rem;">{inputs["condition"]}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =============================================================
        # Local Explanation — Beautiful Custom Chart
        # =============================================================
        st.markdown('<div class="section-header">Prediction Explanation (This Phone)</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <strong>What does this chart show?</strong><br>
            Each bar shows how a single feature pushed the predicted price higher (blue) or
            lower (pink) compared to the average price. Longer bars mean a bigger impact
            on this specific prediction.
        </div>
        """, unsafe_allow_html=True)

        if explainer is not None:
            try:
                import shap

                feature_labels = {
                    "brand_clean_encoded": "Phone Brand",
                    "condition_encoded": "Condition",
                    "location_clean_encoded": "Location",
                    "phone_model_clean_encoded": "Phone Model",
                    "storage_gb": "Storage (GB)",
                    "ram_gb": "RAM (GB)",
                    "is_member_flag": "Verified Seller",
                }
                feature_cols = feature_meta["feature_cols"]
                readable = [feature_labels.get(c, c) for c in feature_cols]

                shap_vals = explainer.shap_values(feature_array)
                base_value = explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = float(base_value[0])
                else:
                    base_value = float(base_value)

                # Generate beautiful custom chart
                fig = create_beautiful_shap_chart(
                    shap_vals[0], readable, base_value, float(predicted_price)
                )

                local_path = os.path.join(PLOTS_DIR, "shap_local_prediction.png")
                fig.savefig(local_path, dpi=150, bbox_inches="tight", facecolor="white")
                plt.close(fig)

                st.image(local_path, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not generate local explanation: {e}")
        else:
            st.info("SHAP explainer not found. Run `python src/explain.py` to generate explanations.")



    # --- Footer ---
    st.markdown("---")
    st.markdown("""
    <div class="footer-text">
        Sri Lanka Mobile Phone Price Prediction | XGBoost ML Model | Data sourced from ikman.lk
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
