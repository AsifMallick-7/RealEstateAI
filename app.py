import json
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

from src.config import DISPLAY_NAMES, METRICS_DIR, MODELS_DIR
from src.data_loader import load_california_housing
from src.validation import validate_dataframe

st.set_page_config(
    page_title="RealEstateAI | Property Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def get_model():
    path = MODELS_DIR / "best_model.joblib"
    return load(path) if path.exists() else None

@st.cache_data
def get_metadata():
    path = MODELS_DIR / "model_metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def get_metrics():
    path = METRICS_DIR / "model_metrics.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_data
def get_feature_importance():
    path = METRICS_DIR / "feature_importance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

model = get_model()
metadata = get_metadata()
metrics = get_metrics()
importance = get_feature_importance()

# ----------------------------- UI styling -----------------------------
st.markdown(
    """
    <style>
    .block-container {max-width: 1250px; padding-top: 1.6rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.45rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(38,42,55,.98), rgba(25,28,38,.98));
        border: 1px solid rgba(255,255,255,.10);
        margin-bottom: 1.15rem;
    }
    .hero h1 {margin: 0; font-size: 2.25rem;}
    .hero p {margin: .45rem 0 0; color: #c9ced9; font-size: 1rem;}
    .eyebrow {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: #8fa1bd;}
    .helper {
        padding: .9rem 1rem;
        border-radius: 12px;
        background: rgba(49,51,63,.42);
        border: 1px solid rgba(255,255,255,.07);
        color: #c7ccd7;
    }
    .step {
        min-height: 145px;
        padding: 1rem;
        border-radius: 15px;
        background: rgba(38,41,52,.62);
        border: 1px solid rgba(255,255,255,.08);
    }
    .step-no {
        display: inline-flex;
        width: 30px; height: 30px;
        align-items: center; justify-content: center;
        border-radius: 50%;
        background: rgba(70,130,180,.25);
        font-weight: 700;
        margin-bottom: .55rem;
    }
    .result {
        padding: 1.35rem 1.45rem;
        border-radius: 18px;
        background: rgba(35,38,49,.82);
        border: 1px solid rgba(255,255,255,.11);
    }
    .result-label {font-size: .76rem; text-transform: uppercase; letter-spacing: .08em; color: #9ba4b6;}
    .result-value {font-size: 2.45rem; font-weight: 750; margin: .2rem 0;}
    .muted {color: #9ca5b6; font-size: .88rem;}
    .mini-card {
        padding: 1rem;
        border-radius: 14px;
        background: rgba(38,41,52,.55);
        border: 1px solid rgba(255,255,255,.07);
    }
    div[data-testid="stMetric"] {
        background: rgba(38,41,52,.40);
        padding: .75rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Sidebar -----------------------------
st.sidebar.markdown("## 🏠 RealEstateAI")
st.sidebar.caption("Machine Learning Property Valuation")
st.sidebar.divider()

page = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "💰 Predict Price", "🤖 Model Performance",
     "📊 Data Analysis", "🔍 Explainability", "ℹ️ About"],
)

st.sidebar.divider()
st.sidebar.markdown("### Quick guide")
st.sidebar.caption("Start with **Predict Price** if you only want an estimate.")
st.sidebar.caption("Use **Model Performance** and **Explainability** to understand how the ML system behaves.")
st.sidebar.divider()
st.sidebar.caption("Educational / demonstration system")

if model is None:
    st.error("No trained model found. Run `python train.py` before using predictions.")
    st.stop()

best_row = metrics.sort_values("RMSE").iloc[0] if not metrics.empty else None
best_name = metadata.get("best_model", "Random Forest")

# ----------------------------- Dashboard -----------------------------
if page == "🏠 Dashboard":
    st.markdown(
        '<div class="hero"><div class="eyebrow">Machine Learning Application</div>'
        '<h1>🏠 RealEstateAI</h1>'
        '<p>Estimate median house value and explore the model behind the prediction.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="helper"><b>New here?</b> Go to <b>Predict Price</b>, enter the eight property/locality inputs, '
        'and select <b>Generate Prediction</b>. The other pages explain the data and model results.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset rows", f"{metadata.get('dataset_rows', 0):,}")
    c2.metric("Input features", metadata.get("feature_count", 8))
    with c3:
        st.markdown(f'<div class="mini-card"><div class="muted">Selected model</div><div style="font-size:1.55rem;font-weight:700;margin-top:.25rem;white-space:nowrap;">{best_name}</div></div>', unsafe_allow_html=True)
    c4.metric("Test R²", f"{float(best_row['R2']):.3f}" if best_row is not None else "—")

    st.markdown("### How the application works")
    steps = [
        ("1", "Enter inputs", "Provide income, age, rooms, bedrooms, population, occupancy and location."),
        ("2", "Generate estimate", "The trained regression pipeline converts the inputs into a model prediction."),
        ("3", "Review result", "See the estimated median house value and model error context."),
        ("4", "Understand the model", "Use performance and explainability pages to interpret the result."),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="step"><div class="step-no">{num}</div><br><b>{title}</b><p class="muted">{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("### ML pipeline")
    st.code(
        "Data → Validation → Preprocessing → Training → Cross-Validation → "
        "Hyperparameter Search → Evaluation → Explainability → Prediction",
        language="text",
    )

    st.markdown("### Important context")
    a, b = st.columns(2)
    with a:
        st.info("**Target:** `MedHouseVal`. The California Housing benchmark expresses the target in units of $100,000.")
    with b:
        if best_row is not None:
            st.info(f"**Current holdout metrics:** MAE {best_row['MAE']:.4f} · RMSE {best_row['RMSE']:.4f} · R² {best_row['R2']:.4f}")

# ----------------------------- Prediction -----------------------------
elif page == "💰 Predict Price":
    st.title("💰 Predict Property Value")
    st.caption("Enter values in the same units/ranges used by the training dataset. Hover over ⓘ for guidance.")

    with st.expander("ℹ️ What are these inputs?", expanded=False):
        st.markdown(
            """
            - **Median Income:** median income represented in tens of thousands of USD.
            - **House Age:** median house age in years for the locality.
            - **Average Rooms / Bedrooms:** average rooms and bedrooms per household.
            - **Population:** locality/block-group population.
            - **Average Occupancy:** average number of people per household.
            - **Latitude / Longitude:** geographic location used by the benchmark dataset.
            """
        )

    with st.form("prediction_form"):
        st.markdown("#### 1. Property characteristics")
        c1, c2 = st.columns(2)
        with c1:
            med_inc = st.number_input("Median Income (10k USD)", 0.0, 15.0, 3.5, 0.1, format="%.2f", help="Median income in units of $10,000.")
            house_age = st.number_input("House Age (years)", 1.0, 52.0, 25.0, 1.0, format="%.0f", help="Median house age for the locality.")
            ave_rooms = st.number_input("Average Rooms", 0.1, 50.0, 5.5, 0.1, format="%.2f", help="Average number of rooms per household.")
            ave_bedrms = st.number_input("Average Bedrooms", 0.1, 20.0, 1.2, 0.1, format="%.2f", help="Average number of bedrooms per household.")
        with c2:
            population = st.number_input("Population", 1.0, 100000.0, 1500.0, 100.0, format="%.0f", help="Population of the locality represented by the dataset record.")
            ave_occup = st.number_input("Average Occupancy", 0.1, 20.0, 3.0, 0.1, format="%.2f", help="Average household occupancy.")
            latitude = st.number_input("Latitude", 32.0, 42.0, 35.5, 0.01, format="%.2f", help="Latitude used in the California Housing dataset.")
            longitude = st.number_input("Longitude", -125.0, -114.0, -119.5, 0.01, format="%.2f", help="Longitude used in the California Housing dataset.")

        st.markdown("#### 2. Generate estimate")
        submitted = st.form_submit_button("🔮 Generate Prediction", use_container_width=True, type="primary")

    if submitted:
        values = {
            "MedInc": med_inc, "HouseAge": house_age, "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms, "Population": population,
            "AveOccup": ave_occup, "Latitude": latitude, "Longitude": longitude,
        }
        input_df = pd.DataFrame([values])
        prediction = float(model.predict(input_df)[0])
        dollar_value = prediction * 100_000
        rmse_value = float(best_row["RMSE"]) * 100_000 if best_row is not None else None

        st.success("Prediction generated successfully.")
        st.markdown("### 3. Your result")
        left, right = st.columns([1.5, 1])
        with left:
            st.markdown(
                f'<div class="result"><div class="result-label">Estimated median house value</div>'
                f'<div class="result-value">${dollar_value:,.0f}</div>'
                f'<div class="muted">Model output: {prediction:.3f} × $100,000</div></div>',
                unsafe_allow_html=True,
            )
        with right:
            st.metric("Model used", best_name)
            if rmse_value is not None:
                st.metric("Test RMSE scale", f"≈ ${rmse_value:,.0f}")

        st.markdown("### Input summary")
        summary = pd.DataFrame({
            "Feature": [DISPLAY_NAMES.get(k, k) for k in values],
            "Input value": list(values.values()),
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.warning(
            "This estimate is based on the California Housing benchmark dataset and represents a median house-value estimate, "
            "not the market price of a specific property. It is not a professional appraisal or live market quote."
        )

# ----------------------------- Performance -----------------------------
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.caption("Compare models using cross-validation and the holdout test set.")

    if metrics.empty:
        st.warning("Run `python train.py` to generate evaluation results.")
        st.stop()

    ordered = metrics.sort_values("RMSE").reset_index(drop=True)
    display = ordered[["Model", "CV_MAE", "CV_RMSE", "CV_R2", "MAE", "RMSE", "R2"]].rename(
        columns={"CV_MAE":"CV MAE", "CV_RMSE":"CV RMSE", "CV_R2":"CV R²", "MAE":"Test MAE", "RMSE":"Test RMSE", "R2":"Test R²"}
    )
    st.dataframe(display.round(4), hide_index=True, use_container_width=True)

    best = ordered.iloc[0]
    st.success(f"Current evaluation selection: **{best['Model']}** · Test RMSE {best['RMSE']:.4f} · Test R² {best['R2']:.4f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", f"{best['MAE']:.4f}")
    c2.metric("Test RMSE", f"{best['RMSE']:.4f}")
    c3.metric("Test R²", f"{best['R2']:.4f}")

    with st.expander("📘 Understand these metrics"):
        st.markdown(
            """
            **MAE** — average absolute prediction error in target units.  
            **RMSE** — similar to MAE, but gives larger errors more weight.  
            **R²** — indicates how much variation in the target is explained relative to a baseline.  
            **CV metrics** — average results across the configured cross-validation folds.
            """
        )

    a, b = st.columns(2)
    with a:
        st.subheader("RMSE comparison")
        rmse_chart = ordered.set_index("Model")["RMSE"].sort_values(ascending=True)
        st.bar_chart(rmse_chart, horizontal=True)
        st.caption("Lower RMSE means smaller average squared-error magnitude on this evaluation.")
    with b:
        st.subheader("R² comparison")
        r2_chart = ordered.set_index("Model")["R2"].sort_values(ascending=True)
        st.bar_chart(r2_chart, horizontal=True)
        st.caption("Higher R² indicates more target variation explained on this evaluation.")

    if metadata.get("best_params"):
        with st.expander("View tuned model parameters"):
            st.json(metadata["best_params"])

# ----------------------------- Data Analysis -----------------------------
elif page == "📊 Data Analysis":
    st.title("📊 Explore the Dataset")
    st.caption("Use this page to understand data quality, distributions and relationships before interpreting predictions.")

    try:
        df = load_california_housing(save_csv=False)
        validation = validate_dataframe(df)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{validation['rows']:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing values", validation["missing_values"])
    c4.metric("Duplicate rows", validation["duplicate_rows"])

    with st.expander("📋 Dataset preview", expanded=True):
        st.dataframe(df.head(20), hide_index=True, use_container_width=True)

    with st.expander("📐 Summary statistics"):
        st.dataframe(df.describe().T.round(4), use_container_width=True)

    st.subheader("Target distribution")
    target_counts = pd.cut(df["MedHouseVal"], bins=20).value_counts().sort_index()
    target_counts.index = target_counts.index.astype(str)
    st.bar_chart(target_counts)
    st.caption("This chart shows how frequently target values fall into each range.")

    st.subheader("Feature relationship with target")
    corr = df.corr(numeric_only=True)["MedHouseVal"].drop("MedHouseVal").sort_values()
    st.bar_chart(corr)
    st.caption("Correlation describes linear association in this dataset; it does not establish causation.")

# ----------------------------- Explainability -----------------------------
elif page == "🔍 Explainability":
    st.title("🔍 Understand the Model")
    st.caption("See which input features are most important to the fitted model under permutation testing.")

    if importance.empty:
        st.warning("Run `python train.py` to generate feature importance.")
        st.stop()

    imp = importance.copy()
    imp["DisplayFeature"] = imp["Feature"].map(lambda x: DISPLAY_NAMES.get(x, x))
    imp = imp.sort_values("Importance", ascending=False)

    st.info(
        "Permutation importance measures how model performance changes when a feature is shuffled. "
        "It is a model-behaviour diagnostic, not proof that a feature causes the target to change."
    )

    top = imp.iloc[0]
    c1, c2 = st.columns(2)
    c1.metric("Most important measured feature", top["DisplayFeature"])
    c2.metric("Importance", f"{top['Importance']:.4f}")

    st.subheader("Feature importance")
    importance_chart = imp.set_index("DisplayFeature")["Importance"].sort_values(ascending=True)
    st.bar_chart(importance_chart, horizontal=True)
    st.caption("Longer bars indicate a larger measured impact on model performance when that feature is shuffled.")

    with st.expander("View detailed values"):
        st.dataframe(
            imp[["DisplayFeature", "Importance", "Std"]].rename(columns={"DisplayFeature":"Feature"}).round(4),
            hide_index=True,
            use_container_width=True,
        )

# ----------------------------- About -----------------------------
else:
    st.title("ℹ️ About RealEstateAI")
    st.markdown(
        """
        ### What is this project?
        RealEstateAI is an end-to-end supervised machine-learning demonstration for a continuous-value prediction task.

        ### Technology stack
        **Python · Pandas · NumPy · Scikit-learn · Joblib · Streamlit**

        ### Workflow
        1. Dataset acquisition and validation
        2. Preprocessing and feature preparation
        3. Baseline model training
        4. Cross-validation
        5. Hyperparameter search
        6. Holdout evaluation
        7. Permutation-based explainability
        8. Interactive prediction

        ### Dataset scope
        The current implementation uses the **California Housing benchmark dataset**. It contains aggregated block-group-level information rather than individual property listings.

        ### Important limitation
        This application is an educational/technical demonstration. It does not provide live market prices or professional property appraisals and should not be treated as the sole basis for a real-estate decision.

        ### Future scope
        A production-oriented system could use representative regional transaction data, richer geospatial features, monitoring, automated retraining, API serving and a database.
        """
    )
