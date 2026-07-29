"""
MRI QC Dashboard
=================
A Streamlit dashboard that scores batches of MRI-derived QC features,
classifies likely artifacts with an XGBoost model, and produces a
PASS / REVIEW / FAIL scorecard with drill-down and reporting.

Run with:
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="MRI QC Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "SNR",
    "Entropy",
    "LaplacianVariance",
    "GLCMContrast",
    "GLCMEnergy",
    "GLCMHomogeneity",
]

ALL_FEATURE_COLS = [
    "Mean", "Std", "Variance", "SNR", "Entropy", "Sharpness",
    "LaplacianVariance", "GradientMagnitude", "GLCMContrast",
    "GLCMCorrelation", "GLCMEnergy", "GLCMHomogeneity",
]

DEFAULT_CLASS_SCORES = {
    "original": 95,
    "blur": 55,
    "bias": 60,
    "motion": 10,
    "noise": 15,
}

STATUS_COLORS = {"PASS": "#2ecc71", "REVIEW": "#f39c12", "FAIL": "#e74c3c"}

SAMPLE_PATH = "final_artifact_features.csv"


# --------------------------------------------------------------------------
# Data / model helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    return df


def validate_columns(df: pd.DataFrame):
    missing = [c for c in FEATURES if c not in df.columns]
    return missing


@st.cache_resource(show_spinner="Training XGBoost artifact classifier...")
def train_model(train_df_json: str):
    """Train the XGBoost classifier on the reference labeled dataset.
    Cached on the (json-serialized) training data so it only retrains
    when the reference dataset actually changes.
    """
    train_df = pd.read_json(io.StringIO(train_df_json), orient="split")

    X = train_df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(train_df["Artifact"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(le.classes_),
        random_state=42,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # refit on the full reference set for best deployed predictions
    model.fit(X, y)

    return model, le, acc, cm, list(le.classes_)


def predict_batch(model, le, df: pd.DataFrame):
    X = df[FEATURES]
    proba = model.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    pred_class = le.inverse_transform(pred_idx)
    confidence = proba.max(axis=1)
    return pred_class, confidence, proba


def compute_qc_score(proba: np.ndarray, classes, class_scores: dict) -> np.ndarray:
    weights = np.array([class_scores.get(c, 50) for c in classes])
    return proba @ weights


def score_to_status(score: float, pass_th: float, review_th: float) -> str:
    if score >= pass_th:
        return "PASS"
    elif score >= review_th:
        return "REVIEW"
    else:
        return "FAIL"


def normalize_features(df: pd.DataFrame, ref_df: pd.DataFrame, cols):
    mins = ref_df[cols].min()
    maxs = ref_df[cols].max()
    rng = (maxs - mins).replace(0, 1)
    return (df[cols] - mins) / rng


# --------------------------------------------------------------------------
# Sidebar — data loading & configuration
# --------------------------------------------------------------------------
st.sidebar.title("🧠 MRI QC Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["📋 Batch Scorecard", "📊 Analytics", "🔍 Scan Drill-down", "🛣️ Roadmap"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data")
uploaded = st.sidebar.file_uploader("Upload QC features CSV", type=["csv"])

try:
    if uploaded is not None:
        raw_df = load_csv(uploaded)
        data_source_label = uploaded.name
    else:
        raw_df = load_csv(SAMPLE_PATH)
        data_source_label = "sample: final_artifact_features.csv (bundled)"
except FileNotFoundError:
    st.sidebar.error(
        "No sample CSV found and no file uploaded. Please upload a QC features CSV."
    )
    st.stop()

missing_cols = validate_columns(raw_df)
if missing_cols:
    st.error(
        f"The loaded CSV is missing required feature columns: {missing_cols}. "
        f"Required columns: {FEATURES}"
    )
    st.stop()

if "Artifact" not in raw_df.columns:
    st.error(
        "The reference/training CSV must include an 'Artifact' label column "
        "for the XGBoost model to train on."
    )
    st.stop()

# unique scan identifier (MRI_ID repeats across synthetic artifact/level variants)
df = raw_df.copy()
if "MRI_ID" in df.columns:
    level_col = df["Level"] if "Level" in df.columns else ""
    df["Scan_ID"] = (
        df["MRI_ID"].astype(str)
        + " | "
        + df.get("Artifact", "").astype(str)
        + " | "
        + df.get("Level", "").astype(str)
    )
else:
    df["Scan_ID"] = df.index.astype(str)

st.sidebar.caption(f"Loaded: **{data_source_label}** ({len(df)} rows)")

st.sidebar.markdown("---")
st.sidebar.subheader("QC Scoring Rules")
with st.sidebar.expander("Class quality weights (0-100)", expanded=False):
    class_scores = {}
    for cls, default in DEFAULT_CLASS_SCORES.items():
        class_scores[cls] = st.slider(f"{cls}", 0, 100, default, key=f"w_{cls}")

pass_th = st.sidebar.slider("PASS threshold (≥)", 0, 100, 75)
review_th = st.sidebar.slider("REVIEW threshold (≥)", 0, 100, 45)
if review_th >= pass_th:
    st.sidebar.warning("REVIEW threshold should be lower than PASS threshold.")

# --------------------------------------------------------------------------
# Train model + run predictions (cached)
# --------------------------------------------------------------------------
train_json = raw_df.to_json(orient="split")
model, le, holdout_acc, cm, classes = train_model(train_json)

pred_class, confidence, proba = predict_batch(model, le, df)
qc_score = compute_qc_score(proba, classes, class_scores)
qc_status = np.array(
    [score_to_status(s, pass_th, review_th) for s in qc_score]
)

df["Predicted_Artifact"] = pred_class
df["Prediction_Confidence"] = confidence
df["QC_Score"] = qc_score.round(1)
df["QC_Status"] = qc_status


# ==========================================================================
# PAGE 1 — Batch Scorecard
# ==========================================================================
if page == "📋 Batch Scorecard":
    st.title("📋 Batch QC Scorecard")
    st.caption(
        "XGBoost artifact classifier trained on the loaded reference dataset "
        f"— hold-out accuracy **{holdout_acc:.1%}** on classes: {', '.join(classes)}."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Scans", len(df))
    c2.metric("✅ PASS", int((df.QC_Status == "PASS").sum()))
    c3.metric("⚠️ REVIEW", int((df.QC_Status == "REVIEW").sum()))
    c4.metric("❌ FAIL", int((df.QC_Status == "FAIL").sum()))
    c5.metric("Avg QC Score", f"{df.QC_Score.mean():.1f}")

    st.markdown("---")

    def highlight_status(val):
        color = STATUS_COLORS.get(val, "")
        return f"background-color: {color}; color: white; font-weight: 600;"

    display_cols = ["Scan_ID", "Predicted_Artifact", "Prediction_Confidence",
                     "QC_Score", "QC_Status"] + FEATURES
    styled = (
        df[display_cols]
        .style.map(highlight_status, subset=["QC_Status"])
        .format({"Prediction_Confidence": "{:.1%}", "QC_Score": "{:.1f}"})
    )
    st.dataframe(styled, use_container_width=True, height=450)

    st.markdown("---")
    st.subheader("Export QC Report")
    report_cols = ["Scan_ID"] + (["MRI_ID"] if "MRI_ID" in df.columns else []) + \
        ["Artifact", "Level"] if "Artifact" in raw_df.columns else ["Scan_ID"]
    report_df = df[["Scan_ID"] + (["MRI_ID"] if "MRI_ID" in df.columns else []) +
                    (["Artifact"] if "Artifact" in df.columns else []) +
                    (["Level"] if "Level" in df.columns else []) +
                    FEATURES +
                    ["Predicted_Artifact", "Prediction_Confidence", "QC_Score", "QC_Status"]]
    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download QC Report (CSV)",
        data=csv_bytes,
        file_name="mri_qc_report.csv",
        mime="text/csv",
    )

    with st.expander("Model performance details (hold-out test split)"):
        cm_fig = px.imshow(
            cm, x=classes, y=classes, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        cm_fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(cm_fig, use_container_width=True)


# ==========================================================================
# PAGE 2 — Analytics
# ==========================================================================
elif page == "📊 Analytics":
    st.title("📊 Batch Analytics")

    tab1, tab2, tab3 = st.tabs(
        ["Artifact Distribution", "Feature Breakdown", "Radar Comparison"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            dist = df["Predicted_Artifact"].value_counts().reset_index()
            dist.columns = ["Artifact", "Count"]
            fig = px.bar(dist, x="Artifact", y="Count", color="Artifact",
                         title="Predicted Artifact Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            status_dist = df["QC_Status"].value_counts().reset_index()
            status_dist.columns = ["Status", "Count"]
            fig2 = px.pie(
                status_dist, names="Status", values="Count", hole=0.45,
                color="Status", color_discrete_map=STATUS_COLORS,
                title="PASS / REVIEW / FAIL Split",
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        feat_choice = st.selectbox("Feature", FEATURES, index=0)
        fig3 = px.box(
            df, x="Predicted_Artifact", y=feat_choice, color="Predicted_Artifact",
            points="outliers", title=f"{feat_choice} by Predicted Artifact Class",
        )
        st.plotly_chart(fig3, use_container_width=True)

        avg_by_class = df.groupby("Predicted_Artifact")[FEATURES].mean().reset_index()
        fig4 = px.bar(
            avg_by_class.melt(id_vars="Predicted_Artifact", var_name="Feature", value_name="Value"),
            x="Feature", y="Value", color="Predicted_Artifact", barmode="group",
            title="Average Feature Profile by Predicted Class",
        )
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.caption("Feature values min-max normalized (0-1) against the reference dataset.")
        norm_all = normalize_features(df, raw_df, FEATURES)
        norm_all["Predicted_Artifact"] = df["Predicted_Artifact"].values
        radar_avg = norm_all.groupby("Predicted_Artifact")[FEATURES].mean()

        fig5 = go.Figure()
        for cls in radar_avg.index:
            vals = radar_avg.loc[cls, FEATURES].tolist()
            fig5.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=FEATURES + [FEATURES[0]],
                fill="toself", name=cls,
            ))
        fig5.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Normalized Feature Radar — Average per Predicted Class",
            showlegend=True,
        )
        st.plotly_chart(fig5, use_container_width=True)


# ==========================================================================
# PAGE 3 — Scan Drill-down
# ==========================================================================
elif page == "🔍 Scan Drill-down":
    st.title("🔍 Scan Drill-down")

    scan_id = st.selectbox("Select scan", df["Scan_ID"].tolist())
    row = df[df["Scan_ID"] == scan_id].iloc[0]

    status = row["QC_Status"]
    color = STATUS_COLORS[status]
    st.markdown(
        f"""
        <div style="padding:16px;border-radius:10px;background-color:{color};color:white;">
        <h3 style="margin:0;">Status: {status}</h3>
        <p style="margin:0;">Predicted artifact: <b>{row['Predicted_Artifact']}</b>
        &nbsp;|&nbsp; Confidence: <b>{row['Prediction_Confidence']:.1%}</b>
        &nbsp;|&nbsp; QC Score: <b>{row['QC_Score']:.1f}</b> / 100</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("###")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Probabilities")
        prob_row = proba[df["Scan_ID"].tolist().index(scan_id)]
        prob_df = pd.DataFrame({"Class": classes, "Probability": prob_row})
        fig6 = px.bar(prob_df, x="Class", y="Probability", color="Class", range_y=[0, 1])
        st.plotly_chart(fig6, use_container_width=True)

    with col2:
        st.subheader("Feature Profile vs. Dataset Average")
        scan_norm = normalize_features(row.to_frame().T, raw_df, FEATURES).iloc[0]
        dataset_norm_mean = normalize_features(raw_df, raw_df, FEATURES).mean()
        fig7 = go.Figure()
        fig7.add_trace(go.Scatterpolar(
            r=scan_norm.tolist() + [scan_norm.tolist()[0]],
            theta=FEATURES + [FEATURES[0]], fill="toself", name="This scan",
        ))
        fig7.add_trace(go.Scatterpolar(
            r=dataset_norm_mean.tolist() + [dataset_norm_mean.tolist()[0]],
            theta=FEATURES + [FEATURES[0]], fill="toself", name="Dataset average",
            opacity=0.5,
        ))
        fig7.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Raw Feature Values")
    st.dataframe(row[ALL_FEATURE_COLS if all(c in df.columns for c in ALL_FEATURE_COLS) else FEATURES]
                 .to_frame().T, use_container_width=True)


# ==========================================================================
# PAGE 4 — Roadmap
# ==========================================================================
else:
    st.title("🛣️ Roadmap — Future Integrations")
    st.caption("Planned capabilities not yet wired into this dashboard.")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.info("**🧊 NIfTI Viewer**\n\nInteractive slice-by-slice viewer for raw "
                "`.nii` / `.nii.gz` / `.hdr+.img` volumes, linked to each scan's QC row.")
    with r2:
        st.info("**🧠 Skull-strip Overlay**\n\nOverlay brain-extraction masks on the "
                "viewer to visually confirm skull-strip quality alongside QC scores.")
    with r3:
        st.info("**🌐 ABIDE Site Analysis**\n\nAggregate QC metrics by acquisition "
                "site/scanner to surface site-level bias in the ABIDE dataset.")

    st.markdown("---")
    st.write(
        "These are placeholders for future development — hook in a NIfTI loader "
        "(e.g. `nibabel`) and a slice renderer (e.g. `plotly`/`matplotlib`) to bring "
        "the NIfTI viewer online, reusing the `extract_slice_features` pipeline "
        "from the model notebook for on-the-fly feature extraction from uploaded volumes."
    )
