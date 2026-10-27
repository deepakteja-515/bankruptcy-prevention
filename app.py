"""
Bankruptcy Prevention - Streamlit App

Run:
    pip install -r requirements.txt
    streamlit run app.py

If model.pkl exists it is loaded directly. Otherwise the app trains from
Bankruptcy_prevention_Dataset.csv and saves model.pkl (or run: python train_model.py).
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from train_model import (
    CSV_NAME,
    FEATURES,
    MODEL_PATH,
    predict_one,
    train_and_save,
)

st.set_page_config(page_title="Bankruptcy Prevention", page_icon="🏦", layout="wide")

RISK_LEVELS = {0.0: "0.0 - Low", 0.5: "0.5 - Medium", 1.0: "1.0 - High"}


@st.cache_resource(show_spinner="Loading model.pkl...")
def load_artifacts():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


st.sidebar.title("🏦 Bankruptcy Prevention")

# Optional: new CSV tho retrain
uploaded = st.sidebar.file_uploader("Upload CSV to retrain (sep=';')", type="csv")
if uploaded is not None and st.sidebar.button("Train & save model.pkl"):
    with st.spinner("Training models..."):
        train_and_save(pd.read_csv(uploaded, sep=";"))
    load_artifacts.clear()
    st.rerun()

artifacts = load_artifacts()

# If model.pkl is missing but the CSV is in the folder, train and save automatically
csv_path = Path(__file__).parent / CSV_NAME
if artifacts is None and csv_path.exists():
    with st.spinner("model.pkl not found - training from CSV and saving it..."):
        train_and_save(pd.read_csv(csv_path, sep=";"))
    load_artifacts.clear()
    artifacts = load_artifacts()

if artifacts is None:
    st.title("🏦 Bankruptcy Prevention")
    st.warning(
        "Neither `model.pkl` nor `Bankruptcy_prevention_Dataset.csv` was found. "
        "Place the CSV in the same folder as app.py, or upload it in the sidebar and train."
    )
    st.stop()

data = artifacts["data"]
results_df = artifacts["results"]
best_name = results_df.loc[0, "Model"]

page = st.sidebar.radio(
    "Page",
    ["🔮 Predict", "📊 Data Explorer", "🏆 Model Comparison", "🧪 Best Model Details"],
)
st.sidebar.caption(
    f"Rows (after removing duplicates): {len(data)}  \n"
    f"Train: {artifacts['train_size']} | Test: {artifacts['test_size']}  \n"
    f"Best model: **{best_name}**"
)


# --------------------------------------------------------------------------
# Page 1: Predict
# --------------------------------------------------------------------------
if page == "🔮 Predict":
    st.title("🔮 Bankruptcy Prediction")
    st.write("Select the company risk factors below to predict bankruptcy risk.")

    model_name = st.selectbox(
        "Model",
        results_df["Model"].tolist(),
        index=0,
        help="By default the model with the highest F1 score is selected.",
    )

    cols = st.columns(3)
    values = {}
    descriptions = {
        "industrial_risk": "Industrial Risk",
        "management_risk": "Management Risk",
        "financial_flexibility": "Financial Flexibility",
        "credibility": "Credibility",
        "competitiveness": "Competitiveness",
        "operating_risk": "Operating Risk",
    }
    for i, feat in enumerate(FEATURES):
        with cols[i % 3]:
            values[feat] = st.select_slider(
                descriptions[feat],
                options=[0.0, 0.5, 1.0],
                value=0.5,
                format_func=lambda v: RISK_LEVELS[v],
            )

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([values])[FEATURES]
        label, probs = predict_one(artifacts, model_name, row)

        if label == "bankruptcy":
            st.error(f"⚠️ Prediction: **{label.upper()}**")
        else:
            st.success(f"✅ Prediction: **{label.upper()}**")

        c1, c2 = st.columns(2)
        for col, (cls, p) in zip((c1, c2), sorted(probs.items())):
            col.metric(f"P({cls})", f"{p:.1%}")
        st.progress(float(probs.get("bankruptcy", 0.0)), text="Bankruptcy probability")

        with st.expander("Input record"):
            st.dataframe(row, hide_index=True)

    st.divider()
    st.subheader("Sample from the test set")
    idx = st.number_input(
        "Test record index (starts at 0)", min_value=0, max_value=artifacts["test_size"] - 1, value=0
    )
    sample = artifacts["X_test"].iloc[[int(idx)]]
    actual = artifacts["y_test"].iloc[int(idx)]
    pred_label, _ = predict_one(artifacts, best_name, sample)
    st.dataframe(sample, hide_index=True)
    st.write(f"**Predicted ({best_name}):** {pred_label}  |  **Actual:** {actual}")


# --------------------------------------------------------------------------
# Page 2: Data Explorer (notebook EDA)
# --------------------------------------------------------------------------
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Class distribution", "Histograms", "Correlation & Outliers"]
    )

    with tab1:
        st.write(f"Raw shape: {artifacts["raw_shape"]}  →  cleaned (duplicates removed): {data.shape}")
        st.dataframe(data.head(20))
        st.write("**Describe**")
        st.dataframe(data.describe())
        st.write("**Null values**")
        st.dataframe(data.isnull().sum().rename("nulls"))

    with tab2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.countplot(data=data, x="class", ax=ax)
        ax.set_title("Bankruptcy Class Distribution")
        st.pyplot(fig)
        st.dataframe(data["class"].value_counts().rename("count"))

    with tab3:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        data[FEATURES].hist(ax=axes.ravel())
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(data.select_dtypes("number").corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.boxplot(data=data.select_dtypes("number"), ax=ax)
        ax.set_title("Box Plot - Outlier Detection")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)


# --------------------------------------------------------------------------
# Page 3: Model comparison
# --------------------------------------------------------------------------
elif page == "🏆 Model Comparison":
    st.title("🏆 Model Comparison")
    st.dataframe(
        results_df.style.format(
            {c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1 Score"]}
        ).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score"], color="#c6efce"),
        
        hide_index=True,
    )

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    plot_df = results_df.melt(
        id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison - All 6 Models")
    plt.xticks(rotation=20)
    plt.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(f"GridSearch best params: {artifacts['best_params']}")
    st.info(
        f"The test set has only {artifacts['test_size']} records, so metrics can be very high "
        "(close to 1.0) - don't rely on them too much."
    )


# --------------------------------------------------------------------------
# Page 4: Best model details
# --------------------------------------------------------------------------
else:
    st.title("🧪 Detailed Evaluation")
    name = st.selectbox("Model", results_df["Model"].tolist(), index=0)
    y_test = artifacts["y_test"]
    y_pred = artifacts["preds"][name]

    st.subheader("Classification report")
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3))

    st.subheader("Confusion matrix")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    st.pyplot(fig)
