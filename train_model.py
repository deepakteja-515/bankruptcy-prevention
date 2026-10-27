"""
Train all 6 models once and save them to model.pkl

Run:
    python train_model.py                      # uses Bankruptcy_prevention_Dataset.csv from the same folder
    python train_model.py path/to/dataset.csv  # or pass a path to the CSV
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

CSV_NAME = "Bankruptcy_prevention_Dataset.csv"
FEATURES = [
    "industrial_risk",
    "management_risk",
    "financial_flexibility",
    "credibility",
    "competitiveness",
    "operating_risk",
]
RISK_LEVELS = {0.0: "0.0 - Low", 0.5: "0.5 - Medium", 1.0: "1.0 - High"}


MODEL_PATH = Path(__file__).parent / "model.pkl"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def train_all(df: pd.DataFrame):
    X = df.drop("class", axis=1)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    def pipe(model):
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    trained = {}

    # 1-3: base models
    base = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    for name, m in base.items():
        p = pipe(m)
        p.fit(X_train, y_train)
        trained[name] = p

    # 4: XGBoost (needs encoded labels)
    xgb = pipe(
        XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
    )
    xgb.fit(X_train, y_train_enc)
    trained["XGBoost"] = xgb

    # 5: SVM
    svm = pipe(SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
    svm.fit(X_train, y_train)
    trained["SVM"] = svm

    # 6: GridSearch Random Forest
    grid = GridSearchCV(
        pipe(RandomForestClassifier(random_state=42)),
        param_grid={
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    trained["GridSearch Random Forest"] = grid.best_estimator_

    # Evaluation (notebook step 5)
    rows, preds = [], {}
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        if name == "XGBoost":
            y_pred = le.inverse_transform(np.asarray(y_pred).astype(int))
        preds[name] = y_pred
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
        )
    results_df = (
        pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    )

    return {
        "models": trained,
        "le": le,
        "results": results_df,
        "preds": preds,
        "X_test": X_test,
        "y_test": y_test,
        "best_params": grid.best_params_,
        "data": df,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def predict_one(art: dict, name: str, row: pd.DataFrame):
    """Return (label, {class: probability}) for one record."""
    model = art["models"][name]
    le = art["le"]
    raw = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    if name == "XGBoost":
        label = le.inverse_transform([int(raw)])[0]
        classes = le.classes_
    else:
        label = raw
        classes = model.classes_
    return label, dict(zip(classes, proba))


def train_and_save(raw: pd.DataFrame, path: Path = MODEL_PATH) -> dict:
    artifacts = train_all(clean_data(raw))
    artifacts["raw_shape"] = raw.shape
    joblib.dump(artifacts, path)
    return artifacts


if __name__ == "__main__":
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / CSV_NAME
    if not csv.exists():
        sys.exit(f"CSV not found: {csv}")
    art = train_and_save(pd.read_csv(csv, sep=";"))
    print(art["results"].round(4).to_string(index=False))
    print(f"\nSaved -> {MODEL_PATH}")
