import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)
import joblib
import os
import time

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
#  MODEL REGISTRY
# ══════════════════════════════════════════════════════

UNSUPERVISED_MODELS = {
    "Isolation Forest"    : "Detects anomalies by isolating data points. Best for high-dimensional sensor data.",
    "Local Outlier Factor": "Detects anomalies based on local density. Great for clustered data.",
    "One-Class SVM"       : "Learns a boundary around normal data. Good for small/clean datasets.",
}

SUPERVISED_MODELS = {
    "Random Forest"      : "Ensemble of decision trees. Most accurate when labels are available.",
    "Decision Tree"      : "Simple tree model. Easy to interpret and explain.",
    "Gradient Boosting"  : "Boosted trees — high accuracy, slower to train.",
    "K-Nearest Neighbors": "Classifies based on similarity to nearest neighbors.",
    "Logistic Regression": "Fast linear baseline model.",
    "Naive Bayes"        : "Probabilistic model. Extremely fast on large datasets.",
}

ALL_MODELS = {**UNSUPERVISED_MODELS, **SUPERVISED_MODELS}


def run_model(name, X_train, X_test, y_train=None, contamination=0.1):
    """
    Train and predict using specified model.
    Returns: predictions (0/1), anomaly_scores, train_time_seconds
    """
    X_tr = X_train.values if hasattr(X_train, "values") else X_train
    X_te = X_test.values  if hasattr(X_test,  "values") else X_test

    start = time.time()

    if name == "Isolation Forest":
        m = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
        m.fit(X_tr)
        preds  = np.where(m.predict(X_te) == -1, 1, 0)
        scores = -m.decision_function(X_te)

    elif name == "Local Outlier Factor":
        m = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True, n_jobs=-1)
        m.fit(X_tr)
        preds  = np.where(m.predict(X_te) == -1, 1, 0)
        scores = -m.decision_function(X_te)

    elif name == "One-Class SVM":
        max_s = min(5000, len(X_tr))
        idx   = np.random.choice(len(X_tr), max_s, replace=False)
        m = OneClassSVM(kernel="rbf", nu=contamination, gamma="scale")
        m.fit(X_tr[idx])
        preds  = np.where(m.predict(X_te) == -1, 1, 0)
        scores = -m.decision_function(X_te)

    elif name == "Random Forest":
        m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_train)
        preds  = m.predict(X_te)
        scores = m.predict_proba(X_te)[:, 1]

    elif name == "Decision Tree":
        m = DecisionTreeClassifier(random_state=42, max_depth=10)
        m.fit(X_tr, y_train)
        preds  = m.predict(X_te)
        scores = m.predict_proba(X_te)[:, 1]

    elif name == "Gradient Boosting":
        m = GradientBoostingClassifier(n_estimators=100, random_state=42)
        m.fit(X_tr, y_train)
        preds  = m.predict(X_te)
        scores = m.predict_proba(X_te)[:, 1]

    elif name == "K-Nearest Neighbors":
        m = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        m.fit(X_tr, y_train)
        preds  = m.predict(X_te)
        scores = m.predict_proba(X_te)[:, 1]

    elif name == "Logistic Regression":
        m = LogisticRegression(random_state=42, max_iter=500, n_jobs=-1)
        m.fit(X_tr, y_train)
        preds  = m.predict(X_te)
        scores = m.predict_proba(X_te)[:, 1]

    elif name == "Naive Bayes":
        m = GaussianNB()
        m.fit(X_tr, y_train)
        preds  = m.predict(X_te)
        scores = m.predict_proba(X_te)[:, 1]

    else:
        raise ValueError(f"Unknown model: {name}")

    train_time = round(time.time() - start, 2)
    safe = name.replace(" ", "_").lower()
    joblib.dump(m, f"{MODEL_DIR}/{safe}.pkl")

    return preds.astype(int), scores, train_time


def evaluate(y_true, y_pred, model_name="", train_time=0):
    """Return dict of evaluation metrics."""
    return {
        "model"            : model_name,
        "accuracy"         : round(accuracy_score(y_true, y_pred), 4),
        "precision"        : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"           : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score"         : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix" : confusion_matrix(y_true, y_pred).tolist(),
        "report"           : classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]),
        "train_time"       : train_time,
    }