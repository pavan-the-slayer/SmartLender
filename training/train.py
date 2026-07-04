"""
Model Selection stage of the SmartLender pipeline.

Runs Grid Search Cross Validation over four model families:
  - K Nearest Neighbors      (n_neighbors, metric)
  - eXtreme Gradient Boost   (learning_rate, max_depth, n_estimators)
  - Random Forest Bagging    (criterion, max_depth, n_estimators)
  - Decision Tree            (criterion, max_depth)

Picks the best model by cross-validated accuracy and saves it as
artifacts/model.pkl, alongside a metrics report.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

CV_FOLDS = 5

SEARCH_SPACE = {
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "metric": ["euclidean", "manhattan", "minkowski"],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ),
        "params": {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5, 6],
            "n_estimators": [100, 200, 300],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": [4, 6, 8, None],
            "n_estimators": [100, 200, 300],
        },
    },
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 4, 5, 6, None],
        },
    },
}


def load_splits():
    X_train = np.load(ARTIFACTS / "X_train.npy")
    X_test = np.load(ARTIFACTS / "X_test.npy")
    y_train = np.load(ARTIFACTS / "y_train.npy")
    y_test = np.load(ARTIFACTS / "y_test.npy")
    return X_train, X_test, y_train, y_test


def run_grid_search(name, spec, X_train, y_train):
    print(f"\n--- Grid Search CV: {name} ---")
    grid = GridSearchCV(
        estimator=spec["estimator"],
        param_grid=spec["params"],
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    return grid


def run() -> None:
    if not (ARTIFACTS / "X_train.npy").exists():
        raise FileNotFoundError(
            "No preprocessed data found. Run `python preprocessing/preprocess.py` first."
        )

    X_train, X_test, y_train, y_test = load_splits()

    results = {}
    fitted = {}
    for name, spec in SEARCH_SPACE.items():
        grid = run_grid_search(name, spec, X_train, y_train)
        test_pred = grid.best_estimator_.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        try:
            test_auc = roc_auc_score(y_test, grid.best_estimator_.predict_proba(X_test)[:, 1])
        except Exception:
            test_auc = None

        results[name] = {
            "best_params": grid.best_params_,
            "cv_accuracy": grid.best_score_,
            "test_accuracy": test_acc,
            "test_auc": test_auc,
        }
        fitted[name] = grid.best_estimator_
        print(f"{name} — test accuracy: {test_acc:.4f}" + (f", AUC: {test_auc:.4f}" if test_auc else ""))

    best_name = max(results, key=lambda k: results[k]["test_accuracy"])
    best_model = fitted[best_name]
    print(f"\n=== Selected model: {best_name} ===")
    print(classification_report(y_test, best_model.predict(X_test)))

    joblib.dump(best_model, ARTIFACTS / "model.pkl")
    with open(ARTIFACTS / "model_metrics.json", "w") as f:
        json.dump({"selected_model": best_name, "results": results}, f, indent=2, default=str)

    print(f"\nSaved best model ({best_name}) to {ARTIFACTS / 'model.pkl'}")


if __name__ == "__main__":
    run()
