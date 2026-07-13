"""
Autonomous Data Scientist
-------------------------
A self-contained Flask backend that cleans data, runs EDA, selects features,
trains/compares models, explains results in plain language, and feeds a
JS dashboard -- all with local Python libraries. No external API keys used.
"""
import io
import json
import traceback

import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, render_template, send_file

from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error, roc_curve, roc_auc_score,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads

# ---------------------------------------------------------------------------
# In-memory pipeline state (single-user local app -- no DB / API keys needed)
# ---------------------------------------------------------------------------
STATE = {
    "df_raw": None,
    "df_clean": None,
    "cleaning_report": None,
    "eda": None,
    "target": None,
    "task_type": None,
    "feature_ranking": None,
    "selected_features": None,
    "leaderboard": None,
    "best_model_name": None,
    "best_pipeline": None,
    "test_data": None,
    "metrics": None,
    "explain": None,
    "label_encoder": None,
}

MAX_HISTOGRAM_COLS = 8
MAX_CATEGORICAL_COLS = 6
MAX_UNIQUE_FOR_CATEGORY = 30


def _reset_state():
    for k in STATE:
        STATE[k] = None


def clean_json(obj):
    """Recursively convert numpy/pandas types into plain JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


def error_response(e, code=400):
    return jsonify({"ok": False, "error": str(e)}), code


def guess_task_type(series: pd.Series) -> str:
    """Heuristic: classification vs regression based on target dtype/cardinality.
    Uses is_numeric_dtype rather than checking for dtype == object, since pandas 3.x
    stores strings under a dedicated 'str' dtype (not 'object') by default."""
    n_unique = series.nunique(dropna=True)
    if pd.api.types.is_bool_dtype(series):
        return "classification"
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"  # text / categorical dtype of any kind
    if pd.api.types.is_integer_dtype(series) and n_unique <= max(20, int(len(series) * 0.05)):
        return "classification"
    if n_unique <= 15 and n_unique / max(len(series), 1) < 0.05:
        return "classification"
    return "regression"


def build_preprocessor(df: pd.DataFrame, feature_cols):
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=25)),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    pre = ColumnTransformer(transformers)
    return pre, numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# Routes: pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes: API
# ---------------------------------------------------------------------------
@app.route("/api/reset", methods=["POST"])
def api_reset():
    _reset_state()
    return jsonify({"ok": True})


@app.route("/api/sample", methods=["POST"])
def api_sample():
    """Load a bundled sample dataset -- no upload / internet required."""
    name = (request.json or {}).get("name", "iris")
    try:
        if name == "iris":
            data = load_iris(as_frame=True)
            df = data.frame
            df["species"] = df["target"].map(dict(enumerate(data.target_names)))
            df = df.drop(columns=["target"])
            suggested_target = "species"
        elif name == "wine":
            data = load_wine(as_frame=True)
            df = data.frame
            df["wine_class"] = df["target"].map(dict(enumerate(data.target_names)))
            df = df.drop(columns=["target"])
            suggested_target = "wine_class"
        elif name == "diabetes":
            data = load_diabetes(as_frame=True)
            df = data.frame
            df = df.rename(columns={"target": "disease_progression"})
            suggested_target = "disease_progression"
        else:
            return error_response(f"Unknown sample dataset '{name}'")

        # sprinkle a few missing values / a duplicate row so cleaning has real work to do
        rng = np.random.RandomState(42)
        df_dirty = df.copy()
        for col in df_dirty.columns[:3]:
            idx = rng.choice(df_dirty.index, size=max(1, len(df_dirty) // 40), replace=False)
            df_dirty.loc[idx, col] = np.nan
        df_dirty = pd.concat([df_dirty, df_dirty.iloc[[0]]], ignore_index=True)

        _reset_state()
        STATE["df_raw"] = df_dirty
        preview = _dataset_summary(df_dirty)
        preview["suggested_target"] = suggested_target
        return jsonify({"ok": True, **preview})
    except Exception as e:
        traceback.print_exc()
        return error_response(e, 500)


def _dataset_summary(df: pd.DataFrame):
    return clean_json({
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing": {c: int(df[c].isna().sum()) for c in df.columns},
        "preview": df.head(8).to_dict(orient="records"),
        "duplicate_rows": int(df.duplicated().sum()),
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return error_response("No file uploaded under field 'file'.")
    f = request.files["file"]
    try:
        raw = f.read()
        df = pd.read_csv(io.BytesIO(raw))
        if df.shape[1] == 0:
            return error_response("Uploaded CSV appears to have no columns.")
        _reset_state()
        STATE["df_raw"] = df
        return jsonify({"ok": True, **_dataset_summary(df)})
    except Exception as e:
        traceback.print_exc()
        return error_response(f"Could not parse CSV: {e}")


@app.route("/api/full_data", methods=["GET"])
def api_full_data():
    stage = request.args.get("stage", "clean")  # 'raw' or 'clean'
    df = STATE["df_raw"] if stage == "raw" else STATE["df_clean"]
    if df is None:
        df = STATE["df_raw"]
    if df is None:
        return error_response("No dataset loaded yet.")
    cap = 5000
    truncated = len(df) > cap
    out = df.head(cap)
    return jsonify({
        "ok": True,
        "columns": list(df.columns),
        "rows": clean_json(out.to_dict(orient="records")),
        "total_rows": int(len(df)),
        "truncated": truncated,
        "shown_rows": int(len(out)),
    })


@app.route("/api/clean", methods=["POST"])
def api_clean():
    if STATE["df_raw"] is None:
        return error_response("Upload or load a dataset first.")
    df = STATE["df_raw"].copy()
    report = {"actions": [], "dropped_columns": [], "before_shape": list(df.shape)}

    # 1. Drop exact duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count:
        df = df.drop_duplicates().reset_index(drop=True)
        report["actions"].append(f"Removed {dup_count} duplicate row(s).")

    # 2. Drop columns that are entirely empty or constant (zero information)
    for col in list(df.columns):
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            df = df.drop(columns=[col])
            report["dropped_columns"].append(col)
            report["actions"].append(f"Dropped constant/empty column '{col}'.")

    # 3. Drop columns with excessive missingness (>60%)
    for col in list(df.columns):
        frac_missing = df[col].isna().mean()
        if frac_missing > 0.6:
            df = df.drop(columns=[col])
            report["dropped_columns"].append(col)
            report["actions"].append(
                f"Dropped column '{col}' ({frac_missing:.0%} missing values)."
            )

    # 4. Drop likely ID columns and other high-cardinality text columns that would
    #    blow up one-hot encoding downstream (e.g. 'CustomerId', 'Surname', free-text notes)
    for col in list(df.columns):
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        n = len(df)
        nunique = df[col].nunique(dropna=True)
        if n == 0:
            continue
        frac_unique = nunique / n
        is_near_unique_id = frac_unique > 0.95 and n > 20
        looks_like_id_name = "id" in col.lower()
        is_high_cardinality_text = nunique > 50 and frac_unique > 0.3
        if is_near_unique_id or looks_like_id_name or is_high_cardinality_text:
            df = df.drop(columns=[col])
            report["dropped_columns"].append(col)
            reason = (
                "likely identifier" if (is_near_unique_id or looks_like_id_name)
                else f"high-cardinality text ({nunique} unique values, unsuitable for one-hot encoding)"
            )
            report["actions"].append(f"Dropped column '{col}' -- {reason}.")

    # 4b. Same idea for numeric near-unique columns explicitly named like an ID,
    #     plus perfectly sequential integer columns (e.g. 'RowNumber') that are just a row index
    for col in list(df.columns):
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        nunique = df[col].nunique(dropna=True)
        n = len(df)
        is_near_unique = nunique > 0.95 * n and n > 20
        named_like_id = "id" in col.lower()
        is_sequential_index = (
            pd.api.types.is_integer_dtype(df[col]) and nunique == n and n > 20
            and (df[col].max() - df[col].min() + 1) == n
        )
        if (is_near_unique and named_like_id) or is_sequential_index:
            df = df.drop(columns=[col])
            report["dropped_columns"].append(col)
            reason = "sequential row index" if is_sequential_index else "likely identifier"
            report["actions"].append(f"Dropped column '{col}' -- {reason}.")

    # 5. Impute remaining missing values
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            report["actions"].append(
                f"Filled {n_missing} missing value(s) in numeric column '{col}' with median ({fill_val:.3g})."
            )
        else:
            mode_series = df[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            report["actions"].append(
                f"Filled {n_missing} missing value(s) in column '{col}' with most frequent value ('{fill_val}')."
            )

    # 6. Strip whitespace from string columns
    obj_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    report["after_shape"] = list(df.shape)
    if not report["actions"]:
        report["actions"].append("Dataset was already clean -- no changes needed.")

    STATE["df_clean"] = df
    STATE["cleaning_report"] = report
    return jsonify({"ok": True, "report": clean_json(report), **_dataset_summary(df)})


@app.route("/api/eda", methods=["POST"])
def api_eda():
    if STATE["df_clean"] is None:
        return error_response("Run cleaning first.")
    df = STATE["df_clean"]
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # Summary statistics for numeric columns
    stats = {}
    for c in numeric_cols:
        desc = df[c].describe()
        stats[c] = {
            "mean": desc.get("mean"), "std": desc.get("std"),
            "min": desc.get("min"), "max": desc.get("max"),
            "median": df[c].median(),
        }

    # Correlation matrix
    corr = {}
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr().round(3)
        corr = {"columns": numeric_cols, "matrix": corr_df.values.tolist()}

    # Histograms (bin counts computed server-side)
    histograms = {}
    for c in numeric_cols[:MAX_HISTOGRAM_COLS]:
        vals = df[c].dropna().values
        if len(vals) == 0:
            continue
        counts, edges = np.histogram(vals, bins=10)
        histograms[c] = {
            "counts": counts.tolist(),
            "edges": [round(float(e), 3) for e in edges],
        }

    # Categorical value counts
    cat_counts = {}
    for c in categorical_cols[:MAX_CATEGORICAL_COLS]:
        vc = df[c].value_counts().head(8)
        cat_counts[c] = {"labels": vc.index.astype(str).tolist(), "counts": vc.values.tolist()}

    eda = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "stats": stats,
        "correlation": corr,
        "histograms": histograms,
        "categorical_counts": cat_counts,
        "row_count": int(df.shape[0]),
        "col_count": int(df.shape[1]),
    }
    STATE["eda"] = eda
    return jsonify({"ok": True, "eda": clean_json(eda)})


@app.route("/api/target", methods=["POST"])
def api_target():
    if STATE["df_clean"] is None:
        return error_response("Run cleaning first.")
    target = (request.json or {}).get("target")
    df = STATE["df_clean"]
    if target not in df.columns:
        return error_response(f"Column '{target}' not found in dataset.")
    task_type = guess_task_type(df[target])
    STATE["target"] = target
    STATE["task_type"] = task_type

    info = {"target": target, "task_type": task_type}
    if task_type == "classification":
        vc = df[target].value_counts()
        info["classes"] = vc.index.astype(str).tolist()
        info["class_counts"] = vc.values.tolist()
    else:
        info["stats"] = {
            "mean": float(df[target].mean()), "std": float(df[target].std()),
            "min": float(df[target].min()), "max": float(df[target].max()),
        }
        counts, edges = np.histogram(df[target].dropna().values, bins=10)
        info["histogram"] = {"counts": counts.tolist(), "edges": [round(float(e), 3) for e in edges]}
    return jsonify({"ok": True, **clean_json(info)})


@app.route("/api/features", methods=["POST"])
def api_features():
    if STATE["target"] is None:
        return error_response("Set a target column first.")
    df = STATE["df_clean"]
    target = STATE["target"]
    task_type = STATE["task_type"]
    candidate_cols = [c for c in df.columns if c != target]

    # Encode a working copy purely for scoring purposes
    work = df[candidate_cols].copy()
    encoders = {}
    for c in work.columns:
        if not pd.api.types.is_numeric_dtype(work[c]):
            le = LabelEncoder()
            work[c] = le.fit_transform(work[c].astype(str))
            encoders[c] = le

    y = df[target]
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        y_enc = LabelEncoder().fit_transform(y.astype(str))
    else:
        y_enc = y.values

    try:
        if task_type == "classification":
            mi = mutual_info_classif(work, y_enc, random_state=42)
        else:
            mi = mutual_info_regression(work, y_enc, random_state=42)
    except Exception:
        mi = np.zeros(work.shape[1])

    # Absolute correlation with target as a secondary, interpretable signal
    corr_scores = []
    for c in work.columns:
        try:
            corr_scores.append(abs(np.corrcoef(work[c], y_enc)[0, 1]))
        except Exception:
            corr_scores.append(0.0)
    corr_scores = np.nan_to_num(np.array(corr_scores))

    mi_norm = mi / mi.max() if mi.max() > 0 else mi
    combined = 0.7 * mi_norm + 0.3 * corr_scores

    ranking = sorted(
        zip(candidate_cols, mi.tolist(), corr_scores.tolist(), combined.tolist()),
        key=lambda t: t[3], reverse=True,
    )
    ranking_out = [
        {"feature": f, "mutual_info": round(m, 4), "abs_correlation": round(c, 4), "score": round(s, 4)}
        for f, m, c, s in ranking
    ]

    k = min(10, len(candidate_cols)) if len(candidate_cols) > 10 else len(candidate_cols)
    # keep only features with a meaningful positive score, but always keep at least 2
    selected = [r["feature"] for r in ranking_out if r["score"] > 0][:k] or [r["feature"] for r in ranking_out[:min(2, len(ranking_out))]]

    STATE["feature_ranking"] = ranking_out
    STATE["selected_features"] = selected
    return jsonify({"ok": True, "ranking": clean_json(ranking_out), "selected_features": selected})


MODEL_ZOO = {
    "classification": {
        "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
        "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=200, random_state=42),
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": lambda: SVC(probability=True, random_state=42),
        "Naive Bayes": lambda: GaussianNB(),
        "AdaBoost": lambda: AdaBoostClassifier(random_state=42),
    },
    "regression": {
        "Linear Regression": lambda: LinearRegression(),
        "Ridge Regression": lambda: Ridge(random_state=42),
        "Lasso Regression": lambda: Lasso(random_state=42),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=42),
        "Extra Trees": lambda: ExtraTreesRegressor(n_estimators=200, random_state=42),
        "Decision Tree": lambda: DecisionTreeRegressor(random_state=42),
        "K-Nearest Neighbors": lambda: KNeighborsRegressor(n_neighbors=5),
        "Support Vector Machine": lambda: SVR(),
        "AdaBoost": lambda: AdaBoostRegressor(random_state=42),
    },
}

if XGBOOST_AVAILABLE:
    MODEL_ZOO["classification"]["XGBoost"] = lambda: XGBClassifier(
        n_estimators=200, random_state=42, eval_metric="logloss", use_label_encoder=False, verbosity=0
    )
    MODEL_ZOO["regression"]["XGBoost"] = lambda: XGBRegressor(
        n_estimators=200, random_state=42, verbosity=0
    )


@app.route("/api/available_models", methods=["GET"])
def api_available_models():
    task_type = request.args.get("task_type") or STATE.get("task_type")
    if task_type not in MODEL_ZOO:
        return error_response("Set a target column first so the task type is known.")
    return jsonify({"ok": True, "task_type": task_type, "models": list(MODEL_ZOO[task_type].keys())})


@app.route("/api/train", methods=["POST"])
def api_train():
    if not STATE.get("selected_features"):
        return error_response("Run feature selection first.")
    df = STATE["df_clean"]
    target = STATE["target"]
    task_type = STATE["task_type"]
    body = request.json or {}
    features = body.get("features") or STATE["selected_features"]
    features = [f for f in features if f in df.columns]
    if len(features) < 1:
        return error_response("No valid features selected.")

    requested_models = body.get("models")
    zoo = MODEL_ZOO[task_type]
    if requested_models:
        zoo = {k: v for k, v in zoo.items() if k in requested_models}
        if not zoo:
            return error_response("None of the requested models are valid for this task type.")

    X = df[features].copy()
    y = df[target].copy()
    label_encoder = None
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)

    stratify = y if task_type == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(df, features)

    leaderboard = []
    fitted_pipelines = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task_type == "classification" else KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = "accuracy" if task_type == "classification" else "r2"

    for name, ctor in zoo.items():
        try:
            pipe = Pipeline([("pre", preprocessor), ("model", ctor())])
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
            pipe.fit(X_train, y_train)
            fitted_pipelines[name] = pipe
            leaderboard.append({
                "model": name,
                "cv_mean": round(float(np.mean(scores)), 4),
                "cv_std": round(float(np.std(scores)), 4),
                "metric": scoring,
            })
        except Exception as e:
            leaderboard.append({"model": name, "error": str(e)})

    valid = [m for m in leaderboard if "cv_mean" in m]
    if not valid:
        return error_response("All models failed to train on this data.", 500)
    valid.sort(key=lambda m: m["cv_mean"], reverse=True)
    best_name = valid[0]["model"]
    best_pipe = fitted_pipelines[best_name]

    y_pred = best_pipe.predict(X_test)
    metrics = {"task_type": task_type, "best_model": best_name}
    if task_type == "classification":
        metrics.update({
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        })
        labels = sorted(pd.unique(y_test))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        class_names = label_encoder.inverse_transform(labels).tolist() if label_encoder else [str(l) for l in labels]
        metrics["confusion_matrix"] = {"labels": class_names, "matrix": cm.tolist()}

        # ROC curve + AUC, only meaningful for binary classification with probability support
        if len(labels) == 2 and hasattr(best_pipe, "predict_proba"):
            try:
                proba = best_pipe.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba, pos_label=labels[1])
                auc = roc_auc_score(y_test, proba)
                step = max(1, len(fpr) // 60)  # thin the curve for a lightweight payload
                metrics["roc_curve"] = {
                    "fpr": [round(float(v), 4) for v in fpr[::step]],
                    "tpr": [round(float(v), 4) for v in tpr[::step]],
                    "auc": round(float(auc), 4),
                    "positive_class": str(class_names[1]),
                }
            except Exception:
                pass

        # class balance, for a quick visual of predicted vs actual distribution
        pred_names = label_encoder.inverse_transform(y_pred.astype(int)).tolist() if label_encoder else y_pred.tolist()
        actual_names = label_encoder.inverse_transform(np.array(y_test).astype(int)).tolist() if label_encoder else list(y_test)
        metrics["class_balance"] = {
            "labels": class_names,
            "actual": [int(np.sum(np.array(actual_names) == c)) for c in class_names],
            "predicted": [int(np.sum(np.array(pred_names) == c)) for c in class_names],
        }
    else:
        metrics.update({
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        })
        metrics["residual_sample"] = {
            "actual": [float(v) for v in list(y_test)[:60]],
            "predicted": [float(v) for v in list(y_pred)[:60]],
        }

    STATE["leaderboard"] = valid + [m for m in leaderboard if "cv_mean" not in m]
    STATE["best_model_name"] = best_name
    STATE["best_pipeline"] = best_pipe
    STATE["metrics"] = metrics
    STATE["test_data"] = {"features": features}
    STATE["label_encoder"] = label_encoder

    return jsonify({"ok": True, "leaderboard": clean_json(STATE["leaderboard"]), "metrics": clean_json(metrics)})


@app.route("/api/explain", methods=["POST"])
def api_explain():
    if STATE["best_pipeline"] is None:
        return error_response("Train a model first.")
    pipe = STATE["best_pipeline"]
    metrics = STATE["metrics"]
    task_type = STATE["task_type"]
    best_name = STATE["best_model_name"]
    features = STATE["test_data"]["features"]

    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    try:
        feature_names = pre.get_feature_names_out().tolist()
    except Exception:
        feature_names = features

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

    def _clean_name(n):
        return n.split("__", 1)[1] if "__" in n else n

    ranked = []
    if importances is not None and len(importances) == len(feature_names):
        order = np.argsort(importances)[::-1]
        total = importances.sum() or 1.0
        for i in order[:12]:
            ranked.append({
                "feature": _clean_name(feature_names[i]),
                "importance": round(float(importances[i]), 4),
                "pct": round(float(importances[i] / total * 100), 1),
            })

    # ---- Natural-language explanation, generated locally (no LLM API calls) ----
    lines = []
    if task_type == "classification":
        lines.append(
            f"The best-performing model was **{best_name}**, reaching "
            f"{metrics['accuracy']*100:.1f}% accuracy and an F1-score of {metrics['f1']:.2f} "
            f"on held-out test data."
        )
        if metrics["precision"] < metrics["recall"] - 0.05:
            lines.append("It catches most true cases (high recall) but produces more false positives than false negatives.")
        elif metrics["recall"] < metrics["precision"] - 0.05:
            lines.append("It is conservative: predictions it makes tend to be correct (high precision), but it misses some true cases.")
        else:
            lines.append("Precision and recall are well balanced, so predictions are trustworthy in both directions.")
    else:
        lines.append(
            f"The best-performing model was **{best_name}**, explaining "
            f"{metrics['r2']*100:.1f}% of the variance (R²) in '{STATE['target']}', "
            f"with an average error of about {metrics['mae']:.3g} units (MAE)."
        )
        if metrics["r2"] > 0.75:
            lines.append("This is a strong fit -- the selected features capture most of the pattern in the target.")
        elif metrics["r2"] > 0.4:
            lines.append("This is a moderate fit -- useful for directional predictions, but sizeable unexplained variance remains.")
        else:
            lines.append("This is a weak fit -- consider adding more informative features or collecting more data.")

    if ranked:
        top = ranked[:3]
        top_desc = ", ".join(f"'{r['feature']}' ({r['pct']:.0f}%)" for r in top)
        lines.append(f"The most influential factors driving predictions were {top_desc}.")

    lines.append(
        f"This model used {len(features)} feature(s) selected automatically for their statistical "
        f"relationship with the target, out of the columns available after cleaning."
    )

    explain = {"narrative": lines, "feature_importance": ranked}
    STATE["explain"] = explain
    return jsonify({"ok": True, **clean_json(explain)})


@app.route("/api/feature_meta", methods=["GET"])
def api_feature_meta():
    if STATE["df_clean"] is None or not STATE.get("test_data"):
        return error_response("Train a model first.")
    df = STATE["df_clean"]
    features = STATE["test_data"]["features"]
    meta = []
    for c in features:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            meta.append({
                "feature": c, "type": "numeric",
                "min": float(col.min()), "max": float(col.max()),
                "mean": float(col.mean()), "median": float(col.median()),
                "step": round(float((col.max() - col.min()) / 100 or 1), 4),
            })
        else:
            options = sorted(col.dropna().astype(str).unique().tolist())
            meta.append({
                "feature": c, "type": "categorical",
                "options": options, "default": col.mode().iloc[0] if not col.mode().empty else options[0],
            })
    return jsonify({"ok": True, "features": meta})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if STATE["best_pipeline"] is None:
        return error_response("Train a model first.")
    body = request.json or {}
    features = STATE["test_data"]["features"]
    df = STATE["df_clean"]
    row = {}
    for c in features:
        if c not in body:
            return error_response(f"Missing value for feature '{c}'.")
        if pd.api.types.is_numeric_dtype(df[c]):
            try:
                row[c] = float(body[c])
            except (TypeError, ValueError):
                return error_response(f"Feature '{c}' expects a numeric value.")
        else:
            row[c] = str(body[c])
    X = pd.DataFrame([row], columns=features)

    pipe = STATE["best_pipeline"]
    task_type = STATE["task_type"]
    label_encoder = STATE.get("label_encoder")
    try:
        pred = pipe.predict(X)[0]
    except Exception as e:
        traceback.print_exc()
        return error_response(f"Prediction failed: {e}", 500)

    result = {"task_type": task_type, "model": STATE["best_model_name"], "inputs": row}
    if task_type == "classification":
        pred_label = label_encoder.inverse_transform([int(pred)])[0] if label_encoder else pred
        result["prediction"] = str(pred_label)
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X)[0]
                classes = pipe.named_steps["model"].classes_
                class_names = label_encoder.inverse_transform(classes.astype(int)) if label_encoder else classes
                probs = sorted(
                    [{"class": str(cn), "probability": round(float(p), 4)} for cn, p in zip(class_names, proba)],
                    key=lambda d: d["probability"], reverse=True,
                )
                result["probabilities"] = probs
            except Exception:
                pass
    else:
        result["prediction"] = round(float(pred), 4)

    return jsonify({"ok": True, **clean_json(result)})


@app.route("/api/download/cleaned_csv", methods=["GET"])
def api_download_cleaned():
    if STATE["df_clean"] is None:
        return error_response("Nothing to export yet -- run cleaning first.")
    buf = io.BytesIO()
    STATE["df_clean"].to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="cleaned_dataset.csv")


@app.route("/api/download/model", methods=["GET"])
def api_download_model():
    if STATE["best_pipeline"] is None:
        return error_response("Train a model first.")
    buf = io.BytesIO()
    joblib.dump({
        "pipeline": STATE["best_pipeline"],
        "label_encoder": STATE.get("label_encoder"),
        "features": STATE["test_data"]["features"],
        "target": STATE["target"],
        "task_type": STATE["task_type"],
        "model_name": STATE["best_model_name"],
    }, buf)
    buf.seek(0)
    safe_name = (STATE["best_model_name"] or "model").lower().replace(" ", "_")
    return send_file(buf, mimetype="application/octet-stream", as_attachment=True,
                      download_name=f"autods_{safe_name}.joblib")


@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    if STATE["df_clean"] is None:
        return error_response("Nothing to show yet -- run the pipeline first.")
    payload = {
        "cleaning_report": STATE["cleaning_report"],
        "eda": STATE["eda"],
        "target": STATE["target"],
        "task_type": STATE["task_type"],
        "feature_ranking": STATE["feature_ranking"],
        "selected_features": STATE["selected_features"],
        "leaderboard": STATE["leaderboard"],
        "metrics": STATE["metrics"],
        "explain": STATE["explain"],
        "dataset_shape": list(STATE["df_clean"].shape),
    }
    return jsonify({"ok": True, **clean_json(payload)})


if __name__ == "__main__":
    # threaded=True lets the UI keep responding (e.g. viewing data) while a model trains;
    # use_reloader=False avoids the dev server restarting mid-session and dropping in-flight requests.
    app.run(debug=True, use_reloader=False, threaded=True, host="0.0.0.0", port=5000)