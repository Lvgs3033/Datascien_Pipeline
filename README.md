# AUTODS — Autonomous Data Scientist

**Docker-Image** : docker pull dhvanik23/autods

A self-contained web app that takes a raw CSV and automatically:

1. **Cleans** the data — drops duplicates, empty/constant columns, likely ID columns, sequential row-index columns, and high-cardinality text columns; imputes missing values.
2. **Explores** it — summary stats, correlation heatmap, distributions, category breakdowns, feature range visualizations.
3. **Selects features** — ranks columns by mutual information + correlation with your chosen target, with manual override.
4. **Trains models** — races up to **10** scikit-learn/XGBoost models per task with cross-validation and picks the best one.
5. **Explains results** — generates a plain-language readout of performance and key drivers (no LLM/API calls — pure Python logic).
6. **Predicts live** — enter feature values in a generated form and get an instant prediction (with class probabilities for classification).
7. **Builds a dashboard** — one page summarizing the whole run, with export buttons.

Backend: Python (Flask, pandas, numpy, scikit-learn, XGBoost, joblib).
Frontend: plain HTML/CSS/JS with Chart.js for charts (loaded from a public CDN).
**No API keys of any kind are required or used.**

## Setup

```bash
cd autods
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

> If `xgboost` fails to install on your machine, the app still runs fine — it just won't list XGBoost as one of the model options.

## Using it

- **Upload a CSV** via the drop zone, or click one of the sample datasets (Iris / Wine / Diabetes) to try it instantly.
- Click **"Run full pipeline"** in the left rail to execute every stage automatically, or step through each stage manually.
- The left rail is a live pipeline tracker: amber = running, teal check = done.

### Model racing (now includes XGBoost)

The Train stage lists every available model for the detected task type:
Logistic/Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, Extra Trees, Decision Tree, KNN, SVM, Naive Bayes/AdaBoost, and **XGBoost**. Tick/untick which ones to race, with "select all" / "none" shortcuts.

### Charts & visualizations

- **EDA**: correlation heatmap, per-column histograms, category count charts, and a min/mean/max **feature range** visualization.
- **Target distribution**: bar chart of class counts (classification) or histogram (regression), shown right after you pick a target.
- **Confusion matrix**: colored heatmap.
- **ROC curve + AUC**: shown automatically for binary classification when the winning model supports probability estimates.
- **Predicted vs. actual class balance**: side-by-side bar chart for classification.
- **Residual scatter plot**: predicted vs. actual for regression.
- **Leaderboard bar chart**: visual comparison of cross-validation scores across every raced model.
- **Feature importance chart**: on the Explain stage.
- All progress-bar-style elements (feature ranking bars, leaderboard bars, prediction probability bars) now use bright, glowing, tiered colors (gold/teal/silver) so they're clearly visible against the dark theme.

### Other features

- **Full dataset viewer** — "View full dataset" toggles from an 8-row preview to a scrollable table (sticky header, vertical scrollbar), both on raw upload and after cleaning.
- **Manual feature selection** — untick any ranked feature to exclude it before training.
- **Export buttons** — download the cleaned dataset as CSV, or the trained model as a `.joblib` file.

## Bug fixes in this version

- **Fixed a pandas 3.x compatibility bug**: newer pandas versions store text columns under a dedicated `str` dtype instead of the legacy `object` dtype. Several checks in the cleaning/task-detection logic still tested `dtype == object`, which silently failed under pandas 3.x — meaning high-cardinality text columns (like a `Surname` column) were never being dropped, and target-type detection could misfire for string targets. All of these now use `pd.api.types.is_numeric_dtype(...)` instead, which works correctly across pandas versions.
- **Fixed high-cardinality columns crashing/hanging training**: a text column like `Surname` or `CustomerId` with hundreds of unique values would explode into hundreds of one-hot columns, making training extremely slow or memory-heavy (this is what was behind the "Failed to fetch" error some users hit — the dev server was dying under the load). The cleaning step now also drops high-cardinality text columns (>50 unique values and >30% unique) and sequential row-index columns (e.g. `RowNumber`), and `OneHotEncoder` is capped at 25 categories per column as a hard safety net regardless.
- **Fixed a "Cannot read properties of null (reading 'classList')" crash**: several DOM lookups weren't defensive against a missing element. All `classList` access now uses optional chaining, and a global error handler shows a friendly toast instead of silently breaking the page.
- **Improved network error messages**: a failed `fetch()` (e.g. the server isn't running) now shows "Could not reach the server. Make sure app.py is still running, then try again." instead of the generic browser message.
- **More robust server startup**: the dev server now runs with `threaded=True` (so the UI stays responsive while a model trains) and `use_reloader=False` (so it doesn't restart mid-session and drop in-flight requests).

## Project structure

```
autods/
├── app.py                  # Flask backend + ML pipeline
├── requirements.txt
├── templates/
│   └── index.html          # single-page app shell
└── static/
    ├── css/style.css       # "instrument console" visual design
    └── js/app.js           # pipeline controller, charts, heatmaps, predict form
```

- All computation runs locally in-process; a single in-memory session holds the current dataset and trained model (fine for local/demo use — restart the server or click "Reset session" to start over).
- Classification vs. regression is detected automatically from the target column's type/cardinality.
- Feature selection uses `mutual_info_classif`/`mutual_info_regression` blended with absolute correlation; you can override the automatic picks before training.
- The downloaded `.joblib` file is a dict: `{pipeline, label_encoder, features, target, task_type, model_name}` — load it with `joblib.load()` and call `.predict()` directly on a DataFrame with the same feature columns.
