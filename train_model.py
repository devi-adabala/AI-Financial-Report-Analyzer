"""Train a revenue forecasting model from a real-world, prepared company-year dataset.

Expected CSV columns:
company_id, company_name, period, Revenue, Net Profit, Operating Expenses,
Current Assets, Current Liabilities, Total Debt, Shareholders Equity,
Total Assets, Operating Cash Flow

The target is next-period Revenue within the same company. A row is used only if
the following period for that company is available.
"""
import argparse
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

FEATURES = ["Revenue", "Net Profit", "Total Assets"]
REQUIRED = ["company_id", "period"] + FEATURES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Prepared real-world company-year CSV")
    parser.add_argument("--output", default="models/revenue_forecast_model.joblib")
    parser.add_argument("--metrics", default="models/training_metrics.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing) +
                         ". Run prepare_sec_dataset.py first for an SEC DERA ZIP, or map your CSV to the documented schema.")
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["company_id", "period", "Revenue"]).copy()
    df = df.sort_values(["company_id", "period"])
    # Avoid duplicate company-year rows from multiple filings.
    df = df.drop_duplicates(["company_id", "period"], keep="last")
    df["target_next_revenue"] = df.groupby("company_id")["Revenue"].shift(-1)
    df["next_period"] = df.groupby("company_id")["period"].shift(-1)
    train = df[df["target_next_revenue"].notna() & (df["next_period"] > df["period"])].copy()
    train = train.dropna(subset=FEATURES + ["target_next_revenue"])
    # Revenue forecasting is trained on positive reported revenue and assets only.
    train = train[(train["Revenue"] > 0) & (train["target_next_revenue"] > 0) & (train["Total Assets"] > 0)].copy()
    if len(train) < 30:
        raise SystemExit(f"Only {len(train)} usable next-period examples were found. Need at least 30 complete company-year examples; provide more years/companies or a better-mapped dataset.")
    X_raw = train[FEATURES].astype(float)
    # Signed log scaling handles large size differences and negative net profit.
    X = np.sign(X_raw) * np.log1p(np.abs(X_raw))
    y_raw = train["target_next_revenue"].astype(float)
    y = np.log1p(y_raw)
    groups = train["company_id"].astype(str)
    # Keep companies separate across train/test where possible.
    if groups.nunique() >= 5:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
    else:
        # Chronological fallback for a limited-company dataset.
        split_at = max(1, int(len(train) * 0.8))
        order = np.arange(len(train))
        train_idx, test_idx = order[:split_at], order[split_at:]
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions_log = model.predict(X.iloc[test_idx]) if len(test_idx) else np.array([])
    predictions = np.maximum(np.expm1(predictions_log), 0)
    y_test = y_raw.iloc[test_idx]
    test_current_revenue = train.iloc[test_idx]["Revenue"].astype(float)
    abs_pct = np.abs((y_test.to_numpy() - predictions) / np.maximum(np.abs(y_test.to_numpy()), 1.0))
    baseline_mae = float(mean_absolute_error(y_test, test_current_revenue)) if len(test_idx) else None
    metrics = {
        "rows_used": int(len(train)),
        "test_rows": int(len(test_idx)),
        "companies": int(groups.nunique()),
        "features": FEATURES,
        "model": "RandomForestRegressor",
        "mae_usd": float(mean_absolute_error(y_test, predictions)) if len(test_idx) else None,
        "rmse_usd": float(mean_squared_error(y_test, predictions) ** 0.5) if len(test_idx) else None,
        "r2": float(r2_score(y_test, predictions)) if len(test_idx) > 1 else None,
        "mean_absolute_percentage_error_pct": float(np.mean(abs_pct) * 100) if len(test_idx) else None,
        "median_absolute_percentage_error_pct": float(np.median(abs_pct) * 100) if len(test_idx) else None,
        "baseline_current_revenue_mae_usd": baseline_mae,
        "split_method": "company-held-out group split" if groups.nunique() >= 5 else "chronological fallback",
        "note": "Metrics are on a held-out test set. Revenue scale varies greatly across companies; interpret USD errors alongside percentage error. Prototype only, not financial advice."
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics) or ".", exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "feature_transform": "signed_log1p", "target_transform": "log1p"}, args.output)
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print("Model saved to:", args.output)

if __name__ == "__main__":
    main()
