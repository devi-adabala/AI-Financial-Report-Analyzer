"""Financial analysis functions and optional trained revenue model."""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import joblib

def calculate_ratios(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["Period", "Ratio", "Value", "Interpretation"])
    wide = data.pivot_table(index="Period", columns="Metric", values="Value", aggfunc="last")
    results = []
    for period, row in wide.iterrows():
        def get(name):
            val = row.get(name, np.nan)
            return float(val) if pd.notna(val) else np.nan
        revenue, profit = get("Revenue"), get("Net Profit")
        ca, cl = get("Current Assets"), get("Current Liabilities")
        debt, equity = get("Total Debt"), get("Shareholders Equity")
        assets = get("Total Assets")
        if pd.notna(ca) and pd.notna(cl) and cl != 0:
            results.append([period, "Current Ratio", ca/cl, "Current assets divided by current liabilities."])
        if pd.notna(debt) and pd.notna(equity) and equity != 0:
            results.append([period, "Debt-to-Equity", debt/equity, "Total debt divided by shareholders' equity."])
        if pd.notna(revenue) and pd.notna(profit) and revenue != 0:
            results.append([period, "Net Profit Margin (%)", profit/revenue*100, "Net profit as a percentage of revenue."])
        if pd.notna(profit) and pd.notna(assets) and assets != 0:
            results.append([period, "Return on Assets (%)", profit/assets*100, "Net profit relative to total assets."])
    return pd.DataFrame(results, columns=["Period", "Ratio", "Value", "Interpretation"])

def compare_periods(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["Metric", "Previous Period", "Latest Period", "Previous Value", "Latest Value", "Change", "Change (%)"]
    if data.empty:
        return pd.DataFrame(columns=columns)
    wide = data.pivot_table(index="Period", columns="Metric", values="Value", aggfunc="last")
    # Sort numeric year labels chronologically where possible.
    wide = wide.copy()
    wide["_sort"] = pd.to_numeric(wide.index, errors="coerce")
    wide = wide.sort_values("_sort", na_position="last").drop(columns="_sort")
    rows = []
    for metric in wide.columns:
        vals = wide[metric].dropna()
        if len(vals) < 2:
            continue
        prev_period, latest_period = vals.index[-2], vals.index[-1]
        prev, latest = float(vals.iloc[-2]), float(vals.iloc[-1])
        pct = ((latest-prev)/abs(prev)*100) if prev != 0 else np.nan
        rows.append([metric, prev_period, latest_period, prev, latest, latest-prev, pct])
    return pd.DataFrame(rows, columns=columns)

def detect_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["Period", "Metric", "Value", "Anomaly"]
    if data.empty:
        return pd.DataFrame(columns=columns)
    out = data.copy()
    out["Anomaly"] = "Not evaluated"
    for metric, idx in out.groupby("Metric").groups.items():
        vals = out.loc[idx, "Value"].astype(float)
        if len(vals) >= 5 and vals.nunique() > 1:
            model = IsolationForest(contamination=0.15, random_state=42)
            labels = model.fit_predict(vals.to_numpy().reshape(-1, 1))
            out.loc[idx, "Anomaly"] = ["Potential anomaly" if x == -1 else "Typical" for x in labels]
        elif len(vals) >= 2:
            median = vals.median()
            scale = (vals - median).abs().median()
            if scale > 0:
                out.loc[idx, "Anomaly"] = ["Potential anomaly" if abs(v-median) > 3*scale else "Typical" for v in vals]
            else:
                out.loc[idx, "Anomaly"] = "Not enough variation"
    return out[columns]

def forecast_metric(data: pd.DataFrame, metric="Revenue", periods_ahead=1) -> pd.DataFrame:
    subset = data[data["Metric"] == metric].copy()
    subset["_sort"] = pd.to_numeric(subset["Period"], errors="coerce")
    subset = subset.sort_values(["_sort", "Period"]).drop(columns="_sort").reset_index(drop=True)
    if len(subset) < 3:
        return pd.DataFrame(columns=["Step", "Forecast"])
    X = np.arange(len(subset)).reshape(-1, 1)
    y = subset["Value"].astype(float).to_numpy()
    model = LinearRegression().fit(X, y)
    pred = model.predict(np.arange(len(subset), len(subset)+periods_ahead).reshape(-1, 1))
    return pd.DataFrame({"Step": [f"Future {i+1} (baseline trend)" for i in range(periods_ahead)], "Forecast": pred})

def model_forecast_revenue(data: pd.DataFrame, model_path: str):
    """Use a trained model only when all feature values can be read from the uploaded PDF."""
    if not os.path.exists(model_path):
        return pd.DataFrame(), "Trained-model prediction skipped: no trained model file is installed."
    try:
        bundle = joblib.load(model_path)
        features = bundle["features"]
        latest_period = data["Period"].drop_duplicates().iloc[-1]
        latest = data[data["Period"].astype(str) == str(latest_period)]
        values = {r["Metric"]: float(r["Value"]) for _, r in latest.iterrows()}
        missing = [f for f in features if f not in values]
        if missing:
            return pd.DataFrame(), "Trained-model prediction skipped because the uploaded PDF does not contain: " + ", ".join(missing) + "."
        X = pd.DataFrame([[values[f] for f in features]], columns=features).astype(float)
        if bundle.get("feature_transform") == "signed_log1p":
            X = np.sign(X) * np.log1p(np.abs(X))
        prediction = float(bundle["model"].predict(X)[0])
        if bundle.get("target_transform") == "log1p":
            prediction = float(np.expm1(prediction))
        prediction = max(prediction, 0.0)
        return pd.DataFrame([{"Step": "Next-period revenue (trained model)", "Forecast": prediction}]), "Revenue prediction was generated by the trained model using only metrics extracted from the uploaded PDF."
    except Exception as exc:
        return pd.DataFrame(), f"Trained-model prediction could not run: {exc}"

def generate_insights(data, changes, ratios, anomalies, forecast) -> list:
    insights = []
    if not changes.empty:
        for _, r in changes.iterrows():
            pct = r["Change (%)"]
            if pd.notna(pct):
                direction = "increased" if pct > 0 else "decreased" if pct < 0 else "remained unchanged"
                if r["Metric"] in ("Revenue", "Net Profit"):
                    insights.append(f"{r['Metric']} {direction} by {abs(pct):.1f}% between {r['Previous Period']} and {r['Latest Period']}.")
                elif abs(pct) >= 20:
                    insights.append(f"{r['Metric']} changed by {pct:.1f}%; review the source statement and business context.")
    if not ratios.empty:
        for _, r in ratios.iterrows():
            if r["Ratio"] == "Current Ratio":
                if r["Value"] < 1:
                    insights.append(f"Current ratio is {r['Value']:.2f}, below 1.0; current liabilities exceed current assets for {r['Period']}.")
            elif r["Ratio"] == "Net Profit Margin (%)":
                insights.append(f"Net profit margin is {r['Value']:.1f}% for {r['Period']}.")
    if not anomalies.empty and (anomalies["Anomaly"] == "Potential anomaly").any():
        n = int((anomalies["Anomaly"] == "Potential anomaly").sum())
        insights.append(f"{n} potential unusual data point(s) were flagged. Verify them against the original PDF.")
    if not forecast.empty:
        insights.append(f"The model estimates the next revenue value at {forecast.iloc[0]['Forecast']:,.2f}; this is an estimate, not a guarantee.")
    if not insights:
        insights.append("Not enough complete information in this PDF to generate detailed insights.")
    return insights
