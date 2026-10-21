import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from pdf_parser import extract_text, parse_financial_values
from financial_analyzer import (
    calculate_ratios, compare_periods, detect_anomalies,
    forecast_metric, generate_insights, model_forecast_revenue
)
from report_generator import create_pdf_report

st.set_page_config(page_title="AI Financial Report Analyzer", page_icon="📊", layout="wide")
st.title("📊 AI-Based Financial Report Analysis System")
st.caption("All displayed financial values must come from the PDF uploaded in this session. No demo or predefined financial data is loaded.")

with st.sidebar:
    st.header("Workflow")
    st.markdown(
        "1. Upload a financial PDF.\n"
        "2. Extract values from that PDF.\n"
        "3. Verify the extracted values.\n"
        "4. Run analysis and download the report."
    )
    st.caption("Scanned PDFs require OCR. Always verify the extracted figures.")

uploaded = st.file_uploader("Upload a financial report (PDF)", type=["pdf"])

if uploaded is None:
    st.info("Upload a financial PDF to begin. The dashboard remains empty until a PDF is provided.")
    st.stop()

try:
    raw_text = extract_text(uploaded)
    if not raw_text.strip():
        st.error("No selectable text was found. This may be a scanned PDF. OCR support is not enabled in this version.")
        st.stop()
    with st.expander("View extracted PDF text"):
        st.text(raw_text[:20000])
    parsed = parse_financial_values(raw_text)
except Exception as exc:
    st.error(f"Could not read this PDF: {exc}")
    st.stop()

if parsed.empty:
    st.error("No supported financial values were detected in this PDF. Try a text-based report with clear metric labels, or improve the parser for this report layout. No sample values will be inserted.")
    st.stop()

st.success(f"Extracted {len(parsed)} candidate value(s) from the uploaded PDF. Please verify them before analysis.")
st.subheader("Review and edit values extracted from this PDF")
edited = st.data_editor(parsed, num_rows="dynamic", use_container_width=True, key="financial_data_editor")

edited = edited.copy()
edited["Value"] = pd.to_numeric(edited["Value"], errors="coerce")
edited = edited.dropna(subset=["Metric", "Value", "Period"])
edited["Metric"] = edited["Metric"].astype(str).str.strip()
edited["Period"] = edited["Period"].astype(str).str.strip()
edited = edited[(edited["Metric"] != "") & (edited["Period"] != "")]

if edited.empty:
    st.warning("No valid extracted values remain. Review the table above.")
    st.stop()

model_path = os.path.join("models", "revenue_forecast_model.joblib")
metrics_path = os.path.join("models", "training_metrics.json")
if os.path.exists(model_path):
    st.success("Trained Random Forest model is installed. It is used only if the uploaded PDF contains Revenue, Net Profit, and Total Assets.")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as metrics_file:
                training_metrics = json.load(metrics_file)
            with st.expander("Model training and validation details"):
                st.write(f"Training examples: {training_metrics.get('rows_used', 'N/A'):,}")
                st.write(f"Companies represented: {training_metrics.get('companies', 'N/A'):,}")
                st.write(f"Test-set R²: {training_metrics.get('r2', float('nan')):.3f}")
                st.write(f"Test-set median absolute percentage error: {training_metrics.get('median_absolute_percentage_error_pct', float('nan')):.2f}%")
                st.caption(training_metrics.get("note", "Prototype forecast; not financial advice."))
        except Exception:
            st.caption("Training metrics could not be loaded.")
else:
    st.info("No trained model file is installed. Ratio analysis and transparent baseline trend analysis can still run from this PDF.")

if st.button("Analyze this PDF", type="primary", use_container_width=True):
    ratios = calculate_ratios(edited)
    changes = compare_periods(edited)
    anomalies = detect_anomalies(edited)
    baseline_forecast = forecast_metric(edited, "Revenue", periods_ahead=1)
    trained_forecast, model_note = model_forecast_revenue(edited, model_path)
    forecast = trained_forecast if not trained_forecast.empty else baseline_forecast
    insights = generate_insights(edited, changes, ratios, anomalies, forecast)
    if model_note:
        insights.append(model_note)

    st.subheader("Key performance indicators")
    period_order = edited["Period"].drop_duplicates().tolist()
    latest_period = period_order[-1]
    latest = edited[edited["Period"] == latest_period]
    metrics = {r["Metric"]: r["Value"] for _, r in latest.iterrows()}
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Revenue ({latest_period})", f"{metrics['Revenue']:,.2f}" if "Revenue" in metrics else "N/A")
    c2.metric(f"Net profit ({latest_period})", f"{metrics['Net Profit']:,.2f}" if "Net Profit" in metrics else "N/A")
    margin = ratios[(ratios["Ratio"] == "Net Profit Margin (%)") & (ratios["Period"].astype(str) == str(latest_period))]
    c3.metric("Net profit margin", f"{margin.iloc[-1]['Value']:.2f}%" if not margin.empty else "N/A")

    st.subheader("Period comparison")
    if not changes.empty:
        st.dataframe(changes, use_container_width=True)
        selected_metric = st.selectbox("Metric trend", sorted(edited["Metric"].unique()))
        chart_data = edited[edited["Metric"] == selected_metric].copy()
        chart_data["_period_sort"] = pd.to_numeric(chart_data["Period"], errors="coerce")
        chart_data = chart_data.sort_values(["_period_sort", "Period"]).drop(columns="_period_sort")
        st.plotly_chart(px.line(chart_data, x="Period", y="Value", markers=True, title=f"{selected_metric} over time"), use_container_width=True)
    else:
        st.info("Only one period was extracted; period-to-period comparison is unavailable.")

    st.subheader("Financial ratios")
    st.dataframe(ratios, use_container_width=True)
    st.subheader("Revenue forecast")
    if not forecast.empty:
        st.dataframe(forecast, use_container_width=True)
        st.caption("Forecasts are estimates. A trained model is used only when available and required PDF features are present.")
    else:
        st.info("No forecast could be produced from the values extracted from this PDF.")

    st.subheader("Analysis insights")
    for insight in insights:
        st.write("• " + insight)

    pdf_bytes = create_pdf_report(edited, ratios, changes, insights)
    st.download_button(
        "Download analysis report (PDF)", data=pdf_bytes,
        file_name="financial_analysis_report.pdf", mime="application/pdf"
    )
    st.caption("This analysis uses only the uploaded PDF's extracted values. Missing metrics are not filled from a training dataset.")
