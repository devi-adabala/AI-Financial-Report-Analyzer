# AI-Based Financial Report Analysis System

## What is included
- Streamlit application for uploading and analyzing a financial PDF.
- PDF text extraction and metric parsing (`pdf_parser.py`).
- Financial ratios, period comparisons, anomaly detection, baseline forecast, and a trained revenue forecasting model (`financial_analyzer.py`).
- SEC-derived training table at `data/sec_company_year.csv`.
- Trained model at `models/revenue_forecast_model.joblib`.
- Held-out evaluation metrics at `models/training_metrics.json`.
- Training and data-preparation scripts for reproducibility.
- Project report and presentation are included in the project deliverables.

## Data integrity rule
The app has **no predefined/sample financial figures in the UI**. Until the user uploads a PDF, the dashboard remains empty. Once a PDF is uploaded, the analysis table is populated from values parsed from that PDF and can be reviewed/edited by the user. The SEC training data is used only to train the forecasting model; it is never used to fill missing PDF values.

PDF extraction and model prediction are different steps. A scanned/image-only PDF may require OCR, which is not enabled in this version. Always verify extracted numbers before relying on the analysis.

## Dataset and training
The training table was prepared from the 16 quarterly SEC Financial Statement Data Sets ZIP files provided with this project workflow, covering **2022 Q1–2025 Q4**. SEC data files contain annual-report facts for SEC registrants and can include comparative facts from earlier fiscal years. The prepared table includes company-year records and selected US-GAAP metrics. Tag coverage varies across companies, so missing fields are expected.

The model is a `RandomForestRegressor` trained to forecast next-period revenue from current-period Revenue, Net Profit, and Total Assets. Features and target use log transforms to reduce scale differences. Evaluation is based on a company-held-out test split. Revenue values vary greatly across companies; use the metrics as a prototype evaluation, not as a guarantee of future accuracy or financial advice.

To reproduce preprocessing:
```bat
python prepare_sec_dataset.py --input data\sec_zips\2022q1.zip data\sec_zips\2022q2.zip data\sec_zips\2022q3.zip data\sec_zips\2022q4.zip data\sec_zips\2023q1.zip data\sec_zips\2023q2.zip data\sec_zips\2023q3.zip data\sec_zips\2023q4.zip data\sec_zips\2024q1.zip data\sec_zips\2024q2.zip data\sec_zips\2024q3.zip data\sec_zips\2024q4.zip data\sec_zips\2025q1.zip data\sec_zips\2025q2.zip data\sec_zips\2025q3.zip data\sec_zips\2025q4.zip --output data\sec_company_year.csv
```

To retrain:
```bat
python train_model.py --input data\sec_company_year.csv --output models\revenue_forecast_model.joblib --metrics models\training_metrics.json
```

## Windows installation and run
1. Install Python 3.10 or newer.
2. Extract the ZIP to a folder.
3. Open Command Prompt in that folder.
4. Run:
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
5. Open the local URL printed by Streamlit in your browser.
6. Upload a text-based financial PDF, review the extracted table, then click **Analyze this PDF**.

## Project structure
- `app.py` — Streamlit interface.
- `pdf_parser.py` — PDF extraction and parsing.
- `financial_analyzer.py` — ratios, comparisons, anomalies, forecasts, insights.
- `report_generator.py` — PDF report export.
- `prepare_sec_dataset.py` — SEC quarterly ZIP to company-year CSV.
- `train_model.py` — train and evaluate model.
- `data/sec_company_year.csv` — processed SEC training table.
- `models/revenue_forecast_model.joblib` — trained model.
- `models/training_metrics.json` — evaluation results.
- `Project_Report.pdf` and `Project_Presentation.pptx` — academic project documentation.
