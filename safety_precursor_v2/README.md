# Safety Precursor Intelligence System v2.0

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create/activate a virtual environment (recommended).
4. Install dependencies:
   `pip install -r requirements.txt`
5. Run:
   `streamlit run app.py`

## Included
- Streamlit multi-page navigation in one app
- TF-IDF + Logistic Regression text classifier
- Synthetic demonstration corpus (clearly labelled)
- Risk screening score and recommended actions
- SQLite persistence for submitted reports
- Live dashboard, audit search/export, site summaries and threshold alerts
- Held-out metrics, per-class F1, confusion matrix and keyword baseline

## Important validation note
The seeded records and model training corpus are synthetic. The evaluation metrics demonstrate the pipeline only and are not evidence of real-world predictive performance. Do not use this prototype as a safety-critical decision system. Validate with manually labelled, representative site data and approved HSE procedures before operational use.

## Data
The app creates `safety_precursor.db` in the working directory on first run. Seed demo reports are inserted only when the database is empty. User-submitted reports are saved persistently in SQLite.
