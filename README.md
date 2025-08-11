## SARIMAX Time Series Forecaster (Streamlit)

This app provides a modern UI to load monthly time series data, fit SARIMAX models (with optional exogenous regressors), run auto hyperparameter search, visualize forecasts, review diagnostics, and export results to CSV.

### Features
- Streamlit + Plotly interactive UI
- CSV upload with columns: `Period` (YYYY-MM), `Value`, a series id column (e.g., `Series`), and optional exogenous columns
- Multi-series selection from a single CSV
- Monthly frequency handling and missing data cleaning (forward-fill + interpolation)
- Auto SARIMAX grid search using AIC/BIC with manual override
- Train/test split (last 12 months) with MAE, RMSE, MAPE
- Residual diagnostics: residual plots, histogram + KDE, ACF/PACF, Ljung–Box test
- 12 month forecast horizon by default (configurable)
- Optional future exogenous CSV for out-of-sample forecasting
- Export per-series or combined results to CSV

### Installation
1) Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run streamlit_app.py
```
Then open the local URL shown in the terminal (usually `http://localhost:8501`).

### Data format
- The main CSV must include:
  - `Period`: monthly in `YYYY-MM` format
  - `Value`: numeric target
  - `Series` (or any series id column name): used to select one or more series to model
  - Optional exogenous columns: any additional numeric columns

Example (wide):

```csv
Period,Series,Value,Promo,Price
2021-01,Series 001,120,1,9.99
2021-02,Series 001,132,0,10.49
2021-03,Series 001,110,0,10.39
...
```

#### Future exogenous CSV (optional)
If you select exogenous variables and want a future forecast, you must provide a second CSV for the forecast horizon, with columns `Period`, the same series id column, and the same exogenous columns. The app aligns periods and validates completeness.

```csv
Period,Series,Promo,Price
2024-01,Series 001,0,10.99
2024-02,Series 001,1,10.89
...
```

### How to use
1. Upload the main CSV in the sidebar.
2. Map the `Series` id column and choose exogenous columns.
3. Configure missing value handling, standardization of exogenous variables, auto/manual SARIMAX, selection criterion (AIC/BIC), seasonal period `s` (default 12), and horizon.
4. Click "Run Modeling".
5. For each selected series:
   - View fitted vs. actuals, test forecasts with confidence intervals, and future forecasts
   - Review metrics on the last 12 months (MAE, RMSE, MAPE)
   - Inspect residual diagnostics and Ljung–Box results
   - Download per-series CSV; or use the combined export at the end

### Notes
- With fewer than ~24 observations, the grid search is restricted via sliders. You can relax or tighten ranges.
- If using exogenous variables, ensure the same set of exog columns is present in both the main CSV and the future exogenous CSV.
- Exogenous variables can be standardized (recommended) for numerical stability.
- For very small series, the test split or diagnostics may be limited.

### Troubleshooting
- "Some Period values could not be parsed": Ensure `Period` is in `YYYY-MM` format.
- "Provide complete future exogenous values": Supply the second CSV with all exog columns filled for each future month in the horizon.
- If the grid search fails often, narrow the ranges or toggle off `enforce_stationarity` / `enforce_invertibility`.

### License
MIT


