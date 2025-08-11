import io
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="SARIMAX Time Series Forecaster",
    page_icon="📈",
    layout="wide",
)


# ----------------------------
# Data Models
# ----------------------------
@dataclass
class SeriesConfig:
    series_id: str
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    enforce_stationarity: bool
    enforce_invertibility: bool


# ----------------------------
# Utilities
# ----------------------------
def parse_month(series: pd.Series) -> pd.DatetimeIndex:
    """Parse YYYY-MM to month period end as Timestamp."""
    parsed = pd.to_datetime(series.astype(str), format="%Y-%m", errors="coerce")
    return parsed


def coerce_monthly_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = parse_month(df[date_col])
    if df[date_col].isna().any():
        raise ValueError("Some Period values could not be parsed. Ensure YYYY-MM format.")
    return df


def ensure_complete_monthly_range(df: pd.DataFrame, date_col: str, series_col: str) -> pd.DataFrame:
    """Reindex each series to complete monthly frequency; keep all columns, fill NaNs."""
    completed_frames: List[pd.DataFrame] = []
    for series_id, sdf in df.groupby(series_col):
        sdf = sdf.sort_values(date_col)
        full_index = pd.date_range(sdf[date_col].min(), sdf[date_col].max(), freq="MS")
        sdf = sdf.set_index(date_col).reindex(full_index)
        sdf.index.name = date_col
        sdf[series_col] = series_id
        completed_frames.append(sdf.reset_index())
    return pd.concat(completed_frames, ignore_index=True)


def clean_missing_values(df: pd.DataFrame, value_col: str, method: str = "ffill_then_interpolate") -> pd.DataFrame:
    df = df.copy()
    if method == "ffill_then_interpolate":
        df[value_col] = df[value_col].ffill()
        df[value_col] = df[value_col].interpolate(method="linear", limit_direction="both")
    elif method == "interpolate_only":
        df[value_col] = df[value_col].interpolate(method="linear", limit_direction="both")
    elif method == "ffill_only":
        df[value_col] = df[value_col].ffill()
    return df


def infer_series_and_exog_columns(df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    candidates = [c for c in df.columns if c.lower() not in {"period", "value"}]
    if not candidates:
        return None, []
    # Heuristic: first non Period/Value is series id; others are exog candidates
    return candidates[0], candidates[1:]


def train_test_split_last_n(
    y: pd.Series,
    exog: Optional[pd.DataFrame],
    test_size: int,
):
    n = len(y)
    # Use last up to 12 months as test, but keep at least 1 train observation
    test_size = max(1, min(test_size, 12, n - 1))
    split_idx = n - test_size
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    if exog is not None:
        x_train, x_test = exog.iloc[:split_idx, :], exog.iloc[split_idx:, :]
    else:
        x_train = x_test = None
    return y_train, y_test, x_train, x_test


def standardize_exog(x_train: Optional[pd.DataFrame], x_test: Optional[pd.DataFrame], x_future: Optional[pd.DataFrame]):
    if x_train is None:
        return None, None, None, None
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
    x_test_scaled = None if x_test is None else pd.DataFrame(
        scaler.transform(x_test), index=x_test.index, columns=x_test.columns
    )
    x_future_scaled = None if x_future is None else pd.DataFrame(
        scaler.transform(x_future), index=x_future.index, columns=x_future.columns
    )
    return scaler, x_train_scaled, x_test_scaled, x_future_scaled


def grid_search_sarimax(
    y_train: pd.Series,
    x_train: Optional[pd.DataFrame],
    p_range: List[int],
    d_range: List[int],
    q_range: List[int],
    P_range: List[int],
    D_range: List[int],
    Q_range: List[int],
    s: int,
    ic: str = "aic",
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], float, Optional[str]]:
    best_ic = np.inf
    best_order = None
    best_seasonal = None
    best_err = None
    for order in itertools.product(p_range, d_range, q_range):
        for seas in itertools.product(P_range, D_range, Q_range):
            seasonal_order = (seas[0], seas[1], seas[2], s)
            try:
                model = SARIMAX(
                    y_train,
                    exog=x_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=enforce_stationarity,
                    enforce_invertibility=enforce_invertibility,
                )
                results = model.fit(disp=False)
                current_ic = getattr(results, ic)
                if current_ic < best_ic:
                    best_ic = current_ic
                    best_order = order
                    best_seasonal = seasonal_order
            except Exception as e:  # noqa: BLE001
                best_err = str(e)
                continue
    if best_order is None:
        raise RuntimeError(f"Grid search failed. Latest error: {best_err}")
    return best_order, best_seasonal, best_ic, best_err


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.nanmean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100)
    return {"MAE": float(mae), "RMSE": rmse, "MAPE": mape}


def make_forecast_plot(
    series_id: str,
    df_series: pd.DataFrame,
    date_col: str,
    value_col: str,
    fitted: pd.Series,
    pred: Optional[pd.Series],
    conf_int: Optional[pd.DataFrame],
    future_forecast: Optional[pd.Series],
    future_conf: Optional[pd.DataFrame],
) -> go.Figure:
    fig = go.Figure()

    # Actuals
    fig.add_trace(
        go.Scatter(
            x=df_series[date_col],
            y=df_series[value_col],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#1f77b4"),
        )
    )

    # Fitted (train)
    if fitted is not None:
        fig.add_trace(
            go.Scatter(
                x=fitted.index,
                y=fitted,
                mode="lines",
                name="Fitted (train)",
                line=dict(color="#2ca02c", dash="dash"),
            )
        )

    # Prediction (test) with CI
    if pred is not None:
        fig.add_trace(
            go.Scatter(
                x=pred.index,
                y=pred,
                mode="lines+markers",
                name="Forecast (test)",
                line=dict(color="#ff7f0e"),
            )
        )
        if conf_int is not None:
            fig.add_trace(
                go.Scatter(
                    x=conf_int.index,
                    y=conf_int.iloc[:, 0],
                    mode="lines",
                    name="Lower CI (test)",
                    line=dict(color="rgba(255,127,14,0.2)"),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=conf_int.index,
                    y=conf_int.iloc[:, 1],
                    mode="lines",
                    name="Upper CI (test)",
                    line=dict(color="rgba(255,127,14,0.2)"),
                    fill="tonexty",
                    fillcolor="rgba(255,127,14,0.1)",
                    showlegend=False,
                )
            )

    # Future forecast with CI
    if future_forecast is not None:
        fig.add_trace(
            go.Scatter(
                x=future_forecast.index,
                y=future_forecast,
                mode="lines+markers",
                name="Forecast (future)",
                line=dict(color="#d62728"),
            )
        )
        if future_conf is not None:
            fig.add_trace(
                go.Scatter(
                    x=future_conf.index,
                    y=future_conf.iloc[:, 0],
                    mode="lines",
                    name="Lower CI (future)",
                    line=dict(color="rgba(214,39,40,0.2)"),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=future_conf.index,
                    y=future_conf.iloc[:, 1],
                    mode="lines",
                    name="Upper CI (future)",
                    line=dict(color="rgba(214,39,40,0.2)"),
                    fill="tonexty",
                    fillcolor="rgba(214,39,40,0.1)",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"Series: {series_id}",
        margin=dict(l=10, r=10, t=40, b=10),
        height=500,
        legend=dict(orientation="h"),
        xaxis_title="Period",
        yaxis_title="Value",
        template="plotly_white",
    )
    return fig


def plot_residual_diagnostics(residuals: pd.Series) -> go.Figure:
    residuals = residuals.dropna()
    lags = min(24, max(2, len(residuals) // 4))

    # Compute ACF and PACF values
    acf_vals = acf(residuals, nlags=lags, fft=False)
    pacf_vals = pacf(residuals, nlags=lags, method="yw")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Residuals", "Histogram", "ACF", "PACF"),
    )

    # Residuals over time
    fig.add_trace(
        go.Scatter(x=residuals.index, y=residuals.values, mode="lines", name="Residuals"),
        row=1,
        col=1,
    )

    # Histogram with KDE
    kde_x = np.linspace(residuals.min(), residuals.max(), 200)
    kde = stats.gaussian_kde(residuals)
    fig.add_trace(
        go.Histogram(x=residuals.values, nbinsx=30, name="Histogram", marker_color="#1f77b4"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=kde_x, y=kde(kde_x), mode="lines", name="KDE", line=dict(color="#d62728")),
        row=1,
        col=2,
    )

    # ACF
    fig.add_trace(
        go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF"),
        row=2,
        col=1,
    )
    # PACF
    fig.add_trace(
        go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF"),
        row=2,
        col=2,
    )

    fig.update_layout(height=700, showlegend=False, template="plotly_white")
    return fig


def ljung_box_table(residuals: pd.Series, lags: int = 12) -> pd.DataFrame:
    residuals = residuals.dropna()
    if len(residuals) < 3:
        return pd.DataFrame({"LB Stat": [], "p-value": []})
    lag = max(1, min(lags, len(residuals) - 1))
    lb = acorr_ljungbox(residuals, lags=[lag], return_df=True)
    lb.rename(columns={"lb_stat": "LB Stat", "lb_pvalue": "p-value"}, inplace=True)
    return lb


def build_download(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")


# ----------------------------
# UI
# ----------------------------
st.title("📈 SARIMAX Time Series Forecaster")
st.write(
    "Upload a monthly time series CSV with columns `Period` (YYYY-MM), `Value`, a series id column, and optional exogenous regressors."
)

with st.sidebar:
    st.header("1) Data")
    uploaded = st.file_uploader("Upload CSV (Period, Value, Series, exogenous columns)", type=["csv"])
    future_exog_upload = st.file_uploader(
        "Optional: Future Exogenous CSV (Period + series id + exog columns)", type=["csv"], help="Required when using exogenous variables for out-of-sample forecasting"
    )

    st.header("2) Preprocessing")
    missing_method = st.selectbox(
        "Missing value handling",
        ["ffill_then_interpolate", "interpolate_only", "ffill_only"],
        index=0,
        help="Applied per series after ensuring monthly frequency",
    )
    standardize = st.checkbox("Standardize exogenous variables", value=True)

    st.header("3) Model")
    auto_mode = st.checkbox("Auto SARIMAX search (p,d,q)(P,D,Q)s", value=True)
    ic = st.radio("Selection criterion", ["aic", "bic"], index=0, horizontal=True)
    s = st.number_input("Seasonal period s", min_value=1, value=12, step=1)
    enforce_stationarity = st.checkbox("Enforce stationarity", value=True)
    enforce_invertibility = st.checkbox("Enforce invertibility", value=True)
    horizon = st.number_input("Forecast horizon (months)", min_value=1, value=12, step=1)

    if auto_mode:
        st.caption("Search ranges (inclusive)")
        max_p = st.slider("max p", 0, 3, 2)
        max_d = st.slider("max d", 0, 2, 1)
        max_q = st.slider("max q", 0, 3, 2)
        max_P = st.slider("max P", 0, 2, 1)
        max_D = st.slider("max D", 0, 1, 1)
        max_Q = st.slider("max Q", 0, 2, 1)
    else:
        p = st.number_input("p", 0, 5, 1)
        d = st.number_input("d", 0, 2, 1)
        q = st.number_input("q", 0, 5, 1)
        P = st.number_input("P", 0, 5, 1)
        D = st.number_input("D", 0, 2, 1)
        Q = st.number_input("Q", 0, 5, 1)

    st.header("4) Run")
    run_button = st.button("Run Modeling")


def read_and_prepare(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, str, List[str]]:
    df = df_raw.copy()
    required_cols = {"Period", "Value"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError("CSV must contain columns: Period, Value")

    df = coerce_monthly_index(df, "Period")
    inferred_series_col, exog_cands = infer_series_and_exog_columns(df)

    # Let user pick columns explicitly, handle case with no series column by creating one
    st.subheader("Column Mapping")
    cols = list(df.columns)
    series_options = [c for c in cols if c not in {"Period", "Value"}]
    if len(series_options) == 0:
        # No series column present; create a default constant series id
        df["Series"] = "Series_1"
        cols = list(df.columns)
        series_options = ["Series"]
        inferred_series_col = "Series"
        st.info("No series ID column detected; using a default 'Series' column.")

    # Series selector
    default_index = 0
    if inferred_series_col in series_options:
        default_index = series_options.index(inferred_series_col)
    series_col = st.selectbox(
        "Series ID column",
        options=series_options,
        index=default_index,
        help="Identifier for multiple series",
    )

    # Exogenous selector
    exog_options = [c for c in cols if c not in {"Period", "Value", series_col}]
    exog_default = [c for c in exog_cands if c in exog_options]
    exog_cols = st.multiselect(
        "Exogenous columns (optional)",
        options=exog_options,
        default=exog_default,
    )

    df = ensure_complete_monthly_range(df[["Period", "Value", series_col] + exog_cols], "Period", series_col)
    df = clean_missing_values(df, "Value")

    # Fill missing exog values similarly
    for c in exog_cols:
        if df[c].isna().any():
            df[c] = df[c].ffill().bfill()

    return df, series_col, exog_cols


def align_future_exog(
    future_df_raw: Optional[pd.DataFrame],
    series_ids: List[str],
    series_col: str,
    exog_cols: List[str],
    last_period: pd.Timestamp,
    horizon: int,
) -> Optional[pd.DataFrame]:
    if not exog_cols:
        return None
    if future_df_raw is None:
        return None
    fdf = future_df_raw.copy()
    if not {"Period", series_col}.issubset(fdf.columns):
        st.warning("Future exogenous CSV must include Period and the series id column.")
        return None
    fdf = coerce_monthly_index(fdf, "Period")
    # Keep only needed columns
    fdf = fdf[["Period", series_col] + [c for c in exog_cols if c in fdf.columns]]
    # Build required future index per series
    all_frames = []
    for sid in series_ids:
        start = (last_period + pd.offsets.MonthBegin(1)).normalize().replace(day=1)
        future_index = pd.date_range(start=start, periods=horizon, freq="MS")
        sdf = fdf[fdf[series_col] == sid].set_index("Period").reindex(future_index)
        sdf.index.name = "Period"
        sdf[series_col] = sid
        all_frames.append(sdf.reset_index())
    merged = pd.concat(all_frames, ignore_index=True)
    # If missing any exog value warn
    missing = merged[exog_cols].isna().any().any()
    if missing:
        st.warning("Some future exogenous values are missing; fill them to enable out-of-sample forecasting.")
    return merged


def build_results_table(
    series_id: str,
    date_col: str,
    value_col: str,
    series_col: str,
    df_series: pd.DataFrame,
    fitted: pd.Series,
    pred: Optional[pd.Series],
    pred_conf: Optional[pd.DataFrame],
    future_forecast: Optional[pd.Series],
    future_conf: Optional[pd.DataFrame],
) -> pd.DataFrame:
    records = []
    # Actuals
    for t, v in zip(df_series[date_col], df_series[value_col]):
        records.append({
            "Series": series_id,
            "Period": t,
            "Split": "actual",
            "Actual": v,
            "Fitted": np.nan,
            "Forecast": np.nan,
            "LowerCI": np.nan,
            "UpperCI": np.nan,
        })

    if fitted is not None:
        for t, v in fitted.items():
            records.append({
                "Series": series_id,
                "Period": t,
                "Split": "train_fitted",
                "Actual": np.nan,
                "Fitted": v,
                "Forecast": np.nan,
                "LowerCI": np.nan,
                "UpperCI": np.nan,
            })

    if pred is not None:
        for t in pred.index:
            lc = pred_conf.loc[t].iloc[0] if pred_conf is not None else np.nan
            uc = pred_conf.loc[t].iloc[1] if pred_conf is not None else np.nan
            records.append({
                "Series": series_id,
                "Period": t,
                "Split": "test_forecast",
                "Actual": np.nan,
                "Fitted": np.nan,
                "Forecast": pred.loc[t],
                "LowerCI": lc,
                "UpperCI": uc,
            })

    if future_forecast is not None:
        for t in future_forecast.index:
            lc = future_conf.loc[t].iloc[0] if future_conf is not None else np.nan
            uc = future_conf.loc[t].iloc[1] if future_conf is not None else np.nan
            records.append({
                "Series": series_id,
                "Period": t,
                "Split": "future_forecast",
                "Actual": np.nan,
                "Fitted": np.nan,
                "Forecast": future_forecast.loc[t],
                "LowerCI": lc,
                "UpperCI": uc,
            })

    out = pd.DataFrame.from_records(records)
    out.sort_values(["Series", "Period", "Split"], inplace=True)
    return out


if uploaded is not None and run_button:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    try:
        df, series_col, exog_cols = read_and_prepare(raw_df)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        st.stop()

    all_series = sorted(df[series_col].astype(str).unique().tolist())
    st.subheader("Series Selection")
    selected_series = st.multiselect("Select series to model", all_series, default=all_series[: min(5, len(all_series))])
    if not selected_series:
        st.warning("Select at least one series.")
        st.stop()

    # Optional future exog alignment if needed
    last_period = df["Period"].max()
    future_exog_df = align_future_exog(future_exog_upload and pd.read_csv(future_exog_upload), selected_series, series_col, exog_cols, last_period, horizon)

    results_frames: List[pd.DataFrame] = []
    metrics_rows: List[Dict[str, object]] = []

    for sid in selected_series:
        st.markdown(f"### Series: `{sid}`")
        sdf = df[df[series_col] == sid].sort_values("Period")

        y = sdf.set_index("Period")["Value"].astype(float)
        X = sdf.set_index("Period")[exog_cols] if exog_cols else None

        y_train, y_test, x_train, x_test = train_test_split_last_n(y, X, test_size=12)

        scaler = None
        x_future = None
        if future_exog_df is not None:
            fdf = future_exog_df[future_exog_df[series_col] == sid].set_index("Period")
            # Ensure all selected exog present
            missing_cols = [c for c in exog_cols if c not in fdf.columns]
            if missing_cols:
                st.warning(f"Future exog missing columns {missing_cols} for series {sid}.")
            else:
                x_future = fdf[exog_cols].iloc[: horizon]

        if standardize:
            scaler, x_train_s, x_test_s, x_future_s = standardize_exog(x_train, x_test, x_future)
            x_train, x_test, x_future = x_train_s, x_test_s, x_future_s

        # Auto or manual configuration
        if auto_mode:
            p_range = list(range(0, max_p + 1))
            d_range = list(range(0, max_d + 1))
            q_range = list(range(0, max_q + 1))
            P_range = list(range(0, max_P + 1))
            D_range = list(range(0, max_D + 1))
            Q_range = list(range(0, max_Q + 1))
            with st.spinner("Searching best SARIMAX hyperparameters..."):
                try:
                    order, seasonal_order, best_ic, _ = grid_search_sarimax(
                        y_train,
                        x_train,
                        p_range,
                        d_range,
                        q_range,
                        P_range,
                        D_range,
                        Q_range,
                        s,
                        ic=ic,
                        enforce_stationarity=enforce_stationarity,
                        enforce_invertibility=enforce_invertibility,
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Grid search failed: {e}")
                    st.stop()
        else:
            order = (int(p), int(d), int(q))
            seasonal_order = (int(P), int(D), int(Q), int(s))
            best_ic = np.nan

        st.write(f"Using order={order}, seasonal_order={seasonal_order} ({ic.upper()}={best_ic:.2f} if auto)")

        # Fit on train
        with st.spinner("Fitting model on training data..."):
            model = SARIMAX(
                y_train,
                exog=x_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
            )
            results = model.fit(disp=False)

        fitted = results.fittedvalues

        # Predict on test
        pred = None
        pred_ci = None
        if len(y_test) > 0:
            try:
                pr = results.get_forecast(steps=len(y_test), exog=x_test)
                pred = pr.predicted_mean
                pred_ci = pr.conf_int(alpha=0.05)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Test prediction failed: {e}")

        # Metrics
        if pred is not None and len(pred) == len(y_test):
            metrics = compute_metrics(y_test, pred)
            metrics_rows.append({"Series": sid, **metrics})
            st.write({k: round(v, 3) for k, v in metrics.items()})
        else:
            st.info("Insufficient test window for metrics.")

        # Refit on full data for future forecast
        future_forecast = None
        future_ci = None
        if horizon > 0:
            if exog_cols and (x_future is None or x_future.isna().any().any()):
                st.warning("Provide complete future exogenous values to enable out-of-sample forecast.")
            else:
                with st.spinner("Refitting on full data and forecasting future horizon..."):
                    # Prepare full-data exogenous (consistent scaling)
                    X_full_use = None
                    future_exog_use = None
                    if X is not None:
                        if standardize:
                            scaler_full = StandardScaler().fit(X)
                            X_full_use = pd.DataFrame(
                                scaler_full.transform(X), index=X.index, columns=X.columns
                            )
                            if x_future is not None:
                                future_exog_use = pd.DataFrame(
                                    scaler_full.transform(x_future), index=x_future.index, columns=x_future.columns
                                )
                        else:
                            X_full_use = X
                            future_exog_use = x_future

                    model_full = SARIMAX(
                        y,
                        exog=X_full_use,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=enforce_stationarity,
                        enforce_invertibility=enforce_invertibility,
                    )
                    res_full = model_full.fit(disp=False)
                    if future_exog_use is not None:
                        prf = res_full.get_forecast(steps=horizon, exog=future_exog_use)
                    else:
                        prf = res_full.get_forecast(steps=horizon)
                    future_forecast = prf.predicted_mean
                    future_ci = prf.conf_int(alpha=0.05)

        # Plot
        fig = make_forecast_plot(sid, sdf, "Period", "Value", fitted, pred, pred_ci, future_forecast, future_ci)
        st.plotly_chart(fig, use_container_width=True)

        # Residual diagnostics
        st.markdown("#### Residual Diagnostics")
        resid = results.resid
        diag_fig = plot_residual_diagnostics(resid)
        st.plotly_chart(diag_fig, use_container_width=True)

        lb_tbl = ljung_box_table(resid, lags=12)
        st.dataframe(lb_tbl)

        # Export table for this series
        table = build_results_table(sid, "Period", "Value", series_col, sdf, fitted, pred, pred_ci, future_forecast, future_ci)
        results_frames.append(table)
        build_download(table, f"results_{sid}.csv", f"Download CSV ({sid})")

    # Combined export
    if results_frames:
        combined = pd.concat(results_frames, ignore_index=True)
        st.subheader("Export All")
        build_download(combined, "results_all_series.csv", "Download All Results (CSV)")

else:
    st.info("Upload a CSV and click Run Modeling to begin.")


