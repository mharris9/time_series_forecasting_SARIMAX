import io
import itertools
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

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


def coerce_flexible_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse flexible date formats and return dataframe with datetime column."""
    df = df.copy()
    df[date_col] = parse_flexible_dates(df[date_col])
    if df[date_col].isna().any():
        raise ValueError("Some date values could not be parsed. Please check date format.")
    return df


def coerce_monthly_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Legacy function - now uses flexible date parsing."""
    return coerce_flexible_dates(df, date_col)


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


# ----------------------------
# Frequency Detection and Conversion
# ----------------------------

def parse_flexible_dates(series: pd.Series) -> pd.DatetimeIndex:
    """Parse various date formats to datetime index."""
    # Try common date formats
    date_formats = [
        "%Y-%m",           # 2023-01
        "%Y-%m-%d",        # 2023-01-15
        "%m/%d/%Y",        # 3/15/2023
        "%m/%d/%y",        # 3/15/23
        "%d/%m/%Y",        # 15/3/2023 (European)
        "%d/%m/%y",        # 15/3/23
        "%Y/%m/%d",        # 2023/3/15
        "%Y-%m-%d %H:%M:%S",  # 2023-01-15 12:30:00
        "%Y-%m-%d %H:%M",     # 2023-01-15 12:30
    ]
    
    # First try pandas' flexible parsing
    try:
        parsed = pd.to_datetime(series, infer_datetime_format=True, errors='coerce')
        if not parsed.isna().any():
            return parsed
    except:
        pass
    
    # Try specific formats
    for fmt in date_formats:
        try:
            parsed = pd.to_datetime(series.astype(str), format=fmt, errors='coerce')
            if not parsed.isna().any():
                return parsed
        except:
            continue
    
    # Last resort: try to parse each format individually and take the one with fewest NaNs
    best_parsed = None
    min_nas = len(series)
    
    for fmt in date_formats:
        try:
            parsed = pd.to_datetime(series.astype(str), format=fmt, errors='coerce')
            na_count = parsed.isna().sum()
            if na_count < min_nas:
                min_nas = na_count
                best_parsed = parsed
        except:
            continue
    
    if best_parsed is not None and min_nas < len(series) * 0.5:  # Accept if <50% NaNs
        return best_parsed
    
    # Final fallback
    return pd.to_datetime(series, errors='coerce')


def detect_frequency(dates: pd.DatetimeIndex) -> Tuple[str, str]:
    """Detect the frequency of a datetime index.
    
    Returns:
        Tuple of (frequency_code, frequency_name)
    """
    if len(dates) < 2:
        return 'D', 'Daily'
    
    # Sort dates to ensure proper diff calculation
    dates_sorted = dates.sort_values()
    diffs = dates_sorted.diff().dropna()
    
    # Get the most common difference
    mode_diff = diffs.mode()
    if len(mode_diff) == 0:
        return 'D', 'Daily'
    
    most_common_diff = mode_diff[0]
    days = most_common_diff.days
    
    # Classify frequency based on most common difference
    if days == 1:
        return 'D', 'Daily'
    elif 6 <= days <= 8:  # Account for weekends
        return 'W', 'Weekly'
    elif 28 <= days <= 32:  # Account for varying month lengths
        return 'M', 'Monthly'
    elif 89 <= days <= 93:  # Quarterly (roughly 91 days)
        return 'Q', 'Quarterly'
    elif 360 <= days <= 370:  # Yearly
        return 'Y', 'Yearly'
    else:
        # For irregular frequencies, default to daily
        return 'D', 'Daily'


def get_frequency_hierarchy() -> Dict[str, int]:
    """Get frequency hierarchy for comparison (higher number = higher frequency)."""
    return {
        'Y': 1,   # Yearly
        'Q': 4,   # Quarterly  
        'M': 12,  # Monthly
        'W': 52,  # Weekly
        'D': 365  # Daily
    }


def recommend_base_frequency(frequencies: Dict[str, str]) -> str:
    """Recommend base frequency from detected frequencies."""
    hierarchy = get_frequency_hierarchy()
    
    # Get the most common frequency
    freq_counts = {}
    for freq_code in frequencies.values():
        freq_counts[freq_code] = freq_counts.get(freq_code, 0) + 1
    
    if not freq_counts:
        return 'M'  # Default to monthly
    
    # Return the most common frequency
    return max(freq_counts.items(), key=lambda x: x[1])[0]


def get_conversion_methods() -> Dict[str, Dict[str, List[str]]]:
    """Get available conversion methods for upsampling and downsampling."""
    return {
        'upsample': {
            'interpolation': ['linear', 'cubic', 'nearest'],
            'fill': ['forward_fill', 'backward_fill', 'repeat']
        },
        'downsample': {
            'aggregation': ['mean', 'sum', 'median', 'last', 'first', 'min', 'max']
        }
    }


def get_recommended_conversion_method(variable_name: str, is_target: bool = False) -> Dict[str, str]:
    """Recommend conversion methods based on variable characteristics."""
    variable_lower = variable_name.lower()
    
    # Default recommendations
    defaults = {
        'upsample_method': 'linear',
        'downsample_method': 'mean'
    }
    
    # Target variable usually benefits from interpolation
    if is_target:
        defaults['upsample_method'] = 'linear'
        defaults['downsample_method'] = 'mean'
        return defaults
    
    # Specific recommendations based on variable name patterns
    if any(word in variable_lower for word in ['price', 'rate', 'temperature', 'score', 'index']):
        defaults['upsample_method'] = 'linear'
        defaults['downsample_method'] = 'mean'
    elif any(word in variable_lower for word in ['sales', 'revenue', 'volume', 'count', 'quantity']):
        defaults['upsample_method'] = 'repeat'
        defaults['downsample_method'] = 'sum'
    elif any(word in variable_lower for word in ['binary', 'flag', 'indicator', 'status']):
        defaults['upsample_method'] = 'forward_fill'
        defaults['downsample_method'] = 'last'
    
    return defaults


def convert_frequency(df: pd.DataFrame, date_col: str, target_freq: str, 
                     conversion_methods: Dict[str, Dict[str, str]] = None) -> pd.DataFrame:
    """Convert dataframe to target frequency."""
    if df.empty:
        return df
    
    df = df.copy()
    df = df.sort_values(date_col)
    
    # Set date column as index
    df_indexed = df.set_index(date_col)
    
    # Determine if we need upsampling or downsampling
    current_freq, _ = detect_frequency(df_indexed.index)
    hierarchy = get_frequency_hierarchy()
    
    target_hierarchy = hierarchy.get(target_freq, 12)
    current_hierarchy = hierarchy.get(current_freq, 12)
    
    # Create target frequency string for pandas
    freq_map = {
        'D': 'D',
        'W': 'W',
        'M': 'MS',  # Month start
        'Q': 'QS',  # Quarter start
        'Y': 'YS'   # Year start
    }
    
    pandas_freq = freq_map.get(target_freq, 'MS')
    
    if target_hierarchy > current_hierarchy:
        # Upsampling (lower to higher frequency)
        return _upsample_dataframe(df_indexed, pandas_freq, conversion_methods)
    elif target_hierarchy < current_hierarchy:
        # Downsampling (higher to lower frequency)
        return _downsample_dataframe(df_indexed, pandas_freq, conversion_methods)
    else:
        # Same frequency, just ensure regular intervals
        return _regularize_frequency(df_indexed, pandas_freq)


def _upsample_dataframe(df: pd.DataFrame, target_freq: str, 
                       conversion_methods: Dict[str, Dict[str, str]] = None) -> pd.DataFrame:
    """Upsample dataframe to higher frequency."""
    # Create full date range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=target_freq)
    df_reindexed = df.reindex(full_range)
    
    # Apply conversion methods per column
    for col in df_reindexed.columns:
        method_info = conversion_methods.get(col, {}) if conversion_methods else {}
        method = method_info.get('upsample_method', 'linear')
        
        if method == 'linear':
            df_reindexed[col] = df_reindexed[col].interpolate(method='linear')
        elif method == 'cubic':
            df_reindexed[col] = df_reindexed[col].interpolate(method='cubic')
        elif method == 'nearest':
            df_reindexed[col] = df_reindexed[col].interpolate(method='nearest')
        elif method == 'forward_fill':
            df_reindexed[col] = df_reindexed[col].fillna(method='ffill')
        elif method == 'backward_fill':
            df_reindexed[col] = df_reindexed[col].fillna(method='bfill')
        elif method == 'repeat':
            df_reindexed[col] = df_reindexed[col].fillna(method='ffill')
    
    return df_reindexed.reset_index().rename(columns={'index': df.index.name or 'Period'})


def _downsample_dataframe(df: pd.DataFrame, target_freq: str,
                         conversion_methods: Dict[str, Dict[str, str]] = None) -> pd.DataFrame:
    """Downsample dataframe to lower frequency."""
    # Group by target frequency and apply aggregation methods
    grouped = df.groupby(pd.Grouper(freq=target_freq))
    
    result_dfs = []
    for col in df.columns:
        method_info = conversion_methods.get(col, {}) if conversion_methods else {}
        method = method_info.get('downsample_method', 'mean')
        
        if method == 'mean':
            col_result = grouped[col].mean()
        elif method == 'sum':
            col_result = grouped[col].sum()
        elif method == 'median':
            col_result = grouped[col].median()
        elif method == 'last':
            col_result = grouped[col].last()
        elif method == 'first':
            col_result = grouped[col].first()
        elif method == 'min':
            col_result = grouped[col].min()
        elif method == 'max':
            col_result = grouped[col].max()
        else:
            col_result = grouped[col].mean()  # Default fallback
        
        result_dfs.append(col_result)
    
    # Combine results
    result = pd.concat(result_dfs, axis=1)
    result = result.dropna(how='all')  # Remove periods with no data
    
    return result.reset_index().rename(columns={'index': df.index.name or 'Period'})


def _regularize_frequency(df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    """Regularize frequency without changing it."""
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=target_freq)
    df_reindexed = df.reindex(full_range)
    
    # Forward fill missing values
    df_reindexed = df_reindexed.fillna(method='ffill')
    
    return df_reindexed.reset_index().rename(columns={'index': df.index.name or 'Period'})


def analyze_frequency_conversion_impact(df_before: pd.DataFrame, df_after: pd.DataFrame, 
                                      value_cols: List[str]) -> Dict[str, Dict[str, Union[str, float]]]:
    """Analyze the impact of frequency conversion."""
    impact = {}
    
    for col in value_cols:
        if col in df_before.columns and col in df_after.columns:
            before_vals = df_before[col].dropna()
            after_vals = df_after[col].dropna()
            
            impact[col] = {
                'data_points_before': len(before_vals),
                'data_points_after': len(after_vals),
                'data_point_change': len(after_vals) - len(before_vals),
                'mean_before': before_vals.mean() if len(before_vals) > 0 else np.nan,
                'mean_after': after_vals.mean() if len(after_vals) > 0 else np.nan,
                'std_before': before_vals.std() if len(before_vals) > 0 else np.nan,
                'std_after': after_vals.std() if len(after_vals) > 0 else np.nan,
            }
            
            # Calculate percentage change in statistics
            if not np.isnan(impact[col]['mean_before']) and impact[col]['mean_before'] != 0:
                mean_pct_change = ((impact[col]['mean_after'] - impact[col]['mean_before']) / 
                                 abs(impact[col]['mean_before'])) * 100
                impact[col]['mean_change_pct'] = mean_pct_change
            else:
                impact[col]['mean_change_pct'] = 0
    
    return impact


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
    
    # Frequency Normalization Section
    st.subheader("Frequency Normalization")
    enable_freq_norm = st.checkbox(
        "Enable frequency normalization", 
        value=False,
        help="Automatically detect and normalize data frequencies across all variables"
    )
    
    if enable_freq_norm:
        freq_strategy = st.radio(
            "Frequency conversion strategy",
            ["auto_detect", "manual_select", "highest_frequency", "lowest_frequency"],
            format_func=lambda x: {
                "auto_detect": "Auto-detect recommended frequency",
                "manual_select": "Manually select target frequency", 
                "highest_frequency": "Convert to highest frequency",
                "lowest_frequency": "Convert to lowest frequency"
            }[x],
            index=0,
            help="Strategy for determining the target frequency"
        )
        
        if freq_strategy == "manual_select":
            manual_frequency = st.selectbox(
                "Target frequency",
                ["D", "W", "M", "Q", "Y"],
                format_func=lambda x: {
                    "D": "Daily",
                    "W": "Weekly", 
                    "M": "Monthly",
                    "Q": "Quarterly",
                    "Y": "Yearly"
                }[x],
                index=2  # Default to Monthly
            )
        
        show_conversion_admin = st.checkbox(
            "Advanced: Customize conversion methods per variable",
            value=False,
            help="Configure upsampling/downsampling methods for each variable"
        )
    
    # Traditional preprocessing options
    st.subheader("Missing Values & Scaling")
    missing_method = st.selectbox(
        "Missing value handling",
        ["ffill_then_interpolate", "interpolate_only", "ffill_only"],
        index=0,
        help="Applied per series after frequency normalization",
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


def read_and_prepare(df_raw: pd.DataFrame, enable_freq_norm: bool = False, 
                    freq_strategy: str = "auto_detect", manual_frequency: str = "M",
                    show_conversion_admin: bool = False) -> Tuple[pd.DataFrame, str, List[str], Dict]:
    df = df_raw.copy()
    required_cols = {"Period", "Value"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError("CSV must contain columns: Period, Value")

    # Parse dates with flexible format support
    df = coerce_flexible_dates(df, "Period")
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

    # Initialize frequency normalization info
    freq_info = {}
    
    # Frequency normalization
    if enable_freq_norm:
        st.subheader("Frequency Analysis")
        
        # Detect frequencies for each series
        detected_frequencies = {}
        all_series = df[series_col].unique()
        
        for series_id in all_series:
            series_data = df[df[series_col] == series_id].sort_values("Period")
            freq_code, freq_name = detect_frequency(series_data["Period"])
            detected_frequencies[series_id] = freq_code
            
        # Show detected frequencies
        freq_summary = {}
        for freq_code in set(detected_frequencies.values()):
            freq_name = {
                'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 
                'Q': 'Quarterly', 'Y': 'Yearly'
            }.get(freq_code, freq_code)
            series_with_freq = [s for s, f in detected_frequencies.items() if f == freq_code]
            freq_summary[freq_name] = len(series_with_freq)
        
        st.write("**Detected Frequencies:**")
        for freq_name, count in freq_summary.items():
            st.write(f"- {freq_name}: {count} series")
        
        # Determine target frequency
        if freq_strategy == "auto_detect":
            target_freq = recommend_base_frequency(detected_frequencies)
            recommended_name = {
                'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly',
                'Q': 'Quarterly', 'Y': 'Yearly'
            }.get(target_freq, target_freq)
            st.info(f"Recommended target frequency: **{recommended_name}**")
        elif freq_strategy == "manual_select":
            target_freq = manual_frequency
        elif freq_strategy == "highest_frequency":
            hierarchy = get_frequency_hierarchy()
            target_freq = max(detected_frequencies.values(), key=lambda x: hierarchy.get(x, 0))
        else:  # lowest_frequency
            hierarchy = get_frequency_hierarchy()
            target_freq = min(detected_frequencies.values(), key=lambda x: hierarchy.get(x, 999))
        
        # Show conversion methods admin panel
        conversion_methods = {}
        if show_conversion_admin:
            st.subheader("Conversion Methods Configuration")
            
            all_variables = ["Value"] + exog_cols
            for var in all_variables:
                st.write(f"**{var}**")
                is_target = (var == "Value")
                recommended = get_recommended_conversion_method(var, is_target)
                
                col1, col2 = st.columns(2)
                with col1:
                    upsample_method = st.selectbox(
                        f"Upsampling method",
                        ["linear", "cubic", "nearest", "forward_fill", "backward_fill", "repeat"],
                        index=["linear", "cubic", "nearest", "forward_fill", "backward_fill", "repeat"].index(
                            recommended['upsample_method']
                        ),
                        key=f"upsample_{var}",
                        help="Method used when increasing frequency (e.g., monthly to daily)"
                    )
                
                with col2:
                    downsample_method = st.selectbox(
                        f"Downsampling method",
                        ["mean", "sum", "median", "last", "first", "min", "max"],
                        index=["mean", "sum", "median", "last", "first", "min", "max"].index(
                            recommended['downsample_method']
                        ),
                        key=f"downsample_{var}",
                        help="Method used when decreasing frequency (e.g., daily to monthly)"
                    )
                
                conversion_methods[var] = {
                    'upsample_method': upsample_method,
                    'downsample_method': downsample_method
                }
        else:
            # Use recommended methods
            all_variables = ["Value"] + exog_cols
            for var in all_variables:
                is_target = (var == "Value")
                conversion_methods[var] = get_recommended_conversion_method(var, is_target)
        
        # Perform frequency conversion
        if target_freq != list(detected_frequencies.values())[0] or len(set(detected_frequencies.values())) > 1:
            st.info(f"Converting all data to {target_freq} frequency...")
            
            # Convert each series separately
            converted_dfs = []
            conversion_impact = {}
            
            for series_id in all_series:
                series_data = df[df[series_col] == series_id][["Period", "Value", series_col] + exog_cols]
                
                # Analyze impact before conversion
                before_stats = analyze_frequency_conversion_impact(
                    series_data, series_data, ["Value"] + exog_cols
                )
                
                # Convert frequency
                converted_series = convert_frequency(
                    series_data, "Period", target_freq, conversion_methods
                )
                
                # Analyze impact after conversion
                after_stats = analyze_frequency_conversion_impact(
                    series_data, converted_series, ["Value"] + exog_cols
                )
                
                conversion_impact[series_id] = after_stats
                converted_dfs.append(converted_series)
            
            # Combine converted series
            df = pd.concat(converted_dfs, ignore_index=True)
            
            # Show conversion impact summary
            st.subheader("Frequency Conversion Impact")
            
            # Check for significant changes
            warnings = []
            for series_id, impact in conversion_impact.items():
                for var, stats in impact.items():
                    if abs(stats.get('mean_change_pct', 0)) > 5:  # >5% change in mean
                        warnings.append(f"Series {series_id}, Variable {var}: {stats['mean_change_pct']:.1f}% change in mean")
                    
                    data_change = stats.get('data_point_change', 0)
                    if data_change != 0:
                        change_type = "increase" if data_change > 0 else "decrease"
                        warnings.append(f"Series {series_id}, Variable {var}: {abs(data_change)} data points {change_type}")
            
            if warnings:
                st.warning("**Information Loss/Change Detected:**")
                for warning in warnings[:5]:  # Show first 5 warnings
                    st.write(f"- {warning}")
                if len(warnings) > 5:
                    st.write(f"... and {len(warnings) - 5} more warnings")
            else:
                st.success("Frequency conversion completed with minimal impact on data statistics.")
        
        freq_info = {
            'enabled': True,
            'target_frequency': target_freq,
            'detected_frequencies': detected_frequencies,
            'conversion_methods': conversion_methods,
            'conversion_impact': conversion_impact if 'conversion_impact' in locals() else {}
        }
    else:
        # Legacy behavior - ensure complete monthly range
        df = ensure_complete_monthly_range(df[["Period", "Value", series_col] + exog_cols], "Period", series_col)
        freq_info = {'enabled': False}

    # Clean missing values
    df = clean_missing_values(df, "Value")

    # Fill missing exog values similarly
    for c in exog_cols:
        if df[c].isna().any():
            df[c] = df[c].ffill().bfill()

    return df, series_col, exog_cols, freq_info


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
    fdf = coerce_flexible_dates(fdf, "Period")
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
        # Get frequency normalization settings from sidebar
        freq_norm_settings = {}
        if 'enable_freq_norm' in locals():
            freq_norm_settings = {
                'enable_freq_norm': enable_freq_norm,
                'freq_strategy': freq_strategy if enable_freq_norm else "auto_detect",
                'manual_frequency': manual_frequency if enable_freq_norm and freq_strategy == "manual_select" else "M",
                'show_conversion_admin': show_conversion_admin if enable_freq_norm else False
            }
        else:
            freq_norm_settings = {
                'enable_freq_norm': False,
                'freq_strategy': "auto_detect", 
                'manual_frequency': "M",
                'show_conversion_admin': False
            }
        
        df, series_col, exog_cols, freq_info = read_and_prepare(raw_df, **freq_norm_settings)
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

    # Show frequency normalization summary if used
    if freq_info.get('enabled', False):
        st.subheader("Frequency Normalization Summary")
        target_freq_name = {
            'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly',
            'Q': 'Quarterly', 'Y': 'Yearly'
        }.get(freq_info.get('target_frequency', 'M'), 'Unknown')
        
        st.info(f"All data converted to: **{target_freq_name}** frequency")
        
        if freq_info.get('conversion_impact'):
            with st.expander("View Conversion Impact Details"):
                for series_id, impact in freq_info['conversion_impact'].items():
                    st.write(f"**{series_id}:**")
                    for var, stats in impact.items():
                        data_change = stats.get('data_point_change', 0)
                        mean_change = stats.get('mean_change_pct', 0)
                        st.write(f"  - {var}: {data_change:+d} data points, {mean_change:+.1f}% mean change")

    # Combined export
    if results_frames:
        combined = pd.concat(results_frames, ignore_index=True)
        st.subheader("Export All")
        build_download(combined, "results_all_series.csv", "Download All Results (CSV)")

else:
    st.info("Upload a CSV and click Run Modeling to begin.")


