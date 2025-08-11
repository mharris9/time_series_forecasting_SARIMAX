# SARIMAX Time Series Forecasting - User Manual

## Table of Contents
1. [Getting Started](#getting-started)
2. [Data Requirements](#data-requirements)
3. [UI Parameters Guide](#ui-parameters-guide)
4. [Model Parameters Explained](#model-parameters-explained)
5. [Model Outputs & Interpretation](#model-outputs--interpretation)
6. [Parameter Selection Guidelines](#parameter-selection-guidelines)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Getting Started

### Installation & Setup
1. Run `update_deps.bat` to install dependencies
2. Run `run_app.bat` to start the application
3. Open your browser to the displayed Streamlit URL (typically `http://localhost:8501`)

### Quick Start
1. Upload a CSV file with columns: `Period` (YYYY-MM), `Value`, and optional series ID/exogenous variables
2. Configure preprocessing and model settings in the sidebar
3. Click "Run Modeling" to generate forecasts

---

## Data Requirements

### Required Columns
- **Period**: Date in YYYY-MM format (e.g., 2023-01, 2023-02)
- **Value**: Numeric time series values to forecast

### Optional Columns
- **Series ID**: Identifier for multiple time series (if missing, app creates default "Series_1")
- **Exogenous Variables**: Additional numeric predictors (e.g., temperature, price, marketing spend)

### Data Format Example
```csv
Period,Value,Region,Temperature,Marketing
2023-01,1250,North,15.2,5000
2023-02,1340,North,18.1,5200
2023-03,1420,North,22.3,4800
```

### Future Exogenous Data
If using exogenous variables, upload a second CSV with future values for out-of-sample forecasting:
```csv
Period,Region,Temperature,Marketing
2024-01,North,16.0,5500
2024-02,North,19.2,5300
```

---

## UI Parameters Guide

### 1. Data Section

#### **Column Mapping**
- **Series ID column**: Select which column identifies different time series
- **Exogenous columns**: Choose additional predictor variables

#### **File Uploads**
- **Main CSV**: Your historical time series data
- **Future Exogenous CSV**: Future values for predictor variables (required for out-of-sample forecasting with exogenous variables)

### 2. Preprocessing Section

#### **Missing Value Handling**
- **ffill_then_interpolate** (Recommended): Forward fill missing values, then use linear interpolation
- **interpolate_only**: Use only linear interpolation
- **ffill_only**: Use only forward filling

#### **Standardize Exogenous Variables**
- ✅ **Checked** (Recommended): Scales exogenous variables to have mean=0, std=1
- ❌ **Unchecked**: Uses raw exogenous values

### 3. Model Section

#### **Auto SARIMAX Search**
- ✅ **Enabled**: Automatically finds best parameters via grid search
- ❌ **Disabled**: Use manual parameter specification

#### **Selection Criterion**
- **AIC** (Akaike Information Criterion): Balances model fit and complexity
- **BIC** (Bayesian Information Criterion): More conservative, penalizes complexity more heavily

#### **Model Constraints**
- **Enforce Stationarity**: Ensures model parameters create a stationary process
- **Enforce Invertibility**: Ensures moving average parameters are invertible

#### **Forecast Settings**
- **Seasonal Period (s)**: Number of periods in seasonal cycle (12 for monthly data)
- **Forecast Horizon**: Number of periods to forecast into the future

---

## Model Parameters Explained

### SARIMAX Model Structure: (p,d,q)(P,D,Q)s

#### **Non-Seasonal Parameters**
- **p (AR order)**: Number of lagged observations in the model
  - Higher p captures more complex autoregressive patterns
  - Range: 0-3 (typically)
  
- **d (Differencing)**: Degree of differencing to make series stationary
  - d=0: No differencing (series is already stationary)
  - d=1: First differencing (removes trend)
  - d=2: Second differencing (removes quadratic trend)
  
- **q (MA order)**: Number of lagged forecast errors
  - Higher q captures more complex error patterns
  - Range: 0-3 (typically)

#### **Seasonal Parameters**
- **P (Seasonal AR)**: Seasonal autoregressive order
  - Captures seasonal patterns in the data
  - Range: 0-2 (typically)
  
- **D (Seasonal Differencing)**: Seasonal differencing degree
  - D=0: No seasonal differencing
  - D=1: Remove seasonal trends
  
- **Q (Seasonal MA)**: Seasonal moving average order
  - Captures seasonal error patterns
  - Range: 0-2 (typically)
  
- **s (Seasonal Period)**: Length of seasonal cycle
  - Monthly data: s=12
  - Quarterly data: s=4
  - Daily data: s=7 or s=365

#### **Exogenous Variables (X)**
External factors that influence your time series:
- Economic indicators
- Weather data
- Marketing campaigns
- Price changes

---

## Model Outputs & Interpretation

### 1. Model Summary
- **Selected Parameters**: Optimal (p,d,q)(P,D,Q)s configuration
- **Information Criterion**: AIC/BIC value (lower is better)

### 2. Forecast Plot
- **Actual (Blue)**: Historical observed values
- **Fitted (Green, dashed)**: Model's fit to training data
- **Forecast Test (Orange)**: Predictions on held-out test data
- **Forecast Future (Red)**: Out-of-sample predictions
- **Confidence Intervals**: Uncertainty bands around forecasts

### 3. Performance Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error

### 4. Residual Diagnostics

#### **Residuals Plot**
- Should appear random with no clear patterns
- Constant variance over time

#### **Histogram**
- Should approximate normal distribution
- KDE (Kernel Density Estimate) overlay shows distribution shape

#### **ACF (Autocorrelation Function)**
- Measures correlation between residuals and their lags
- Should be close to zero for all lags (good model)

#### **PACF (Partial Autocorrelation Function)**
- Measures direct correlation between residuals and their lags
- Should be close to zero for all lags

#### **Ljung-Box Test**
- Tests for autocorrelation in residuals
- p-value > 0.05 indicates no significant autocorrelation (good)
- p-value ≤ 0.05 suggests model may need adjustment

---

## Parameter Selection Guidelines

### Automatic vs Manual Selection

#### **Use Automatic Search When:**
- You're new to time series modeling
- You have sufficient data (>50 observations)
- You want to explore different parameter combinations
- Computational time is not a constraint

#### **Use Manual Selection When:**
- You have domain knowledge about the data
- Working with limited data
- Need faster results
- Have specific model requirements

### Data-Driven Parameter Guidelines

#### **Seasonal Period (s)**
- **Monthly data**: s=12
- **Quarterly data**: s=4
- **Weekly data**: s=52
- **Daily data**: s=7 (weekly) or s=365 (yearly)

#### **Differencing (d, D)**
- **Check stationarity first**:
  - Trending data usually needs d=1
  - Seasonal trends may need D=1
  - Over-differencing can hurt performance

#### **AR/MA Orders (p, q, P, Q)**
- **Start simple**: Try (1,1,1)(1,1,1)s
- **Examine ACF/PACF plots**:
  - ACF cuts off at lag q → suggests MA(q)
  - PACF cuts off at lag p → suggests AR(p)
- **Keep orders low**: Higher orders risk overfitting

### Search Ranges (Auto Mode)
- **Conservative**: p,q ≤ 2, P,Q ≤ 1
- **Moderate**: p,q ≤ 3, P,Q ≤ 2 (default)
- **Extensive**: p,q ≤ 5, P,Q ≤ 3 (slow but thorough)

### Exogenous Variables

#### **When to Include:**
- Strong theoretical relationship with target variable
- Variables are available for future periods
- Correlation analysis shows significant relationships

#### **When to Exclude:**
- Variables are not available for forecasting horizon
- Weak or spurious correlations
- Variables that are outcomes rather than drivers

---

## Troubleshooting

### Common Issues & Solutions

#### **"Grid search failed" Error**
- **Cause**: No parameter combination converged
- **Solutions**:
  - Reduce search ranges (lower max p,q,P,Q)
  - Check data for outliers or missing values
  - Try manual parameter selection
  - Ensure sufficient data (>30 observations)

#### **Poor Forecast Performance**
- **High MAPE/RMSE**:
  - Try different parameter combinations
  - Add relevant exogenous variables
  - Check for structural breaks in data
  - Consider data transformations (log, Box-Cox)

#### **Residual Diagnostics Show Patterns**
- **Non-random residuals**:
  - Increase AR/MA orders
  - Add seasonal components
  - Include missing exogenous variables
  - Check for outliers

#### **Confidence Intervals Too Wide**
- **High uncertainty**:
  - Add more historical data
  - Include relevant exogenous variables
  - Reduce forecast horizon
  - Consider ensemble methods

#### **Model Takes Too Long**
- **Performance issues**:
  - Reduce search ranges
  - Use fewer exogenous variables
  - Decrease data frequency (monthly vs daily)
  - Switch to manual parameter selection

### Data Quality Checks

#### **Before Modeling:**
1. **Completeness**: Check for missing periods
2. **Outliers**: Identify and handle extreme values
3. **Seasonality**: Confirm seasonal patterns match period setting
4. **Stationarity**: Verify data characteristics

#### **Exogenous Variables:**
1. **Future Availability**: Ensure variables are available for forecasting
2. **Correlation**: Check relationships with target variable
3. **Multicollinearity**: Avoid highly correlated predictors
4. **Stationarity**: May need differencing like target variable

---

## Best Practices

### Data Preparation
1. **Clean Data**: Handle outliers and missing values appropriately
2. **Consistent Frequency**: Ensure regular time intervals
3. **Sufficient History**: Use at least 2-3 seasonal cycles for seasonal data
4. **Feature Engineering**: Create relevant derived variables

### Model Selection
1. **Start Simple**: Begin with basic ARIMA before adding complexity
2. **Cross-Validation**: Use time series cross-validation for model selection
3. **Multiple Models**: Compare different approaches
4. **Domain Knowledge**: Incorporate business understanding

### Forecasting
1. **Validate Assumptions**: Check residual diagnostics
2. **Monitor Performance**: Track forecast accuracy over time
3. **Update Regularly**: Retrain models with new data
4. **Communicate Uncertainty**: Always show confidence intervals

### Model Interpretation
1. **Understand Components**: Separate trend, seasonal, and irregular components
2. **Scenario Analysis**: Test different exogenous variable scenarios
3. **Sensitivity Analysis**: Understand parameter impact
4. **Business Context**: Relate results to business decisions

### Deployment Considerations
1. **Data Pipeline**: Automate data collection and preprocessing
2. **Model Monitoring**: Track model performance degradation
3. **Retraining Schedule**: Establish regular model updates
4. **Backup Models**: Maintain alternative approaches

---

## Advanced Tips

### For Experienced Users

#### **Model Diagnostics**
- Use information criteria (AIC/BIC) for model comparison
- Examine residual ACF/PACF for model adequacy
- Perform Ljung-Box tests for residual autocorrelation
- Check forecast accuracy on multiple holdout periods

#### **Parameter Tuning**
- Consider seasonal decomposition before modeling
- Use Box-Cox transformations for variance stabilization
- Experiment with different exogenous variable lags
- Try ensemble methods combining multiple models

#### **Advanced Features**
- State space representation for missing data handling
- Intervention analysis for structural breaks
- Transfer function models for complex exogenous relationships
- Regime-switching models for changing relationships

Remember: The best model is one that provides actionable insights and reliable forecasts for your specific business context. Always validate results against domain knowledge and business requirements.
