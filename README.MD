AirPassengers Time Series Forecasting 

This project implements a complete classical time series forecasting pipeline on 
the famous AirPassengers dataset (monthly airline passenger counts from 1949–1960).

## Project Highlights

- Built a complete **time series forecasting pipeline** on the classic AirPassengers 
dataset (monthly airline passenger counts, 1949–1960). 
- Performed **seasonal decomposition** to separate trend, seasonality, and residuals, 
revealing strong yearly patterns and an increasing long‑term trend. 
- Implemented multiple **baseline models** (naive, seasonal naive, 12‑month moving average)
with a proper **time‑based train–test split**.
- Applied **Holt‑Winters exponential smoothing** (additive & multiplicative) and achieved 
significantly lower MAPE compared to all baselines. 
- Visualized train, test, and forecasts together for clear comparison 
and interpretation of model performance over time. 


Source: AirPassengers CSV (Month, #Passengers).

Frequency: Monthly (MS – month start).

Target: Number of airline passengers per month.

Objectives:

Explore the temporal structure of the series (trend, seasonality, residuals).

Build and evaluate baseline forecasting models.

Apply Holt‑Winters exponential smoothing to model trend and seasonality.

Compare models using a proper time‑based train–test split and error metrics.

 Methods and Workflow
Data Preparation

Loaded AirPassengers.csv with Pandas.

Converted Month to datetime, set it as index, and enforced monthly frequency.

Split data into:

Train: 1949‑01 to 1958‑12

Test: 1959‑01 to 1960‑12

Time Series Decomposition

Used seasonal_decompose to perform:

Multiplicative decomposition (works well due to increasing variance).

Optional additive decomposition for comparison.

Visualized Observed, Trend, Seasonality, and Residuals components.

Baseline Forecasting Models

Implemented three classic baselines:

Naive forecast – repeats the last observed value.

Seasonal naive forecast – uses the value from the same month one year earlier.

12‑month moving average – smooths the last 12 months and uses that as the forecast.

Evaluated each baseline using MAPE (Mean Absolute Percentage Error).

Holt‑Winters (Triple Exponential Smoothing)

Built two Holt‑Winters models using statsmodels:

Additive: level + trend + additive seasonality.

Multiplicative: level + trend + multiplicative seasonality.

Forecasted the test period and evaluated using MAPE.

Plotted train vs test vs forecasts for visual comparison.

Results (MAPE on Test Set)
Naive: 23.58%

Seasonal Naive: 42.10%

Moving Average (12M): 15.47%

Holt‑Winters Additive: 6.64%

Holt‑Winters Multiplicative: 6.39%

Holt‑Winters models, especially the multiplicative one, significantly outperform the simple baselines by explicitly modeling both trend and strong yearly seasonality in passenger counts.


Key Takeaways:
Proper time‑based splitting is crucial for realistic evaluation in time series.

Simple baselines are essential reference points; they are easy to implement but can miss complex seasonal patterns.

Holt‑Winters exponential smoothing is a powerful classical method that can achieve low error on well‑behaved seasonal data like AirPassengers.

This project demonstrates a full, interpretable pipeline without using ARIMA/SARIMA or deep learning, making it ideal for learning and explaining core time‑series concepts.
