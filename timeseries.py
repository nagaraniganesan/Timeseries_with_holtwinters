import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ===================== 1. Load and prepare data =====================

df = pd.read_csv("AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df = df.set_index("Month").asfreq("MS").sort_index()

ts = df["#Passengers"]

print("--- Head of series ---")
print(ts.head())
print("\nSeries info:")
print(ts.describe())

# ===================== 2. Decomposition =====================

#  multiplicative (works well here)
result_mul = seasonal_decompose(ts, model="multiplicative", period=12)

plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(result_mul.observed)
plt.title("Observed")

plt.subplot(4, 1, 2)
plt.plot(result_mul.trend)
plt.title("Trend")

plt.subplot(4, 1, 3)
plt.plot(result_mul.seasonal)
plt.title("Seasonality")

plt.subplot(4, 1, 4)
plt.plot(result_mul.resid)
plt.title("Residuals")

plt.tight_layout()
plt.show()

#  also check additive model for comparison
result_add = seasonal_decompose(ts, model="additive", period=12)
#we can repeat similar plots for additive if needed

# ===================== 3. Train–test split by time =====================

# Using last 24 months as test set
train = ts[:-24]
test = ts[-24:]

print("\nTrain period:", train.index.min(), "to", train.index.max())
print("Test period :", test.index.min(), "to", test.index.max())

# Helper metric
def mape(y_true, y_pred):
    return (np.abs((y_true - y_pred) / y_true).mean()) * 100

# ===================== 4. Baseline models =====================

# 4.1 Naive forecast
naive_forecast = test.copy()
naive_forecast[:] = train.iloc[-1]

naive_mape = mape(test, naive_forecast)
print("\nNaive MAPE:", naive_mape)

# 4.2 Seasonal naive (same month last year)
seasonal_naive = test.copy()
for t in range(len(test)):
    seasonal_naive.iloc[t] = train.iloc[-12 + t]

seasonal_mape = mape(test, seasonal_naive)
print("Seasonal naive MAPE:", seasonal_mape)

# 4.3 Moving average baseline (12-month rolling mean)
window = 12
moving_avg_forecast = test.copy()
last_train_ma = train.rolling(window).mean().iloc[-1]

moving_avg_forecast[:] = last_train_ma

ma_mape = mape(test, moving_avg_forecast)
print("Moving average (12-month) MAPE:", ma_mape)

# ===================== 5. Holt-Winters models =====================

# 5.1 Holt-Winters additive
hw_add = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=12
).fit()

hw_add_forecast = hw_add.forecast(len(test))
mape_hw_add = mape(test, hw_add_forecast)
print("\nHolt-Winters additive MAPE:", mape_hw_add)

# 5.2 Holt-Winters multiplicative
hw_mul = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="mul",
    seasonal_periods=12
).fit()

hw_mul_forecast = hw_mul.forecast(len(test))
mape_hw_mul = mape(test, hw_mul_forecast)
print("Holt-Winters multiplicative MAPE:", mape_hw_mul)

# ===================== 6. Plot forecasts vs actual =====================

plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test", color="black")

plt.plot(test.index, seasonal_naive, label="Seasonal Naive", color="red")
plt.plot(test.index, hw_add_forecast, label="HW Additive", color="green")
plt.plot(test.index, hw_mul_forecast, label="HW Multiplicative", color="orange")

plt.title("AirPassengers: Baselines vs Holt-Winters Forecasts")
plt.xlabel("Date")
plt.ylabel("#Passengers")
plt.legend()
plt.tight_layout()
plt.show()

# ===================== 7. Summary printout =====================

print("\n=== MAPE Summary ===")
print(f"Naive:                 {naive_mape:.2f}")
print(f"Seasonal Naive:        {seasonal_mape:.2f}")
print(f"Moving Average (12M):  {ma_mape:.2f}")
print(f"Holt-Winters Additive: {mape_hw_add:.2f}")
print(f"Holt-Winters Multipl.: {mape_hw_mul:.2f}")
