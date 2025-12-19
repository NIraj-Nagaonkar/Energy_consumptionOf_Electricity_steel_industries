import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("steel_plant_2_years_weather.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# ===============================
# FEATURE ENGINEERING (CRITICAL)
# ===============================
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

df["Energy_per_Ton"] = (
    df["Electricity_Consumption_MWh"] / df["Production_Tons"]
)

TARGET = "Electricity_Consumption_MWh"

FEATURES = [
    "Production_Tons",
    "Energy_per_Ton"
]

X = df[FEATURES]
y = df[TARGET]

# ===============================
# TIME SPLIT
# ===============================
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

# ===============================
# TRAIN MODEL
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)

# ===============================
# EVALUATE
# ===============================
preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("R² :", r2_score(y_test, preds))

# ===============================
# SAVE MODEL
# ===============================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved")
