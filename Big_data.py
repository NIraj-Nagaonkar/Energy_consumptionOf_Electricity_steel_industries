import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os


DATA_FILE = "steel_plant_big.csv"  
MODEL_FILE = "model.pkl"
TARGET = "Electricity_Consumption_MWh"
EPS = 1e-8

SHIFT_SELECTED = "All Shifts"    
YEAR_SELECTED = "All Years"       
BACKTEST_DAYS = 30                



def load_model(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Model file '{filepath}' not found.")
        return None
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_data(filepath):
    print(f"Loading data from {filepath} (This might take a moment for large CSVs)...")
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
        
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    return df


def backtest_accuracy(model, df, days):
    test_df = df.iloc[-days:].copy()
    X_test = test_df[["Production_Tons"]]
    y_test = test_df[TARGET]

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    smape = np.mean(
        2 * np.abs(preds - y_test) /
        (np.abs(y_test) + np.abs(preds) + EPS)
    ) * 100

    accuracy = (1 - mae / np.mean(y_test)) * 100
    tolerance_accuracy = np.mean(np.abs(y_test - preds) / (np.abs(y_test) + EPS) <= 0.05) * 100

    result_df = test_df[["Date", TARGET]].copy()
    result_df["Predicted_Energy_MWh"] = preds.round(2)

    return result_df, mae, r2, smape, accuracy, tolerance_accuracy


def plot_energy_dashboard(df, target_col):
    """Generates a 2x2 presentation dashboard. Groups data to prevent large-file freezing."""
    df_plot = df.copy()
    df_plot['Year'] = df_plot['Date'].dt.year
    df_plot['Quarter'] = 'Q' + df_plot['Date'].dt.quarter.astype(str)

  
    quarterly_trend = df_plot.groupby(['Year', 'Quarter'])[target_col].sum().unstack(fill_value=0)
    yearly_total = df_plot.groupby('Year')[target_col].sum()
    
    years = sorted(df_plot['Year'].unique())
    if len(years) > 1:
        first_year, last_year = years[0], years[-1]
        growth_rates = {
            q: ((quarterly_trend.loc[last_year, q] - quarterly_trend.loc[first_year, q]) / 
                (quarterly_trend.loc[first_year, q] + EPS)) * 100 
            for q in quarterly_trend.columns
        }
        growth_df = pd.Series(growth_rates)
    else:
        growth_df = pd.Series({q: 0 for q in quarterly_trend.columns})


    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Steel Plant Energy Volume Analysis', fontsize=16, fontweight='bold')

    # Top-Left: Trend
    ax1 = axs[0, 0]
    for i, quarter in enumerate(quarterly_trend.columns):
        ax1.plot(quarterly_trend.index, quarterly_trend[quarter], marker='o', linewidth=2, label=quarter)
    ax1.set_title('Quarter-wise Energy Consumption Trend', fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Energy (MWh)')
    ax1.set_xticks(years)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Top-Right: Year Comparison
    ax2 = axs[0, 1]
    x = np.arange(len(years))
    width = 0.2
    for i, quarter in enumerate(quarterly_trend.columns):
        ax2.bar(x + (width * i), quarterly_trend[quarter], width, label=quarter)
    ax2.set_title('Year-wise Comparison by Quarter', fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_xticks(x + width * 1.5, years)
    ax2.legend()

    # Bottom-Left: Yearly Total
    ax3 = axs[1, 0]
    bars = ax3.bar(yearly_total.index.astype(str), yearly_total.values, color=['#f8766d', '#00bfc4', '#00ba38', '#619cff'][:len(years)], edgecolor='black')
    ax3.set_title('Total Yearly Energy Volume', fontweight='bold')
    ax3.set_ylabel('Total Energy (MWh)')
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01, f"{int(bar.get_height())}", ha='center', va='bottom', fontweight='bold')

    # Bottom-Right: Growth
    ax4 = axs[1, 1]
    hbars = ax4.barh(growth_df.index, growth_df.values, color='#fec058', edgecolor='black')
    ax4.set_title(f'Energy Growth Rate ({years[0]}-{years[-1]})', fontweight='bold')
    ax4.set_xlabel('Growth Rate (%)')
    for bar in hbars:
        ax4.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}%", ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig


if __name__ == "__main__":
    print("\n" + "="*50)
    print(" STEEL PLANT ENERGY ANALYTICS ENGINE ")
    print("="*50)

    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        exit()
        
    df = load_data(DATA_FILE)

    # 2. Apply Filters
    if SHIFT_SELECTED != "All Shifts" and "Shift" in df.columns:
        df = df[df["Shift"] == SHIFT_SELECTED]
    if YEAR_SELECTED != "All Years":
        df = df[df["Date"].dt.year == YEAR_SELECTED]

    print(f" Data loaded successfully. Total records: {len(df):,}")
    print("-" * 50)

    # 3 Load Model
    model = load_model(MODEL_FILE)
    if model is None:
        print("Cannot proceed without a valid model file.")
        exit()

    # 4 Model Performance Math
    print("\nCalculating AI Model Performance...")
    X = df[["Production_Tons"]]
    y = df[TARGET]

    split_idx = int(len(df) * 0.8)
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    preds = model.predict(X_test)

    # 5. Print Presentation Metrics
    print("\n--- Model Validation Metrics ---")
    print(f"MAE (MWh):               {mean_absolute_error(y_test, preds):.2f}")
    print(f"R² Score:                {r2_score(y_test, preds):.3f}")
    print(f"Prediction Accuracy:     {(1 - mean_absolute_error(y_test, preds) / np.mean(y_test)) * 100:.2f}%")

    print(f"\n--- Recent Validation (Last {BACKTEST_DAYS} Records) ---")
    bt_df, bt_mae, bt_r2, bt_smape, bt_acc, bt_tol_acc = backtest_accuracy(model, df, BACKTEST_DAYS)
    print(f"Recent Accuracy:         {bt_acc:.2f}%")
    print(f"Within ±5% Tolerance:    {bt_tol_acc:.1f}%")

    # 6. Generate Visuals
    print("\nGenerating Presentation Dashboards... (Please check new windows)")
    
    # Figure 1: The 2x2 Dashboard
    fig1 = plot_energy_dashboard(df, TARGET)
    
    # Figure 2: Model Accuracy (Actual vs Predicted)
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bt_df["Date"], bt_df[TARGET], label="Actual Electricity", marker="o", color="blue")
    ax.plot(bt_df["Date"], bt_df["Predicted_Energy_MWh"], label="AI Prediction", marker="x", color="red", linestyle="--")
    ax.set_title(f"AI Prediction Accuracy (Last {BACKTEST_DAYS} Records)", fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Electricity (MWh)")
    ax.legend()
    fig2.autofmt_xdate(rotation=45)
    plt.tight_layout()

    plt.show()

    # 7. Save output
    output_csv = "predicted_vs_actual_energy.csv"
    bt_df.to_csv(output_csv, index=False)
    print(f"\nSaved predicted values to '{output_csv}' for further review.")