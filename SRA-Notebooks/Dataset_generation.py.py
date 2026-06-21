"""
=========================================================
SIMULATION DATASET GENERATOR
Single Stress-Based Simulation Dataset
=========================================================

Purpose:
Creates ONE simulation dataset for:
- stress testing
- robustness evaluation
- model behavior analysis

Derived from:
master_v6.csv

Output:
D_simulation.csv
=========================================================
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# =========================================================
# LOAD MASTER DATASET
# =========================================================

MASTER_PATH = r"C:\Users\mmuku\Downloads\master_v6.csv"

df = pd.read_csv(MASTER_PATH)

# =========================================================
# OUTPUT DIRECTORY
# Saves into Downloads/simulation_dataset
# =========================================================

OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "simulation_dataset"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# CREATE SIMULATION DATASET
# =========================================================

simulation_df = df.sample(
    3000,
    random_state=42
).copy()

# =========================================================
# APPLY STRESS CONDITIONS
# =========================================================

# Reduce financial aid
simulation_df["aid_amount"] *= np.random.uniform(
    0.4,
    0.8,
    len(simulation_df)
)

# Increase debt burden
simulation_df["synthetic_debt_load"] *= np.random.uniform(
    1.1,
    1.5,
    len(simulation_df)
)

# Increase income volatility
simulation_df["cash_flow_cv"] = np.clip(
    simulation_df["cash_flow_cv"]
    + np.random.normal(0.12, 0.05, len(simulation_df)),
    0,
    1
)

# Reduce payment punctuality
simulation_df["bill_payment_rate"] = np.clip(
    simulation_df["bill_payment_rate"]
    - np.random.normal(0.08, 0.04, len(simulation_df)),
    0,
    1
)

# Increase delinquency probability
simulation_df["prior_delinquency"] = np.random.choice(
    [0, 1, 2],
    p=[0.60, 0.30, 0.10],
    size=len(simulation_df)
)

# Increase late-semester pressure
simulation_df["semester_phase"] = np.random.choice(
    ["Mid", "Late"],
    p=[0.35, 0.65],
    size=len(simulation_df)
)

# =========================================================
# RECOMPUTE ENGINEERED FEATURES
# =========================================================

simulation_df["cfri"] = (
    1 - simulation_df["cash_flow_cv"]
)

simulation_df["opr"] = (
    simulation_df["bill_payment_rate"]
)

simulation_df["acr"] = (
    simulation_df["aid_amount"]
    + (simulation_df["monthly_inflow_mean"] * 4)
) / (
    simulation_df["estimated_tuition"]
    + simulation_df["estimated_living_cost"]
)

gdi_map = {
    "Improving": 1,
    "Stable": 0,
    "Declining": -1
}

simulation_df["gdi"] = (
    simulation_df["gpa_direction"].map(gdi_map)
)

simulation_df["ecs"] = (
    simulation_df["semesters_enrolled"]
    / simulation_df["expected_total_semesters"]
)

simulation_df["tgb"] = (
    np.log1p(simulation_df["remaining_semesters"])
    /
    np.log1p(simulation_df["synthetic_debt_load"])
)

# =========================================================
# RECOMPUTE DEFAULT PROBABILITY
# =========================================================

risk_score = (
    -1.5 * simulation_df["ecs"]
    -0.8 * simulation_df["gdi"]
    -1.2 * simulation_df["acr"]
    -2.5 * simulation_df["cfri"]
    -3.0 * simulation_df["opr"]
    +1.8 * simulation_df["tgb"]
    +1.5 * simulation_df["prior_delinquency"]
)

# Add noise
risk_score += np.random.normal(
    0,
    0.5,
    len(simulation_df)
)

# Sigmoid probability
prob = 1 / (1 + np.exp(-risk_score))

# Scale close to target default rate
scale = 0.035 / prob.mean()

prob = np.clip(
    prob * scale,
    0.001,
    0.95
)

simulation_df["default_probability"] = prob

simulation_df["default_flag"] = (
    np.random.rand(len(simulation_df)) < prob
).astype(int)

# =========================================================
# SAVE DATASET
# =========================================================

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "D_simulation.csv"
)

simulation_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# =========================================================
# VALIDATION OUTPUT
# =========================================================

print("\n================================================")
print("SIMULATION DATASET GENERATED")
print("================================================")

print("\nDataset Shape:")
print(simulation_df.shape)

print("\nDefault Rate:")
print(round(
    simulation_df["default_flag"].mean(),
    4
))

print("\nMissing Values:")
print(simulation_df.isnull().sum().sum())

print("\nSaved To:")
print(OUTPUT_PATH)

print("\n================================================")
print("COMPLETED SUCCESSFULLY")
print("================================================")