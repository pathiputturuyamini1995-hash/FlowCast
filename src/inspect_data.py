import pandas as pd
from pathlib import Path

# Find the FlowCast project and dataset
project_dir = Path(__file__).resolve().parent.parent
data_file = project_dir / "data" / "traffic_sensor_log.csv"

print("Loading FlowCast traffic dataset...")

df = pd.read_csv(data_file)

print("\n--- DATASET SUMMARY ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())