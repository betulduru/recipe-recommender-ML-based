import pandas as pd

df = pd.read_csv("recipes.csv")

print("=== SHAPE ===")
print(df.shape)

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== BASIC STATS ===")
print(df.describe())

print("\n=== ANY MISSING VALUES? ===")
print(df.isnull().sum())