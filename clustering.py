import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# 1. Load and scale data
df = pd.read_csv("recipes.csv")
features = ["calories", "protein", "fat", "carbs"]
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# 2. Train KMeans with K=4
model = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(df_scaled)

# 3. Describe each cluster
print("Cluster profiles:\n")
for c in range(4):
    group = df[df["cluster"] == c]
    print(f"Cluster {c} ({len(group)} recipes)")
    print(f"  Avg calories : {group['calories'].mean():.0f}")
    print(f"  Avg protein  : {group['protein'].mean():.1f}g")
    print(f"  Avg fat      : {group['fat'].mean():.1f}g")
    print(f"  Avg carbs    : {group['carbs'].mean():.1f}g")
    print(f"  Examples     : {list(group['title'].head(3).values)}\n")

# 4. Save updated dataset
df.to_csv("recipes.csv", index=False)
print("recipes.csv updated with cluster column")