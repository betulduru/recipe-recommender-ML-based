import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# 1. Load and scale data
df = pd.read_csv("recipes.csv")
features = ["calories", "protein", "fat", "carbs"]
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# 2. Train KMeans
model = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(df_scaled)

# 3. User liked recipes
liked_titles = ["Pasta With Tuna", "Tomato and lentil soup", "Chicken Lo Mein"]

liked_indices = []
for title in liked_titles:
    matches = df[df["title"].str.lower() == title.lower()]
    if not matches.empty:
        liked_indices.append(matches.index[0])

print(f"You liked {len(liked_indices)} recipes:")
for i in liked_indices:
    print(f"  - {df.loc[i, 'title']} (cluster {df.loc[i, 'cluster']})")

# 4. Build user profile vector
user_profile = df_scaled.iloc[liked_indices].mean().values

# 5. Find user's cluster
user_cluster = model.predict(pd.DataFrame([user_profile], columns=features))[0]
print(f"\nYour taste profile belongs to cluster {user_cluster}")

# 6. Recommend from same cluster only
cluster_df = df[df["cluster"] == user_cluster].copy()
cluster_scaled = df_scaled.loc[cluster_df.index]

distances = np.linalg.norm(cluster_scaled.values - user_profile, axis=1)
cluster_df["distance"] = distances
cluster_df = cluster_df[~cluster_df.index.isin(liked_indices)]
cluster_df = cluster_df.sort_values("distance")

print(f"\n🍽️  Recommended for you (from cluster {user_cluster}):")
print("-" * 60)
for count, (_, row) in enumerate(cluster_df.head(5).iterrows()):
    print(f"{count+1}. {row['title']}")
    print(f"   Calories: {row['calories']} | Protein: {row['protein']}g | Fat: {row['fat']}g | Carbs: {row['carbs']}g")
    print(f"   Distance: {row['distance']:.4f}")